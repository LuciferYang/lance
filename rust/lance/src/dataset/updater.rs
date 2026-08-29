// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::array::{ArrayData, ArrayDataBuilder};
use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array, cast::AsArray, make_array};
use arrow_buffer::{ArrowNativeType, MutableBuffer};
use arrow_schema::{DataType, Field};
use futures::StreamExt;
use lance_arrow::blank::minimal_value;
use lance_arrow::interleave_batches;
use lance_core::datatypes::{OnMissing, OnTypeMismatch};
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result, datatypes::Schema};
use lance_table::format::{DataFile, Fragment};
use lance_table::utils::stream::ReadBatchFutStream;
use std::sync::Arc;

use super::Dataset;
use super::fragment::FragmentReader;
use super::scanner::get_default_batch_size;
use super::versions;
use super::write::{GenericWriter, cleanup_data_fragments};
use crate::dataset::FileFragment;
use crate::dataset::utils::SchemaAdapter;

/// Update or insert a new column.
///
/// To use, call [`Updater::next`] to get the next [`RecordBatch`] as input,
/// then call [`Updater::update`] to update the batch. Repeat until
/// [`Updater::next`] returns `None`.
///
/// `write_schema` dictates the schema of the new file, while `final_schema` is
/// the schema of the full fragment after the update. These are optional and if
/// not specified, the updater will infer the write schema from the first batch
/// of results and will append them to the current schema to get the final schema.
pub struct Updater {
    fragment: FileFragment,

    /// The reader over the [`Fragment`]
    input_stream: ReadBatchFutStream,

    /// The last batch read from the file, with deleted rows removed
    last_input: Option<RecordBatch>,

    writer: Option<Box<dyn GenericWriter>>,

    /// The final schema of the fragment after the update.
    final_schema: Option<Schema>,

    /// The schema the new files will be written in. This only contains new columns.
    write_schema: Option<Schema>,

    /// The adapter to convert the logical data to physical data.
    schema_adapter: Option<SchemaAdapter>,

    allow_external_blob_outside_bases: bool,

    finished: bool,

    deletion_restorer: DeletionRestorer,
}

impl Updater {
    /// Create a new updater with source reader, and destination writer.
    ///
    /// The `schemas` parameter is a tuple of the write schema (just the new fields)
    /// and the final schema (all the fields).
    ///
    /// If the schemas are not known, they can be None and will be inferred from
    /// the first batch of results.
    pub(super) async fn try_new(
        fragment: FileFragment,
        reader: FragmentReader,
        deletion_vector: DeletionVector,
        schemas: Option<(Schema, Schema)>,
        batch_size: Option<u32>,
    ) -> Result<Self> {
        let (write_schema, final_schema) = if let Some((write_schema, final_schema)) = schemas {
            (Some(write_schema), Some(final_schema))
        } else {
            (None, None)
        };

        let storage_version = fragment
            .dataset()
            .manifest()
            .data_storage_format
            .lance_file_format();
        let legacy_batch_size =
            versions::row_group_size_for_rewrite(storage_version, &fragment).await?;

        let batch_size = match (&legacy_batch_size, batch_size) {
            // If this is a v1 dataset we must use the row group size of the file
            (Some(legacy_batch_size), _) => *legacy_batch_size,
            // If this is a v2 dataset, let the user pick the batch size
            (None, Some(user_specified_batch_size)) => user_specified_batch_size,
            // Otherwise, default to 1024 if the user didn't specify anything
            (None, None) => get_default_batch_size().unwrap_or(1024) as u32,
        };

        let input_stream = reader.read_all(batch_size).await?;

        Ok(Self {
            fragment,
            input_stream,
            last_input: None,
            writer: None,
            write_schema,
            final_schema,
            // The schema adapter needs the data schema, not the logical schema, so it can't be
            // created until after the first batch is read.
            schema_adapter: None,
            allow_external_blob_outside_bases: false,
            finished: false,
            deletion_restorer: DeletionRestorer::new(deletion_vector, legacy_batch_size),
        })
    }

    pub fn fragment(&self) -> &FileFragment {
        &self.fragment
    }

    pub fn dataset(&self) -> &Dataset {
        self.fragment.dataset()
    }

    /// Returns the next [`RecordBatch`] as input for updater.
    pub async fn next(&mut self) -> Result<Option<&RecordBatch>> {
        if self.finished {
            return Ok(None);
        }
        let batch = self.input_stream.next().await;
        match batch {
            None => {
                if !self.deletion_restorer.is_exhausted() {
                    // This can happen only if there is a batch size (e.g. v1 file) and the
                    // last batch(es) are entirely deleted.
                    return Err(Error::not_supported_source("Missing too many rows in merge, run compaction to materialize deletions first".into()));
                }
                self.finished = true;
                Ok(None)
            }
            Some(batch) => {
                self.last_input = Some(batch.await?);
                Ok(self.last_input.as_ref())
            }
        }
    }

    /// Create a new Writer for new columns.
    ///
    /// After it is called, this Fragment contains the metadata of the new DataFile,
    /// containing the columns, even the data has not written yet.
    ///
    /// It is the caller's responsibility to close the [`FileWriter`].
    ///
    /// Internal use only.
    async fn new_writer(&mut self, schema: Schema) -> Result<Box<dyn GenericWriter>> {
        let data_storage_version = self
            .dataset()
            .manifest()
            .data_storage_format
            .lance_file_format();

        versions::open_update_writer(
            data_storage_version,
            self.dataset(),
            &schema,
            self.allow_external_blob_outside_bases,
        )
        .await
    }

    /// Allow trusted existing external blob references to pass through an update rewrite.
    /// Callers must separately validate any newly supplied references before writing.
    pub(super) fn allow_external_blob_outside_bases(&mut self) {
        self.allow_external_blob_outside_bases = true;
    }

    /// Update one batch.
    pub async fn update(&mut self, batch: RecordBatch) -> Result<()> {
        let Some(last) = self.last_input.as_ref() else {
            return Err(Error::invalid_input(
                "Fragment Updater: no input data is available before update".to_string(),
            ));
        };

        if last.num_rows() != batch.num_rows() {
            return Err(Error::invalid_input(format!(
                "Fragment Updater: new batch has different size with the source batch: {} != {}",
                last.num_rows(),
                batch.num_rows()
            )));
        };

        // Add back in deleted rows
        let batch = self.deletion_restorer.restore(batch)?;

        if self.writer.is_none() {
            if self.write_schema.is_none() {
                // Need to infer the schema.
                let output_schema = batch.schema();
                let mut final_schema = self.fragment.schema().merge(output_schema.as_ref())?;
                final_schema.set_field_id(Some(self.fragment.dataset().manifest.max_field_id()));
                self.final_schema = Some(final_schema);
                self.final_schema.as_ref().unwrap().validate()?;
                self.write_schema = Some(self.final_schema.as_ref().unwrap().project_by_schema(
                    output_schema.as_ref(),
                    OnMissing::Error,
                    OnTypeMismatch::Error,
                )?);
            }

            self.writer = Some(
                self.new_writer(self.write_schema.as_ref().unwrap().clone())
                    .await?,
            );
        }

        let schema_adapter = if let Some(schema_adapter) = self.schema_adapter.as_ref() {
            schema_adapter
        } else {
            self.schema_adapter = Some(SchemaAdapter::new(batch.schema()));
            self.schema_adapter.as_ref().unwrap()
        };

        let batch = schema_adapter.to_physical_batch(batch)?;

        let writer = self.writer.as_mut().unwrap();

        writer.write(&[batch]).await?;

        Ok(())
    }

    /// Finish updating this fragment, and returns the updated [`Fragment`].
    pub async fn finish(&mut self) -> Result<Fragment> {
        if let Some(writer) = self.writer.as_mut() {
            let (_, data_file) = writer.finish().await?;
            self.fragment.metadata.files.push(data_file);
        }

        Ok(self.fragment.metadata().clone())
    }

    /// Clean up any data file and blob sidecars created by the current unfinished writer.
    pub(super) async fn cleanup_unfinished_writer(&mut self) {
        let Some(writer) = self.writer.take() else {
            return;
        };
        let (path, base_id) = writer.data_file_path();
        let path = path.to_string();
        drop(writer);

        if path.is_empty() {
            return;
        }

        let mut fragment = Fragment::new(self.fragment.id() as u64);
        let storage_version = self
            .dataset()
            .manifest()
            .data_storage_format
            .lance_file_format();
        // cleanup_data_fragments only needs path/base_id to remove the unfinished
        // data file and any blob sidecars. Build a minimal synthetic fragment so
        // we can reuse the shared cleanup path without fabricating full metadata.
        fragment.files.push(DataFile::new(
            path,
            vec![],
            vec![],
            storage_version,
            None,
            base_id,
        ));
        cleanup_data_fragments(
            &self.dataset().object_store,
            &self.dataset().base,
            None,
            &[fragment],
        )
        .await;
    }

    /// Get the final schema of the fragment after the update.
    ///
    /// This may be None if the schema is not known. This can happen if it was
    /// not specified up front and the first batch of results has not yet been
    /// processed.
    pub fn schema(&self) -> Option<&Schema> {
        self.final_schema.as_ref()
    }
}

/// Restores deleted rows.
///
/// All data files in a fragment must have the same # of rows (including deleted rows)
/// When we run the update process the next/update methods don't actually calculate on
/// deleted rows.  This means the updated batches will have fewer rows than the original
/// data files.  This struct restores the deleted rows, inserting arbitrary values into the
/// batches where the deleted rows should be.
///
/// To do this we scan through the deletion vector in sorted order, merging deleted rows
/// in as appropriate.
struct DeletionRestorer {
    current_row_id: u32,

    /// Number of rows in each batch, only used in legacy files for validation
    legacy_batch_size: Option<u32>,

    deletion_vector_iter: Option<Box<dyn Iterator<Item = u32> + Send>>,

    last_deleted_row_id: Option<u32>,
}

impl DeletionRestorer {
    fn new(deletion_vector: DeletionVector, legacy_batch_size: Option<u32>) -> Self {
        Self {
            current_row_id: 0,
            legacy_batch_size,
            deletion_vector_iter: Some(deletion_vector.into_sorted_iter()),
            last_deleted_row_id: None,
        }
    }

    fn is_exhausted(&self) -> bool {
        self.deletion_vector_iter.is_none()
    }

    fn is_full(batch_size: Option<u32>, num_rows: u32) -> bool {
        if let Some(legacy_batch_size) = batch_size {
            // We should never encounter the case that `batch_size < num_rows` because
            // that would mean we have a v1 writer and it generated a batch with more rows
            // than expected
            debug_assert!(legacy_batch_size >= num_rows);
            legacy_batch_size == num_rows
        } else {
            false
        }
    }

    /// Given a batch of `num_rows`, walk through the deletion vector, and figure out where blanks
    /// should be inserted.
    ///
    /// For example, if self.current_row_id is 10 and the deletion vector is [11, 12, 19, 25] and
    /// num_rows is 7 then this function will at least return [1, 2] and the batch will at least
    /// span row ids 10..18.
    ///
    /// Then, in the example we need to choose whether the returned batch should include
    /// row 19 (and have 10 rows) or not (and have 9 rows).  This is only a concern in v1 files
    /// where we want to match the original row group size (which is the batch size).  If the
    /// batch size is 9 then we do not include 19 and return as above.
    ///
    /// If the batch size is 10 (or unset) then we do include 19 and the return will be [1, 2, 9]
    ///
    /// In v2 files, since the batch size will be unset, we will always include as many deleted
    /// rows at the end as we can.
    fn deleted_batch_offsets_in_range(&mut self, mut num_rows: u32) -> Vec<u32> {
        let mut deleted = Vec::new();
        let first_row_id = self.current_row_id;
        // The last row id (exclusive) in the batch
        let mut last_row_id = first_row_id + num_rows;
        // If there are zero deleted rows then the range covered will be first_row_id..last_row_id
        if self.deletion_vector_iter.is_none() {
            return deleted;
        }
        let deletion_vector_iter = self.deletion_vector_iter.as_mut().unwrap();

        // Now we need to walk through our deletion vector and figure out where to insert blanks
        let mut next_deleted_id = if self.last_deleted_row_id.is_some() {
            self.last_deleted_row_id
        } else {
            deletion_vector_iter.next()
        };
        loop {
            if let Some(next_deleted_id) = next_deleted_id {
                if next_deleted_id > last_row_id
                    || (next_deleted_id == last_row_id
                        && Self::is_full(self.legacy_batch_size, num_rows))
                {
                    // Either the next deleted id is out of range or it is the next row but
                    // we are full.  Either way, stash it and return
                    self.last_deleted_row_id = Some(next_deleted_id);
                    return deleted;
                }
                // Otherwise, the deleted row is in range, and we have space in our batch
                // and so we include it
                deleted.push(next_deleted_id - first_row_id);
                last_row_id += 1;
                num_rows += 1;
            } else {
                // Deleted row ids iterator is exhausted
                self.deletion_vector_iter = None;
                return deleted;
            }
            next_deleted_id = deletion_vector_iter.next();
        }
    }

    fn restore(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        // Because of deleted rows, the number of row ids in the batch might not
        // match the length.
        let deleted_batch_offsets = self.deleted_batch_offsets_in_range(batch.num_rows() as u32);
        let batch = add_blanks(batch, &deleted_batch_offsets)?;

        if let Some(batch_size) = self.legacy_batch_size {
            // validation just in case, when the input has a fixed batch size then the
            // output should have the same fixed batch size (except the last batch)
            let is_last = self.is_exhausted();
            if batch.num_rows() != batch_size as usize && !is_last {
                return Err(Error::internal(format!(
                    "Fragment Updater: batch size mismatch: {} != {}",
                    batch.num_rows(),
                    batch_size
                )));
            }
        }

        self.current_row_id += batch.num_rows() as u32;
        Ok(batch)
    }
}

/// Builds the one-row batch whose values fill the blank rows.
///
/// Each column gets the smallest value its field allows — a null where the field
/// is nullable, and otherwise the smallest non-null value of its type, so a blank
/// row costs no payload for a byte or list column and a zeroed value elsewhere.
fn blank_row_for(batch: &RecordBatch) -> Result<RecordBatch> {
    let columns = batch
        .schema()
        .fields()
        .iter()
        .map(|field| {
            // Only a type no Lance schema can hold returns `None`, and copying the
            // batch's first row instead — which is what every column used to get
            // — would put back the payload duplication this avoids.
            minimal_value(field).ok_or_else(|| {
                Error::not_supported(format!(
                    "Cannot add blank rows for column '{}' of type {}",
                    field.name(),
                    field.data_type()
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

/// The two `interleave` sources `add_blanks` splices together: the live batch,
/// and the one-row batch of placeholders.
const LIVE: usize = 0;
const BLANK: usize = 1;

/// Splices a blank row into `batch` at every offset in `batch_offsets`, which
/// must be ascending and are the positions the deleted rows used to occupy.
pub(crate) fn add_blanks(batch: RecordBatch, batch_offsets: &[u32]) -> Result<RecordBatch> {
    // Fast early return
    if batch_offsets.is_empty() {
        return Ok(batch);
    }

    if batch.num_rows() == 0 {
        // TODO: implement adding blanks for an empty batch.
        // There is no live row to take a dictionary's keys from.
        return Err(Error::not_supported_source(
            "Missing too many rows in merge, run compaction to materialize deletions first".into(),
        ));
    }

    let blank_row = blank_row_for(&batch)?;

    let mut indices = Vec::with_capacity(batch.num_rows() + batch_offsets.len());
    let mut batch_pos = 0usize;
    let mut next_id = 0;
    for batch_offset in batch_offsets {
        // `interleave` panics on a row index past the end of its source, so an
        // offset that goes backwards, or that asks for more live rows than are
        // left, has to be turned away here.
        let live_rows_left = batch.num_rows() - batch_pos;
        let num_rows = batch_offset
            .checked_sub(next_id)
            .map(|num_rows| num_rows as usize)
            .filter(|num_rows| *num_rows <= live_rows_left)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "add_blanks: batch offset {} is outside [{}, {}]; offsets must ascend and must not run past the {} rows of the batch",
                    batch_offset,
                    next_id,
                    next_id as usize + live_rows_left,
                    batch.num_rows()
                ))
            })?;
        indices.extend((batch_pos..batch_pos + num_rows).map(|row| (LIVE, row)));
        indices.push((BLANK, 0));
        next_id = *batch_offset + 1;
        batch_pos += num_rows;
    }
    indices.extend((batch_pos..batch.num_rows()).map(|row| (LIVE, row)));

    let with_blanks = interleave_batches(&[stub_dictionaries(&batch)?, blank_row], &indices)
        .map_err(|e| Error::arrow(format!("Failed to add blanks: {}", e)))?;
    restore_dictionary_columns(with_blanks, &batch, &indices)
        .map_err(|e| Error::arrow(format!("Failed to add blanks: {}", e)))
}

/// Rebuilds the dictionary parts of `with_blanks` from `batch` with `take`.
///
/// `interleave` merges the two sources' dictionaries and renumbers their keys,
/// but the v1 writer persists a dictionary's values once from the schema and
/// each batch's keys as they come, so renumbered keys would decode against the
/// wrong values. `take` keeps the values array.
///
/// This goes leaf by leaf rather than column by column: what `take` duplicates
/// for a blank row is everything under the node it is given, so handing it a
/// whole `Struct<{Dictionary, Binary}>` would put the binary payload back on the
/// blank rows and with it the offset overflow. Only dictionary leaves are taken,
/// apart from the fallback arm below.
fn restore_dictionary_columns(
    with_blanks: RecordBatch,
    batch: &RecordBatch,
    indices: &[(usize, usize)],
) -> Result<RecordBatch> {
    let dictionary_columns = batch
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| contains_dictionary(field.data_type()))
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    if dictionary_columns.is_empty() {
        return Ok(with_blanks);
    }

    let slots = indices
        .iter()
        .map(|(source, row)| (*source == LIVE).then_some(*row as u32))
        .collect::<Vec<_>>();
    let mut columns = with_blanks.columns().to_vec();
    for idx in dictionary_columns {
        columns[idx] = restore_dictionaries(&columns[idx], batch.column(idx), &slots)?;
    }
    Ok(RecordBatch::try_new(with_blanks.schema(), columns)?)
}

fn restore_dictionaries(
    interleaved: &ArrayRef,
    live: &ArrayRef,
    slots: &[Option<u32>],
) -> Result<ArrayRef> {
    match interleaved.data_type() {
        // A blank takes row 0's key, which is what makes rebuilding this leaf
        // cheap enough to do instead of interleaving it.
        DataType::Dictionary(_, _) => take_slots(live, slots),
        DataType::Struct(fields) => {
            let children = fields
                .iter()
                .enumerate()
                .map(|(idx, field)| {
                    let interleaved = interleaved.as_struct().column(idx).clone();
                    if contains_dictionary(field.data_type()) {
                        restore_dictionaries(&interleaved, live.as_struct().column(idx), slots)
                    } else {
                        Ok(interleaved)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            with_children(interleaved, children)
        }
        DataType::List(_) => {
            let slots = element_slots(live.as_list::<i32>().value_offsets(), slots);
            let values = restore_dictionaries(
                interleaved.as_list::<i32>().values(),
                live.as_list::<i32>().values(),
                &slots,
            )?;
            with_children(interleaved, vec![values])
        }
        DataType::LargeList(_) => {
            let slots = element_slots(live.as_list::<i64>().value_offsets(), slots);
            let values = restore_dictionaries(
                interleaved.as_list::<i64>().values(),
                live.as_list::<i64>().values(),
                &slots,
            )?;
            with_children(interleaved, vec![values])
        }
        DataType::Map(_, _) => {
            let slots = element_slots(live.as_map().value_offsets(), slots);
            let entries = restore_dictionaries(
                &(Arc::new(interleaved.as_map().entries().clone()) as ArrayRef),
                &(Arc::new(live.as_map().entries().clone()) as ArrayRef),
                &slots,
            )?;
            with_children(interleaved, vec![entries])
        }
        DataType::FixedSizeList(_, width) if *width > 0 => {
            // Unlike a list, the placeholder occupies the full width here, so
            // every row owns `width` child slots. A blank's stay blank: a leaf
            // below turns one into row 0, but a list below owes `interleave`'s
            // offsets an empty entry, not row 0's elements.
            let stride = *width as u32;
            let slots = slots
                .iter()
                .flat_map(|slot| {
                    (0..stride).map(move |offset| slot.map(|row| row * stride + offset))
                })
                .collect::<Vec<_>>();
            let values = restore_dictionaries(
                interleaved.as_fixed_size_list().values(),
                live.as_fixed_size_list().values(),
                &slots,
            )?;
            with_children(interleaved, vec![values])
        }
        // Only a zero-width `FixedSizeList` reaches this: it owns no element, so
        // there is no key to translate, and taking the node keeps the values
        // array that `interleave` replaced with a stub. The other types
        // `contains_dictionary` answers for — `Union`, `RunEndEncoded`, the view
        // lists — are rejected by `blank_row_for` before the restore runs.
        _ => take_slots(live, slots),
    }
}

/// Puts `children` in place of `array`'s, keeping its length, offsets and
/// validity.
///
/// This goes through `ArrayData` rather than the typed constructors because
/// those reject more than arrow's own validation does — `ListArray::try_new`
/// turns away a non-nullable item field whose dictionary child merely has an
/// unreferenced null among its values, and the `take` this replaces never
/// checked that.
fn with_children(array: &ArrayRef, children: Vec<ArrayRef>) -> Result<ArrayRef> {
    let children = children.iter().map(|child| child.to_data()).collect();
    Ok(make_array(
        ArrayDataBuilder::from(array.to_data())
            .child_data(children)
            .build()?,
    ))
}

/// Replaces every dictionary in `batch` with a stub of the same shape: the same
/// keys buffer length, all zero, against a single values entry.
///
/// `interleave` merges the dictionaries of its sources, and the merged values
/// array has to stay indexable by the key type: a `UInt8` dictionary already
/// using all 256 keys leaves no room for the placeholder's value, and the path
/// that concatenates instead of merging fails once the two arrays together reach
/// 256 entries. The first returns `DictionaryKeyOverflowError`; the second
/// panics inside `MutableArrayData`. `restore_dictionary_columns` throws the merged
/// result away regardless, so give `interleave` a dictionary that cannot
/// overflow and keep the real one for the restore.
fn stub_dictionaries(batch: &RecordBatch) -> Result<RecordBatch> {
    let columns = batch
        .columns()
        .iter()
        .map(|column| {
            if contains_dictionary(column.data_type()) {
                Ok(make_array(stub_dictionary_data(&column.to_data())?))
            } else {
                Ok(column.clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

fn stub_dictionary_data(data: &ArrayData) -> Result<ArrayData> {
    if let DataType::Dictionary(key_type, value_type) = data.data_type() {
        let width = key_type.primitive_width().ok_or_else(|| {
            Error::not_supported(format!("Dictionary key type {} has no width", key_type))
        })?;
        let values = minimal_value(&Field::new("item", value_type.as_ref().clone(), false))
            .ok_or_else(|| {
                Error::not_supported(format!(
                    "Cannot build a placeholder for dictionary values of type {}",
                    value_type
                ))
            })?;
        return Ok(ArrayDataBuilder::new(data.data_type().clone())
            .len(data.len())
            .nulls(data.nulls().cloned())
            .add_buffer(MutableBuffer::from_len_zeroed(data.len() * width).into())
            .add_child_data(values.to_data())
            .build()?);
    }
    let children = data
        .child_data()
        .iter()
        .map(|child| {
            if contains_dictionary(child.data_type()) {
                stub_dictionary_data(child)
            } else {
                Ok(child.clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ArrayDataBuilder::from(data.clone())
        .child_data(children)
        .build()?)
}

fn take_slots(live: &ArrayRef, slots: &[Option<u32>]) -> Result<ArrayRef> {
    let indices = UInt32Array::from_iter_values(slots.iter().map(|slot| slot.unwrap_or(0)));
    Ok(arrow::compute::take(live.as_ref(), &indices, None)?)
}

/// Translates a container's row slots into its child's element slots: a live row
/// contributes its own elements and a blank contributes none, because the
/// placeholder's list is empty. That keeps the child aligned with the offsets
/// `interleave` produced.
fn element_slots<O: ArrowNativeType>(offsets: &[O], slots: &[Option<u32>]) -> Vec<Option<u32>> {
    slots
        .iter()
        .flatten()
        .flat_map(|row| {
            let row = *row as usize;
            (offsets[row].as_usize() as u32..offsets[row + 1].as_usize() as u32).map(Some)
        })
        .collect()
}

/// Whether a dictionary sits anywhere in this type. Every type that can hold a
/// child is listed, so the `false` below only ever answers for a leaf: a type
/// missing from this match would keep `interleave`'s renumbered keys silently.
fn contains_dictionary(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(_, _) => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _)
        | DataType::ListView(field)
        | DataType::LargeListView(field) => contains_dictionary(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| contains_dictionary(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| contains_dictionary(field.data_type())),
        DataType::RunEndEncoded(_, values) => contains_dictionary(values.data_type()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::add_blanks;
    use arrow::array::{ArrayDataBuilder, AsArray, make_array};
    use arrow::datatypes::Int32Type;
    use arrow_array::{
        Array, ArrayRef, BinaryArray, DictionaryArray, FixedSizeListArray, Int8Array, Int32Array,
        LargeBinaryArray, LargeListArray, ListArray, MapArray, NullArray, RecordBatch, StringArray,
        StructArray, UInt8Array, types::UInt8Type,
    };
    use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer};
    use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
    use lance_datagen::RowCount;
    use std::sync::Arc;

    #[test]
    fn test_restore_deletes() {
        for batch_size in &[None, Some(10)] {
            let mut restorer = super::DeletionRestorer::new(
                vec![11, 12, 19, 20, 25].into_iter().collect(),
                *batch_size,
            );

            let batch = lance_datagen::gen_batch()
                .col("x", lance_datagen::array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(10))
                .unwrap();
            // First batch is rows ids 0..9 so nothing is restored
            let restored = restorer.restore(batch.clone()).unwrap();
            assert_eq!(restored, batch);

            let batch = lance_datagen::gen_batch()
                .col("x", lance_datagen::array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(7))
                .unwrap();
            // Next batch is rows ids 10..18 so we need to restore 11, 12
            // 19, and maybe 20 (depends on batch size)
            let restored = restorer.restore(batch).unwrap();
            let values = restored.column(0).as_primitive::<Int32Type>();
            assert_eq!(values.value(0), 0);
            assert_eq!(values.value(1), 0);
            assert_eq!(values.value(2), 0);
            assert_eq!(values.value(3), 1);
            assert_eq!(values.value(4), 2);
            assert_eq!(values.value(5), 3);
            assert_eq!(values.value(6), 4);
            assert_eq!(values.value(7), 5);
            assert_eq!(values.value(8), 6);
            assert_eq!(values.value(9), 0);
            if *batch_size == Some(10) {
                assert_eq!(values.len(), 10);
            } else {
                assert_eq!(values.value(10), 0);
                assert_eq!(values.len(), 11);
            }
        }
    }

    #[test]
    fn test_add_blanks() {
        // Values start at 100 so a blank (0) cannot be mistaken for a copy of
        // row 0, which is what this used to insert.
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step_custom::<Int32Type>(100, 1))
            .into_batch_rows(RowCount::from(10))
            .unwrap();

        let with_blanks = add_blanks(batch.clone(), &[5, 7]).unwrap();

        assert_eq!(with_blanks.num_rows(), 12);
        let values = with_blanks.column(0).as_primitive::<Int32Type>();
        for i in 0..5 {
            assert_eq!(values.value(i), 100 + i as i32);
        }
        assert_eq!(values.value(5), 0);
        assert_eq!(values.value(6), 105);
        assert_eq!(values.value(7), 0);
        for i in 8..12 {
            assert_eq!(values.value(i), 100 + (i - 2) as i32);
        }

        let with_blanks = add_blanks(batch, &[0, 11]).unwrap();
        let values = with_blanks.column(0).as_primitive::<Int32Type>();
        assert_eq!(values.value(0), 0);
        for i in 1..11 {
            assert_eq!(values.value(i), 100 + (i - 1) as i32);
        }
        assert_eq!(values.value(11), 0);
    }

    /// `interleave` panics rather than erroring on an out-of-range row index, so
    /// `add_blanks` has to reject the offsets that would produce one.
    #[test]
    fn add_blanks_rejects_offsets_it_cannot_honor() {
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(10))
            .unwrap();

        // Repeated, descending, and past the end. The last pair is the one that
        // stops rejecting if `live_rows_left` forgets `batch_pos`.
        for offsets in [vec![5, 5], vec![5, 3], vec![11], vec![0, 12], vec![3, 12]] {
            let err = add_blanks(batch.clone(), &offsets).unwrap_err();
            assert!(
                matches!(err, lance_core::Error::InvalidInput { .. }),
                "{offsets:?} gave {err}"
            );
            assert!(
                err.to_string().contains("offsets must ascend"),
                "{offsets:?} gave {err}"
            );
        }
    }

    /// Blank rows must carry each column's minimal value, not a copy of row 0.
    /// The old implementation duplicated row 0, which for a large value made one
    /// row pay for every blank.
    #[test]
    fn add_blanks_uses_minimal_values_not_row_zero() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![100, 200, 300])),
                Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"])),
            ],
        )
        .unwrap();
        let live_string_bytes = "alpha".len() + "beta".len() + "gamma".len();

        let with_blanks = add_blanks(batch, &[1, 4]).unwrap();

        assert_eq!(with_blanks.num_rows(), 5);
        let ints = with_blanks.column(0).as_primitive::<Int32Type>();
        assert_eq!(ints.values(), &[100, 0, 200, 300, 0]);

        let strings = with_blanks.column(1).as_string::<i32>();
        assert_eq!(strings.value(0), "alpha");
        assert_eq!(strings.value(1), "", "a blank must not copy row 0's value");
        assert_eq!(strings.value(2), "beta");
        assert_eq!(strings.value(3), "gamma");
        assert_eq!(strings.value(4), "");
        assert_eq!(
            arrow_array::Array::null_count(strings),
            0,
            "the column is not nullable"
        );
        assert_eq!(
            strings.value_offsets().last().copied().unwrap() as usize,
            live_string_bytes,
            "blanks must not consume offset space"
        );
    }

    /// The payload of a variable-width column must not grow with the number of
    /// blanks. Under the old implementation each blank copied row 0, so this
    /// batch's value bytes grew by `blanks * len(row 0)`.
    #[test]
    fn add_blanks_does_not_grow_binary_payload() {
        const VALUE_SIZE: usize = 1024 * 1024;
        let big = vec![0xABu8; VALUE_SIZE];
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "blob",
            DataType::Binary,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BinaryArray::from(vec![
                Some(big.as_slice()),
                Some(&b"small"[..]),
            ]))],
        )
        .unwrap();

        // Eight blanks around two live rows: the old code would have copied the
        // 1 MiB value eight more times.
        let with_blanks = add_blanks(batch, &[0, 1, 2, 3, 6, 7, 8, 9]).unwrap();

        assert_eq!(with_blanks.num_rows(), 10);
        let blobs = with_blanks.column(0).as_binary::<i32>();
        assert_eq!(
            blobs.value_offsets().last().copied().unwrap() as usize,
            VALUE_SIZE + b"small".len(),
            "blanks must contribute no bytes"
        );
        assert_eq!(blobs.value(4).len(), VALUE_SIZE);
        assert_eq!(blobs.value(5), b"small");
        for blank in [0, 1, 2, 3, 6, 7, 8, 9] {
            assert!(blobs.value(blank).is_empty(), "row {blank} should be blank");
        }
    }

    /// A `Binary` column whose live bytes plus duplicated blanks would cross
    /// `i32::MAX` used to fail with `Offset overflow error`, which is what the
    /// `Failed to add blanks` reports in production.
    #[test]
    #[ignore = "allocates ~384MiB; run manually with --ignored"]
    fn add_blanks_survives_binary_offset_overflow() {
        const VALUE_SIZE: usize = 128 * 1024 * 1024;
        // 17 copies of a 128 MiB value is 2.125 GiB, past `i32::MAX`.
        const BLANKS: u32 = 17;

        let big = vec![0xABu8; VALUE_SIZE];
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "blob",
            DataType::Binary,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BinaryArray::from(vec![Some(big.as_slice())]))],
        )
        .unwrap();

        let offsets: Vec<u32> = (1..=BLANKS).collect();
        let with_blanks = add_blanks(batch, &offsets).unwrap();

        assert_eq!(with_blanks.num_rows(), 1 + BLANKS as usize);
        let blobs = with_blanks.column(0).as_binary::<i32>();
        assert_eq!(
            blobs.value_offsets().last().copied().unwrap() as usize,
            VALUE_SIZE,
            "only the live row contributes bytes"
        );
    }

    /// The v1 writer persists a dictionary's values once from the schema and
    /// each batch's keys as they come, so `add_blanks` must leave the keys
    /// indexing the values array it was handed.
    #[test]
    fn add_blanks_keeps_dictionary_keys_valid() {
        let entries = ["alpha", "beta", "gamma", "delta"];
        let dictionary = || {
            Arc::new(
                DictionaryArray::<UInt8Type>::try_new(
                    UInt8Array::from(vec![3u8, 1, 0]),
                    Arc::new(StringArray::from(entries.to_vec())),
                )
                .unwrap(),
            ) as ArrayRef
        };
        let nested = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new(
                "d",
                DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)),
                false,
            )),
            dictionary(),
        )])) as ArrayRef;
        let batch =
            RecordBatch::try_from_iter(vec![("flat", dictionary()), ("nested", nested)]).unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();

        for column in [
            with_blanks.column(0).clone(),
            with_blanks.column(1).as_struct().column(0).clone(),
        ] {
            let dict = column.as_dictionary::<UInt8Type>();
            // The blank at position 1 copies row 0's key, which costs a key and
            // not a payload; the live rows keep the keys they came with.
            assert_eq!(dict.keys().values(), &[3, 3, 1, 0]);
            assert_eq!(
                dict.values().as_string::<i32>(),
                &StringArray::from(entries.to_vec()),
                "the values array must survive untouched"
            );
        }
    }

    /// A blank row must leave a nullable child null: the blob writer dispatches
    /// on `uri` being present, and an empty string there is an external
    /// reference to `""`, not an absent one.
    #[test]
    fn add_blanks_leaves_a_nullable_child_null() {
        let fields = Fields::from(vec![
            Field::new("data", DataType::LargeBinary, true),
            Field::new("uri", DataType::Utf8, true),
        ]);
        let descriptor = StructArray::new(
            fields,
            vec![
                Arc::new(LargeBinaryArray::from(vec![Some(b"payload".as_slice())])) as ArrayRef,
                Arc::new(StringArray::new_null(1)) as ArrayRef,
            ],
            None,
        );
        let batch = RecordBatch::try_from_iter_with_nullable(vec![(
            "blob",
            Arc::new(descriptor) as ArrayRef,
            false,
        )])
        .unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();

        let blob = with_blanks.column(0).as_struct();
        assert_eq!(blob.null_count(), 0, "the column itself is not nullable");
        assert!(blob.column(1).is_null(0), "the live row had no uri");
        assert!(blob.column(1).is_null(1), "and neither has the blank");
        assert!(
            blob.column(0).is_null(1),
            "an absent payload, not an empty one"
        );
    }

    /// A dictionary sibling must not drag a large payload back onto the blank
    /// rows. `take` duplicates everything under the node it is handed, so the
    /// restore has to reach the dictionary leaf and leave the binary child as
    /// `interleave` built it.
    #[test]
    fn add_blanks_does_not_grow_a_dictionary_siblings_payload() {
        const VALUE_SIZE: usize = 1024 * 1024;
        const BLANKS: u32 = 8;

        let tag = DictionaryArray::<UInt8Type>::try_new(
            UInt8Array::from(vec![1u8]),
            Arc::new(StringArray::from(vec!["x", "y"])),
        )
        .unwrap();
        let blob = BinaryArray::from(vec![Some(vec![7u8; VALUE_SIZE].as_slice())]);
        let fields = Fields::from(vec![
            Field::new(
                "tag",
                DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)),
                false,
            ),
            Field::new("blob", DataType::Binary, false),
        ]);
        let batch = RecordBatch::try_from_iter_with_nullable(vec![(
            "s",
            Arc::new(StructArray::new(
                fields,
                vec![Arc::new(tag) as ArrayRef, Arc::new(blob) as ArrayRef],
                None,
            )) as ArrayRef,
            false,
        )])
        .unwrap();

        let with_blanks = add_blanks(batch, &(1..=BLANKS).collect::<Vec<_>>()).unwrap();

        let out = with_blanks.column(0).as_struct();
        assert_eq!(out.len(), 1 + BLANKS as usize);
        assert_eq!(
            out.column(1)
                .as_binary::<i32>()
                .value_offsets()
                .last()
                .copied()
                .unwrap() as usize,
            VALUE_SIZE,
            "only the live row contributes bytes"
        );
        let tag = out.column(0).as_dictionary::<UInt8Type>();
        assert_eq!(tag.keys().values(), &[1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(
            tag.values().as_string::<i32>(),
            &StringArray::from(vec!["x", "y"]),
            "the values array must survive untouched"
        );
    }

    /// A dictionary under a list keeps the keys and values it came with, and the
    /// blank contributes no element at all.
    #[test]
    fn add_blanks_keeps_a_listed_dictionary_valid() {
        // The fourth value is a null no key points at. `ListArray::try_new`
        // rejects that under a non-nullable item field while `ArrayData` accepts
        // it, which is why the restore rebuilds through `ArrayData` — and why
        // this fixture has to be built the way a reader builds one.
        let values = DictionaryArray::<UInt8Type>::try_new(
            UInt8Array::from(vec![2u8, 0, 1]),
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                Some("c"),
                None,
            ])),
        )
        .unwrap();
        let item = Arc::new(Field::new(
            "item",
            DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)),
            false,
        ));
        let list = make_array(
            ArrayDataBuilder::new(DataType::List(item))
                .len(2)
                .add_buffer(Buffer::from_slice_ref([0i32, 2, 3]))
                .add_child_data(values.to_data())
                .build()
                .unwrap(),
        );
        let batch = RecordBatch::try_from_iter(vec![("l", list)]).unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();

        let out = with_blanks.column(0).as_list::<i32>();
        assert_eq!(out.len(), 3);
        let entries = out.values().as_dictionary::<UInt8Type>();
        assert_eq!(
            entries.values().as_string::<i32>(),
            &StringArray::from(vec![Some("a"), Some("b"), Some("c"), None]),
            "the values array must survive untouched"
        );
        // Row 0 is [c, a] and row 1 is [b]; the blank is an empty list, so no
        // element is duplicated and no key is rewritten.
        assert_eq!(entries.keys().values(), &[2, 0, 1]);
        assert_eq!(out.value_length(0), 2);
        assert_eq!(out.value_length(1), 0, "the blank");
        assert_eq!(out.value_length(2), 1);
    }

    /// The same property one level down: a dictionary inside a list element must
    /// not drag its element's payload onto the blank rows either.
    #[test]
    fn add_blanks_does_not_grow_a_listed_dictionary_siblings_payload() {
        const VALUE_SIZE: usize = 1024 * 1024;
        const BLANKS: u32 = 8;

        let fields = Fields::from(vec![
            Field::new(
                "tag",
                DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)),
                false,
            ),
            Field::new("blob", DataType::Binary, false),
        ]);
        let element = StructArray::new(
            fields.clone(),
            vec![
                Arc::new(
                    DictionaryArray::<UInt8Type>::try_new(
                        UInt8Array::from(vec![1u8]),
                        Arc::new(StringArray::from(vec!["x", "y"])),
                    )
                    .unwrap(),
                ) as ArrayRef,
                Arc::new(BinaryArray::from(vec![Some(
                    vec![7u8; VALUE_SIZE].as_slice(),
                )])) as ArrayRef,
            ],
            None,
        );
        let list = ListArray::new(
            Arc::new(Field::new("item", DataType::Struct(fields), false)),
            OffsetBuffer::new(vec![0, 1].into()),
            Arc::new(element) as ArrayRef,
            None,
        );
        let batch = RecordBatch::try_from_iter(vec![("l", Arc::new(list) as ArrayRef)]).unwrap();

        let with_blanks = add_blanks(batch, &(1..=BLANKS).collect::<Vec<_>>()).unwrap();

        let out = with_blanks.column(0).as_list::<i32>();
        assert_eq!(out.len(), 1 + BLANKS as usize);
        let element = out.values().as_struct();
        assert_eq!(
            element
                .column(1)
                .as_binary::<i32>()
                .value_offsets()
                .last()
                .copied()
                .unwrap() as usize,
            VALUE_SIZE,
            "only the live element contributes bytes"
        );
        let tag = element.column(0).as_dictionary::<UInt8Type>();
        assert_eq!(tag.keys().values(), &[1]);
        assert_eq!(
            tag.values().as_string::<i32>(),
            &StringArray::from(vec!["x", "y"])
        );
    }

    /// The recursion has to reach a dictionary nested two structs deep, and the
    /// rebuild has to keep the live rows' own nulls.
    #[test]
    fn add_blanks_keeps_a_twice_nested_dictionary_and_its_nulls() {
        let dictionary_type =
            DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8));
        let inner_fields = Fields::from(vec![Field::new("d", dictionary_type, false)]);
        let inner = StructArray::new(
            inner_fields.clone(),
            vec![Arc::new(
                DictionaryArray::<UInt8Type>::try_new(
                    UInt8Array::from(vec![2u8, 0]),
                    Arc::new(StringArray::from(vec!["a", "b", "c"])),
                )
                .unwrap(),
            ) as ArrayRef],
            None,
        );
        let outer_fields = Fields::from(vec![Field::new(
            "inner",
            DataType::Struct(inner_fields),
            true,
        )]);
        let outer = StructArray::new(
            outer_fields,
            vec![Arc::new(inner) as ArrayRef],
            // The second live row is null.
            Some(NullBuffer::from(vec![true, false])),
        );
        let batch = RecordBatch::try_from_iter(vec![("s", Arc::new(outer) as ArrayRef)]).unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();

        let out = with_blanks.column(0).as_struct();
        assert_eq!(out.len(), 3);
        assert!(!out.is_null(0), "the live row that had a value");
        assert!(out.is_null(2), "the live row that was null stays null");
        let dictionary = out
            .column(0)
            .as_struct()
            .column(0)
            .as_dictionary::<UInt8Type>();
        assert_eq!(dictionary.keys().values(), &[2, 2, 0]);
        assert_eq!(
            dictionary.values().as_string::<i32>(),
            &StringArray::from(vec!["a", "b", "c"]),
            "the values array must survive untouched"
        );
    }

    /// A blank under a `FixedSizeList` owes the width its slots, but a list below
    /// that owes `interleave`'s offsets an empty entry — not row 0's elements.
    #[test]
    fn add_blanks_keeps_a_dictionary_under_a_fixed_size_list_aligned() {
        let dictionary_type =
            DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8));
        let tags = ListArray::new(
            Arc::new(Field::new("item", dictionary_type, false)),
            OffsetBuffer::new(vec![0, 1, 2, 3, 3].into()),
            Arc::new(
                DictionaryArray::<UInt8Type>::try_new(
                    UInt8Array::from(vec![0u8, 1, 2]),
                    Arc::new(StringArray::from(vec!["a", "b", "c"])),
                )
                .unwrap(),
            ) as ArrayRef,
            None,
        );
        let list_type = tags.data_type().clone();
        let element_fields = Fields::from(vec![Field::new("tags", list_type, false)]);
        let element = StructArray::new(
            element_fields.clone(),
            vec![Arc::new(tags) as ArrayRef],
            None,
        );
        // Two rows of two elements each: [[a], [b]] and [[c], []].
        let outer = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Struct(element_fields), false)),
            2,
            Arc::new(element) as ArrayRef,
            None,
        );
        let batch = RecordBatch::try_from_iter(vec![("f", Arc::new(outer) as ArrayRef)]).unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();

        let out = with_blanks.column(0).as_fixed_size_list();
        assert_eq!(out.len(), 3);
        let tags = out.values().as_struct().column(0).as_list::<i32>();
        let entries = tags.values().as_dictionary::<UInt8Type>();
        assert_eq!(
            entries.keys().values(),
            &[0, 1, 2],
            "a blank contributes no element, so no key moves"
        );
        assert_eq!(
            entries.values().as_string::<i32>(),
            &StringArray::from(vec!["a", "b", "c"])
        );
        assert_eq!(
            (0..tags.len())
                .map(|i| tags.value_length(i))
                .collect::<Vec<_>>(),
            vec![1, 1, 0, 0, 1, 0],
            "the blank's two elements are empty lists"
        );
    }

    /// The same for the two containers `List` does not cover.
    #[test]
    fn add_blanks_keeps_a_mapped_and_large_listed_dictionary_valid() {
        let dictionary_type =
            DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8));
        let dictionary = || {
            Arc::new(
                DictionaryArray::<UInt8Type>::try_new(
                    UInt8Array::from(vec![2u8, 0]),
                    Arc::new(StringArray::from(vec!["a", "b", "c"])),
                )
                .unwrap(),
            ) as ArrayRef
        };
        let entry_fields = Fields::from(vec![
            Field::new("keys", DataType::Utf8, false),
            Field::new("values", dictionary_type.clone(), true),
        ]);
        let map = MapArray::new(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(entry_fields.clone()),
                false,
            )),
            OffsetBuffer::new(vec![0, 2].into()),
            StructArray::new(
                entry_fields,
                vec![
                    Arc::new(StringArray::from(vec!["k0", "k1"])) as ArrayRef,
                    dictionary(),
                ],
                None,
            ),
            None,
            false,
        );
        let large = LargeListArray::new(
            Arc::new(Field::new("item", dictionary_type, false)),
            OffsetBuffer::new(vec![0i64, 2].into()),
            dictionary(),
            None,
        );
        let batch = RecordBatch::try_from_iter(vec![
            ("m", Arc::new(map) as ArrayRef),
            ("l", Arc::new(large) as ArrayRef),
        ])
        .unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();

        let map = with_blanks.column(0).as_map();
        assert_eq!(map.value_length(1), 0, "the blank is an empty map");
        let large = with_blanks.column(1).as_list::<i64>();
        assert_eq!(large.value_length(1), 0, "the blank is an empty list");
        for entries in [
            map.entries().column(1).as_dictionary::<UInt8Type>(),
            large.values().as_dictionary::<UInt8Type>(),
        ] {
            assert_eq!(entries.keys().values(), &[2, 0]);
            assert_eq!(
                entries.values().as_string::<i32>(),
                &StringArray::from(vec!["a", "b", "c"]),
                "the values array must survive untouched"
            );
        }
    }

    /// The offsets a sliced batch reports are absolute, and a null row keeps its
    /// null: both are properties the element translation has to respect.
    #[test]
    fn add_blanks_keeps_a_sliced_list_and_its_nulls() {
        let list = ListArray::new(
            Arc::new(Field::new(
                "item",
                DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)),
                false,
            )),
            OffsetBuffer::new(vec![0, 1, 2, 3].into()),
            Arc::new(
                DictionaryArray::<UInt8Type>::try_new(
                    UInt8Array::from(vec![0u8, 1, 2]),
                    Arc::new(StringArray::from(vec!["a", "b", "c"])),
                )
                .unwrap(),
            ) as ArrayRef,
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let batch = RecordBatch::try_from_iter(vec![("l", Arc::new(list) as ArrayRef)]).unwrap();
        // Drop row 0, so the remaining rows' offsets start at 1.
        let batch = batch.slice(1, 2);

        let with_blanks = add_blanks(batch, &[1]).unwrap();

        let out = with_blanks.column(0).as_list::<i32>();
        assert_eq!(out.len(), 3);
        assert!(out.is_null(1), "the blank of a nullable column is a null");
        assert!(out.is_null(2), "the live row that was null stays null");
        let entries = out.values().as_dictionary::<UInt8Type>();
        assert_eq!(
            entries.keys().values(),
            &[1, 2],
            "the sliced rows own elements 1 and 2"
        );
    }

    /// A dictionary that fills its key type cannot be merged with a second one,
    /// so `interleave` must never be handed the real values array: the flat case
    /// errors inside `merge_dictionary_values`, and the `FixedSizeList` case
    /// panics inside `MutableArrayData`.
    #[test]
    fn add_blanks_survives_a_saturated_dictionary() {
        // Every key of a `UInt8` dictionary, all referenced.
        let entries = (0..=255u16).map(|i| i.to_string()).collect::<Vec<_>>();
        let values = Arc::new(StringArray::from(entries.clone()));
        let dictionary = || {
            Arc::new(
                DictionaryArray::<UInt8Type>::try_new(
                    UInt8Array::from((0..=255u8).collect::<Vec<_>>()),
                    values.clone(),
                )
                .unwrap(),
            ) as ArrayRef
        };
        // Two elements per row, so its child references every key twice.
        let listed = FixedSizeListArray::new(
            Arc::new(Field::new(
                "item",
                DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)),
                false,
            )),
            2,
            Arc::new(
                DictionaryArray::<UInt8Type>::try_new(
                    UInt8Array::from((0..=255u8).chain(0..=255u8).collect::<Vec<_>>()),
                    values.clone(),
                )
                .unwrap(),
            ) as ArrayRef,
            None,
        );
        let batch = RecordBatch::try_from_iter(vec![
            ("flat", dictionary()),
            ("listed", Arc::new(listed) as ArrayRef),
        ])
        .unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();

        assert_eq!(with_blanks.num_rows(), 257);
        let flat = with_blanks.column(0).as_dictionary::<UInt8Type>();
        assert_eq!(
            flat.values().as_string::<i32>(),
            &StringArray::from(entries),
            "the values array must survive untouched"
        );
        // Row 0 keeps key 0, and the blank at position 1 copies it.
        assert_eq!(&flat.keys().values()[..3], &[0, 0, 1]);
    }

    /// A dictionary of nulls never reaches the merge — its values type is neither
    /// byte nor primitive, so `interleave` concatenates instead, and that
    /// overflows on the values count alone. The stub has to cover it too.
    #[test]
    fn add_blanks_survives_a_saturated_dictionary_of_nulls() {
        let dictionary = DictionaryArray::<arrow_array::types::Int8Type>::try_new(
            Int8Array::from(vec![0i8]),
            Arc::new(NullArray::new(130)) as ArrayRef,
        )
        .unwrap();
        let batch =
            RecordBatch::try_from_iter(vec![("d", Arc::new(dictionary) as ArrayRef)]).unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();

        assert_eq!(with_blanks.num_rows(), 2);
        let out = with_blanks
            .column(0)
            .as_dictionary::<arrow_array::types::Int8Type>();
        assert_eq!(out.keys().values(), &[0, 0]);
        assert_eq!(
            out.values().len(),
            130,
            "the values array must survive untouched"
        );
    }
}
