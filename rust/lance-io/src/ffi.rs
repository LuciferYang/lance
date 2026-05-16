// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::VecDeque;

use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, SchemaRef};
use futures::StreamExt;
use lance_core::Result;

use crate::stream::RecordBatchStream;

const PREFETCH_BATCH_COUNT: usize = 16;

#[pin_project::pin_project]
struct RecordBatchIteratorAdaptor<S: RecordBatchStream> {
    schema: SchemaRef,

    #[pin]
    stream: S,

    handle: tokio::runtime::Handle,

    buffer: VecDeque<std::result::Result<RecordBatch, ArrowError>>,
}

impl<S: RecordBatchStream> RecordBatchIteratorAdaptor<S> {
    fn new(stream: S, schema: SchemaRef, handle: tokio::runtime::Handle) -> Self {
        Self {
            schema,
            stream,
            handle,
            buffer: VecDeque::with_capacity(PREFETCH_BATCH_COUNT),
        }
    }
}

impl<S: RecordBatchStream + Unpin> arrow::record_batch::RecordBatchReader
    for RecordBatchIteratorAdaptor<S>
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl<S: RecordBatchStream + Unpin> Iterator for RecordBatchIteratorAdaptor<S> {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.buffer.pop_front() {
            return Some(item);
        }
        self.handle.block_on(async {
            for _ in 0..PREFETCH_BATCH_COUNT {
                match self.stream.next().await {
                    Some(r) => self.buffer.push_back(
                        r.map_err(|e| ArrowError::ExternalError(Box::new(e))),
                    ),
                    None => break,
                }
            }
        });
        self.buffer.pop_front()
    }
}

/// Wrap a [`RecordBatchStream`] into an [FFI_ArrowArrayStream].
pub fn to_ffi_arrow_array_stream(
    stream: impl RecordBatchStream + std::marker::Unpin + 'static,
    handle: tokio::runtime::Handle,
) -> Result<FFI_ArrowArrayStream> {
    let schema = stream.schema();
    let arrow_stream = RecordBatchIteratorAdaptor::new(stream, schema, handle);
    let reader = FFI_ArrowArrayStream::new(Box::new(arrow_stream));

    Ok(reader)
}
