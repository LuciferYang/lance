/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance;

/**
 * Per-fragment column-layout statistics of a dataset version as parallel primitive arrays: {@code
 * fragmentIds[i]}, {@code liveFileCounts[i]} and {@code overlayCounts[i]} describe the same
 * fragment. Returned by {@link Dataset#getColumnLayoutStatistics()}. Compaction repacks a
 * fragment's columns into fewer files when its live file count is above {@code
 * maxDataFilesPerFragment}.
 */
public final class ColumnLayoutStatistics {
  private final long[] fragmentIds;
  private final int[] liveFileCounts;
  private final int[] overlayCounts;

  ColumnLayoutStatistics(long[] fragmentIds, int[] liveFileCounts, int[] overlayCounts) {
    this.fragmentIds = fragmentIds;
    this.liveFileCounts = liveFileCounts;
    this.overlayCounts = overlayCounts;
  }

  /** Fragment IDs in manifest order. */
  public long[] getFragmentIds() {
    return fragmentIds;
  }

  /**
   * Number of data files holding at least one column of the schema, per fragment, aligned with
   * {@link #getFragmentIds()}. A file left holding only tombstones or dropped columns, or kept only
   * for the fragment's spilled row lineage, is not counted.
   */
  public int[] getLiveFileCounts() {
    return liveFileCounts;
  }

  /** Number of data overlay files per fragment, aligned with {@link #getFragmentIds()}. */
  public int[] getOverlayCounts() {
    return overlayCounts;
  }

  /** Number of fragments described. */
  public int size() {
    return fragmentIds.length;
  }
}
