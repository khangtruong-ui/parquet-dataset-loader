# parquet-dataset-loader

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Version: 0.3.0](https://img.shields.io/badge/version-0.3.0-orange.svg)](pyproject.toml)

A high-performance, memory-efficient Python library dedicated to loading large Parquet datasets from Hugging Face Hub and local storage. Featuring **instant random-access index-based streaming** without downloading multi-gigabyte or terabyte files to disk.

---

## The Problem

When working with massive datasets on the Hugging Face Hub (e.g. [`KhangTruong/COCO-inpainted`](https://huggingface.co/datasets/KhangTruong/COCO-inpainted), which is **62.4 GB across 65 Parquet files**), the standard Hugging Face `datasets` library forces you into an painful trade-off:

| Approach | Limitations |
| :--- | :--- |
| **`load_dataset("repo", streaming=False)`** | Downloads **all 62+ GB** to local disk before yielding a single row. Consumes tens of gigabytes of disk and RAM, leading to `No space left on device` crashes. |
| **`load_dataset("repo", streaming=True)`** | Returns an `IterableDataset` that: <br>1. **Has no length** (`len(ds)` raises `TypeError`).<br>2. **Cannot do random access** (`ds[50000]` raises `TypeError`).<br>3. **Cannot seek or slice**: Skipping to index 50,000 sequentially downloads and parses all rows 0..49,999, taking dozens of minutes. |

### The Solution: `parquet-dataset-loader`

Because Parquet stores its metadata in the **file footer** (at the very end of each file), we don't need to download the full Parquet files to know their structure!

`parquet-dataset-loader` uses **HTTP Range requests** (`Range: bytes=-65536`) to fetch only the metadata footer (typically ~26 KB) for each file. It constructs an in-memory **global row index** in seconds.

With this index:
- **`len(dataset)` is known instantly** (e.g., exactly 122,184 rows in `KhangTruong/COCO-inpainted` in ~0.5s).
- **$O(\log M)$ Random Access Indexing (`ds[50000]`)**: Binary-searches the exact file and row group, then issues an HTTP Range request for *only that row group*. Files 0–24 are **never touched**.
- **Bounded RAM footprint**: Decoded row groups are cached in a thread-safe **LRU memory cache** (default 2 row groups). Memory stays strictly capped at ~100–200 MB even for 100+ GB datasets.
- **Zero Disk Footprint**: Streaming mode downloads 0 bytes of full Parquet files to disk.
- **Column Projection**: Request only metadata or subset columns (`columns=["mask"]`), reading ~250 KB instead of 98 MB per row group.

---

## Benchmark: `KhangTruong/COCO-inpainted` (62.4 GB)

| Metric | Hugging Face `load_dataset` | Hugging Face `streaming=True` | `parquet-dataset-loader` (Ours) |
| :--- | :--- | :--- | :--- |
| **Disk Required** | ~62.4 GB | 0 GB | **0 GB** (or cached metadata ~50 KB) |
| **Time to First Row** | ~15–30 minutes (full download) | ~2–5 seconds | **~0.4 seconds** |
| **`len(dataset)`** | Available only after 62 GB dl | ❌ Not supported (`TypeError`) | **122,184 rows (Instant)** |
| **Access Row 50,000** | Must wait for 62 GB download | ❌ Sequential scan (~10+ mins) | **~3.5 seconds** (direct jump) |
| **Consecutive Row 50,001**| Memory-mapped cache | ~20 ms | **0.06 ms** (LRU cache hit) |
| **Memory Footprint** | Tens of GBs | Unbounded streaming buffers | **Strictly Bounded** (e.g. <200 MB) |

---

## Installation

```bash
# Install directly from the repository
pip install .

# Or install in editable / development mode
pip install -e .
```

Requirements:
- Python $\ge$ 3.9
- `pyarrow >= 14.0.0`
- `requests >= 2.28.0`
- `fsspec >= 2023.1.0`
- `huggingface-hub >= 0.20.0`
- `datasets >= 2.14.0`

---

## Architecture & How It Works

```
                                  [ User Request: ds[50000] ]
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │    MetadataIndex (Precomputed)   │
                              │    O(log M) Binary Search        │
                              └──────────────────────────────────┘
                                               │
                                               ├─ File: train-00025.parquet
                                               ├─ Row Group: 0
                                               └─ Local Row Offset: 0
                                               │
                                               ▼
                                  ┌───────────────────────────┐
                                  │   RowGroupMemoryCache     │
                                  │   (Thread-Safe LRU)       │
                                  └─────────────┬─────────────┘
                                                │
                                       Cache Hit? (0.06 ms)
                                      ┌─────────┴─────────┐
                                      ▼                   ▼
                                    [Yes]                [No]
                                  Return row              │
                                                          ▼
                                            ┌───────────────────────────┐
                                            │ HTTP Range Request        │
                                            │ Fetch ONLY Row Group 0    │
                                            │ (Skip Files 0..24)        │
                                            └───────────────────────────┘
                                                          │
                                                          ▼
                                            ┌───────────────────────────┐
                                            │ Store in LRU Cache        │
                                            │ Return Row Dictionary     │
                                            └───────────────────────────┘
```

---

## Quickstart

### 1. Random-Access Streaming

```python
import parquet_dataset_loader as pdl

# Load dataset in streaming mode with column projection
ds = pdl.load_dataset(
    "KhangTruong/COCO-inpainted",
    split="train",
    streaming=True,
    columns=["mask"],  # Column projection: downloads only mask, not 98MB image payloads
    max_cached_row_groups=2,
)

print(f"Total rows: {len(ds):,}")  # 122,184 rows (known immediately!)

# Random access to index 0
sample_0 = ds[0]
print(sample_0["mask"]["path"])

# Jump directly to index 50,000 without touching files 0-24!
sample_50k = ds[50000]
print(sample_50k["mask"]["path"])

# Consecutive accesses hit the in-memory LRU cache in 0.06 ms:
sample_50001 = ds[50001]
```

### 2. Slicing Matching Hugging Face Format

Slicing returns a dictionary of lists `{column_name: [values]}`, exactly matching Hugging Face `Dataset` behavior:

```python
# Slice 5 samples across row group boundaries
batch = ds[50000:50005]
print(batch.keys())  # dict_keys(['mask'])
print(len(batch["mask"]))  # 5
```

### 3. Seekable Streaming (`iter_from`, `stream`, `start_index`)

Resume training or start reading from any arbitrary index instantly without reading prior row groups:

```python
# Stream starting from index 100,000 via iter_from() or stream()
for row in ds.iter_from(start_index=100000):
    process(row)

# Or specify start_index directly when loading (both streaming and non-streaming):
ds = pdl.load_dataset(
    "KhangTruong/COCO-inpainted",
    split="train",
    streaming=True,
    start_index=100000,
)
```

### 4. Reproducible Shuffling with Seed (`shuffle`, `seed`)

Fully supported across **both streaming (`streaming=True`) and non-streaming (`streaming=False`)** modes:

- **Reproducible Ordering**: Providing `seed` guarantees 100% deterministic, reproducible sample ordering across runs.
- **Instant Random Access**: Index permutation shuffle enables instant $O(\log M)$ index lookup `shuffled_ds[i]`, slicing, and PyTorch `DataLoader` compatibility without downloading full files.
- **Combined with Streaming from Index**: Resume training on a shuffled stream from an exact checkpoint step!
- **Streaming Buffer Shuffle**: For unbounded streams or very small memory environments, specify `buffer_size` to shuffle within a rolling buffer.

```python
# Approach A: Direct parameter in load_dataset (streaming or non-streaming)
ds_shuffled = pdl.load_dataset(
    "KhangTruong/COCO-inpainted",
    split="train",
    streaming=True,
    shuffle=True,
    seed=42,
)

# Approach B: Call .shuffle(seed=...) on loaded dataset or dataset dictionary
ds_shuffled = ds.shuffle(seed=42)

# Resume / stream from index 50,000 of the shuffled dataset
for row in ds_shuffled.iter_from(start_index=50000):
    process(row)

# Or combine shuffle and resume directly in load_dataset:
ds_resume = pdl.load_dataset(
    "KhangTruong/COCO-inpainted",
    split="train",
    streaming=True,
    shuffle=True,
    seed=42,
    start_index=50000,  # Starts streaming from index 50,000 of the seed-42 permutation
)

# Streaming buffer shuffle (rolling window of 10,000 rows):
ds_buf = ds.shuffle(seed=42, buffer_size=10000)
for row in ds_buf:
    process(row)
```

### 5. Zero-Copy Views (`take`, `skip`, `slice`)

Create sub-dataset views without copying data:

```python
train_sub = ds.take(1000)      # First 1,000 rows
train_tail = ds.skip(100000)   # Rows from 100,000 onwards
train_slice = ds.slice(500, 200) # 200 rows starting at 500
```

### 6. Conversion to Hugging Face, Arrow, and Pandas

```python
# Convert a slice or small dataset to a Hugging Face Dataset
hf_dataset = ds.take(100).to_hf_dataset()

# Convert to PyArrow Table
arrow_table = ds.take(100).to_arrow()

# Convert to Pandas DataFrame
df = ds.take(100).to_pandas()
```

### 7. Full Split Dictionary Handling (`split=None`)

```python
# Loads all splits into a ParquetDatasetDict
ds_dict = pdl.load_dataset("KhangTruong/COCO-inpainted", split=None, streaming=True)

print(ds_dict.keys())  # ['train', 'validation']
print(f"Train: {len(ds_dict['train']):,} | Validation: {len(ds_dict['validation']):,}")

# Shuffle all splits with seed simultaneously:
shuffled_dict = ds_dict.shuffle(seed=42)
```

### 8. Disk Loading & Saving

```python
# Save dataset to disk in native Parquet and Feather format
ds.take(500).save_to_disk("./my_local_data")

# Or save without arguments to the default cache directory (~/.cache/parquet_dataset_loader/saved)
saved_dir = ds_dict.save_to_disk()

# Reload from disk (or call load_from_disk() without arguments to reload from default cache)
reloaded = pdl.load_from_disk("./my_local_data")
print(len(reloaded))  # 500

# Multi-split DatasetDict reloading with full Hugging Face compatibility
reloaded_dict = pdl.load_from_disk(saved_dir)
print(reloaded_dict.num_rows)  # {'train': 100, 'validation': 30}
```

### 9. Streaming + Save to Disk Simultaneously (Zero Initial Wait)

If you want a local copy on disk, standard Hugging Face `load_dataset("...", streaming=False)` forces you to wait for the entire 62 GB download before you can access a single sample.

With `parquet-dataset-loader`, pass `save_to_disk="./my_archive"`:
- **Instant access**: Step 0 starts in <0.5 seconds without waiting!
- **Simultaneous saving**: Each row group is saved to disk as it is streamed.
- **Background prefetching** (optional): Set `background_download=True` to download remaining row groups in a background worker thread while you process data unblocked.

```python
# Start streaming immediately while saving to disk in parallel
ds = pdl.load_dataset(
    "KhangTruong/COCO-inpainted",
    split="train",
    streaming=True,
    save_to_disk="./coco_archive",
    columns=["mask"],
    background_download=True,  # Prefetches remaining row groups in the background
)

# Access row 0 instantly without waiting for download!
sample_0 = ds[0]

# Check progress anytime:
print(f"Disk save progress: {ds.save_progress * 100:.1f}%")
print(f"Is fully saved: {ds.is_fully_saved}")

# Stream through and ensure everything is saved:
ds.stream_and_save(show_progress=True)

# Later, reload from disk completely offline:
offline_ds = pdl.load_from_disk("./coco_archive")
print(len(offline_ds))
```

### 10. Stream-from-Index + Forward Background Downloading (Prefetching)

When you resume training or evaluate a slice from an offset using `start_index` (or `from_index`), background downloading automatically prioritizes row groups **forward from your start index**:

```
Dataset Total Rows: [ RG 0 ] [ RG 1 ] [ RG 2 ] [ RG 3 ] [ RG 4 ] [ RG 5 ]
Consumer start_index ───────────────► [Starts at RG 2]
Background Worker Download Order:    (1st)    (2nd)    (3rd)    (4th) ──┐
                                     RG 2  ─► RG 3  ─► RG 4  ─► RG 5   │
                                  ┌────────────────────────────────────┘
                                  ▼ (Wrap around for offline completeness)
                                 (5th)   (6th)
                                 RG 0 ─► RG 1
```

```python
# Starts streaming at row index 50,000 while prefetching forward from RG containing row 50,000
ds = pdl.load_dataset(
    "KhangTruong/COCO-inpainted",
    split="train",
    streaming=True,
    start_index=50000,
    save_to_disk="./coco_stream_prefetch",
    background_download=True,
)

# Row 50,000 is available instantly, and upcoming rows (50,001+) are prefetched ahead in RAM/disk
print(ds[0])  # Accesses relative row 0 (global row 50,000)

# Dynamic control over background prefetcher:
ds.stop_background_download()
ds.start_background_download(start_index=60000)
```

---

### 11. Resuming Incomplete Progressive Datasets on Disk

If training or downloading is interrupted (e.g. spot instance preemption, network disconnect, or process termination), progressive saves are **never corrupted**:
1. **Atomic File Writes**: Row groups are saved atomically via `.tmp` files and committed only when fully validated.
2. **Disk Reconciliation**: On reload, `ProgressiveDiskSaver` sweeps the directory, discovers all valid completed `.feather` row groups, and purges orphaned `.tmp` files.
3. **Seamless Remote Resumption**: Manifest files store the upstream repo `source_path`. Resuming unblocks streaming immediately while downloading only the missing row groups.

```python
# Option A: Universal resume_dataset() helper
ds_resumed = pdl.resume_dataset(
    dataset_path="./coco_partial_archive",
    background_download=True,  # Finishes missing row groups in background
)

# Option B: load_from_disk with resume=True (default is True)
ds_resumed = pdl.load_from_disk(
    "./coco_partial_archive",
    resume=True,
    background_download=True,
)

# Access any row immediately (saved rows load from disk; unsaved stream from remote)
sample = ds_resumed[1000]

# If you only want to load already-downloaded local rows without touching the network:
ds_offline = pdl.load_from_disk("./coco_partial_archive", allow_incomplete=True, resume=False)
```

---

### 12. Multiple Dataset Instance Management (`DatasetManager`)

When training multi-modal architectures or running joint evaluation, you often instantiate multiple datasets (e.g. `ds_train`, `ds_val`, `ds_labels`). `parquet-dataset-loader` provides a centralized, thread-safe **lifecycle registry**:

- **Automatic Registration**: Datasets created via `load_dataset` or `load_from_disk` are tracked in the global `DatasetManager`.
- **Introspection**: Inspect active datasets, download progress, row counts, and memory caches with `pdl.list_active_datasets()`.
- **Clean Teardown**: Stop all background threads, close open reader file descriptors, and release memory with `close_all_datasets()` or `ds.close()`.

```python
import parquet_dataset_loader as pdl

# Create multiple managed datasets with explicit or auto-assigned IDs
ds1 = pdl.load_dataset("repo1", split="train", streaming=True, dataset_id="ds_train")
ds2 = pdl.load_dataset("repo2", split="validation", streaming=True, dataset_id="ds_val")

# Introspect all active datasets across your process:
active = pdl.list_active_datasets()
for ds_id, status in active.items():
    print(f"ID: {ds_id} | Rows: {status['num_rows']:,} | Background Worker: {status['background_downloading']}")

# Clean up individually:
ds1.close()
assert ds1.is_closed

# Or clean up all datasets in one call:
pdl.close_all_datasets()
```

#### Scoped Lifecycles with Context Managers

```python
# Scoped Dataset context:
with pdl.load_dataset("repo1", split="train", streaming=True) as ds:
    train_step(ds[0])
# ds is automatically closed, and background downloaders stopped!

# Multi-dataset scoped block:
with pdl.managed_datasets() as mgr:
    ds_a = pdl.load_dataset("repo_a", streaming=True)
    ds_b = pdl.load_dataset("repo_b", streaming=True)
    evaluate(ds_a, ds_b)
# All datasets registered inside this block are cleanly closed upon exit!
```

---

### 13. Hugging Face Authentication & High Quotas (`HF_TOKEN`)

Authenticated Hugging Face accounts benefit from significantly higher rate limits, increased download throughput, and access to private or gated repositories. `parquet-dataset-loader` seamlessly supports authentication:

- **Automatic Environment Detection**: Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in your environment, and it is automatically applied to all operations:
  ```bash
  export HF_TOKEN="hf_your_token_here"
  ```
- **Login Cache Detection**: If you have logged in via `huggingface-cli login` or Colab secrets, your stored token is detected automatically.
- **Explicit Parameter**:
  ```python
  # Pass token string explicitly
  ds = pdl.load_dataset("org/private-dataset", token="hf_...")

  # Explicitly disable authentication
  ds = pdl.load_dataset("org/public-dataset", token=False)
  ```

Authentication headers (`Authorization: Bearer <token>`) are automatically applied to:
- Hugging Face repository and Parquet split resolution API requests.
- HTTP Range requests for Parquet file footers during metadata indexing.
- Concurrent HTTP Range requests made by `fsspec` when streaming row groups.
- Direct downloads and background prefetching workers.

---

## PyTorch DataLoader Integration

Because `IndexedParquetDataset` implements standard Python sequence protocols (`__len__` and `__getitem__`), you can plug it straight into PyTorch `DataLoader` — including with deterministic shuffling:

```python
from torch.utils.data import DataLoader
import parquet_dataset_loader as pdl

# Load dataset in streaming mode with deterministic shuffling
ds = pdl.load_dataset(
    "KhangTruong/COCO-inpainted",
    split="train",
    streaming=True,
    columns=["mask"],
    shuffle=True,
    seed=42,
    max_cached_row_groups=4,
)

dataloader = DataLoader(ds, batch_size=32)

for batch in dataloader:
    # batch['mask'] is yielded in deterministic shuffled order
    pass
```

---

## API Reference

### `load_dataset(...)`

```python
def load_dataset(
    path: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]],
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    save_to_disk: Union[bool, str] = False,
    background_download: bool = False,
    columns: Optional[Sequence[str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    max_cached_row_groups: int = 2,
    disk_cache: bool = False,
    max_workers: int = 16,
    show_progress: bool = False,
    shuffle: Optional[bool] = None,
    seed: Optional[int] = None,
    buffer_size: Optional[int] = None,
    start_index: Optional[int] = None,
    from_index: Optional[int] = None,
    dataset_id: Optional[str] = None,
    manage: bool = True,
    **kwargs,
) -> Union[IndexedParquetDataset, ParquetDatasetDict]
```

- **`path`**: Hugging Face repo ID, local path, directory, or direct URL.
- **`split`**: Split name (e.g. `'train'`, `'validation'`), split slice (`'train[:1000]'`), or `None` for all splits.
- **`streaming`**: If `True`, enables random-access streaming with zero full-file disk downloads. If `False`, downloads files to disk and opens them locally.
- **`shuffle`**: If `True` (or if `seed` is passed without `shuffle=False`), enables deterministic shuffling. Supported with both streaming and non-streaming.
- **`seed`**: Integer seed for 100% reproducible shuffling.
- **`buffer_size`**: Optional buffer size for streaming buffer-based shuffle.
- **`start_index` / `from_index`**: Row index to start/resume streaming from (0-indexed). Forward background prefetching automatically prioritizes row groups starting from this index.
- **`save_to_disk`**: Path string or `True`. Enables progressive saving to disk while streaming immediately without blocking. Defaults to `False` (disabled by default when `streaming=True`). If `True`, saves to `~/.cache/parquet_dataset_loader/saved`.
- **`background_download`**: If `True`, starts a background worker thread to prefetch and archive remaining row groups to disk.
- **`dataset_id`**: Optional unique name to register this dataset with `DatasetManager`.
- **`manage`**: Whether to register instance with `DatasetManager` (default `True`).
- **`columns`**: Column projection list.
- **`max_cached_row_groups`**: Number of decoded row group tables to keep in RAM simultaneously (default `2`).
- **`disk_cache`**: If `True`, caches fetched row groups to SSD in Feather format for sub-millisecond repeated reads.
- **`max_workers`**: Concurrency level for metadata indexing or file downloads.
- **`show_progress`**: Whether to display progress bars.

### `load_from_disk(...)`

```python
def load_from_disk(
    dataset_path: Optional[str] = None,
    split: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    shuffle: Optional[bool] = None,
    seed: Optional[int] = None,
    buffer_size: Optional[int] = None,
    start_index: Optional[int] = None,
    from_index: Optional[int] = None,
    resume: bool = True,
    allow_incomplete: bool = True,
    background_download: bool = False,
    token: Optional[Union[bool, str]] = None,
    dataset_id: Optional[str] = None,
    manage: bool = True,
    **kwargs,
) -> Union[IndexedParquetDataset, ParquetDatasetDict]
```

### `resume_dataset(...)`

```python
def resume_dataset(
    dataset_path: str,
    source_path: Optional[str] = None,
    split: Optional[str] = None,
    background_download: bool = True,
    columns: Optional[Sequence[str]] = None,
    token: Optional[Union[bool, str]] = None,
    dataset_id: Optional[str] = None,
    manage: bool = True,
    **kwargs,
) -> Union[IndexedParquetDataset, ParquetDatasetDict]
```

### Lifecycle Functions & `DatasetManager`

- **`get_dataset_manager()`**: Access the global `DatasetManager` singleton instance.
- **`list_active_datasets()`**: Return dictionary mapping active dataset IDs to status metadata.
- **`close_all_datasets()`**: Stop background workers and close all open datasets across the process.
- **`managed_datasets()`**: Context manager ensuring all datasets created inside are closed on block exit.
- **`dataset.status`**: Dictionary containing `dataset_id`, `split`, `num_rows`, `save_progress`, `background_downloading`, and `is_closed`.
- **`dataset.close()`**: Cleanly unregister dataset and terminate any background worker.
- **`dataset.start_background_download(start_index=...)`** / **`dataset.stop_background_download()`**: Dynamically control prefetching workers.

---

## Running Tests

The test suite contains 68 comprehensive unit and integration tests:

```bash
# Run unit tests (offline, fast)
pytest -m "not integration"

# Run all tests including live COCO-inpainted integration test
pytest
```

---

## Versioning & Changelog

This project follows [Semantic Versioning](https://semver.org/).

### Version 0.3.0
- **Multiple Dataset Instance Management (`DatasetManager`)**:
  - Centralized thread-safe registry tracking active dataset instances (`ds1`, `ds2`) with unique IDs.
  - Added `get_dataset_manager()`, `list_active_datasets()`, and `close_all_datasets()`.
  - Added `managed_datasets()` and dataset context manager protocol (`with load_dataset(...) as ds:`).
  - Added `status`, `is_closed`, and clean `close()` methods releasing background threads and file descriptors.
- **Stream-from-Index + Forward Background Downloading**:
  - `BackgroundDownloader` prioritizes prefetching row groups forward from `start_index` before wrapping around to earlier row groups.
  - Added dynamic prefetch controls: `ds.start_background_download(start_index=...)` and `ds.stop_background_download()`.
- **Incomplete Dataset Resumption & Disk Reconciliation**:
  - Added `resume_dataset(...)` top-level function for resuming partially-saved datasets.
  - Enhanced `load_from_disk(..., resume=True)` to reconnect to remote sources automatically.
  - Added disk reconciliation in `ProgressiveDiskSaver`: auto-discovers completed `.feather` row groups even if manifests were interrupted, and cleans up orphaned `.tmp` files.
- **9 New Tests & Runnable Example**:
  - Added `tests/test_manager_and_resume.py` (68 total passing tests).
  - Added `examples/manager_and_resume_example.py`.

### Version 0.2.2
- **Reproducible Shuffling with Seed**: Added `shuffle` and `seed` support to `load_dataset`, `IndexedParquetDataset.shuffle(seed=...)`, and `ParquetDatasetDict.shuffle(seed=...)`.
- **Streaming from Index**: Added `start_index` and `from_index` parameter to `load_dataset`, as well as `ds.iter_from(start_index)` and `ds.stream(start_index)`.
- **Streaming Buffer Shuffle**: Added `buffer_size` support to `shuffle(seed=..., buffer_size=...)`.

### Version 0.2.1
- **Full Hugging Face Authentication Support (`HF_TOKEN`)**: Automatic resolution and propagation of authentication tokens (`HF_TOKEN`, Colab secrets, login cache).

### Version 0.2.0
- Simultaneous Streaming + Save-to-Disk with `ProgressiveDiskSaver` and `BackgroundDownloader`.

### Version 0.1.0
- Initial release with HTTP Range-based metadata index extraction and $O(\log M)$ binary-search random-access row locator.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
