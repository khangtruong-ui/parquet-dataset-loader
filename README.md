# parquet-dataset-loader

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-orange.svg)](pyproject.toml)

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

### 3. Seekable Streaming (`iter_from`)

Resume training or start reading from any arbitrary index instantly:

```python
# Stream starting from index 100,000 without reading earlier rows
for row in ds.iter_from(start_index=100000):
    process(row)
```

### 4. Zero-Copy Views (`take`, `skip`, `slice`)

Create sub-dataset views without copying data:

```python
train_sub = ds.take(1000)      # First 1,000 rows
train_tail = ds.skip(100000)   # Rows from 100,000 onwards
train_slice = ds.slice(500, 200) # 200 rows starting at 500
```

### 5. Conversion to Hugging Face, Arrow, and Pandas

```python
# Convert a slice or small dataset to a Hugging Face Dataset
hf_dataset = ds.take(100).to_hf_dataset()

# Convert to PyArrow Table
arrow_table = ds.take(100).to_arrow()

# Convert to Pandas DataFrame
df = ds.take(100).to_pandas()
```

### 6. Full Split Dictionary Handling (`split=None`)

```python
# Loads all splits into a ParquetDatasetDict
ds_dict = pdl.load_dataset("KhangTruong/COCO-inpainted", split=None, streaming=True)

print(ds_dict.keys())  # ['train', 'validation']
print(f"Train: {len(ds_dict['train']):,} | Validation: {len(ds_dict['validation']):,}")
```

### 7. Disk Loading & Saving

```python
# Save dataset to disk in Arrow IPC (Feather) format
ds.take(500).save_to_disk("./my_local_data")

# Reload from disk
reloaded = pdl.load_from_disk("./my_local_data")
print(len(reloaded))  # 500
```

---

## PyTorch DataLoader Integration

Because `IndexedParquetDataset` implements standard Python sequence protocols (`__len__` and `__getitem__`), you can plug it straight into PyTorch `DataLoader`:

```python
from torch.utils.data import DataLoader
import parquet_dataset_loader as pdl

ds = pdl.load_dataset(
    "KhangTruong/COCO-inpainted",
    split="train",
    streaming=True,
    columns=["mask"],
    max_cached_row_groups=4,
)

dataloader = DataLoader(ds, batch_size=32, shuffle=False)

for batch in dataloader:
    # batch['mask'] is yielded batch by batch
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
    columns: Optional[Sequence[str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    max_cached_row_groups: int = 2,
    disk_cache: bool = False,
    max_workers: int = 16,
    show_progress: bool = False,
    **kwargs,
) -> Union[IndexedParquetDataset, ParquetDatasetDict]
```

- **`path`**: Hugging Face repo ID, local path, directory, or direct URL.
- **`split`**: Split name (e.g. `'train'`, `'validation'`), split slice (`'train[:1000]'`), or `None` for all splits.
- **`streaming`**: If `True`, enables random-access streaming with zero full-file disk downloads. If `False`, downloads files to disk and opens them locally.
- **`columns`**: Column projection list.
- **`max_cached_row_groups`**: Number of decoded row group tables to keep in RAM simultaneously (default `2`).
- **`disk_cache`**: If `True`, caches fetched row groups to SSD in Feather format for sub-millisecond repeated reads.

### `load_from_disk(dataset_path, columns=None)`
Reloads a dataset or multi-split dataset directory saved via `dataset.save_to_disk(...)`.

---

## Running Tests

The test suite contains 32 comprehensive unit and integration tests:

```bash
# Run unit tests (offline, fast)
pytest -m "not integration"

# Run all tests including live COCO-inpainted integration test
pytest
```

---

## Versioning & Changelog

This project follows [Semantic Versioning](https://semver.org/).

### Version 0.1.0
- Initial release.
- HTTP Range-based metadata index extraction (`Range: bytes=-65536`).
- $O(\log M)$ binary-search random-access row locator.
- `IndexedParquetDataset` and `ParquetDatasetDict` abstractions.
- Thread-safe `RowGroupMemoryCache` (LRU) and `DiskCache` (Feather).
- Full column projection support.
- Hugging Face `load_dataset`-compatible API and split slice syntax (`train[:1000]`).
- Verification on `KhangTruong/COCO-inpainted` (62.4 GB).

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
