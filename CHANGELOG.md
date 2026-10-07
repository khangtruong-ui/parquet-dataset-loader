# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-07

### Added
- **Multiple Dataset Instance Management (`DatasetManager`)**:
  - Centralized, thread-safe lifecycle registry (`DatasetManager`) tracking all active dataset instances (`IndexedParquetDataset` and `ParquetDatasetDict`).
  - Added `dataset_id` tracking, with automatic disambiguation on ID collisions (e.g. `ds_name_1`).
  - Exposed module-level lifecycle controls:
    - `get_dataset_manager()`: Access the global registry singleton.
    - `list_active_datasets()`: Return status dictionaries (row count, columns, download progress, cache states, background workers).
    - `close_all_datasets()`: Coordinated shutdown of all active datasets, file descriptors, and worker threads.
    - `managed_datasets()`: Context manager ensuring all datasets created inside are cleanly unmounted and closed upon exit.
  - Added context manager protocol to `IndexedParquetDataset` and `ParquetDatasetDict` (`with load_dataset(...) as ds:`).
  - Added `status` and `is_closed` properties and explicit `close()` method to dataset classes.
- **Stream-from-Index + Forward Background Downloading**:
  - Updated `BackgroundDownloader` with `start_rg_index`: prioritized prefetching downloads forward from the row group of `start_index` to optimize upcoming reads, before wrapping around to earlier row groups.
  - Added dynamic worker controls: `ds.start_background_download(start_index=...)` and `ds.stop_background_download()`.
- **Incomplete Dataset Resumption & Disk Reconciliation**:
  - Manifest format now records upstream repository `source_path` and `total_row_groups`.
  - Added disk reconciliation to `ProgressiveDiskSaver`: discovers already-saved `.feather` row groups on disk (recovering from lost manifests) and cleans up stale `.tmp` files.
  - Added `resume_dataset(dataset_path, ...)` top-level function.
  - Added `resume=True` and `allow_incomplete=True` support to `load_from_disk(...)`.
- **Unit Tests and Documentation**:
  - Added `tests/test_manager_and_resume.py` with 9 unit tests verifying multi-instance tracking, prefetching priority, disk recovery, and resumption.
  - Added `examples/manager_and_resume_example.py` runnable demonstration script.
  - Full test suite now consists of 68 passing unit and integration tests.

## [0.2.2] - 2026-10-07

### Added
- **Reproducible Shuffling with Seed (`shuffle` / `seed`)**:
  - Full support for `load_dataset(..., shuffle=True, seed=42)` across both streaming (`streaming=True`) and non-streaming (`streaming=False`) modes.
  - Added `.shuffle(seed=...)` to `IndexedParquetDataset` and `ParquetDatasetDict`.
  - 100% deterministic pseudo-random ordering when provided a seed; different seeds produce distinct permutations.
  - Preserves instant $O(\log M)$ random-access indexing, batch slicing matching HF format, and PyTorch `DataLoader` compatibility on shuffled datasets.
  - Streaming buffer shuffle (`buffer_size=N`) for bounded-memory randomized streaming over remote datasets.
  - Optimized PyArrow, Pandas, and Hugging Face Dataset conversions (`to_arrow()`, `to_pandas()`, `to_hf_dataset()`) grouping row group decodes to eliminate cache evictions on shuffled datasets.
- **Streaming from Index (`start_index` / `from_index` / `iter_from` / `stream`)**:
  - Added `start_index` and `from_index` parameters to `load_dataset(...)` and `load_from_disk(...)`.
  - Added `stream()` and `stream_from()` convenience methods to `IndexedParquetDataset` aliasing `iter_from()`.
  - Seamlessly combines with shuffling: stream or resume training from an arbitrary index of a seed-permuted dataset.
  - Added `skip()`, `take()`, and `slice()` utility methods on `ParquetDatasetDict`.
- **Comprehensive Test Suite**:
  - Added `tests/test_shuffle_and_stream.py` with 15 tests verifying reproducibility, indexing, slicing, Arrow/Pandas/HF conversions, disk saving, and both streaming/non-streaming parameter configurations.
- **Runnable Example**:
  - Added `examples/shuffle_and_streaming.py` demonstrating shuffle with seed and streaming from index.

## [0.2.1] - 2026-10-07

### Added
- **Full Hugging Face Authentication Support (`HF_TOKEN`)**:
  - Added `resolve_hf_token()` supporting `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` environment variables, `huggingface-cli login` cached credentials, Colab secrets, explicit token strings, and `token=False` to explicitly disable auth.
  - Automatically propagates authenticated headers (`Authorization: Bearer <token>`) across all components:
    - Hugging Face split resolution API requests.
    - HTTP Range requests for Parquet file footers during metadata indexing (`build_metadata_index`).
    - Concurrent HTTP Range requests made by `fsspec` when streaming row groups (`RowGroupReader`).
    - File downloads and background prefetching workers (`download_parquet_files`).
  - Enables access to gated/private repositories and unlocks higher rate limits and download quotas for authenticated accounts.
- Unit tests in `tests/test_reader.py` covering LRU file handle eviction and default capacity eviction beyond 8 files.
- Unit test in `tests/test_progressive.py` covering multi-file dataset `stream_and_save` beyond `max_open_files` followed by offline `load_from_disk`.
- Unit tests in `tests/test_hf_resolver.py` and `tests/test_api.py` covering token resolution and propagation across components.

### Fixed
- Fixed `TypeError: cannot unpack non-iterable ParquetFile object` in `RowGroupReader._get_parquet_file` when open file handles reach `max_open_files` capacity (default 8). Corrected tuple unpacking by using `OrderedDict.popitem(last=False)`.
- Implemented true LRU eviction ordering for open file handles with `OrderedDict.move_to_end()` on cache hits.
- Added file descriptor cleanup if `ParquetFile` initialization fails.
- Fixed file ordering in `load_from_disk()` for progressively saved row group files.

## [0.2.0] - 2026-10-06

### Added
- **Simultaneous Streaming + Save-to-Disk**:
  - `save_to_disk` parameter in `load_dataset`: allows callers to start streaming immediately with zero initial wait time (unblocked step 0), while progressively persisting accessed row groups to local disk in Feather format.
  - `ProgressiveDiskSaver`: thread-safe manager maintaining an atomic manifest of saved row groups on disk, enabling seamless resuming and offline usage.
  - `BackgroundDownloader`: optional background worker thread (`background_download=True`) that prefetches and archives remaining row groups asynchronously while foreground iteration proceeds unblocked.
  - `ds.stream_and_save()`: streams through the entire dataset with optional progress bar, ensuring all row groups are persisted to disk.
  - `ds.save_progress` and `ds.is_fully_saved` monitoring properties on `IndexedParquetDataset`.
  - Full compatibility in `load_from_disk()` to reload progressively saved dataset directories.
  - New example `examples/streaming_and_saving.py` demonstrating zero-wait streaming while archiving to disk.
  - Dedicated unit tests in `tests/test_progressive.py`.

## [0.1.0] - 2026-10-06

### Added
- **Core Package**: `parquet_dataset_loader` pip installable module.
- **Instant Metadata Indexing**:
  - HTTP Range request (`Range: bytes=-65536`) suffix footer parsing for remote Parquet files.
  - Concurrently indexes multi-gigabyte/terabyte datasets in seconds without downloading data bodies.
  - Generates cumulative global row boundaries and per-row-group byte ranges.
- **$O(\log M)$ Random-Access Indexing**:
  - `IndexedParquetDataset` supporting random access `ds[i]`, negative indexing, and slicing.
  - Direct seeking to arbitrary global rows (e.g., `ds[50000]`) without scanning or downloading earlier files.
  - Hugging Face-compatible batch return format on slices (`{column: [values]}`).
- **Memory & Bandwidth Control**:
  - Thread-safe `RowGroupMemoryCache` (LRU eviction) bounding memory usage to configurable row group count.
  - Column projection: fetch only selected columns (e.g. metadata) over HTTP range requests, avoiding large binary payloads.
- **Disk Support**:
  - `streaming=False` mode for full-file downloading and local loading.
  - `save_to_disk()` and `load_from_disk()` with fast memory-mapped Feather format.
  - Hybrid `disk_cache=True` for caching fetched row groups during streaming.
- **Hugging Face Compatibility**:
  - `load_dataset()` API matching Hugging Face signature and split slice syntax (`train[:1000]`).
  - `ParquetDatasetDict` for multi-split datasets.
  - Conversion methods: `to_hf_dataset()`, `to_arrow()`, `to_pandas()`.
- **Test Suite**:
  - 32 pytest unit and integration tests covering metadata indexing, LRU caching, disk caching, resolver, sequence protocol, and live `KhangTruong/COCO-inpainted` streaming.
