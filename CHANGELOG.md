# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
