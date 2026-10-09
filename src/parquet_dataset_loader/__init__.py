"""parquet_dataset_loader.

A high-performance, memory-efficient loader for large Hugging Face Parquet datasets
featuring instant random-access index streaming without downloading multi-gigabyte files.
"""

from parquet_dataset_loader.api import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SAVE_DIR,
    load_dataset,
    load_from_disk,
    resume_dataset,
)
from parquet_dataset_loader.dataset import (
    BlockShuffledSampler,
    IndexedParquetDataset,
    ParquetDatasetDict,
)
from parquet_dataset_loader.exceptions import (
    CorruptParquetError,
    DatasetNotFoundError,
    IndexOutOfBoundsError,
    NetworkRangeError,
    ParquetDatasetError,
    RateLimitError,
    SplitNotFoundError,
)
from parquet_dataset_loader.hf_resolver import (
    infer_split_name,
    parse_split_slice,
    resolve_hf_token,
    resolve_parquet_dataset,
)
from parquet_dataset_loader.index import (
    MetadataIndex,
    ParquetFileInfo,
    RowGroupInfo,
    build_metadata_index,
)
from parquet_dataset_loader.manager import (
    DatasetManager,
    cleanup_background_tasks,
    close_all_datasets,
    get_dataset_manager,
    list_active_datasets,
    managed_datasets,
    stop_all_background_tasks,
)
from parquet_dataset_loader.progressive import BackgroundDownloader, ProgressiveDiskSaver
from parquet_dataset_loader.reader import RowGroupReader, download_parquet_files

__version__ = "0.3.2"
__all__ = [
    "__version__",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_SAVE_DIR",
    "load_dataset",
    "load_from_disk",
    "resume_dataset",
    "BlockShuffledSampler",
    "IndexedParquetDataset",
    "ParquetDatasetDict",
    "DatasetManager",
    "get_dataset_manager",
    "list_active_datasets",
    "close_all_datasets",
    "stop_all_background_tasks",
    "cleanup_background_tasks",
    "managed_datasets",
    "MetadataIndex",
    "ParquetFileInfo",
    "RowGroupInfo",
    "RowGroupMemoryCache",
    "DiskCache",
    "ProgressiveDiskSaver",
    "BackgroundDownloader",
    "RowGroupReader",
    "download_parquet_files",
    "build_metadata_index",
    "resolve_parquet_dataset",
    "resolve_hf_token",
    "parse_split_slice",
    "infer_split_name",
    "ParquetDatasetError",
    "DatasetNotFoundError",
    "SplitNotFoundError",
    "IndexOutOfBoundsError",
    "CorruptParquetError",
    "NetworkRangeError",
    "RateLimitError",
]

