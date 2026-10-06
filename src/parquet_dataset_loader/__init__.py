"""parquet_dataset_loader.

A high-performance, memory-efficient loader for large Hugging Face Parquet datasets
featuring instant random-access index streaming without downloading multi-gigabyte files.
"""

from parquet_dataset_loader.api import load_dataset, load_from_disk
from parquet_dataset_loader.cache import DiskCache, RowGroupMemoryCache
from parquet_dataset_loader.dataset import IndexedParquetDataset, ParquetDatasetDict
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
    resolve_parquet_dataset,
)
from parquet_dataset_loader.index import (
    MetadataIndex,
    ParquetFileInfo,
    RowGroupInfo,
    build_metadata_index,
)
from parquet_dataset_loader.reader import RowGroupReader, download_parquet_files

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "load_dataset",
    "load_from_disk",
    "IndexedParquetDataset",
    "ParquetDatasetDict",
    "MetadataIndex",
    "ParquetFileInfo",
    "RowGroupInfo",
    "RowGroupMemoryCache",
    "DiskCache",
    "RowGroupReader",
    "download_parquet_files",
    "build_metadata_index",
    "resolve_parquet_dataset",
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
