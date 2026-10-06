"""Custom exceptions for parquet_dataset_loader.

This module defines the exception hierarchy used across parquet_dataset_loader,
enabling callers to catch specific error conditions such as dataset resolution
failures, missing splits, index out-of-bounds errors, and network range request issues.
"""

from typing import Optional


class ParquetDatasetError(Exception):
    """Base exception for all errors raised by parquet_dataset_loader."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class DatasetNotFoundError(ParquetDatasetError):
    """Raised when a dataset repository, path, or file cannot be located."""

    def __init__(self, dataset_path: str, reason: Optional[str] = None) -> None:
        msg = f"Dataset not found at '{dataset_path}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, details={"dataset_path": dataset_path, "reason": reason})
        self.dataset_path = dataset_path


class SplitNotFoundError(ParquetDatasetError):
    """Raised when the requested dataset split does not exist."""

    def __init__(self, split: str, available_splits: Optional[list] = None) -> None:
        avail_str = f" Available splits: {available_splits}" if available_splits else ""
        msg = f"Split '{split}' not found.{avail_str}"
        super().__init__(msg, details={"split": split, "available_splits": available_splits})
        self.split = split
        self.available_splits = available_splits or []


class IndexOutOfBoundsError(IndexError, ParquetDatasetError):
    """Raised when an index is outside the valid range of the dataset."""

    def __init__(self, index: int, total_rows: int) -> None:
        msg = f"Index {index} is out of bounds for dataset of length {total_rows} (valid range: 0 to {total_rows - 1})"
        # Initialize both parent classes appropriately
        IndexError.__init__(self, msg)
        ParquetDatasetError.__init__(self, msg, details={"index": index, "total_rows": total_rows})
        self.index = index
        self.total_rows = total_rows


class CorruptParquetError(ParquetDatasetError):
    """Raised when a Parquet file's metadata or data is malformed or unreadable."""

    def __init__(self, source: str, reason: str) -> None:
        msg = f"Failed to read Parquet data from '{source}': {reason}"
        super().__init__(msg, details={"source": source, "reason": reason})
        self.source = source
        self.reason = reason


class NetworkRangeError(ParquetDatasetError):
    """Raised when an HTTP range request fails or returns an unexpected status code."""

    def __init__(self, url: str, status_code: int, message: str) -> None:
        msg = f"HTTP range request to '{url}' failed with status {status_code}: {message}"
        super().__init__(msg, details={"url": url, "status_code": status_code})
        self.url = url
        self.status_code = status_code


class RateLimitError(NetworkRangeError):
    """Raised when the remote server (e.g. Hugging Face) returns HTTP 429 Too Many Requests."""

    def __init__(self, url: str, retry_after: Optional[int] = None) -> None:
        msg = f"Rate limit exceeded (HTTP 429) for '{url}'."
        if retry_after:
            msg += f" Retry after {retry_after} seconds."
        super().__init__(url=url, status_code=429, message=msg)
        self.retry_after = retry_after
