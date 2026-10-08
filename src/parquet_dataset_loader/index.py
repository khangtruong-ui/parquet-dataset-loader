"""Metadata indexing and fast random-access locator for Parquet datasets.

This module provides the core data structures and HTTP range request logic to
index remote or local Parquet files by their footers. By fetching only the footer
(typically 20-60 KB) of each Parquet file, it determines the exact schema, total
rows, and row group boundaries without downloading multi-gigabyte data files.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from parquet_dataset_loader.exceptions import (
    CorruptParquetError,
    IndexOutOfBoundsError,
    NetworkRangeError,
    RateLimitError,
)
from parquet_dataset_loader.hf_resolver import create_retry_session, resolve_hf_token


@dataclass
class RowGroupInfo:
    """Metadata for a single row group within a Parquet file.

    Attributes:
        file_index: Zero-based index of the file within the dataset split.
        file_url: URL or local path of the Parquet file.
        rg_index: Zero-based index of this row group within its file.
        num_rows: Total rows contained in this row group.
        global_start_row: Starting row index in the dataset split (inclusive).
        global_end_row: Ending row index in the dataset split (exclusive).
        byte_start: Byte offset of the first column chunk in the row group.
        byte_end: Byte offset of the end of the last column chunk in the row group.
    """

    file_index: int
    file_url: str
    rg_index: int
    num_rows: int
    global_start_row: int
    global_end_row: int
    byte_start: int
    byte_end: int

    @property
    def byte_length(self) -> int:
        """Total byte size of the compressed column data in this row group."""
        return max(0, self.byte_end - self.byte_start)


@dataclass
class ParquetFileInfo:
    """Metadata for a single Parquet file.

    Attributes:
        file_index: Zero-based index of the file within the dataset split.
        url_or_path: Remote URL or local filesystem path.
        file_size: Total file size in bytes.
        num_rows: Total rows contained in this file.
        num_row_groups: Number of row groups in this file.
        global_start_row: Starting row index in the dataset split (inclusive).
        global_end_row: Ending row index in the dataset split (exclusive).
        row_groups: List of RowGroupInfo objects in order.
    """

    file_index: int
    url_or_path: str
    file_size: int
    num_rows: int
    num_row_groups: int
    global_start_row: int
    global_end_row: int
    row_groups: List[RowGroupInfo]


class MetadataIndex:
    """Global index covering all files and row groups in a dataset split.

    Enables O(log M) random-access lookups from any global dataset row index
    directly to the exact Parquet file, row group, and local row offset.

    Attributes:
        split_name: Name of the split (e.g. 'train', 'validation').
        total_rows: Total row count across all files in the split.
        schema: PyArrow Schema for the dataset.
        column_names: Ordered list of column names.
        files: Ordered list of ParquetFileInfo descriptors.
        row_groups: Flattened list of all RowGroupInfo descriptors in global order.
        features_info: Optional dictionary extracted from Hugging Face metadata.
    """

    def __init__(
        self,
        split_name: str,
        total_rows: int,
        schema: pa.Schema,
        files: List[ParquetFileInfo],
        row_groups: List[RowGroupInfo],
        features_info: Optional[dict] = None,
    ) -> None:
        self.split_name = split_name
        self.total_rows = total_rows
        self.schema = schema
        self.column_names = [field.name for field in schema]
        self.files = files
        self.row_groups = row_groups
        self.features_info = features_info

        # Precompute end boundary list for O(log M) binary search
        self._rg_end_rows = [rg.global_end_row for rg in self.row_groups]

    def locate_row(self, global_row_index: int) -> Tuple[RowGroupInfo, int]:
        """Locate the row group and in-group row offset for a global row index.

        Args:
            global_row_index: Zero-based global index in the split. Negative
                indices are supported and resolve from the end of the split.

        Returns:
            A tuple of (RowGroupInfo, local_row_index) where local_row_index is
            the zero-based offset within the row group table.

        Raises:
            IndexOutOfBoundsError: If the index is out of the valid range.
        """
        if global_row_index < 0:
            global_row_index += self.total_rows

        if global_row_index < 0 or global_row_index >= self.total_rows:
            raise IndexOutOfBoundsError(global_row_index, self.total_rows)

        # Binary search for row group
        rg_idx = bisect_right(self._rg_end_rows, global_row_index)
        rg = self.row_groups[rg_idx]
        local_row = global_row_index - rg.global_start_row
        return rg, local_row

    def locate_range(
        self, start: int, stop: int
    ) -> List[Tuple[RowGroupInfo, int, int]]:
        """Locate all row groups and their local slices for a global range [start, stop).

        Args:
            start: Global start index (inclusive, 0 <= start <= total_rows).
            stop: Global stop index (exclusive, 0 <= stop <= total_rows).

        Returns:
            List of tuples (RowGroupInfo, local_start, local_stop) representing
            the segments across row groups that cover [start, stop).
        """
        start = max(0, min(start, self.total_rows))
        stop = max(start, min(stop, self.total_rows))

        if start >= stop:
            return []

        first_rg_idx = bisect_right(self._rg_end_rows, start)
        last_rg_idx = bisect_right(self._rg_end_rows, stop - 1)

        result: List[Tuple[RowGroupInfo, int, int]] = []
        for i in range(first_rg_idx, last_rg_idx + 1):
            rg = self.row_groups[i]
            rg_start = max(start, rg.global_start_row)
            rg_stop = min(stop, rg.global_end_row)
            local_start = rg_start - rg.global_start_row
            local_stop = rg_stop - rg.global_start_row
            result.append((rg, local_start, local_stop))

        return result

    def to_dict(self) -> dict:
        """Serialize the index structure to a JSON-compatible dictionary."""
        schema_bytes = self.schema.serialize().to_pybytes()
        return {
            "split_name": self.split_name,
            "total_rows": self.total_rows,
            "schema_hex": schema_bytes.hex(),
            "features_info": self.features_info,
            "files": [asdict(f) for f in self.files],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MetadataIndex:
        """Reconstruct a MetadataIndex from a dictionary."""
        schema_bytes = bytes.fromhex(data["schema_hex"])
        schema = pa.ipc.read_schema(pa.py_buffer(schema_bytes))

        files: List[ParquetFileInfo] = []
        all_row_groups: List[RowGroupInfo] = []

        for f_data in data["files"]:
            row_groups = [RowGroupInfo(**rg_dict) for rg_dict in f_data["row_groups"]]
            all_row_groups.extend(row_groups)
            files.append(
                ParquetFileInfo(
                    file_index=f_data["file_index"],
                    url_or_path=f_data["url_or_path"],
                    file_size=f_data["file_size"],
                    num_rows=f_data["num_rows"],
                    num_row_groups=f_data["num_row_groups"],
                    global_start_row=f_data["global_start_row"],
                    global_end_row=f_data["global_end_row"],
                    row_groups=row_groups,
                )
            )

        return cls(
            split_name=data["split_name"],
            total_rows=data["total_rows"],
            schema=schema,
            files=files,
            row_groups=all_row_groups,
            features_info=data.get("features_info"),
        )

    def save_cache(self, cache_file: str) -> None:
        """Save this metadata index to a local JSON cache file."""
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_cache(cls, cache_file: str) -> Optional[MetadataIndex]:
        """Load a metadata index from a local JSON cache file if it exists."""
        if not os.path.exists(cache_file):
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception:
            return None


def calculate_row_group_byte_span(rg: pq.RowGroupMetaData) -> Tuple[int, int]:
    """Calculate the byte range [byte_start, byte_end] occupied by a row group.

    Args:
        rg: PyArrow RowGroupMetaData object.

    Returns:
        Tuple of (byte_start, byte_end) offsets.
    """
    min_offset = float("inf")
    max_offset = 0

    for i in range(rg.num_columns):
        col = rg.column(i)
        offset = col.data_page_offset
        if col.has_dictionary_page and col.dictionary_page_offset is not None and col.dictionary_page_offset > 0:
            offset = min(offset, col.dictionary_page_offset)

        min_offset = min(min_offset, offset)
        max_offset = max(max_offset, offset + col.total_compressed_size)

    if min_offset == float("inf"):
        min_offset = 0
    return int(min_offset), int(max_offset)


def fetch_remote_parquet_footer(
    url: str,
    session: requests.Session,
    token: Optional[Union[bool, str]] = None,
    timeout: int = 20,
) -> Tuple[int, pq.FileMetaData]:
    """Fetch and parse the Parquet FileMetaData footer via HTTP Range requests.

    Uses a single suffix byte range request (last 64 KB) to retrieve the footer.
    If the footer exceeds 64 KB, it performs an exact range request for the remainder.

    Args:
        url: Remote HTTP/HTTPS Parquet URL.
        session: Configured requests.Session.
        token: Optional authentication Bearer token.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (total_file_size, FileMetaData).

    Raises:
        NetworkRangeError: If range requests fail or return non-206 / non-200.
        RateLimitError: If HTTP 429 Too Many Requests is encountered.
        CorruptParquetError: If the footer magic bytes are invalid.
    """
    auth_token = resolve_hf_token(token)
    headers = {"Range": "bytes=-65536"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        resp = session.get(url, headers=headers, timeout=timeout)
    except Exception as e:
        raise NetworkRangeError(url, 0, f"Connection error: {e}") from e

    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        secs = int(retry_after) if retry_after and retry_after.isdigit() else None
        raise RateLimitError(url, retry_after=secs)

    if resp.status_code not in (200, 206):
        raise NetworkRangeError(url, resp.status_code, resp.text[:200])

    content = resp.content
    if len(content) < 8:
        raise CorruptParquetError(url, "Response too short to contain Parquet footer.")

    magic = content[-4:]
    if magic != b"PAR1":
        raise CorruptParquetError(url, f"Invalid Parquet magic bytes: {magic!r}")

    footer_len = struct.unpack("<I", content[-8:-4])[0]

    # Extract total file size from Content-Range header
    content_range = resp.headers.get("Content-Range", "")
    total_size = 0
    if "/" in content_range:
        try:
            total_size = int(content_range.split("/")[-1])
        except ValueError:
            pass

    if total_size == 0 and resp.status_code == 200:
        total_size = len(content)

    # Check if footer was fully included in the 64 KB tail
    if footer_len + 8 <= len(content):
        footer_bytes = content[-(footer_len + 8) :]
    else:
        # Footer is larger than 64KB, fetch exact byte range
        if total_size == 0:
            head_headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
            head = session.head(url, headers=head_headers)
            total_size = int(head.headers.get("content-length", 0))

        start_byte = total_size - (footer_len + 8)
        headers2 = {"Range": f"bytes={start_byte}-{total_size - 1}"}
        if auth_token:
            headers2["Authorization"] = f"Bearer {auth_token}"
        resp2 = session.get(url, headers=headers2, timeout=timeout)
        if resp2.status_code not in (200, 206):
            raise NetworkRangeError(url, resp2.status_code, "Failed fetching full footer")
        footer_bytes = resp2.content

    try:
        meta = pq.read_metadata(io.BytesIO(b"PAR1" + footer_bytes))
    except Exception as e:
        raise CorruptParquetError(url, f"Failed parsing metadata thrift buffer: {e}") from e

    return total_size, meta


def read_local_parquet_metadata(path: str) -> Tuple[int, pq.FileMetaData]:
    """Read Parquet metadata from a local file.

    Args:
        path: Path to the local Parquet file.

    Returns:
        Tuple of (file_size, FileMetaData).
    """
    abs_path = os.path.abspath(os.path.expanduser(path))
    file_size = os.path.getsize(abs_path)
    meta = pq.read_metadata(abs_path)
    return file_size, meta


def _index_single_file(
    file_idx: int,
    url_or_path: str,
    session: requests.Session,
    token: Optional[Union[bool, str]] = None,
) -> Tuple[int, int, pq.FileMetaData]:
    """Index a single file (remote or local).

    Returns:
        Tuple of (file_idx, file_size, FileMetaData).
    """
    is_remote = url_or_path.startswith(("http://", "https://"))
    if is_remote:
        file_size, meta = fetch_remote_parquet_footer(url_or_path, session=session, token=token)
    else:
        file_size, meta = read_local_parquet_metadata(url_or_path)
    return file_idx, file_size, meta


def build_metadata_index(
    files: Sequence[str],
    split_name: str = "train",
    cache_dir: Optional[str] = None,
    max_workers: int = 16,
    token: Optional[Union[bool, str]] = None,
    use_cache: bool = True,
) -> MetadataIndex:
    """Build a complete MetadataIndex across a sequence of Parquet files.

    Concurrently retrieves the metadata footer of each file, computes cumulative
    row offsets, extracts the PyArrow schema, and builds a fast binary-searchable
    row-group lookup index.

    Args:
        files: List of file URLs or local file paths.
        split_name: Split name (e.g. 'train').
        cache_dir: Optional directory to read/write cached metadata index.
        max_workers: Maximum threads for concurrent footer requests.
        token: Optional Hugging Face auth token (or True to use stored/env token).
        use_cache: Whether to use disk caching for the index.

    Returns:
        An initialized MetadataIndex.
    """
    if not files:
        raise ValueError("Cannot build MetadataIndex from an empty list of files.")

    auth_token = resolve_hf_token(token)

    # Determine cache path
    cache_file: Optional[str] = None
    if cache_dir and use_cache:
        hasher = hashlib.sha256()
        hasher.update(split_name.encode("utf-8"))
        for f in files:
            hasher.update(f.encode("utf-8"))
            if not f.startswith(("http://", "https://")) and os.path.exists(f):
                try:
                    stat = os.stat(f)
                    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
                    hasher.update(str(stat.st_size).encode("utf-8"))
                except OSError:
                    pass
        cache_key = hasher.hexdigest()[:24]
        cache_file = os.path.join(cache_dir, "indices", f"{split_name}_{cache_key}.json")

        cached_index = MetadataIndex.load_cache(cache_file)
        if cached_index is not None and cached_index.total_rows > 0:
            is_valid = True
            for file_info in cached_index.files:
                if not file_info.url_or_path.startswith(("http://", "https://")):
                    if not os.path.exists(file_info.url_or_path):
                        is_valid = False
                        break
                    if os.path.getsize(file_info.url_or_path) != file_info.file_size:
                        is_valid = False
                        break
            if is_valid:
                return cached_index

    session = create_retry_session()
    raw_results: List[Optional[Tuple[int, pq.FileMetaData]]] = [None] * len(files)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(files))) as executor:
        futures = {
            executor.submit(_index_single_file, idx, url, session, auth_token): idx
            for idx, url in enumerate(files)
        }
        for future in as_completed(futures):
            idx, file_size, meta = future.result()
            raw_results[idx] = (file_size, meta)

    # Assemble ParquetFileInfo and RowGroupInfo with global row offsets
    file_infos: List[ParquetFileInfo] = []
    all_row_groups: List[RowGroupInfo] = []
    current_global_row = 0
    reference_schema: Optional[pa.Schema] = None
    features_info: Optional[dict] = None

    for idx, url in enumerate(files):
        res = raw_results[idx]
        assert res is not None
        file_size, meta = res

        if reference_schema is None:
            reference_schema = meta.schema.to_arrow_schema()
            # Extract HF features metadata if available
            raw_meta = reference_schema.metadata or {}
            if b"huggingface" in raw_meta:
                try:
                    features_info = json.loads(raw_meta[b"huggingface"].decode("utf-8"))
                except Exception:
                    pass

        file_start_row = current_global_row
        rg_infos: List[RowGroupInfo] = []

        for rg_idx in range(meta.num_row_groups):
            rg = meta.row_group(rg_idx)
            rg_rows = rg.num_rows
            rg_start = current_global_row
            rg_end = current_global_row + rg_rows
            current_global_row += rg_rows

            byte_start, byte_end = calculate_row_group_byte_span(rg)
            rg_info = RowGroupInfo(
                file_index=idx,
                file_url=url,
                rg_index=rg_idx,
                num_rows=rg_rows,
                global_start_row=rg_start,
                global_end_row=rg_end,
                byte_start=byte_start,
                byte_end=byte_end,
            )
            rg_infos.append(rg_info)
            all_row_groups.append(rg_info)

        file_info = ParquetFileInfo(
            file_index=idx,
            url_or_path=url,
            file_size=file_size,
            num_rows=meta.num_rows,
            num_row_groups=meta.num_row_groups,
            global_start_row=file_start_row,
            global_end_row=current_global_row,
            row_groups=rg_infos,
        )
        file_infos.append(file_info)

    assert reference_schema is not None

    index = MetadataIndex(
        split_name=split_name,
        total_rows=current_global_row,
        schema=reference_schema,
        files=file_infos,
        row_groups=all_row_groups,
        features_info=features_info,
    )

    if cache_file:
        index.save_cache(cache_file)

    return index
