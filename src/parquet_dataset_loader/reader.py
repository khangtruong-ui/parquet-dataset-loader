"""Row group reader and concurrent downloader.

This module provides the I/O layer for reading individual row groups over HTTP
range requests or local disk, applying column projection, and managing connection pools.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from parquet_dataset_loader.cache import DiskCache, RowGroupMemoryCache
from parquet_dataset_loader.exceptions import CorruptParquetError, ParquetDatasetError
from parquet_dataset_loader.hf_resolver import create_retry_session, resolve_hf_token


class RowGroupReader:
    """Reads individual row groups from local or remote Parquet files with caching.

    Maintains open ParquetFile handles in an LRU connection pool to avoid redundant
    HTTP connection handshakes and metadata parsing.

    Attributes:
        memory_cache: LRU memory cache for decoded PyArrow tables.
        disk_cache: Optional persistent disk cache.
        token: Optional authentication Bearer token.
        block_size: Read block size for remote fsspec HTTP streams.
    """

    def __init__(
        self,
        memory_cache: Optional[RowGroupMemoryCache] = None,
        disk_cache: Optional[DiskCache] = None,
        progressive_saver: Optional[Any] = None,
        token: Optional[Union[bool, str]] = None,
        block_size: int = 2 * 1024 * 1024,  # 2 MB default block size
        max_open_files: int = 8,
    ) -> None:
        self.memory_cache = (
            memory_cache if memory_cache is not None else RowGroupMemoryCache(max_entries=2)
        )
        self.disk_cache = disk_cache
        self.progressive_saver = progressive_saver
        self.token = resolve_hf_token(token)
        self.block_size = block_size
        self.max_open_files = max(1, max_open_files)

        self._open_files: OrderedDict[str, Tuple[Any, pq.ParquetFile]] = OrderedDict()
        self._lock = threading.Lock()

        # Configure filesystem
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self._http_fs = fsspec.filesystem("http", headers=headers)

    def _get_parquet_file(self, file_url_or_path: str) -> pq.ParquetFile:
        """Obtain a reusable ParquetFile instance for the specified file."""
        with self._lock:
            if file_url_or_path in self._open_files:
                self._open_files.move_to_end(file_url_or_path)
                _, pf = self._open_files[file_url_or_path]
                return pf

            # Close oldest file handle if capacity reached
            if len(self._open_files) >= self.max_open_files:
                oldest_url, (fp, _) = self._open_files.popitem(last=False)
                try:
                    fp.close()
                except Exception:
                    pass

            is_remote = file_url_or_path.startswith(("http://", "https://"))
            fp = None
            try:
                if is_remote:
                    fp = self._http_fs.open(file_url_or_path, "rb", block_size=self.block_size)
                    pf = pq.ParquetFile(fp)
                else:
                    abs_path = os.path.abspath(os.path.expanduser(file_url_or_path))
                    fp = open(abs_path, "rb")
                    pf = pq.ParquetFile(fp, memory_map=True)
            except Exception:
                if fp is not None:
                    try:
                        fp.close()
                    except Exception:
                        pass
                raise

            self._open_files[file_url_or_path] = (fp, pf)
            return pf

    def read_row_group(
        self,
        file_url: str,
        rg_index: int,
        columns: Optional[Sequence[str]] = None,
        file_index: int = 0,
    ) -> pa.Table:
        """Read a single row group table, checking memory, progressive, and disk caches first.

        Args:
            file_url: URL or local path to the Parquet file.
            rg_index: Row group index within the file.
            columns: Optional subset of columns to read.
            file_index: Index of the file within the dataset split (for progressive saving).

        Returns:
            Decoded PyArrow Table containing the rows of this row group.

        Raises:
            CorruptParquetError: If reading or decoding the row group fails.
        """
        # 1. Check in-memory LRU cache
        cached = self.memory_cache.get(file_url, rg_index, columns)
        if cached is not None:
            return cached

        # 2. Check progressive disk saver if configured
        if self.progressive_saver is not None:
            cached_prog = self.progressive_saver.get(file_index, rg_index, columns)
            if cached_prog is not None:
                self.memory_cache.put(file_url, rg_index, cached_prog, columns)
                return cached_prog

        # 3. Check disk cache if configured
        if self.disk_cache is not None:
            cached_disk = self.disk_cache.get(file_url, rg_index, columns)
            if cached_disk is not None:
                self.memory_cache.put(file_url, rg_index, cached_disk, columns)
                return cached_disk

        # 4. Read from source (remote HTTP range request or local disk)
        try:
            pf = self._get_parquet_file(file_url)
            table = pf.read_row_group(rg_index, columns=list(columns) if columns else None)
        except Exception as e:
            raise CorruptParquetError(
                file_url, f"Failed reading row group {rg_index}: {e}"
            ) from e

        # 5. Save to progressive disk saver if configured
        if self.progressive_saver is not None:
            self.progressive_saver.save(file_index, rg_index, table, columns)

        # 6. Save to disk cache if configured
        if self.disk_cache is not None:
            self.disk_cache.put(file_url, rg_index, table, columns)

        # 7. Save to memory cache
        self.memory_cache.put(file_url, rg_index, table, columns)

        return table

    def close(self) -> None:
        """Close all open file handles and clear caches."""
        with self._lock:
            for _, (fp, _) in self._open_files.items():
                try:
                    fp.close()
                except Exception:
                    pass
            self._open_files.clear()
        self.memory_cache.clear()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_open_files", None)
        state.pop("_http_fs", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._open_files = OrderedDict()
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self._http_fs = fsspec.filesystem("http", headers=headers)



def download_single_file(
    url: str,
    target_path: str,
    session: requests.Session,
    token: Optional[Union[bool, str]] = None,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Download a single remote Parquet file with resume and atomic write.

    Args:
        url: File URL to download.
        target_path: Destination path on local disk.
        session: requests.Session.
        token: Optional auth token.
        chunk_size: Streaming chunk size in bytes.

    Returns:
        The target path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    temp_path = f"{target_path}.part"

    auth_token = resolve_hf_token(token)
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    # Check remote size
    head = session.head(url, headers=headers, allow_redirects=True, timeout=15)
    remote_size = int(head.headers.get("content-length", 0))

    if os.path.exists(target_path) and remote_size > 0:
        if os.path.getsize(target_path) == remote_size:
            return target_path  # Already downloaded completely

    resume_offset = 0
    mode = "wb"
    if os.path.exists(temp_path):
        resume_offset = os.path.getsize(temp_path)
        if 0 < resume_offset < remote_size:
            headers["Range"] = f"bytes={resume_offset}-"
            mode = "ab"
        else:
            resume_offset = 0

    with session.get(url, headers=headers, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with open(temp_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    os.replace(temp_path, target_path)
    return target_path


def download_parquet_files(
    urls: Sequence[str],
    target_dir: str,
    token: Optional[Union[bool, str]] = None,
    max_workers: int = 8,
    show_progress: bool = True,
) -> List[str]:
    """Concurrently download a sequence of Parquet files to local disk.

    Args:
        urls: List of Parquet URLs to download.
        target_dir: Local destination directory.
        token: Optional Hugging Face auth token.
        max_workers: Concurrency level.
        show_progress: Whether to show a progress bar.

    Returns:
        List of local downloaded file paths in corresponding order.
    """
    os.makedirs(target_dir, exist_ok=True)
    session = create_retry_session()
    auth_token = resolve_hf_token(token)

    local_paths = [
        os.path.join(target_dir, os.path.basename(url.split("?")[0])) for url in urls
    ]

    tasks = list(zip(urls, local_paths))
    with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as executor:
        futures = {
            executor.submit(download_single_file, url, path, session, auth_token): idx
            for idx, (url, path) in enumerate(tasks)
        }

        pbar = (
            tqdm(total=len(urls), desc="Downloading Parquet files", unit="file")
            if show_progress
            else None
        )

        for future in as_completed(futures):
            future.result()
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

    return local_paths
