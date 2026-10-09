"""Row group reader and concurrent downloader.

This module provides the I/O layer for reading individual row groups over HTTP
range requests or local disk, applying column projection, and managing connection pools.
"""

from __future__ import annotations

import logging
import os
import threading
import time
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

logger = logging.getLogger("parquet_dataset_loader.reader")


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
        max_retries: int = 5,
        retry_delay: float = 0.5,
        max_retry_delay: float = 8.0,
    ) -> None:
        self.memory_cache = (
            memory_cache if memory_cache is not None else RowGroupMemoryCache(max_entries=2)
        )
        self.disk_cache = disk_cache
        self.progressive_saver = progressive_saver
        self.token = resolve_hf_token(token)
        self.block_size = block_size
        self.max_open_files = max(1, max_open_files)
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = float(retry_delay)
        self.max_retry_delay = float(max_retry_delay)

        self._open_files: OrderedDict[str, Tuple[Any, pq.ParquetFile]] = OrderedDict()
        self._lock = threading.Lock()
        self._http_fs = None
        self._http_fs_pid = None
        # Initialize filesystem for current process
        _ = self.http_fs

    @property
    def http_fs(self) -> Any:
        """Obtain a fork-safe fsspec HTTP filesystem for current process."""
        current_pid = os.getpid()
        if (
            not hasattr(self, "_http_fs_pid")
            or self._http_fs_pid != current_pid
            or getattr(self, "_http_fs", None) is None
        ):
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            self._http_fs = fsspec.filesystem("http", headers=headers)
            self._http_fs_pid = current_pid
            self._lock = threading.Lock()
            if hasattr(self, "_open_files"):
                self._open_files.clear()
        return self._http_fs

    def _evict_file(self, file_url_or_path: str) -> None:
        """Safely close and remove an open file handle from cache."""
        with self._lock:
            if file_url_or_path in self._open_files:
                fp, _ = self._open_files.pop(file_url_or_path)
                try:
                    fp.close()
                except Exception:
                    pass

    def reset_filesystem(self) -> None:
        """Close all open files and reset HTTP filesystem connection pool."""
        with self._lock:
            for _, (fp, _) in list(self._open_files.items()):
                try:
                    fp.close()
                except Exception:
                    pass
            self._open_files.clear()
            self._http_fs = None
            self._http_fs_pid = None

    def _get_parquet_file(self, file_url_or_path: str, max_attempts: int = 3) -> pq.ParquetFile:
        """Obtain a reusable ParquetFile instance for the specified file with retry on remote open."""
        is_remote = file_url_or_path.startswith(("http://", "https://"))
        attempts = max_attempts if is_remote else 1
        last_exc: Optional[Exception] = None

        for attempt in range(attempts):
            _ = self.http_fs
            with self._lock:
                if file_url_or_path in self._open_files:
                    self._open_files.move_to_end(file_url_or_path)
                    _, pf = self._open_files[file_url_or_path]
                    return pf

                if len(self._open_files) >= self.max_open_files:
                    oldest_url, (fp, _) = self._open_files.popitem(last=False)
                    try:
                        fp.close()
                    except Exception:
                        pass

            fp = None
            try:
                if is_remote:
                    fp = self.http_fs.open(file_url_or_path, "rb", block_size=self.block_size)
                    pf = pq.ParquetFile(fp)
                else:
                    abs_path = os.path.abspath(os.path.expanduser(file_url_or_path))
                    fp = open(abs_path, "rb")
                    pf = pq.ParquetFile(fp, memory_map=True)

                with self._lock:
                    self._open_files[file_url_or_path] = (fp, pf)
                return pf
            except Exception as exc:
                last_exc = exc
                if fp is not None:
                    try:
                        fp.close()
                    except Exception:
                        pass
                if is_remote and attempt < attempts - 1:
                    logger.warning(
                        f"⚠️ Error opening remote ParquetFile '{file_url_or_path}' "
                        f"(attempt {attempt + 1}/{attempts}): {exc}. Resetting filesystem and retrying..."
                    )
                    self.reset_filesystem()
                    time.sleep(min(self.max_retry_delay, self.retry_delay * (2 ** attempt)))
                else:
                    break

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Failed to open ParquetFile for {file_url_or_path}")

    def read_row_group(
        self,
        file_url: str,
        rg_index: int,
        columns: Optional[Sequence[str]] = None,
        file_index: int = 0,
    ) -> pa.Table:
        """Read a single row group table, checking memory, progressive, and disk caches first.

        Automatically handles transient HTTP stream disconnects, payload truncations, and
        network errors when reading remote Parquet files by performing exponential backoff
        retries and recycling open file handles.

        Args:
            file_url: URL or local path to the Parquet file.
            rg_index: Row group index within the file.
            columns: Optional subset of columns to read.
            file_index: Index of the file within the dataset split (for progressive saving).

        Returns:
            Decoded PyArrow Table containing the rows of this row group.

        Raises:
            CorruptParquetError: If reading or decoding the row group fails after exhausting retries.
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

        # 4. Read from source with automatic retries on transient network/IO errors
        is_remote = file_url.startswith(("http://", "https://"))
        max_attempts = self.max_retries if is_remote else 1
        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                pf = self._get_parquet_file(file_url)
                table = pf.read_row_group(rg_index, columns=list(columns) if columns else None)
                break
            except Exception as e:
                last_error = e
                # Evict the potentially severed/corrupted file handle from pool
                self._evict_file(file_url)
                if is_remote and attempt < max_attempts - 1:
                    delay = min(self.max_retry_delay, self.retry_delay * (2 ** attempt))
                    logger.warning(
                        f"⚠️ Transient error reading row group {rg_index} from '{file_url}' "
                        f"(attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Resetting connection and retrying in {delay:.2f}s..."
                    )
                    # For payload/network stream errors, reset filesystem to establish fresh HTTP session
                    self.reset_filesystem()
                    time.sleep(delay)
                else:
                    raise CorruptParquetError(
                        file_url,
                        f"Failed reading row group {rg_index} after {attempt + 1} attempt(s): {e}",
                    ) from e
        else:
            raise CorruptParquetError(
                file_url,
                f"Failed reading row group {rg_index} after exhausting {max_attempts} retries: {last_error}",
            ) from last_error

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
        self.reset_filesystem()
        self.memory_cache.clear()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_open_files", None)
        state.pop("_http_fs", None)
        state.pop("_http_fs_pid", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._open_files = OrderedDict()
        self._http_fs = None
        self._http_fs_pid = None



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
