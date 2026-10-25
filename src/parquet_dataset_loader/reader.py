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

from parquet_dataset_loader.cache import DiskCache, FileCache, RowGroupMemoryCache
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
        file_cache: Optional[FileCache] = None,
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
        self.file_cache = file_cache
        self.token = resolve_hf_token(token)
        self.block_size = block_size
        self.max_open_files = max(1, max_open_files)
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = float(retry_delay)
        self.max_retry_delay = float(max_retry_delay)

        self._local = threading.local()
        self._lock = threading.Lock()
        self._all_active_fps: set = set()
        # Initialize filesystem for current thread/process
        _ = self.http_fs

    @property
    def _open_files(self) -> OrderedDict[str, Tuple[Any, pq.ParquetFile]]:
        if not hasattr(self._local, "open_files") or self._local.open_files is None:
            self._local.open_files = OrderedDict()
        return self._local.open_files

    @_open_files.setter
    def _open_files(self, value: OrderedDict[str, Tuple[Any, pq.ParquetFile]]) -> None:
        self._local.open_files = value

    @property
    def _http_fs(self) -> Any:
        return getattr(self._local, "http_fs", None)

    @_http_fs.setter
    def _http_fs(self, value: Any) -> None:
        self._local.http_fs = value

    @property
    def _http_fs_pid(self) -> Any:
        return getattr(self._local, "http_fs_pid", None)

    @_http_fs_pid.setter
    def _http_fs_pid(self, value: Any) -> None:
        self._local.http_fs_pid = value

    @property
    def http_fs(self) -> Any:
        """Obtain a fork-safe and thread-isolated fsspec HTTP filesystem for current process and thread."""
        current_pid = os.getpid()
        local_pid = getattr(self._local, "http_fs_pid", None)
        local_fs = getattr(self._local, "http_fs", None)
        if local_pid != current_pid or local_fs is None:
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            self._local.http_fs = fsspec.filesystem("http", headers=headers)
            self._local.http_fs_pid = current_pid
            if hasattr(self._local, "open_files") and self._local.open_files:
                for _, (fp, _) in list(self._local.open_files.items()):
                    with self._lock:
                        self._all_active_fps.discard(fp)
                    try:
                        fp.close()
                    except Exception:
                        pass
                self._local.open_files.clear()
            else:
                self._local.open_files = OrderedDict()
        return self._local.http_fs

    def _evict_file(self, file_url_or_path: str) -> None:
        """Safely close and remove an open file handle from current thread's cache."""
        open_files = self._open_files
        if file_url_or_path in open_files:
            fp, _ = open_files.pop(file_url_or_path)
            with self._lock:
                self._all_active_fps.discard(fp)
            try:
                fp.close()
            except Exception:
                pass
            if self.file_cache is not None:
                self.file_cache.release_file(file_url_or_path)

    def reset_filesystem(self) -> None:
        """Close current thread's open files and reset its HTTP filesystem connection pool."""
        if hasattr(self._local, "open_files") and self._local.open_files:
            for url_or_path, (fp, _) in list(self._local.open_files.items()):
                with self._lock:
                    self._all_active_fps.discard(fp)
                try:
                    fp.close()
                except Exception:
                    pass
                if self.file_cache is not None:
                    self.file_cache.release_file(url_or_path)
            self._local.open_files.clear()
        self._local.http_fs = None
        self._local.http_fs_pid = None

    def _get_parquet_file(self, file_url_or_path: str, max_attempts: Optional[int] = None) -> pq.ParquetFile:
        """Obtain a reusable ParquetFile instance for current thread with retry on remote open."""
        is_remote = file_url_or_path.startswith(("http://", "https://"))

        # Check local file cache if enabled
        if is_remote and self.file_cache is not None:
            try:
                local_path = self.file_cache.get_or_download(file_url_or_path)
                self.file_cache.acquire_file(file_url_or_path)
                file_url_or_path = local_path
                is_remote = False
            except Exception as e:
                logger.warning(
                    "⚠️ FileCache could not retrieve '%s': %s. Streaming directly over HTTP.",
                    file_url_or_path,
                    e,
                )

        attempts = max_attempts if max_attempts is not None else (self.max_retries if is_remote else 1)
        last_exc: Optional[Exception] = None

        for attempt in range(attempts):
            _ = self.http_fs
            open_files = self._open_files
            if file_url_or_path in open_files:
                open_files.move_to_end(file_url_or_path)
                _, pf = open_files[file_url_or_path]
                return pf

            if len(open_files) >= self.max_open_files:
                oldest_url, (fp, _) = open_files.popitem(last=False)
                with self._lock:
                    self._all_active_fps.discard(fp)
                try:
                    fp.close()
                except Exception:
                    pass
                if self.file_cache is not None and oldest_url:
                    self.file_cache.release_file(oldest_url)

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
                    self._all_active_fps.add(fp)
                open_files[file_url_or_path] = (fp, pf)
                return pf
            except Exception as exc:
                last_exc = exc
                if fp is not None:
                    with self._lock:
                        self._all_active_fps.discard(fp)
                    try:
                        fp.close()
                    except Exception:
                        pass
                is_rate_limit = "429" in str(exc) or "too many requests" in str(exc).lower()
                effective_attempts = max(attempts, 6) if is_rate_limit else attempts
                if is_remote and attempt < effective_attempts - 1 and "cannot schedule new futures" not in str(exc).lower():
                    delay = (
                        min(self.max_retry_delay * 4, 3.0 * (2 ** attempt))
                        if is_rate_limit
                        else min(self.max_retry_delay, self.retry_delay * (2 ** attempt))
                    )
                    logger.warning(
                        f"⚠️ {'Rate limit (429)' if is_rate_limit else 'Error'} opening remote ParquetFile '{file_url_or_path}' "
                        f"(attempt {attempt + 1}/{effective_attempts}): {exc}. Resetting filesystem and retrying in {delay:.2f}s..."
                    )
                    self.reset_filesystem()
                    time.sleep(delay)
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
                is_rate_limit = "429" in str(e) or "too many requests" in str(e).lower()
                effective_attempts = max(max_attempts, 6) if is_rate_limit else max_attempts
                if is_remote and attempt < effective_attempts - 1 and "cannot schedule new futures" not in str(e).lower():
                    delay = (
                        min(self.max_retry_delay * 4, 3.0 * (2 ** attempt))
                        if is_rate_limit
                        else min(self.max_retry_delay, self.retry_delay * (2 ** attempt))
                    )
                    logger.warning(
                        f"⚠️ {'Rate limit (429)' if is_rate_limit else 'Transient error'} reading row group {rg_index} from '{file_url}' "
                        f"(attempt {attempt + 1}/{effective_attempts}): {e}. "
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
        with self._lock:
            for fp in list(self._all_active_fps):
                try:
                    fp.close()
                except Exception:
                    pass
            self._all_active_fps.clear()
        self.memory_cache.clear()
        if self.file_cache is not None:
            self.file_cache.close()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_local", None)
        state.pop("_all_active_fps", None)
        state.pop("_open_files", None)
        state.pop("_http_fs", None)
        state.pop("_http_fs_pid", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all_active_fps = set()



def download_single_file(
    url: str,
    target_path: str,
    session: Optional[requests.Session] = None,
    token: Optional[Union[bool, str]] = None,
    chunk_size: int = 1024 * 1024,
    max_retries: int = 5,
) -> str:
    """Download a single remote Parquet file with resume and atomic write.

    Handles transient network errors, socket disconnects, and HTTP 429 rate limits
    with exponential backoff and Retry-After awareness.

    Args:
        url: File URL to download.
        target_path: Destination path on local disk.
        session: Optional requests.Session.
        token: Optional auth token.
        chunk_size: Streaming chunk size in bytes.
        max_retries: Number of retry attempts.

    Returns:
        The target path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    temp_path = f"{target_path}.part"

    sess = session or create_retry_session()
    auth_token = resolve_hf_token(token)
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    # Check remote size with retry on 429
    remote_size = 0
    for attempt in range(max_retries):
        try:
            head = sess.head(url, headers=headers, allow_redirects=True, timeout=15)
            if head.status_code == 429:
                retry_after = head.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 2.0 * (2 ** attempt))
                logger.warning("⚠️ 429 Too Many Requests on HEAD %s (attempt %d/%d). Sleeping %0.1fs...", url, attempt + 1, max_retries, delay)
                time.sleep(delay)
                continue
            head.raise_for_status()
            remote_size = int(head.headers.get("content-length", 0))
            break
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(30.0, 1.5 * (2 ** attempt)))

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

    for attempt in range(max_retries):
        try:
            with sess.get(url, headers=headers, stream=True, timeout=30) as resp:
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 2.0 * (2 ** attempt))
                    logger.warning("⚠️ 429 Too Many Requests on GET %s (attempt %d/%d). Sleeping %0.1fs...", url, attempt + 1, max_retries, delay)
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                with open(temp_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            break
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(30.0, 1.5 * (2 ** attempt)))

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
