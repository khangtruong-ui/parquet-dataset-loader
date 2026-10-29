"""Caching mechanisms for row groups in memory and on disk.

This module provides:
1. RowGroupMemoryCache: A thread-safe, bounded LRU cache for decoded PyArrow tables,
   ensuring that memory usage remains strictly bounded regardless of dataset size.
2. DiskCache: An optional on-disk cache using PyArrow Feather (IPC) format for
   memory-mapped zero-copy re-reads of remote row groups.
3. FileCache: An LRU and forward-caching manager for remote Parquet files on local disk,
   maintaining a bounded window of downloaded files regulated by file count or size (GBs).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pyarrow as pa
import pyarrow.feather as feather

logger = logging.getLogger(__name__)


class RowGroupMemoryCache:
    """Thread-safe LRU in-memory cache for decoded PyArrow row group tables.

    Prevents unbounded memory growth by evicting the least recently used row group
    whenever the cache reaches max_entries.

    Attributes:
        max_entries: Maximum number of row group tables to keep in RAM simultaneously.
    """

    def __init__(self, max_entries: int = 2) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._cache: OrderedDict[Tuple[str, int, Optional[Tuple[str, ...]]], pa.Table] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        file_url: str,
        rg_index: int,
        columns: Optional[Sequence[str]] = None,
    ) -> Tuple[str, int, Optional[Tuple[str, ...]]]:
        """Generate a hashable cache key."""
        col_key = tuple(columns) if columns is not None else None
        return (file_url, rg_index, col_key)

    def get(
        self,
        file_url: str,
        rg_index: int,
        columns: Optional[Sequence[str]] = None,
    ) -> Optional[pa.Table]:
        """Retrieve a row group table from the cache if present.

        Args:
            file_url: URL or path of the Parquet file.
            rg_index: Row group index within the file.
            columns: Optional projected column list.

        Returns:
            The cached PyArrow Table, or None on cache miss.
        """
        key = self.make_key(file_url, rg_index, columns)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def put(
        self,
        file_url: str,
        rg_index: int,
        table: pa.Table,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        """Store a row group table into the LRU cache.

        Args:
            file_url: URL or path of the Parquet file.
            rg_index: Row group index within the file.
            table: PyArrow Table for this row group.
            columns: Optional projected column list.
        """
        key = self.make_key(file_url, rg_index, columns)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = table
            else:
                if len(self._cache) >= self.max_entries:
                    self._cache.popitem(last=False)  # Evict oldest entry
                self._cache[key] = table

    def clear(self) -> None:
        """Clear all entries and reset hit/miss counters."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        """Return hit, miss, and hit ratio statistics."""
        with self._lock:
            total = self.hits + self.misses
            ratio = (self.hits / total) if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total,
                "hit_ratio": ratio,
                "current_size": len(self._cache),
                "max_entries": self.max_entries,
            }

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        # Avoid duplicating cached tables into spawned worker processes
        state["_cache"] = OrderedDict()
        state["hits"] = 0
        state["misses"] = 0
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()



class DiskCache:
    """Persistent on-disk cache for row groups using PyArrow Feather (IPC).

    Feather allows instant memory-mapped reading of cached tables without
    Parquet decompression or parsing overhead.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = os.path.join(os.path.abspath(os.path.expanduser(cache_dir)), "row_groups")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _cache_path(
        self,
        file_url: str,
        rg_index: int,
        columns: Optional[Sequence[str]] = None,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(file_url.encode("utf-8"))
        hasher.update(str(rg_index).encode("utf-8"))
        if columns is not None:
            hasher.update(",".join(columns).encode("utf-8"))
        key_hash = hasher.hexdigest()[:32]
        return os.path.join(self.cache_dir, f"{key_hash}.feather")

    def contains(
        self,
        file_url: str,
        rg_index: int,
        columns: Optional[Sequence[str]] = None,
    ) -> bool:
        """Check if a row group is cached on disk."""
        path = self._cache_path(file_url, rg_index, columns)
        return os.path.exists(path)

    def get(
        self,
        file_url: str,
        rg_index: int,
        columns: Optional[Sequence[str]] = None,
    ) -> Optional[pa.Table]:
        """Read a cached row group table from disk."""
        path = self._cache_path(file_url, rg_index, columns)
        if not os.path.exists(path):
            return None
        try:
            return feather.read_table(path, memory_map=True)
        except Exception:
            return None

    def put(
        self,
        file_url: str,
        rg_index: int,
        table: pa.Table,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        """Write a row group table to disk cache."""
        path = self._cache_path(file_url, rg_index, columns)
        temp_path = f"{path}.tmp.{os.getpid()}"
        try:
            feather.write_feather(table, temp_path, compression="zstd")
            os.replace(temp_path, path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def clear(self) -> None:
        """Remove all cached files from the disk cache directory."""
        with self._lock:
            if os.path.exists(self.cache_dir):
                for f in os.listdir(self.cache_dir):
                    if f.endswith(".feather"):
                        try:
                            os.remove(os.path.join(self.cache_dir, f))
                        except OSError:
                            pass

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()


class FileCache:
    """Thread-safe LRU and forward-caching manager for remote Parquet files on local disk.

    Maintains a bounded window of downloaded remote Parquet files to eliminate
    concurrent HTTP socket thrashing, connection timeouts, and 429 rate limits
    during multi-worker streaming iterations.

    Attributes:
        cache_dir: Directory where remote Parquet files are stored.
        max_cached_files: Maximum number of Parquet files to keep on disk simultaneously.
            If None, file count is unlimited (regulated only by size).
        max_cache_bytes: Maximum total size on disk in bytes (or max_cache_gb).
            If None, disk usage is regulated only by file count.
    """

    def __init__(
        self,
        cache_dir: str,
        max_cached_files: Optional[int] = 4,
        max_cache_gb: Optional[float] = None,
        max_cache_bytes: Optional[int] = None,
        token: Optional[Union[bool, str]] = None,
        prefetch_ahead: int = 1,
        save_to_disk: bool = False,
    ) -> None:
        self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(self.cache_dir, exist_ok=True)

        self.max_cached_files = (
            max(1, int(max_cached_files)) if max_cached_files is not None else None
        )
        if max_cache_bytes is not None:
            self.max_cache_bytes: Optional[int] = max(1, int(max_cache_bytes))
        elif max_cache_gb is not None:
            self.max_cache_bytes = max(1, int(float(max_cache_gb) * 1024 * 1024 * 1024))
        else:
            self.max_cache_bytes = None

        self.token = token
        self.prefetch_ahead = max(0, int(prefetch_ahead))
        self.save_to_disk = bool(save_to_disk)

        self._lock = threading.RLock()
        self._file_locks: Dict[str, threading.Lock] = {}
        self._active_refs: Dict[str, int] = {}
        self._access_times: Dict[str, float] = {}  # url -> last access time
        self._consumed_files: Set[str] = set()  # url -> consumed by reader
        self._file_sequence: List[str] = []
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_futures: Dict[str, Future] = {}
        self._session: Optional[Any] = None

        self._discover_existing_files()

    def _url_to_path(self, file_url: str) -> str:
        clean_url = file_url.split("?")[0]
        base = os.path.basename(clean_url)
        if not base.endswith((".parquet", ".pq")):
            base = f"{base}.parquet"
        hasher = hashlib.sha256(clean_url.encode("utf-8"))
        h_prefix = hasher.hexdigest()[:12]
        return os.path.join(self.cache_dir, f"{h_prefix}_{base}")

    def _discover_existing_files(self) -> None:
        """Scan cache directory for already downloaded Parquet files and clean stale temps."""
        if not os.path.isdir(self.cache_dir):
            return
        with self._lock:
            try:
                for fname in os.listdir(self.cache_dir):
                    fpath = os.path.join(self.cache_dir, fname)
                    if fname.endswith((".part", ".tmp")):
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass
                    elif fname.endswith((".parquet", ".pq")):
                        try:
                            mtime = os.path.getmtime(fpath)
                            self._access_times[fpath] = mtime
                        except OSError:
                            pass
            except Exception as e:
                logger.warning("Error discovering existing cached files in %s: %s", self.cache_dir, e)
            self._enforce_limits()

    @property
    def session(self) -> Any:
        if self._session is None:
            from parquet_dataset_loader.hf_resolver import create_retry_session
            self._session = create_retry_session()
        return self._session

    def acquire_file(self, file_url: str) -> None:
        """Increment active reference count to prevent eviction while an open file handle exists."""
        with self._lock:
            self._active_refs[file_url] = self._active_refs.get(file_url, 0) + 1

    def release_file(self, file_url: str) -> None:
        """Decrement active reference count when a file handle is closed."""
        with self._lock:
            if file_url in self._active_refs:
                self._active_refs[file_url] = max(0, self._active_refs[file_url] - 1)
                if self._active_refs[file_url] == 0:
                    del self._active_refs[file_url]
                    if not self.save_to_disk and file_url in self._consumed_files:
                        self._evict_consumed_file(file_url)

    def mark_consumed(self, file_url: str) -> None:
        """Mark a cached file as consumed by the reader/consumer.

        If save_to_disk is True, the file remains permanently on disk ('still there, not removed').
        If save_to_disk is False, the file is evicted and deleted once no longer active.
        """
        with self._lock:
            self._consumed_files.add(file_url)
            if not self.save_to_disk:
                self._evict_consumed_file(file_url)

    def is_consumed(self, file_url: str) -> bool:
        """Check whether a file has been marked as consumed."""
        with self._lock:
            return file_url in self._consumed_files

    def reset_consumption(self) -> None:
        """Reset consumed files tracking (e.g. for multi-epoch training iteration)."""
        with self._lock:
            self._consumed_files.clear()

    def _evict_consumed_file(self, file_url: str) -> None:
        """Evict a consumed file from disk when save_to_disk is False."""
        if self.save_to_disk:
            return
        if self._active_refs.get(file_url, 0) > 0:
            return
        path = self._url_to_path(file_url) if file_url.startswith(("http://", "https://")) else file_url
        try:
            if os.path.exists(path):
                os.remove(path)
            self._access_times.pop(file_url, None)
            logger.debug("Removed consumed file '%s' from disk (save_to_disk=False).", path)
        except OSError as e:
            logger.warning("Could not remove consumed file '%s': %s", path, e)

    def set_file_sequence(self, file_urls: Sequence[str]) -> None:
        """Set the expected sequence of files to enable forward prefetching."""
        with self._lock:
            self._file_sequence = list(file_urls)

    def is_cached(self, file_url: str) -> bool:
        """Check if a remote file is already downloaded and present in local cache."""
        if not file_url.startswith(("http://", "https://")):
            return True
        path = self._url_to_path(file_url)
        return os.path.exists(path) and os.path.getsize(path) > 0

    def get_or_download(self, file_url: str) -> str:
        """Get local cached path for a remote Parquet file, downloading it if necessary."""
        if not file_url.startswith(("http://", "https://")):
            return os.path.abspath(os.path.expanduser(file_url))

        target_path = self._url_to_path(file_url)

        with self._lock:
            if file_url not in self._file_locks:
                self._file_locks[file_url] = threading.Lock()
            flock = self._file_locks[file_url]

        with flock:
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                with self._lock:
                    self._access_times[file_url] = time.time()
                self._trigger_forward_prefetch(file_url)
                return target_path

            # Check if active prefetch is downloading it
            fut = None
            with self._lock:
                if file_url in self._prefetch_futures:
                    fut = self._prefetch_futures[file_url]

            if fut is not None:
                try:
                    fut.result()
                    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                        with self._lock:
                            self._access_times[file_url] = time.time()
                            self._prefetch_futures.pop(file_url, None)
                        self._trigger_forward_prefetch(file_url)
                        return target_path
                except Exception as e:
                    logger.warning("Prefetch for '%s' failed: %s. Downloading directly...", file_url, e)
                    with self._lock:
                        self._prefetch_futures.pop(file_url, None)

            # Synchronous download
            self._download_file(file_url, target_path)

            with self._lock:
                self._access_times[file_url] = time.time()
                self._enforce_limits(exclude={file_url})

            self._trigger_forward_prefetch(file_url)
            return target_path

    def _download_file(self, file_url: str, target_path: str) -> str:
        from parquet_dataset_loader.reader import download_single_file
        return download_single_file(
            url=file_url,
            target_path=target_path,
            session=self.session,
            token=self.token,
        )

    def _trigger_forward_prefetch(self, current_url: str) -> None:
        if self.prefetch_ahead <= 0 or not self._file_sequence:
            return
        with self._lock:
            try:
                curr_idx = self._file_sequence.index(current_url)
            except ValueError:
                return
            upcoming = self._file_sequence[curr_idx + 1 : curr_idx + 1 + self.prefetch_ahead]
        if upcoming:
            self.prefetch(upcoming)

    def prefetch(self, file_urls: Sequence[str]) -> None:
        """Submit background download tasks for upcoming remote files waiting for consumption."""
        with self._lock:
            if self._prefetch_executor is None:
                self._prefetch_executor = ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="pdl-file-prefetch",
                )
            for url in file_urls:
                if not url.startswith(("http://", "https://")):
                    continue
                path = self._url_to_path(url)
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    continue
                if url in self._prefetch_futures and not self._prefetch_futures[url].done():
                    continue

                # Throttle prefetching by unconsumed cache limits
                # max_cache_bytes/max_cache_gb is the max cache size waiting for consumption!
                pending_count = len([f for f in self._prefetch_futures.values() if not f.done()])
                if (
                    self.max_cached_files is not None
                    and (self.unconsumed_files_count + pending_count) >= self.max_cached_files
                ):
                    break

                if self.max_cache_bytes is not None and self.unconsumed_size_bytes >= self.max_cache_bytes:
                    break

                def _task(u=url, p=path) -> None:
                    with self._lock:
                        if u not in self._file_locks:
                            self._file_locks[u] = threading.Lock()
                        flock = self._file_locks[u]
                    with flock:
                        if not os.path.exists(p) or os.path.getsize(p) == 0:
                            try:
                                self._download_file(u, p)
                                with self._lock:
                                    self._access_times[u] = time.time()
                                    self._enforce_limits(exclude={u})
                            except Exception as ex:
                                if "429" in str(ex) or "too many requests" in str(ex).lower():
                                    logger.error("429 Too Many Requests during background prefetch of %s: %s", u, ex)
                                    raise
                                logger.warning("Background forward-prefetch failed for %s: %s", u, ex)

                self._prefetch_futures[url] = self._prefetch_executor.submit(_task)

    def _enforce_limits(self, exclude: Optional[Set[str]] = None) -> None:
        """Evict cached files to satisfy file count and GB limits.

        - If save_to_disk is True: files that have been consumed are PRESERVED ON DISK
          ('still there, not removed'). Unconsumed files waiting for consumption are
          bounded during prefetch.
        - If save_to_disk is False: files that have been consumed are removed from disk,
          and any excess unconsumed files are evicted LRU.
        """
        exclude_set = exclude or set()

        # 1. When save_to_disk is False, clean up any consumed files first
        if not self.save_to_disk:
            for url in list(self._consumed_files):
                if url in exclude_set or self._active_refs.get(url, 0) > 0:
                    continue
                path = self._url_to_path(url) if url.startswith(("http://", "https://")) else url
                try:
                    if os.path.exists(path):
                        os.remove(path)
                    self._access_times.pop(url, None)
                    logger.debug("Removed consumed file '%s' from disk (save_to_disk=False).", path)
                except OSError as e:
                    logger.warning("Could not evict consumed file '%s': %s", path, e)

        # 2. When save_to_disk is False, evict unconsumed unreferenced files if limits are exceeded
        if not self.save_to_disk:
            unconsumed: List[Tuple[str, str, int, float]] = []
            unconsumed_total = 0

            for url, last_time in list(self._access_times.items()):
                if url in self._consumed_files:
                    continue
                path = self._url_to_path(url) if url.startswith(("http://", "https://")) else url
                if os.path.exists(path):
                    try:
                        sz = os.path.getsize(path)
                        unconsumed.append((url, path, sz, last_time))
                        unconsumed_total += sz
                    except OSError:
                        pass
                else:
                    self._access_times.pop(url, None)

            # Sort by access time ascending (oldest first)
            unconsumed.sort(key=lambda item: item[3])

            for url, path, sz, _ in unconsumed:
                file_limit_exceeded = (
                    self.max_cached_files is not None and len(unconsumed) > self.max_cached_files
                )
                size_limit_exceeded = (
                    self.max_cache_bytes is not None and unconsumed_total > self.max_cache_bytes
                )
                if not (file_limit_exceeded or size_limit_exceeded):
                    break

                if url in exclude_set or self._active_refs.get(url, 0) > 0:
                    continue

                try:
                    if os.path.exists(path):
                        os.remove(path)
                    self._access_times.pop(url, None)
                    unconsumed = [c for c in unconsumed if c[0] != url]
                    unconsumed_total -= sz
                    logger.debug("Evicted cached remote file '%s' to respect unconsumed cache limits.", path)
                except OSError as e:
                    logger.warning("Could not evict cached file '%s': %s", path, e)

    @property
    def unconsumed_size_bytes(self) -> int:
        """Total size in bytes of cached files waiting for consumption."""
        with self._lock:
            total = 0
            for url in list(self._access_times.keys()):
                if url not in self._consumed_files:
                    path = self._url_to_path(url) if url.startswith(("http://", "https://")) else url
                    if os.path.exists(path):
                        try:
                            total += os.path.getsize(path)
                        except OSError:
                            pass
            return total

    @property
    def unconsumed_size_gb(self) -> float:
        """Total size in gigabytes of cached files waiting for consumption."""
        return self.unconsumed_size_bytes / (1024 * 1024 * 1024)

    @property
    def unconsumed_files_count(self) -> int:
        """Number of cached files on disk currently waiting for consumption."""
        with self._lock:
            count = 0
            for url in list(self._access_times.keys()):
                if url not in self._consumed_files:
                    path = self._url_to_path(url) if url.startswith(("http://", "https://")) else url
                    if os.path.exists(path):
                        count += 1
            return count

    @property
    def consumed_files_count(self) -> int:
        """Number of files marked as consumed."""
        with self._lock:
            return len(self._consumed_files)

    @property
    def total_size_bytes(self) -> int:
        """Total size in bytes of files currently cached on disk."""
        with self._lock:
            total = 0
            for url in list(self._access_times.keys()):
                path = self._url_to_path(url) if url.startswith(("http://", "https://")) else url
                if os.path.exists(path):
                    try:
                        total += os.path.getsize(path)
                    except OSError:
                        pass
            return total

    @property
    def total_size_gb(self) -> float:
        """Total size in gigabytes of files currently cached on disk."""
        return self.total_size_bytes / (1024 * 1024 * 1024)

    @property
    def cached_files_count(self) -> int:
        """Number of files currently cached on disk."""
        with self._lock:
            count = 0
            for url in list(self._access_times.keys()):
                path = self._url_to_path(url) if url.startswith(("http://", "https://")) else url
                if os.path.exists(path):
                    count += 1
            return count

    def clear(self) -> None:
        """Remove all files from the cache directory and reset state."""
        with self._lock:
            if self._prefetch_executor is not None:
                self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
                self._prefetch_executor = None
            self._prefetch_futures.clear()

            if os.path.exists(self.cache_dir):
                for f in os.listdir(self.cache_dir):
                    if f.endswith((".parquet", ".pq", ".part", ".tmp")):
                        try:
                            os.remove(os.path.join(self.cache_dir, f))
                        except OSError:
                            pass
            self._access_times.clear()
            self._active_refs.clear()
            self._consumed_files.clear()

    def close(self) -> None:
        """Stop background prefetch executor."""
        with self._lock:
            if self._prefetch_executor is not None:
                self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
                self._prefetch_executor = None
            self._prefetch_futures.clear()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_file_locks", None)
        state.pop("_prefetch_executor", None)
        state.pop("_prefetch_futures", None)
        state.pop("_session", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._file_locks = {}
        self._prefetch_executor = None
        self._prefetch_futures = {}
        self._session = None
        if not hasattr(self, "_consumed_files"):
            self._consumed_files = set()
        if not hasattr(self, "save_to_disk"):
            self.save_to_disk = False

