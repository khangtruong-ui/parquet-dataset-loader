"""Caching mechanisms for row groups in memory and on disk.

This module provides:
1. RowGroupMemoryCache: A thread-safe, bounded LRU cache for decoded PyArrow tables,
   ensuring that memory usage remains strictly bounded regardless of dataset size.
2. DiskCache: An optional on-disk cache using PyArrow Feather (IPC) format for
   memory-mapped zero-copy re-reads of remote row groups.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple, Union

import pyarrow as pa
import pyarrow.feather as feather


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

