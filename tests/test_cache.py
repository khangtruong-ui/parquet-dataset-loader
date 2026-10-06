"""Unit tests for memory and disk caching layers."""

import os
import threading
import pyarrow as pa
import pytest

from parquet_dataset_loader.cache import DiskCache, RowGroupMemoryCache


def test_row_group_memory_cache_lru() -> None:
    cache = RowGroupMemoryCache(max_entries=2)
    t1 = pa.Table.from_arrays([pa.array([1, 2])], names=["a"])
    t2 = pa.Table.from_arrays([pa.array([3, 4])], names=["a"])
    t3 = pa.Table.from_arrays([pa.array([5, 6])], names=["a"])

    # Put t1, t2
    cache.put("file1", 0, t1)
    cache.put("file1", 1, t2)
    assert len(cache) == 2

    # Access t1 (making t2 the LRU)
    res = cache.get("file1", 0)
    assert res is not None
    assert cache.stats["hits"] == 1

    # Put t3 -> should evict t2
    cache.put("file2", 0, t3)
    assert len(cache) == 2
    assert cache.get("file1", 1) is None  # t2 evicted
    assert cache.get("file1", 0) is not None  # t1 still present
    assert cache.get("file2", 0) is not None  # t3 present


def test_cache_column_projection_keys() -> None:
    cache = RowGroupMemoryCache(max_entries=5)
    t_all = pa.Table.from_arrays([pa.array([1]), pa.array(["x"])], names=["a", "b"])
    t_a = pa.Table.from_arrays([pa.array([1])], names=["a"])

    cache.put("file1", 0, t_all, columns=None)
    cache.put("file1", 0, t_a, columns=["a"])

    assert len(cache) == 2
    res_all = cache.get("file1", 0, columns=None)
    res_a = cache.get("file1", 0, columns=["a"])

    assert res_all is not None and res_all.num_columns == 2
    assert res_a is not None and res_a.num_columns == 1


def test_row_group_memory_cache_thread_safety() -> None:
    cache = RowGroupMemoryCache(max_entries=4)
    table = pa.Table.from_arrays([pa.array([1])], names=["val"])

    def worker(worker_id: int) -> None:
        for i in range(50):
            cache.put(f"file_{worker_id}", i % 5, table)
            cache.get(f"file_{worker_id}", i % 5)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(cache) <= 4


def test_disk_cache(temp_dir: str) -> None:
    cache_dir = os.path.join(temp_dir, "disk_cache")
    dc = DiskCache(cache_dir=cache_dir)

    table = pa.Table.from_arrays([pa.array([10, 20, 30])], names=["num"])
    url = "https://example.com/test.parquet"

    assert not dc.contains(url, 0)
    assert dc.get(url, 0) is None

    dc.put(url, 0, table)
    assert dc.contains(url, 0)

    loaded = dc.get(url, 0)
    assert loaded is not None
    assert loaded.num_rows == 3
    assert loaded.column("num").to_pylist() == [10, 20, 30]

    # Test clear
    dc.clear()
    assert not dc.contains(url, 0)
