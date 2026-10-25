"""Unit tests for FileCache forward caching, disk bounds, and file locality."""

import os
import shutil
import tempfile
import threading
import time
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_dataset_loader.api import load_dataset
from parquet_dataset_loader.cache import FileCache
from parquet_dataset_loader.dataset import BlockShuffledSampler, IndexedParquetDataset
from parquet_dataset_loader.reader import RowGroupReader


def create_sample_parquet(path: str, num_rows: int = 20, start_id: int = 0) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array(list(range(start_id, start_id + num_rows))),
            pa.array([f"item_{i}" for i in range(start_id, start_id + num_rows)]),
        ],
        names=["id", "text"],
    )
    pq.write_table(table, path, row_group_size=10)
    return path


@pytest.fixture
def temp_cache_dir():
    d = tempfile.mkdtemp(prefix="pdl_test_file_cache_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_file_cache_local_and_download(temp_cache_dir: str) -> None:
    cache = FileCache(cache_dir=temp_cache_dir, max_cached_files=3)
    local_f = os.path.join(temp_cache_dir, "local_source.parquet")
    create_sample_parquet(local_f, num_rows=10)

    # Local file should return as-is
    res = cache.get_or_download(local_f)
    assert res == local_f
    cache.close()


def test_file_cache_max_cached_files_eviction(temp_cache_dir: str) -> None:
    source_dir = os.path.join(temp_cache_dir, "source")
    cache_store_dir = os.path.join(temp_cache_dir, "store")
    os.makedirs(source_dir, exist_ok=True)

    f1 = create_sample_parquet(os.path.join(source_dir, "file1.parquet"), 10, start_id=0)
    f2 = create_sample_parquet(os.path.join(source_dir, "file2.parquet"), 10, start_id=10)
    f3 = create_sample_parquet(os.path.join(source_dir, "file3.parquet"), 10, start_id=20)

    cache = FileCache(cache_dir=cache_store_dir, max_cached_files=2)

    # Mock _download_file to copy from source_dir
    def mock_download(url: str, target: str) -> str:
        fname = os.path.basename(url)
        src = os.path.join(source_dir, fname)
        shutil.copy2(src, target)
        return target

    cache._download_file = mock_download

    u1 = "https://example.com/data/file1.parquet"
    u2 = "https://example.com/data/file2.parquet"
    u3 = "https://example.com/data/file3.parquet"

    p1 = cache.get_or_download(u1)
    assert os.path.exists(p1)
    assert cache.cached_files_count == 1

    time.sleep(0.01)
    p2 = cache.get_or_download(u2)
    assert os.path.exists(p2)
    assert cache.cached_files_count == 2

    time.sleep(0.01)
    p3 = cache.get_or_download(u3)
    assert os.path.exists(p3)

    # Cache should be bounded at max_cached_files=2
    assert cache.cached_files_count == 2
    # Oldest (file1) should have been evicted and deleted
    assert not os.path.exists(p1)
    assert os.path.exists(p2)
    assert os.path.exists(p3)
    cache.close()


def test_file_cache_active_file_protection(temp_cache_dir: str) -> None:
    source_dir = os.path.join(temp_cache_dir, "source")
    cache_store_dir = os.path.join(temp_cache_dir, "store")
    os.makedirs(source_dir, exist_ok=True)

    f1 = create_sample_parquet(os.path.join(source_dir, "file1.parquet"), 10)
    f2 = create_sample_parquet(os.path.join(source_dir, "file2.parquet"), 10)
    f3 = create_sample_parquet(os.path.join(source_dir, "file3.parquet"), 10)

    cache = FileCache(cache_dir=cache_store_dir, max_cached_files=2)

    def mock_download(url: str, target: str) -> str:
        fname = os.path.basename(url)
        shutil.copy2(os.path.join(source_dir, fname), target)
        return target

    cache._download_file = mock_download

    u1 = "https://example.com/file1.parquet"
    u2 = "https://example.com/file2.parquet"
    u3 = "https://example.com/file3.parquet"

    p1 = cache.get_or_download(u1)
    # Acquire active ref on u1
    cache.acquire_file(u1)

    time.sleep(0.01)
    p2 = cache.get_or_download(u2)
    time.sleep(0.01)
    p3 = cache.get_or_download(u3)

    # u1 is active so u2 should be evicted instead of u1!
    assert os.path.exists(p1)
    assert not os.path.exists(p2)
    assert os.path.exists(p3)

    cache.release_file(u1)
    cache.close()


def test_file_cache_max_cache_bytes_limit(temp_cache_dir: str) -> None:
    source_dir = os.path.join(temp_cache_dir, "source")
    cache_store_dir = os.path.join(temp_cache_dir, "store")
    os.makedirs(source_dir, exist_ok=True)

    f1 = create_sample_parquet(os.path.join(source_dir, "file1.parquet"), 100)
    f2 = create_sample_parquet(os.path.join(source_dir, "file2.parquet"), 100)
    f1_size = os.path.getsize(f1)

    # Set byte limit allowing only ~1.5 files
    cache = FileCache(cache_dir=cache_store_dir, max_cached_files=None, max_cache_bytes=int(f1_size * 1.5))

    def mock_download(url: str, target: str) -> str:
        fname = os.path.basename(url)
        shutil.copy2(os.path.join(source_dir, fname), target)
        return target

    cache._download_file = mock_download

    u1 = "https://example.com/file1.parquet"
    u2 = "https://example.com/file2.parquet"

    p1 = cache.get_or_download(u1)
    assert os.path.exists(p1)

    time.sleep(0.01)
    p2 = cache.get_or_download(u2)
    assert os.path.exists(p2)
    assert not os.path.exists(p1)  # u1 evicted due to byte size limit
    assert cache.total_size_bytes <= int(f1_size * 1.5)
    cache.close()


def test_file_cache_concurrent_multi_thread(temp_cache_dir: str) -> None:
    source_dir = os.path.join(temp_cache_dir, "source")
    cache_store_dir = os.path.join(temp_cache_dir, "store")
    os.makedirs(source_dir, exist_ok=True)

    f1 = create_sample_parquet(os.path.join(source_dir, "file1.parquet"), 50)

    cache = FileCache(cache_dir=cache_store_dir, max_cached_files=4)
    download_counts = 0
    lock = threading.Lock()

    def mock_download(url: str, target: str) -> str:
        nonlocal download_counts
        time.sleep(0.05)  # Simulate network latency
        with lock:
            download_counts += 1
        shutil.copy2(f1, target)
        return target

    cache._download_file = mock_download
    url = "https://example.com/concurrent.parquet"

    results = []

    def worker():
        p = cache.get_or_download(url)
        results.append(p)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert all(os.path.exists(p) for p in results)
    # Exactly one download happened across all concurrent threads!
    assert download_counts == 1
    cache.close()


def test_row_group_reader_with_file_cache(temp_cache_dir: str) -> None:
    source_dir = os.path.join(temp_cache_dir, "source")
    cache_store_dir = os.path.join(temp_cache_dir, "store")
    os.makedirs(source_dir, exist_ok=True)

    f1 = create_sample_parquet(os.path.join(source_dir, "file1.parquet"), 30)

    file_cache = FileCache(cache_dir=cache_store_dir, max_cached_files=2)

    def mock_download(url: str, target: str) -> str:
        shutil.copy2(f1, target)
        return target

    file_cache._download_file = mock_download

    reader = RowGroupReader(file_cache=file_cache)
    try:
        url = "https://example.com/data/test.parquet"
        tbl = reader.read_row_group(url, 0)
        assert tbl.num_rows == 10
        assert file_cache.cached_files_count == 1
        # Re-reading same URL uses cached file
        tbl2 = reader.read_row_group(url, 1)
        assert tbl2.num_rows == 10
    finally:
        reader.close()


def test_block_shuffled_sampler_file_locality(temp_cache_dir: str) -> None:
    # Create 3 files with 2 row groups each (total 6 row groups, 60 rows)
    files = [
        create_sample_parquet(os.path.join(temp_cache_dir, f"file_{i}.parquet"), num_rows=20, start_id=i * 20)
        for i in range(3)
    ]

    class MockDataset:
        def __len__(self):
            return 60
        def __getitem__(self, idx):
            return {"id": idx}
        def get_row_group_indices(self):
            # 6 row groups: [0..9], [10..19], [20..29], [30..39], [40..49], [50..59]
            return [list(range(i * 10, (i + 1) * 10)) for i in range(6)]
        def get_row_group_file_indices(self):
            # RG 0,1 belong to file 0; RG 2,3 to file 1; RG 4,5 to file 2
            return [0, 0, 1, 1, 2, 2]

    ds = MockDataset()
    sampler = BlockShuffledSampler(ds, seed=42, shuffle=True)
    assert len(sampler) == 60
    assert sampler.file_indices == [0, 0, 1, 1, 2, 2]

    indices = list(sampler)
    assert len(indices) == 60
    assert set(indices) == set(range(60))

    # Verify that samples from the same file remain grouped consecutively!
    file_visited = []
    for idx in indices:
        f_id = idx // 20
        if not file_visited or file_visited[-1] != f_id:
            file_visited.append(f_id)

    # Because file locality is preserved, each file is entered exactly once!
    assert len(file_visited) == 3
    assert set(file_visited) == {0, 1, 2}
