"""Unit tests for RowGroupReader and I/O handling."""

import os
import pytest

from parquet_dataset_loader.cache import RowGroupMemoryCache
from parquet_dataset_loader.exceptions import CorruptParquetError
from parquet_dataset_loader.reader import RowGroupReader
from tests.conftest import create_sample_parquet_file


def test_row_group_reader_local(temp_dir: str) -> None:
    f1 = os.path.join(temp_dir, "test.parquet")
    create_sample_parquet_file(f1, num_rows=50, row_group_size=25)

    cache = RowGroupMemoryCache(max_entries=2)
    reader = RowGroupReader(memory_cache=cache)

    try:
        # Read RG 0
        rg0 = reader.read_row_group(f1, 0)
        assert rg0.num_rows == 25
        assert rg0.column_names == ["id", "text", "value", "payload"]
        assert cache.stats["misses"] == 1

        # Second read of RG 0 should hit cache
        rg0_cached = reader.read_row_group(f1, 0)
        assert rg0_cached.num_rows == 25
        assert cache.stats["hits"] == 1

        # Read RG 1 with column projection
        rg1_proj = reader.read_row_group(f1, 1, columns=["id", "value"])
        assert rg1_proj.num_rows == 25
        assert rg1_proj.column_names == ["id", "value"]
    finally:
        reader.close()


def test_row_group_reader_error(temp_dir: str) -> None:
    reader = RowGroupReader()
    try:
        with pytest.raises(CorruptParquetError):
            reader.read_row_group(os.path.join(temp_dir, "missing.parquet"), 0)
    finally:
        reader.close()


def test_row_group_reader_lru_file_eviction(temp_dir: str) -> None:
    """Test that file handles are evicted according to LRU policy when max_open_files is reached."""
    # Create 6 files
    files = [
        create_sample_parquet_file(
            os.path.join(temp_dir, f"file_{i}.parquet"),
            num_rows=20,
            row_group_size=10,
            start_id=i * 20,
        )
        for i in range(6)
    ]

    reader = RowGroupReader(max_open_files=3)
    try:
        # Read files 0, 1, 2 -> pool is full (size 3)
        reader.read_row_group(files[0], 0)
        reader.read_row_group(files[1], 0)
        reader.read_row_group(files[2], 0)
        assert len(reader._open_files) == 3
        assert list(reader._open_files.keys()) == [files[0], files[1], files[2]]

        # Re-read file 0 -> moves file 0 to MRU position: order becomes [files[1], files[2], files[0]]
        reader.read_row_group(files[0], 1)
        assert list(reader._open_files.keys()) == [files[1], files[2], files[0]]

        # Read file 3 -> should evict files[1] (oldest), keeping files[2], files[0], files[3]
        reader.read_row_group(files[3], 0)
        assert len(reader._open_files) == 3
        assert list(reader._open_files.keys()) == [files[2], files[0], files[3]]
        assert files[1] not in reader._open_files

        # Re-read evicted file 1 -> evicts files[2]
        tbl = reader.read_row_group(files[1], 0)
        assert tbl.num_rows == 10
        assert len(reader._open_files) == 3
        assert list(reader._open_files.keys()) == [files[0], files[3], files[1]]
    finally:
        reader.close()
        assert len(reader._open_files) == 0


def test_row_group_reader_default_capacity_eviction(temp_dir: str) -> None:
    """Verify that default max_open_files=8 evicts cleanly without unpack errors at file 8 and beyond."""
    files = [
        create_sample_parquet_file(
            os.path.join(temp_dir, f"split_file_{i}.parquet"),
            num_rows=10,
            row_group_size=10,
            start_id=i * 10,
        )
        for i in range(12)
    ]

    reader = RowGroupReader(max_open_files=8)
    try:
        for i, f in enumerate(files):
            tbl = reader.read_row_group(f, 0)
            assert tbl.num_rows == 10
            expected_len = min(i + 1, 8)
            assert len(reader._open_files) == expected_len

        # Pool size remains bounded at 8
        assert len(reader._open_files) == 8
        # The oldest files (0 through 3) should have been evicted; newest 8 kept (4 through 11)
        assert files[0] not in reader._open_files
        assert files[3] not in reader._open_files
        assert files[11] in reader._open_files
    finally:
        reader.close()


def test_row_group_reader_retry_on_transient_error(temp_dir: str) -> None:
    """Test that RowGroupReader retries and recovers from transient remote reading errors."""
    f1 = os.path.join(temp_dir, "test_retry.parquet")
    create_sample_parquet_file(f1, num_rows=30, row_group_size=10)

    reader = RowGroupReader(max_retries=3, retry_delay=0.01, max_retry_delay=0.05)
    call_count = 0
    original_get = reader._get_parquet_file

    def flaky_get_parquet_file(url: str):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionResetError(f"Simulated network drop on attempt {call_count}")
        return original_get(f1)

    reader._get_parquet_file = flaky_get_parquet_file

    try:
        # Use remote URL prefix to trigger retry logic
        tbl = reader.read_row_group("https://example.com/dataset/flaky.parquet", 0)
        assert tbl.num_rows == 10
        assert call_count == 3  # Succeeded on 3rd attempt
    finally:
        reader.close()


def test_row_group_reader_exhausted_retries_raises_corrupt_parquet_error() -> None:
    """Test that exhausting all retries on persistent errors raises CorruptParquetError."""
    reader = RowGroupReader(max_retries=3, retry_delay=0.01, max_retry_delay=0.02)

    def always_fail_get(url: str):
        raise OSError("Connection timed out completely")

    reader._get_parquet_file = always_fail_get

    try:
        with pytest.raises(CorruptParquetError) as exc_info:
            reader.read_row_group("https://example.com/bad.parquet", 0)
        assert "attempt" in str(exc_info.value).lower()
    finally:
        reader.close()


def test_row_group_reader_evict_and_reset_filesystem(temp_dir: str) -> None:
    """Test explicit handle eviction and filesystem connection pool reset."""
    f1 = os.path.join(temp_dir, "test_evict.parquet")
    create_sample_parquet_file(f1, num_rows=10, row_group_size=10)

    reader = RowGroupReader()
    try:
        reader.read_row_group(f1, 0)
        assert f1 in reader._open_files
        reader._evict_file(f1)
        assert f1 not in reader._open_files

        # Test reset_filesystem
        reader.memory_cache.clear()
        reader.read_row_group(f1, 0)
        assert len(reader._open_files) == 1
        reader.reset_filesystem()
        assert len(reader._open_files) == 0
        assert reader._http_fs is None
    finally:
        reader.close()


