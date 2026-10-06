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
