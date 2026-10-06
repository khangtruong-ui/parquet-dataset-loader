"""Unit tests for MetadataIndex and row location logic."""

import os
import pyarrow as pa
import pytest

from parquet_dataset_loader.exceptions import IndexOutOfBoundsError
from parquet_dataset_loader.index import (
    MetadataIndex,
    ParquetFileInfo,
    RowGroupInfo,
    build_metadata_index,
)
from tests.conftest import create_sample_parquet_file


def test_row_group_info_properties() -> None:
    rg = RowGroupInfo(
        file_index=0,
        file_url="file:///tmp/test.parquet",
        rg_index=1,
        num_rows=50,
        global_start_row=25,
        global_end_row=75,
        byte_start=1000,
        byte_end=5000,
    )
    assert rg.byte_length == 4000
    assert rg.num_rows == 50
    assert rg.global_start_row == 25
    assert rg.global_end_row == 75


def test_build_metadata_index_local(temp_dir: str) -> None:
    f1 = os.path.join(temp_dir, "f1.parquet")
    f2 = os.path.join(temp_dir, "f2.parquet")
    create_sample_parquet_file(f1, num_rows=40, row_group_size=20, start_id=0)
    create_sample_parquet_file(f2, num_rows=60, row_group_size=30, start_id=40)

    index = build_metadata_index([f1, f2], split_name="train", use_cache=False)

    assert index.total_rows == 100
    assert index.split_name == "train"
    assert len(index.files) == 2
    assert len(index.row_groups) == 4  # 2 in f1, 2 in f2
    assert index.column_names == ["id", "text", "value", "payload"]

    # Verify cumulative offsets
    assert index.row_groups[0].global_start_row == 0
    assert index.row_groups[0].global_end_row == 20
    assert index.row_groups[1].global_start_row == 20
    assert index.row_groups[1].global_end_row == 40
    assert index.row_groups[2].global_start_row == 40
    assert index.row_groups[2].global_end_row == 70
    assert index.row_groups[3].global_start_row == 70
    assert index.row_groups[3].global_end_row == 100


def test_locate_row_binary_search(temp_dir: str) -> None:
    f1 = os.path.join(temp_dir, "f1.parquet")
    f2 = os.path.join(temp_dir, "f2.parquet")
    create_sample_parquet_file(f1, num_rows=40, row_group_size=20, start_id=0)
    create_sample_parquet_file(f2, num_rows=60, row_group_size=30, start_id=40)

    index = build_metadata_index([f1, f2], split_name="train", use_cache=False)

    # First row
    rg, local = index.locate_row(0)
    assert rg.file_index == 0
    assert rg.rg_index == 0
    assert local == 0

    # End of first row group
    rg, local = index.locate_row(19)
    assert rg.file_index == 0
    assert rg.rg_index == 0
    assert local == 19

    # Start of second row group
    rg, local = index.locate_row(20)
    assert rg.file_index == 0
    assert rg.rg_index == 1
    assert local == 0

    # Start of second file
    rg, local = index.locate_row(40)
    assert rg.file_index == 1
    assert rg.rg_index == 0
    assert local == 0

    # Middle of second file second row group
    rg, local = index.locate_row(85)
    assert rg.file_index == 1
    assert rg.rg_index == 1
    assert local == 15

    # Last row
    rg, local = index.locate_row(99)
    assert rg.file_index == 1
    assert rg.rg_index == 1
    assert local == 29

    # Negative index
    rg, local = index.locate_row(-1)
    assert rg.file_index == 1
    assert rg.rg_index == 1
    assert local == 29


def test_locate_row_out_of_bounds(temp_dir: str) -> None:
    f1 = os.path.join(temp_dir, "f1.parquet")
    create_sample_parquet_file(f1, num_rows=20, row_group_size=10)
    index = build_metadata_index([f1], split_name="train", use_cache=False)

    with pytest.raises(IndexOutOfBoundsError):
        index.locate_row(20)

    with pytest.raises(IndexOutOfBoundsError):
        index.locate_row(100)

    with pytest.raises(IndexOutOfBoundsError):
        index.locate_row(-21)


def test_locate_range(temp_dir: str) -> None:
    f1 = os.path.join(temp_dir, "f1.parquet")
    f2 = os.path.join(temp_dir, "f2.parquet")
    create_sample_parquet_file(f1, num_rows=40, row_group_size=20, start_id=0)
    create_sample_parquet_file(f2, num_rows=60, row_group_size=30, start_id=40)
    index = build_metadata_index([f1, f2], split_name="train", use_cache=False)

    # Range fully within row group 0
    ranges = index.locate_range(5, 15)
    assert len(ranges) == 1
    rg, l_start, l_stop = ranges[0]
    assert rg.rg_index == 0
    assert l_start == 5
    assert l_stop == 15

    # Range spanning across row groups and files (15 to 45)
    # rg 0 (0..20): 15..20 (5 rows)
    # rg 1 (20..40): 0..20 (20 rows)
    # rg 2 (40..70): 0..5 (5 rows)
    ranges = index.locate_range(15, 45)
    assert len(ranges) == 3
    assert ranges[0][1] == 15 and ranges[0][2] == 20
    assert ranges[1][1] == 0 and ranges[1][2] == 20
    assert ranges[2][1] == 0 and ranges[2][2] == 5


def test_index_serialization_and_cache(temp_dir: str) -> None:
    f1 = os.path.join(temp_dir, "f1.parquet")
    create_sample_parquet_file(f1, num_rows=30, row_group_size=10)

    cache_file = os.path.join(temp_dir, "cache", "index.json")
    index1 = build_metadata_index([f1], split_name="train", cache_dir=temp_dir, use_cache=True)
    index1.save_cache(cache_file)

    loaded = MetadataIndex.load_cache(cache_file)
    assert loaded is not None
    assert loaded.total_rows == 30
    assert loaded.split_name == "train"
    assert len(loaded.row_groups) == 3
    assert loaded.column_names == index1.column_names
    assert loaded.schema.names == index1.schema.names

    rg, local = loaded.locate_row(25)
    assert rg.rg_index == 2
    assert local == 5
