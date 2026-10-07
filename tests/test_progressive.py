"""Unit tests for progressive streaming and simultaneous save-to-disk."""

import os
import time
import pyarrow as pa
import pytest

from parquet_dataset_loader.api import load_dataset, load_from_disk
from parquet_dataset_loader.index import RowGroupInfo
from parquet_dataset_loader.progressive import (
    BackgroundDownloader,
    ProgressiveDiskSaver,
)
from parquet_dataset_loader.reader import RowGroupReader
from tests.conftest import create_sample_parquet_file


def test_progressive_disk_saver_basics(temp_dir: str) -> None:
    save_dir = os.path.join(temp_dir, "progressive_test")
    saver = ProgressiveDiskSaver(target_dir=save_dir, split_name="train", total_row_groups=2)

    assert saver.saved_count == 0
    assert saver.save_progress == 0.0
    assert not saver.is_complete
    assert not saver.is_saved(0, 0)
    assert saver.get(0, 0) is None

    t0 = pa.Table.from_arrays([pa.array([1, 2, 3])], names=["id"])
    t1 = pa.Table.from_arrays([pa.array([4, 5, 6])], names=["id"])

    # Save first row group
    saver.save(file_idx=0, rg_idx=0, table=t0)
    assert saver.is_saved(0, 0)
    assert saver.saved_count == 1
    assert saver.save_progress == 0.5
    assert not saver.is_complete

    loaded_0 = saver.get(0, 0)
    assert loaded_0 is not None
    assert loaded_0.column("id").to_pylist() == [1, 2, 3]

    # Save second row group
    saver.save(file_idx=0, rg_idx=1, table=t1)
    assert saver.is_saved(0, 1)
    assert saver.saved_count == 2
    assert saver.save_progress == 1.0
    assert saver.is_complete

    # Manifest reload test (create new instance pointing to same dir)
    saver_reloaded = ProgressiveDiskSaver(target_dir=save_dir, split_name="train", total_row_groups=2)
    assert saver_reloaded.is_saved(0, 0)
    assert saver_reloaded.is_saved(0, 1)
    assert saver_reloaded.is_complete

    # Finalize
    info_file = saver.finalize()
    assert os.path.exists(info_file)


def test_load_dataset_stream_and_save_to_disk(temp_dir: str, sample_dataset_dir: str) -> None:
    save_path = os.path.join(temp_dir, "stream_saved")

    # Load with save_to_disk parameter
    # Should return immediately with streaming=True without blocking
    ds = load_dataset(
        sample_dataset_dir,
        split="train",
        streaming=True,
        save_to_disk=save_path,
    )

    assert ds.progressive_saver is not None
    assert ds.save_progress == 0.0
    assert not ds.is_fully_saved

    # Access index 0 -> triggers fetch and save of row group 0
    row_0 = ds[0]
    assert row_0["id"] == 0
    assert ds.save_progress > 0.0
    assert ds.progressive_saver.is_saved(0, 0)

    # Calling ds.stream_and_save() completes saving the whole dataset
    ds.stream_and_save(show_progress=False)
    assert ds.is_fully_saved
    assert ds.save_progress == 1.0

    # Reload from the progressive save directory using load_from_disk
    reloaded = load_from_disk(save_path)
    assert len(reloaded) == 100
    assert reloaded[0]["id"] == 0
    assert reloaded[-1]["id"] == 99


def test_background_downloader(temp_dir: str, sample_dataset_dir: str) -> None:
    save_path = os.path.join(temp_dir, "bg_saved")

    # Load with background_download=True
    ds = load_dataset(
        sample_dataset_dir,
        split="train",
        streaming=True,
        save_to_disk=save_path,
        background_download=True,
    )

    # First row is accessible immediately
    assert ds[0]["id"] == 0

    # Wait for background thread to complete the small sample dataset
    if ds.background_downloader is not None:
        ds.background_downloader.join(timeout=3.0)

    assert ds.save_progress == 1.0
    assert ds.is_fully_saved

    ds.close()


def test_stream_and_save_multi_file_dataset_beyond_max_open_files(temp_dir: str) -> None:
    """Test stream_and_save on a dataset with 10 files (exceeding max_open_files=8).

    Guarantees no unpack exceptions occur during full dataset streaming and that
    all row groups are saved and loadable offline.
    """
    data_dir = os.path.join(temp_dir, "multi_file_data")
    os.makedirs(data_dir, exist_ok=True)
    num_files = 10
    rows_per_file = 15

    for i in range(num_files):
        f = os.path.join(data_dir, f"train-{i:05d}.parquet")
        create_sample_parquet_file(
            f,
            num_rows=rows_per_file,
            row_group_size=rows_per_file,
            start_id=i * rows_per_file,
        )

    save_path = os.path.join(temp_dir, "multi_saved")
    ds = load_dataset(
        data_dir,
        split="train",
        streaming=True,
        save_to_disk=save_path,
        columns=["id", "text"],
    )

    assert len(ds) == num_files * rows_per_file
    assert ds.save_progress == 0.0

    # Stream through all 10 files (triggers eviction at file 8 and 9)
    ds.stream_and_save(show_progress=False)
    assert ds.is_fully_saved
    assert ds.save_progress == 1.0

    # Verify offline reload
    offline_ds = load_from_disk(save_path)
    assert len(offline_ds) == num_files * rows_per_file
    assert offline_ds[0]["id"] == 0
    assert offline_ds[-1]["id"] == (num_files * rows_per_file) - 1
    assert list(offline_ds[0].keys()) == ["id", "text"]

