"""Unit tests for dataset manager lifecycle, stream-from-index prefetching, and disk resume."""

import json
import os
import time
from typing import Generator
import pyarrow as pa
import pytest

import parquet_dataset_loader as pdl
from parquet_dataset_loader.api import load_dataset, load_from_disk, resume_dataset
from parquet_dataset_loader.dataset import IndexedParquetDataset, ParquetDatasetDict
from parquet_dataset_loader.exceptions import ParquetDatasetError
from parquet_dataset_loader.manager import (
    DatasetManager,
    close_all_datasets,
    get_dataset_manager,
    list_active_datasets,
    managed_datasets,
)
from parquet_dataset_loader.progressive import BackgroundDownloader, ProgressiveDiskSaver
from tests.conftest import create_sample_parquet_file


@pytest.fixture(autouse=True)
def clean_manager_state() -> Generator[None, None, None]:
    """Ensure manager state is reset before and after every test."""
    close_all_datasets()
    yield
    close_all_datasets()


def test_manager_multiple_dataset_instances(sample_dataset_dir: str) -> None:
    """Test managing multiple dataset instances simultaneously."""
    mgr = get_dataset_manager()
    assert mgr.active_count == 0

    ds1 = load_dataset(sample_dataset_dir, split="train", dataset_id="ds_alpha")
    ds2 = load_dataset(sample_dataset_dir, split="train", dataset_id="ds_beta")
    ds3 = load_dataset(sample_dataset_dir, split="train")  # auto-assigned ID

    assert mgr.active_count == 3
    assert ds1.dataset_id == "ds_alpha"
    assert ds2.dataset_id == "ds_beta"
    assert ds3.dataset_id.startswith("dataset_")

    active = list_active_datasets()
    assert "ds_alpha" in active
    assert "ds_beta" in active
    assert ds3.dataset_id in active

    assert active["ds_alpha"]["split"] == "train"
    assert not active["ds_alpha"]["is_closed"]

    # Close one dataset individually
    ds1.close()
    assert ds1.is_closed
    assert mgr.active_count == 2
    assert "ds_alpha" not in list_active_datasets()
    assert "ds_beta" in list_active_datasets()

    # Manager close by ID
    assert mgr.close("ds_beta") is True
    assert ds2.is_closed
    assert mgr.active_count == 1

    # Closing nonexistent returns False
    assert mgr.close("nonexistent_id") is False

    # close_all_datasets closes remaining
    closed_count = close_all_datasets()
    assert closed_count == 1
    assert ds3.is_closed
    assert mgr.active_count == 0


def test_manager_id_collision_resolution(sample_dataset_dir: str) -> None:
    """Test that duplicate dataset IDs are automatically disambiguated with unique suffixes."""
    mgr = get_dataset_manager()
    ds1 = load_dataset(sample_dataset_dir, split="train", dataset_id="custom_id")
    ds2 = load_dataset(sample_dataset_dir, split="train", dataset_id="custom_id")

    assert ds1.dataset_id == "custom_id"
    assert ds2.dataset_id == "custom_id_1"
    assert mgr.active_count == 2
    assert "custom_id" in mgr
    assert "custom_id_1" in mgr


def test_dataset_context_manager_protocol(sample_dataset_dir: str) -> None:
    """Test using `with load_dataset(...) as ds:` cleanly closes on exit."""
    mgr = get_dataset_manager()

    with load_dataset(sample_dataset_dir, split="train", dataset_id="ctx_ds") as ds:
        assert not ds.is_closed
        assert "ctx_ds" in mgr
        assert len(ds) > 0
        _ = ds[0]

    assert ds.is_closed
    assert "ctx_ds" not in mgr


def test_managed_datasets_context_manager(sample_dataset_dir: str) -> None:
    """Test managed_datasets() context manager closes datasets created within its scope."""
    mgr = get_dataset_manager()

    # Create one dataset outside the block
    ds_outside = load_dataset(sample_dataset_dir, split="train", dataset_id="ds_outside")
    assert not ds_outside.is_closed

    with managed_datasets():
        ds_inside_1 = load_dataset(sample_dataset_dir, split="train", dataset_id="ds_in_1")
        ds_inside_2 = load_dataset(sample_dataset_dir, split="train", dataset_id="ds_in_2")
        assert not ds_inside_1.is_closed
        assert not ds_inside_2.is_closed
        assert mgr.active_count == 3

    # Inside datasets must be closed automatically
    assert ds_inside_1.is_closed
    assert ds_inside_2.is_closed
    # Outside dataset remains open and active
    assert not ds_outside.is_closed
    assert mgr.active_count == 1

    ds_outside.close()
    assert ds_outside.is_closed
    assert mgr.active_count == 0


def test_parquet_dataset_dict_managed(sample_dataset_dir: str) -> None:
    """Test ParquetDatasetDict integration with DatasetManager."""
    mgr = get_dataset_manager()

    ds_dict = load_dataset(sample_dataset_dir, dataset_id="my_dict")
    assert isinstance(ds_dict, ParquetDatasetDict)
    assert ds_dict.dataset_id == "my_dict"
    assert "my_dict" in mgr

    status = ds_dict.status
    assert status["dataset_id"] == "my_dict"
    assert "splits" in status
    assert not ds_dict.is_closed

    ds_dict.close()
    assert ds_dict.is_closed
    assert "my_dict" not in mgr


def test_stream_from_index_forward_prefetching(temp_dir: str) -> None:
    """Test that BackgroundDownloader prefetches starting from the stream's start_rg_index."""
    # Create a 4-file dataset (1 row group per file = 4 row groups total)
    files = []
    for i in range(4):
        fpath = os.path.join(temp_dir, f"chunk_{i}.parquet")
        create_sample_parquet_file(fpath, num_rows=5, row_group_size=5, start_id=i * 5)
        files.append(fpath)

    save_dir = os.path.join(temp_dir, "prefetch_target")
    saver = ProgressiveDiskSaver(
        target_dir=save_dir,
        split_name="train",
        total_row_groups=4,
        source_path=temp_dir,
    )

    class MockReader:
        def __init__(self, fpaths: list[str], saver: ProgressiveDiskSaver) -> None:
            self.read_order: list[tuple[int, int]] = []
            self.fpaths = fpaths
            self.saver = saver

        def read_row_group(
            self,
            file_url: str = "",
            rg_index: int = 0,
            columns=None,
            file_index: int = 0,
        ) -> pa.Table:
            self.read_order.append((file_index, rg_index))
            t = pa.Table.from_arrays([pa.array([1, 2, 3])], names=["val"])
            self.saver.save(file_idx=file_index, rg_idx=rg_index, table=t)
            time.sleep(0.01)
            return t

    reader = MockReader(files, saver)
    row_groups = [(0, 0), (1, 0), (2, 0), (3, 0)]

    # Start prefetcher with start_rg_index=2 (so row group 2 and 3 are downloaded first!)
    downloader = BackgroundDownloader(
        reader=reader,  # type: ignore
        saver=saver,
        row_groups=row_groups,
        start_rg_index=2,
    )
    downloader.start()
    downloader.wait(timeout=5.0)

    # Verify that the read order prioritized row groups starting from index 2
    assert reader.read_order[0] == (2, 0)
    assert reader.read_order[1] == (3, 0)
    assert reader.read_order[2] == (0, 0)
    assert reader.read_order[3] == (1, 0)
    assert saver.is_complete


def test_progressive_disk_saver_tmp_file_cleanup_and_recovery(temp_dir: str) -> None:
    """Test that ProgressiveDiskSaver recovers saved files and cleans up orphaned .tmp files."""
    save_dir = os.path.join(temp_dir, "recovery_test")
    os.makedirs(save_dir, exist_ok=True)

    # Simulate an interrupted save:
    # 1 valid saved feather file
    saver1 = ProgressiveDiskSaver(
        target_dir=save_dir,
        split_name="train",
        total_row_groups=3,
        source_path="fake/source",
    )
    t0 = pa.Table.from_arrays([pa.array([10, 20])], names=["val"])
    saver1.save(file_idx=0, rg_idx=0, table=t0)
    assert saver1.saved_count == 1

    # Simulate a crash leaving an orphaned temporary file
    orphan_tmp = os.path.join(save_dir, "train_000001_000000.feather.tmp.123456")
    with open(orphan_tmp, "w") as f:
        f.write("corrupt partial data")
    assert os.path.exists(orphan_tmp)

    # Initialize a second saver on the same directory (simulating restart / reload)
    saver2 = ProgressiveDiskSaver(
        target_dir=save_dir,
        split_name="train",
        total_row_groups=3,
        source_path="fake/source",
    )

    # Orphaned tmp file must be purged
    assert not os.path.exists(orphan_tmp)
    # Already-saved row group 0 must be recognized
    assert saver2.is_saved(0, 0)
    assert saver2.saved_count == 1
    assert not saver2.is_saved(0, 1)

    # Can read row group 0 from disk
    recovered_table = saver2.get(0, 0)
    assert recovered_table is not None
    assert recovered_table.column("val").to_pylist() == [10, 20]

    # Save remaining row groups
    t1 = pa.Table.from_arrays([pa.array([30, 40])], names=["val"])
    t2 = pa.Table.from_arrays([pa.array([50, 60])], names=["val"])
    saver2.save(file_idx=0, rg_idx=1, table=t1)
    saver2.save(file_idx=0, rg_idx=2, table=t2)
    assert saver2.is_complete


def test_resume_dataset_from_incomplete_disk(temp_dir: str) -> None:
    """Test resuming an incomplete progressive save on disk via resume_dataset."""
    # Create a source dataset with 3 files (1 row group each, 5 rows each = 15 rows)
    src_dir = os.path.join(temp_dir, "source_repo")
    os.makedirs(src_dir, exist_ok=True)
    for i in range(3):
        create_sample_parquet_file(
            os.path.join(src_dir, f"data_{i}.parquet"),
            num_rows=5,
            row_group_size=5,
            start_id=i * 5,
        )

    save_dir = os.path.join(temp_dir, "target_incomplete")

    # Start streaming with save_to_disk, but read only 1 row group (first 5 rows) then close
    ds = load_dataset(
        src_dir,
        split="train",
        streaming=True,
        save_to_disk=save_dir,
        background_download=False,
    )
    assert len(ds) == 15
    # Read row 0 (triggers loading row group 0)
    _ = ds[0]
    # Check that disk saver recorded row group 0
    assert ds.progressive_saver is not None
    assert ds.progressive_saver.saved_count == 1
    assert not ds.progressive_saver.is_complete
    ds.close()

    # Manifest exists and records incomplete state
    manifest_path = os.path.join(save_dir, "train_manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["is_complete"] is False
    assert len(manifest["saved_row_groups"]) == 1

    # Resume the dataset using resume_dataset()
    resumed_ds = resume_dataset(
        dataset_path=save_dir,
        background_download=False,
    )
    assert isinstance(resumed_ds, IndexedParquetDataset)
    assert len(resumed_ds) == 15

    # Access rows from incomplete parts (row 5 and row 10)
    row_5 = resumed_ds[5]
    assert row_5 is not None
    row_14 = resumed_ds[14]
    assert row_14 is not None

    # Verify that all row groups are now saved
    assert resumed_ds.progressive_saver.is_complete
    resumed_ds.close()

    # Now load_from_disk can load it completely offline
    final_ds = load_from_disk(save_dir)
    assert len(final_ds) == 15
    assert final_ds[0]["id"] == 0
    final_ds.close()


def test_load_from_disk_resume_flag(temp_dir: str) -> None:
    """Test load_from_disk(..., resume=True) unblocks incomplete datasets."""
    src_dir = os.path.join(temp_dir, "source_repo_2")
    os.makedirs(src_dir, exist_ok=True)
    for i in range(2):
        create_sample_parquet_file(
            os.path.join(src_dir, f"part_{i}.parquet"),
            num_rows=4,
            row_group_size=4,
            start_id=i * 4,
        )

    save_dir = os.path.join(temp_dir, "incomplete_save_2")
    ds = load_dataset(
        src_dir,
        split="train",
        streaming=True,
        save_to_disk=save_dir,
        background_download=False,
    )
    _ = ds[0]  # triggers row group 0 save
    ds.close()

    # Calling load_from_disk without resume=True or allow_incomplete=True should raise ParquetDatasetError
    with pytest.raises(ParquetDatasetError) as exc_info:
        load_from_disk(save_dir, resume=False, allow_incomplete=False)
    assert "Incomplete progressive dataset" in str(exc_info.value)

    # Calling load_from_disk with resume=True resumes it seamlessly
    resumed_ds = load_from_disk(save_dir, resume=True, background_download=False)
    assert len(resumed_ds) == 8
    # Read remaining rows
    assert resumed_ds[7] is not None
    resumed_ds.close()


def test_stop_all_background_tasks_and_cleanup(temp_dir: str, sample_dataset_dir: str) -> None:
    from parquet_dataset_loader.manager import (
        close_all_datasets,
        cleanup_background_tasks,
        get_dataset_manager,
        stop_all_background_tasks,
    )
    from parquet_dataset_loader.cli import cli_kill, cleanup_stale_cache_files

    # Create dummy temp file in cache dir
    dummy_tmp = os.path.join(temp_dir, "test.tmp.feather")
    with open(dummy_tmp, "w") as f:
        f.write("temporary data")
    assert os.path.exists(dummy_tmp)
    removed = cleanup_stale_cache_files(temp_dir)
    assert removed == 1
    assert not os.path.exists(dummy_tmp)

    # Test load with background download and manager stopping
    ds = load_dataset(
        sample_dataset_dir,
        streaming=True,
        save_to_disk=os.path.join(temp_dir, "ds_stop"),
        background_download=True,
        manage=True,
    )
    assert get_dataset_manager().active_count >= 1

    # Call stop_all_background_tasks
    stopped = stop_all_background_tasks()
    assert stopped >= 1

    # Call cleanup_background_tasks
    cleaned = cleanup_background_tasks()
    assert cleaned >= 1
    assert get_dataset_manager().active_count == 0

    # Call cli_kill
    exit_code = cli_kill(["--cache-dir", temp_dir, "-q"])
    assert exit_code == 0

