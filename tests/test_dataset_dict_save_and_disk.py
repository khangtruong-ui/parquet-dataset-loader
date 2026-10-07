"""Tests for dataset dict saving to disk, loading from disk, and default save paths."""

import json
import os
import pytest

from parquet_dataset_loader.api import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SAVE_DIR,
    load_dataset,
    load_from_disk,
)
from parquet_dataset_loader.dataset import IndexedParquetDataset, ParquetDatasetDict
from parquet_dataset_loader.exceptions import DatasetNotFoundError, SplitNotFoundError


def test_default_save_to_disk_is_false_on_streaming(sample_dataset_dir: str) -> None:
    """Verify that save_to_disk defaults to False when streaming=True."""
    # 1. streaming=True with default save_to_disk
    ds = load_dataset(sample_dataset_dir, split="train", streaming=True)
    assert ds.progressive_saver is None

    # 2. streaming=True with explicit save_to_disk=False
    ds_no_save = load_dataset(
        sample_dataset_dir, split="train", streaming=True, save_to_disk=False
    )
    assert ds_no_save.progressive_saver is None


def test_save_to_disk_true_uses_default_save_dir(
    sample_dataset_dir: str, monkeypatch: pytest.MonkeyPatch, temp_dir: str
) -> None:
    """Verify that save_to_disk=True saves to DEFAULT_SAVE_DIR in ~/.cache/parquet_dataset_loader."""
    custom_save_dir = os.path.join(temp_dir, "custom_cache_saved")
    monkeypatch.setattr("parquet_dataset_loader.api.DEFAULT_SAVE_DIR", custom_save_dir)

    ds = load_dataset(
        sample_dataset_dir,
        split="train",
        streaming=True,
        save_to_disk=True,
    )
    assert ds.progressive_saver is not None
    assert ds.progressive_saver.target_dir.startswith(custom_save_dir)
    # Stream one row group
    _ = ds[0]
    assert ds.progressive_saver.saved_count >= 1
    ds.close()


def test_save_and_load_from_disk_multi_split_full(
    temp_dir: str, sample_dataset_dir: str
) -> None:
    """Test saving and loading a multi-split DatasetDict to disk."""
    ds_dict = load_dataset(sample_dataset_dir, split=None, streaming=True)
    assert isinstance(ds_dict, ParquetDatasetDict)
    assert "train" in ds_dict
    assert "validation" in ds_dict

    save_dir = os.path.join(temp_dir, "saved_dataset_dict")
    saved_path = ds_dict.save_to_disk(save_dir)
    assert saved_path == os.path.abspath(save_dir)

    # 1. dataset_dict.json should exist with split names
    manifest_path = os.path.join(save_dir, "dataset_dict.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["splits"] == ["train", "validation"]

    # 2. Both .parquet and .feather exist for each split
    for s in ["train", "validation"]:
        assert os.path.exists(os.path.join(save_dir, s, f"{s}.parquet"))
        assert os.path.exists(os.path.join(save_dir, s, f"{s}.feather"))
        assert os.path.exists(os.path.join(save_dir, s, f"{s}_info.json"))

    # 3. Reload full DatasetDict
    reloaded = load_from_disk(save_dir)
    assert isinstance(reloaded, ParquetDatasetDict)
    assert set(reloaded.keys()) == {"train", "validation"}
    assert len(reloaded["train"]) == 100
    assert len(reloaded["validation"]) == 30
    assert reloaded["train"][0]["id"] == 0
    assert reloaded["validation"][0]["id"] == 1000

    # 4. Reload specific split
    train_only = load_from_disk(save_dir, split="train")
    assert isinstance(train_only, IndexedParquetDataset)
    assert len(train_only) == 100

    val_only = load_from_disk(save_dir, split="validation")
    assert isinstance(val_only, IndexedParquetDataset)
    assert len(val_only) == 30

    # 5. Reload nonexistent split raises SplitNotFoundError
    with pytest.raises(SplitNotFoundError):
        load_from_disk(save_dir, split="nonexistent_split")


def test_save_and_load_custom_split_names(
    temp_dir: str, sample_dataset_dir: str
) -> None:
    """Test saving and loading datasets with arbitrary/custom split names."""
    train_ds = load_dataset(sample_dataset_dir, split="train", streaming=True)
    val_ds = load_dataset(sample_dataset_dir, split="validation", streaming=True)

    custom_dict = ParquetDatasetDict({
        "alpha_split": train_ds.take(20),
        "beta_eval": val_ds.take(15),
    })

    save_dir = os.path.join(temp_dir, "custom_splits")
    custom_dict.save_to_disk(save_dir)

    reloaded = load_from_disk(save_dir)
    assert isinstance(reloaded, ParquetDatasetDict)
    assert set(reloaded.keys()) == {"alpha_split", "beta_eval"}
    assert len(reloaded["alpha_split"]) == 20
    assert len(reloaded["beta_eval"]) == 15
    assert reloaded["alpha_split"].split == "alpha_split"
    assert reloaded["beta_eval"].split == "beta_eval"


def test_default_save_path_no_argument(
    sample_dataset_dir: str, monkeypatch: pytest.MonkeyPatch, temp_dir: str
) -> None:
    """Test calling save_to_disk() without arguments saves to DEFAULT_SAVE_DIR and load_from_disk() without arguments loads it."""
    custom_saved = os.path.join(temp_dir, "pdl_saved_dir")
    monkeypatch.setattr("parquet_dataset_loader.api.DEFAULT_SAVE_DIR", custom_saved)

    ds_dict = load_dataset(
        sample_dataset_dir, split=None, streaming=True, dataset_id="my_test_dict"
    )
    saved_dir = ds_dict.save_to_disk()
    assert saved_dir.startswith(custom_saved)
    assert os.path.exists(os.path.join(saved_dir, "dataset_dict.json"))

    # load_from_disk without path loads the dataset
    reloaded = load_from_disk()
    assert isinstance(reloaded, ParquetDatasetDict)
    assert "train" in reloaded
    assert len(reloaded["train"]) == 100


def test_single_dataset_default_save_path_no_argument(
    sample_dataset_dir: str, monkeypatch: pytest.MonkeyPatch, temp_dir: str
) -> None:
    """Test calling IndexedParquetDataset.save_to_disk() without arguments saves to DEFAULT_SAVE_DIR."""
    custom_saved = os.path.join(temp_dir, "single_saved_dir")
    monkeypatch.setattr("parquet_dataset_loader.api.DEFAULT_SAVE_DIR", custom_saved)

    ds = load_dataset(sample_dataset_dir, split="train", streaming=True, dataset_id="ds_train")
    saved_dir = ds.save_to_disk()
    assert saved_dir.startswith(custom_saved)
    assert os.path.exists(os.path.join(saved_dir, "train.parquet"))
    assert os.path.exists(os.path.join(saved_dir, "train.feather"))

    reloaded = load_from_disk(saved_dir)
    assert isinstance(reloaded, IndexedParquetDataset)
    assert len(reloaded) == 100


def test_parquet_dataset_dict_properties_and_methods(sample_dataset_dir: str) -> None:
    """Test num_rows, column_names, save_progress, and stream_and_save on ParquetDatasetDict."""
    ds_dict = load_dataset(sample_dataset_dir, split=None, streaming=True)
    assert ds_dict.num_rows == {"train": 100, "validation": 30}
    assert "id" in ds_dict.column_names["train"]
    assert "text" in ds_dict.column_names["validation"]
    assert ds_dict.save_progress == 0.0
    assert not ds_dict.is_fully_saved

    # Test stream_and_save
    ds_dict.stream_and_save(show_progress=False)
    # Background prefetching methods should not crash
    ds_dict.start_background_download()
    ds_dict.stop_background_download()
    ds_dict.close()
    assert ds_dict.is_closed


def test_progressive_multi_split_stream_save_and_reload(
    sample_dataset_dir: str, temp_dir: str
) -> None:
    """Test streaming and progressive persistence of multiple splits simultaneously, then loading from disk."""
    save_path = os.path.join(temp_dir, "progressive_multi")
    ds_dict = load_dataset(
        sample_dataset_dir,
        split=None,
        streaming=True,
        save_to_disk=save_path,
    )
    assert isinstance(ds_dict, ParquetDatasetDict)

    # Stream through both splits
    train_rows = [row for row in ds_dict["train"]]
    val_rows = [row for row in ds_dict["validation"]]
    assert len(train_rows) == 100
    assert len(val_rows) == 30
    ds_dict.close()

    # Now reload from disk
    reloaded = load_from_disk(save_path)
    assert isinstance(reloaded, ParquetDatasetDict)
    assert set(reloaded.keys()) == {"train", "validation"}
    assert len(reloaded["train"]) == 100
    assert len(reloaded["validation"]) == 30
    assert reloaded["train"][0]["id"] == 0
    assert reloaded["validation"][0]["id"] == 1000
    reloaded.close()
