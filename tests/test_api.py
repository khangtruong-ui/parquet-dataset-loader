"""Unit tests for the top-level load_dataset and load_from_disk API."""

import os
import pytest

from parquet_dataset_loader.api import load_dataset, load_from_disk
from parquet_dataset_loader.dataset import IndexedParquetDataset, ParquetDatasetDict


def test_load_dataset_local_dir_splits(sample_dataset_dir: str) -> None:
    # Load all splits (split=None)
    ds_dict = load_dataset(sample_dataset_dir, split=None, streaming=True)
    assert isinstance(ds_dict, ParquetDatasetDict)
    assert "train" in ds_dict
    assert "validation" in ds_dict
    assert len(ds_dict["train"]) == 100
    assert len(ds_dict["validation"]) == 30

    # Load single split
    train_ds = load_dataset(sample_dataset_dir, split="train", streaming=True)
    assert isinstance(train_ds, IndexedParquetDataset)
    assert len(train_ds) == 100
    assert train_ds[0]["id"] == 0
    assert train_ds[-1]["id"] == 99


def test_load_dataset_split_slice_syntax(sample_dataset_dir: str) -> None:
    # Sliced split: 'train[:20]'
    sliced_ds = load_dataset(sample_dataset_dir, split="train[:20]", streaming=True)
    assert len(sliced_ds) == 20
    assert sliced_ds[0]["id"] == 0
    assert sliced_ds[-1]["id"] == 19

    # Sliced split: 'train[30:50]'
    range_ds = load_dataset(sample_dataset_dir, split="train[30:50]", streaming=True)
    assert len(range_ds) == 20
    assert range_ds[0]["id"] == 30
    assert range_ds[-1]["id"] == 49


def test_load_dataset_column_projection(sample_dataset_dir: str) -> None:
    ds = load_dataset(
        sample_dataset_dir,
        split="train",
        streaming=True,
        columns=["id", "text"],
    )
    assert ds.column_names == ["id", "text"]
    assert set(ds[0].keys()) == {"id", "text"}


def test_load_dataset_non_streaming_local(sample_dataset_dir: str) -> None:
    # Local files with streaming=False
    ds = load_dataset(sample_dataset_dir, split="train", streaming=False)
    assert len(ds) == 100
    assert ds[0]["id"] == 0


def test_save_and_load_from_disk(temp_dir: str, sample_dataset_dir: str) -> None:
    ds = load_dataset(sample_dataset_dir, split="train", streaming=True)
    save_dir = os.path.join(temp_dir, "saved_single")
    ds.save_to_disk(save_dir)

    reloaded = load_from_disk(save_dir)
    assert len(reloaded) == 100
    assert reloaded[0]["id"] == 0
    assert reloaded[-1]["id"] == 99


def test_save_and_load_from_disk_multi_split(temp_dir: str, sample_dataset_dir: str) -> None:
    ds_dict = load_dataset(sample_dataset_dir, split=None, streaming=True)
    save_dir = os.path.join(temp_dir, "saved_multi")
    ds_dict.save_to_disk(save_dir)

    reloaded_dict = load_from_disk(save_dir)
    assert isinstance(reloaded_dict, ParquetDatasetDict)
    assert "train" in reloaded_dict
    assert len(reloaded_dict["train"]) == 100


def test_load_dataset_token_propagation(sample_dataset_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Automatic resolution from HF_TOKEN env var
    monkeypatch.setenv("HF_TOKEN", "hf_propagated_token")
    ds = load_dataset(sample_dataset_dir, split="train", streaming=True)
    assert ds.reader.token == "hf_propagated_token"

    # 2. Explicit token=False disables auth
    ds_no_auth = load_dataset(sample_dataset_dir, split="train", streaming=True, token=False)
    assert ds_no_auth.reader.token is None

    # 3. Explicit string override
    ds_override = load_dataset(
        sample_dataset_dir, split="train", streaming=True, token="hf_explicit_override"
    )
    assert ds_override.reader.token == "hf_explicit_override"

