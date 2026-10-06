"""Unit tests for Hugging Face resolver and split inference."""

import os
import pytest

from parquet_dataset_loader.exceptions import DatasetNotFoundError, SplitNotFoundError
from parquet_dataset_loader.hf_resolver import (
    infer_split_name,
    parse_split_slice,
    resolve_local_files,
    resolve_parquet_dataset,
)
from tests.conftest import create_sample_parquet_file


def test_infer_split_name() -> None:
    assert infer_split_name("data/train-00000.parquet") == "train"
    assert infer_split_name("train_data.parquet") == "train"
    assert infer_split_name("validation-00001.parquet") == "validation"
    assert infer_split_name("val_0.parquet") == "validation"
    assert infer_split_name("dev-set.parquet") == "validation"
    assert infer_split_name("eval_features.parquet") == "validation"
    assert infer_split_name("test-00000.parquet") == "test"
    assert infer_split_name("testing.parquet") == "test"
    assert infer_split_name("unnamed.parquet") == "train"


def test_parse_split_slice() -> None:
    assert parse_split_slice("train") == ("train", None)
    assert parse_split_slice("train[:100]") == ("train", slice(None, 100, None))
    assert parse_split_slice("train[50:150]") == ("train", slice(50, 150, None))
    assert parse_split_slice("train[::2]") == ("train", slice(None, None, 2))
    assert parse_split_slice("validation[-20:]") == ("validation", slice(-20, None, None))
    assert parse_split_slice(None) == (None, None)


def test_resolve_local_files(temp_dir: str) -> None:
    f1 = os.path.join(temp_dir, "train-00000.parquet")
    f2 = os.path.join(temp_dir, "validation-00000.parquet")
    create_sample_parquet_file(f1, num_rows=10)
    create_sample_parquet_file(f2, num_rows=10)

    splits = resolve_local_files(temp_dir)
    assert "train" in splits
    assert "validation" in splits
    assert splits["train"] == [f1]
    assert splits["validation"] == [f2]

    # Single file
    single = resolve_local_files(f1)
    assert "train" in single
    assert single["train"] == [f1]


def test_resolve_local_files_not_found(temp_dir: str) -> None:
    with pytest.raises(DatasetNotFoundError):
        resolve_local_files(os.path.join(temp_dir, "nonexistent"))

    txt = os.path.join(temp_dir, "test.txt")
    with open(txt, "w") as f:
        f.write("hello")

    with pytest.raises(DatasetNotFoundError):
        resolve_local_files(txt)


def test_resolve_parquet_dataset_with_data_files(temp_dir: str) -> None:
    f1 = os.path.join(temp_dir, "data_a.parquet")
    f2 = os.path.join(temp_dir, "data_b.parquet")
    create_sample_parquet_file(f1, num_rows=10)
    create_sample_parquet_file(f2, num_rows=10)

    splits = resolve_parquet_dataset(
        path="dummy",
        data_files={"train": f1, "test": f2},
    )
    assert splits["train"] == [f1]
    assert splits["test"] == [f2]
