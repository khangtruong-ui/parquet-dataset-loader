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


def test_resolve_hf_token_explicit_string() -> None:
    from parquet_dataset_loader.hf_resolver import resolve_hf_token

    assert resolve_hf_token("hf_explicit_123") == "hf_explicit_123"
    assert resolve_hf_token("  hf_with_spaces  ") == "hf_with_spaces"


def test_resolve_hf_token_disabled_with_false(monkeypatch: pytest.MonkeyPatch) -> None:
    from parquet_dataset_loader.hf_resolver import resolve_hf_token

    monkeypatch.setenv("HF_TOKEN", "hf_should_be_ignored")
    assert resolve_hf_token(False) is None


def test_resolve_hf_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from parquet_dataset_loader.hf_resolver import resolve_hf_token

    # 1. HF_TOKEN env var
    monkeypatch.setenv("HF_TOKEN", "hf_from_env_token")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert resolve_hf_token() == "hf_from_env_token"
    assert resolve_hf_token(True) == "hf_from_env_token"

    # 2. HUGGING_FACE_HUB_TOKEN fallback
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_from_legacy_env")
    assert resolve_hf_token() == "hf_from_legacy_env"

    # 3. HF_TOKEN takes precedence over HUGGING_FACE_HUB_TOKEN
    monkeypatch.setenv("HF_TOKEN", "hf_primary")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_secondary")
    assert resolve_hf_token() == "hf_primary"

    # 4. Explicit string takes precedence over env var
    assert resolve_hf_token("hf_override") == "hf_override"

    # 5. Empty / whitespace string in env var is treated as None
    monkeypatch.setenv("HF_TOKEN", "   ")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    # Without cached login token, should be None
    monkeypatch.setattr("parquet_dataset_loader.hf_resolver.get_token", lambda: None)
    assert resolve_hf_token() is None


def test_row_group_reader_inherits_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from parquet_dataset_loader.reader import RowGroupReader

    monkeypatch.setenv("HF_TOKEN", "hf_auth_secret_xyz")
    reader = RowGroupReader()
    try:
        assert reader.token == "hf_auth_secret_xyz"
        # Verify filesystem storage options contain Authorization header
        assert reader._http_fs.kwargs.get("headers", {}).get("Authorization") == "Bearer hf_auth_secret_xyz"
    finally:
        reader.close()


def test_row_group_reader_disabled_auth_with_false(monkeypatch: pytest.MonkeyPatch) -> None:
    from parquet_dataset_loader.reader import RowGroupReader

    monkeypatch.setenv("HF_TOKEN", "hf_auth_secret_xyz")
    reader = RowGroupReader(token=False)
    try:
        assert reader.token is None
        assert "Authorization" not in reader._http_fs.kwargs.get("headers", {})
    finally:
        reader.close()

