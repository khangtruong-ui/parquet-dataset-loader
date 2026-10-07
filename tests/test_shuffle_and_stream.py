"""Tests for shuffle with seed and streaming from index across streaming and non-streaming modes."""

import os
import pytest

from parquet_dataset_loader.api import load_dataset, load_from_disk
from parquet_dataset_loader.dataset import IndexedParquetDataset, ParquetDatasetDict
from parquet_dataset_loader.index import build_metadata_index
from parquet_dataset_loader.reader import RowGroupReader
from tests.conftest import create_sample_parquet_file


@pytest.fixture
def sample_dataset(temp_dir: str) -> IndexedParquetDataset:
    f1 = os.path.join(temp_dir, "train-00000.parquet")
    f2 = os.path.join(temp_dir, "train-00001.parquet")
    create_sample_parquet_file(f1, num_rows=50, row_group_size=25, start_id=0)
    create_sample_parquet_file(f2, num_rows=50, row_group_size=25, start_id=50)

    index = build_metadata_index([f1, f2], split_name="train", use_cache=False)
    reader = RowGroupReader()
    return IndexedParquetDataset(index=index, reader=reader, split="train")


def test_shuffle_with_seed_reproducibility(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset
    assert len(ds) == 100

    # Shuffling with the same seed produces identical order
    shuffled_1 = ds.shuffle(seed=42)
    shuffled_2 = ds.shuffle(seed=42)
    assert len(shuffled_1) == 100
    assert len(shuffled_2) == 100

    order_1 = [shuffled_1[i]["id"] for i in range(100)]
    order_2 = [shuffled_2[i]["id"] for i in range(100)]
    assert order_1 == order_2
    # Ensure it is actually shuffled and not identical to original
    assert order_1 != list(range(100))
    # All 100 original elements should still be present
    assert sorted(order_1) == list(range(100))

    # Shuffling with a different seed produces a different order
    shuffled_diff = ds.shuffle(seed=999)
    order_diff = [shuffled_diff[i]["id"] for i in range(100)]
    assert order_diff != order_1
    assert sorted(order_diff) == list(range(100))


def test_shuffle_random_seed(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset
    # Shuffling without seed (seed=None) works
    shuffled = ds.shuffle(seed=None)
    assert len(shuffled) == 100
    order = [shuffled[i]["id"] for i in range(100)]
    assert sorted(order) == list(range(100))


def test_shuffled_dataset_indexing_and_slicing(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset
    shuffled = ds.shuffle(seed=123)

    # Single indexing
    r0 = shuffled[0]
    assert r0["id"] == shuffled.indices[0]

    # Negative indexing
    r_last = shuffled[-1]
    assert r_last["id"] == shuffled.indices[-1]

    # Slicing matching HF batch format
    batch = shuffled[10:15]
    assert isinstance(batch, dict)
    expected_ids = [shuffled.indices[i] for i in range(10, 15)]
    assert batch["id"] == expected_ids

    # List indexing
    batch_list = shuffled[[0, 5, 20]]
    assert batch_list["id"] == [shuffled.indices[0], shuffled.indices[5], shuffled.indices[20]]


def test_stream_from_index_unshuffled(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset

    # Stream from index 60 via iter_from
    stream_iter = list(ds.iter_from(start_index=60))
    assert len(stream_iter) == 40
    assert [r["id"] for r in stream_iter] == list(range(60, 100))

    # Stream from index 60 via stream alias
    stream_alias = list(ds.stream(start_index=60))
    assert [r["id"] for r in stream_alias] == list(range(60, 100))

    # Stream from index 60 via from_index keyword
    stream_from_kw = list(ds.stream(from_index=60))
    assert [r["id"] for r in stream_from_kw] == list(range(60, 100))

    # Stream via stream_from alias
    stream_from_alias = list(ds.stream_from(60))
    assert [r["id"] for r in stream_from_alias] == list(range(60, 100))


def test_stream_from_index_shuffled(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset
    shuffled = ds.shuffle(seed=42)

    full_order = [shuffled[i]["id"] for i in range(100)]

    # Stream from index 50
    streamed_50 = list(shuffled.iter_from(start_index=50))
    assert len(streamed_50) == 50
    assert [r["id"] for r in streamed_50] == full_order[50:]

    # Stream via stream alias
    streamed_alias = list(shuffled.stream(50))
    assert [r["id"] for r in streamed_alias] == full_order[50:]

    # Full iteration via iter()
    iter_all = list(shuffled)
    assert len(iter_all) == 100
    assert [r["id"] for r in iter_all] == full_order


def test_shuffled_take_skip_slice(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset
    shuffled = ds.shuffle(seed=42)
    full_order = [shuffled[i]["id"] for i in range(100)]

    # take
    taken = shuffled.take(25)
    assert len(taken) == 25
    assert [taken[i]["id"] for i in range(25)] == full_order[:25]

    # skip
    skipped = shuffled.skip(70)
    assert len(skipped) == 30
    assert [skipped[i]["id"] for i in range(30)] == full_order[70:]
    # Iterating on skip yields the same as iter_from(70)
    assert [r["id"] for r in skipped] == [r["id"] for r in shuffled.iter_from(70)]

    # slice
    sliced = shuffled.slice(start=20, length=15)
    assert len(sliced) == 15
    assert [sliced[i]["id"] for i in range(15)] == full_order[20:35]


def test_shuffled_to_arrow_and_pandas(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset
    shuffled = ds.shuffle(seed=42).take(30)
    expected_ids = [shuffled[i]["id"] for i in range(30)]

    # to_arrow preserves shuffle order
    table = shuffled.to_arrow()
    assert table.num_rows == 30
    assert table.column("id").to_pylist() == expected_ids

    # to_pandas preserves shuffle order
    df = shuffled.to_pandas()
    assert list(df["id"]) == expected_ids

    # to_hf_dataset preserves shuffle order
    import datasets
    hf_ds = shuffled.to_hf_dataset()
    assert isinstance(hf_ds, datasets.Dataset)
    assert list(hf_ds["id"]) == expected_ids


def test_shuffled_save_and_load_from_disk(temp_dir: str, sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset
    shuffled = ds.shuffle(seed=42).take(40)
    expected_ids = [shuffled[i]["id"] for i in range(40)]

    save_path = os.path.join(temp_dir, "saved_shuffled")
    shuffled.save_to_disk(save_path)

    reloaded = load_from_disk(save_path)
    assert len(reloaded) == 40
    assert [reloaded[i]["id"] for i in range(40)] == expected_ids


def test_streaming_buffer_shuffle(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset

    # Shuffle with buffer_size
    buf_shuffled = ds.shuffle(seed=42, buffer_size=20)
    streamed = list(buf_shuffled)
    assert len(streamed) == 100
    streamed_ids = [r["id"] for r in streamed]
    assert sorted(streamed_ids) == list(range(100))
    # It should not be strictly sequential
    assert streamed_ids != list(range(100))

    # Buffer shuffle with start_index
    streamed_from_50 = list(buf_shuffled.iter_from(start_index=50))
    assert len(streamed_from_50) == 50
    streamed_from_50_ids = [r["id"] for r in streamed_from_50]
    assert sorted(streamed_from_50_ids) == list(range(50, 100))


def test_load_dataset_streaming_shuffle_and_seed(sample_dataset_dir: str) -> None:
    # streaming=True with shuffle=True and seed=42
    ds1 = load_dataset(sample_dataset_dir, split="train", streaming=True, shuffle=True, seed=42)
    ds2 = load_dataset(sample_dataset_dir, split="train", streaming=True, shuffle=True, seed=42)
    assert len(ds1) == 100
    assert len(ds2) == 100

    order1 = [ds1[i]["id"] for i in range(100)]
    order2 = [ds2[i]["id"] for i in range(100)]
    assert order1 == order2
    assert order1 != list(range(100))
    assert sorted(order1) == list(range(100))


def test_load_dataset_non_streaming_shuffle_and_seed(sample_dataset_dir: str) -> None:
    # streaming=False with shuffle=True and seed=42
    ds1 = load_dataset(sample_dataset_dir, split="train", streaming=False, shuffle=True, seed=42)
    ds2 = load_dataset(sample_dataset_dir, split="train", streaming=False, shuffle=True, seed=42)
    assert len(ds1) == 100
    assert len(ds2) == 100

    order1 = [ds1[i]["id"] for i in range(100)]
    order2 = [ds2[i]["id"] for i in range(100)]
    assert order1 == order2
    assert order1 != list(range(100))
    assert sorted(order1) == list(range(100))


def test_load_dataset_seed_without_explicit_shuffle(sample_dataset_dir: str) -> None:
    # Passing seed=42 automatically enables shuffle
    ds = load_dataset(sample_dataset_dir, split="train", streaming=True, seed=42)
    order = [ds[i]["id"] for i in range(100)]
    assert order != list(range(100))

    # Passing shuffle=False with seed=42 disables shuffle
    ds_unshuffled = load_dataset(
        sample_dataset_dir, split="train", streaming=True, shuffle=False, seed=42
    )
    assert [ds_unshuffled[i]["id"] for i in range(100)] == list(range(100))


def test_load_dataset_stream_from_start_index_both_modes(sample_dataset_dir: str) -> None:
    # streaming=True with start_index=75
    ds_stream = load_dataset(
        sample_dataset_dir, split="train", streaming=True, start_index=75
    )
    assert len(ds_stream) == 25
    assert ds_stream[0]["id"] == 75
    assert [r["id"] for r in ds_stream] == list(range(75, 100))

    # streaming=False with start_index=75
    ds_non_stream = load_dataset(
        sample_dataset_dir, split="train", streaming=False, start_index=75
    )
    assert len(ds_non_stream) == 25
    assert ds_non_stream[0]["id"] == 75
    assert [r["id"] for r in ds_non_stream] == list(range(75, 100))

    # from_index alias
    ds_from_kw = load_dataset(
        sample_dataset_dir, split="train", streaming=True, from_index=80
    )
    assert len(ds_from_kw) == 20
    assert ds_from_kw[0]["id"] == 80


def test_load_dataset_shuffle_with_start_index_both_modes(sample_dataset_dir: str) -> None:
    # First get the reference shuffled order for seed 42
    ds_ref = load_dataset(
        sample_dataset_dir, split="train", streaming=True, shuffle=True, seed=42
    )
    ref_order = [ds_ref[i]["id"] for i in range(100)]

    # streaming=True with shuffle=True, seed=42, start_index=60
    ds_stream = load_dataset(
        sample_dataset_dir, split="train", streaming=True, shuffle=True, seed=42, start_index=60
    )
    assert len(ds_stream) == 40
    assert [r["id"] for r in ds_stream] == ref_order[60:]
    assert ds_stream[0]["id"] == ref_order[60]

    # streaming=False with shuffle=True, seed=42, start_index=60
    ds_non_stream = load_dataset(
        sample_dataset_dir, split="train", streaming=False, shuffle=True, seed=42, start_index=60
    )
    assert len(ds_non_stream) == 40
    assert [r["id"] for r in ds_non_stream] == ref_order[60:]
    assert ds_non_stream[0]["id"] == ref_order[60]


def test_parquet_dataset_dict_shuffle(sample_dataset_dir: str) -> None:
    # split=None with streaming=True, shuffle=True, seed=42
    ds_dict_stream = load_dataset(
        sample_dataset_dir, split=None, streaming=True, shuffle=True, seed=42
    )
    assert isinstance(ds_dict_stream, ParquetDatasetDict)
    assert len(ds_dict_stream["train"]) == 100
    assert len(ds_dict_stream["validation"]) == 30
    assert [ds_dict_stream["train"][i]["id"] for i in range(100)] != list(range(100))

    # split=None with streaming=False, shuffle=True, seed=42
    ds_dict_non_stream = load_dataset(
        sample_dataset_dir, split=None, streaming=False, shuffle=True, seed=42
    )
    assert isinstance(ds_dict_non_stream, ParquetDatasetDict)
    assert len(ds_dict_non_stream["train"]) == 100
    assert len(ds_dict_non_stream["validation"]) == 30

    # Calling .shuffle(seed=...) directly on ParquetDatasetDict
    ds_dict_unshuffled = load_dataset(sample_dataset_dir, split=None, streaming=True)
    shuffled_dict = ds_dict_unshuffled.shuffle(seed=42)
    assert isinstance(shuffled_dict, ParquetDatasetDict)
    assert [shuffled_dict["train"][i]["id"] for i in range(100)] == [
        ds_dict_stream["train"][i]["id"] for i in range(100)
    ]
