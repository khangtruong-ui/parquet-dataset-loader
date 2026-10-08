"""Unit tests for IndexedParquetDataset and ParquetDatasetDict."""

import os
import pytest

from parquet_dataset_loader.dataset import IndexedParquetDataset, ParquetDatasetDict
from parquet_dataset_loader.exceptions import IndexOutOfBoundsError
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


def test_dataset_len_and_indexing(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset
    assert len(ds) == 100
    assert ds.num_rows == 100

    # First row
    r0 = ds[0]
    assert isinstance(r0, dict)
    assert r0["id"] == 0
    assert r0["text"] == "sample_text_0"

    # Row 24 (end of RG 0)
    r24 = ds[24]
    assert r24["id"] == 24

    # Row 25 (start of RG 1)
    r25 = ds[25]
    assert r25["id"] == 25

    # Row 50 (start of file 1, RG 0)
    r50 = ds[50]
    assert r50["id"] == 50

    # Last row
    r99 = ds[99]
    assert r99["id"] == 99

    # Negative index
    r_last = ds[-1]
    assert r_last["id"] == 99

    # Out of bounds
    with pytest.raises(IndexOutOfBoundsError):
        _ = ds[100]

    with pytest.raises(IndexOutOfBoundsError):
        _ = ds[-101]


def test_dataset_slicing_hf_format(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset

    # Slicing returns dict of lists matching Hugging Face Dataset format
    batch = ds[10:15]
    assert isinstance(batch, dict)
    assert set(batch.keys()) == set(ds.column_names)
    assert batch["id"] == [10, 11, 12, 13, 14]
    assert batch["text"] == [f"sample_text_{i}" for i in range(10, 15)]

    # Slicing across row group boundary (23 to 27)
    batch_cross = ds[23:27]
    assert batch_cross["id"] == [23, 24, 25, 26]

    # Index list
    batch_list = ds[[0, 25, 50]]
    assert batch_list["id"] == [0, 25, 50]


def test_dataset_iter_from(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset

    # Stream from index 80
    stream = list(ds.iter_from(80))
    assert len(stream) == 20
    assert [r["id"] for r in stream] == list(range(80, 100))

    # Full iteration
    count = 0
    for r in ds:
        count += 1
    assert count == 100


def test_dataset_column_projection(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset
    proj = ds.select_columns(["id", "value"])
    assert proj.column_names == ["id", "value"]

    r0 = proj[0]
    assert set(r0.keys()) == {"id", "value"}

    batch = proj[0:3]
    assert set(batch.keys()) == {"id", "value"}


def test_dataset_take_and_skip(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset

    # Take first 30 rows
    taken = ds.take(30)
    assert len(taken) == 30
    assert taken[0]["id"] == 0
    assert taken[-1]["id"] == 29

    # Skip first 80 rows
    skipped = ds.skip(80)
    assert len(skipped) == 20
    assert skipped[0]["id"] == 80
    assert skipped[-1]["id"] == 99

    # Slice view
    sliced = ds.slice(start=20, length=15)
    assert len(sliced) == 15
    assert sliced[0]["id"] == 20
    assert sliced[-1]["id"] == 34


def test_conversions_arrow_pandas_hf(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset.take(10)

    # Arrow Table
    tbl = ds.to_arrow()
    assert tbl.num_rows == 10
    assert tbl.column_names == ["id", "text", "value", "payload"]

    # Pandas DataFrame
    df = ds.to_pandas()
    assert len(df) == 10
    assert list(df["id"]) == list(range(10))

    # Hugging Face Dataset
    import datasets
    hf_ds = ds.to_hf_dataset()
    assert isinstance(hf_ds, datasets.Dataset)
    assert len(hf_ds) == 10
    assert hf_ds[0]["id"] == 0


def test_save_to_disk(temp_dir: str, sample_dataset: IndexedParquetDataset) -> None:
    save_path = os.path.join(temp_dir, "saved_ds")
    sample_dataset.save_to_disk(save_path)

    assert os.path.exists(os.path.join(save_path, "train.feather"))
    assert os.path.exists(os.path.join(save_path, "train_info.json"))


def test_parquet_dataset_dict(sample_dataset: IndexedParquetDataset) -> None:
    ds_dict = ParquetDatasetDict({
        "train": sample_dataset,
        "test": sample_dataset.take(10),
    })

    assert len(ds_dict["train"]) == 100
    assert len(ds_dict["test"]) == 10

    proj_dict = ds_dict.select_columns(["id"])
    assert proj_dict["train"].column_names == ["id"]
    assert proj_dict["test"].column_names == ["id"]

    import datasets
    hf_dict = ds_dict.to_hf_dataset()
    assert isinstance(hf_dict, datasets.DatasetDict)
    assert len(hf_dict["train"]) == 100
    assert len(hf_dict["test"]) == 10


def test_dataset_pickle_and_multiprocessing(sample_dataset: IndexedParquetDataset) -> None:
    import pickle
    dumped = pickle.dumps(sample_dataset)
    restored = pickle.loads(dumped)
    assert len(restored) == len(sample_dataset)
    assert restored[0]["id"] == sample_dataset[0]["id"]
    assert restored[99]["id"] == sample_dataset[99]["id"]


def test_dataset_select_and_shard(sample_dataset: IndexedParquetDataset) -> None:
    ds = sample_dataset
    assert ds.n_shards == 2

    # select arbitrary indices
    sub = ds.select([0, 10, 50, 99])
    assert len(sub) == 4
    assert sub[0]["id"] == 0
    assert sub[1]["id"] == 10
    assert sub[2]["id"] == 50
    assert sub[3]["id"] == 99

    # contiguous shard
    shard0 = ds.shard(num_shards=2, index=0, contiguous=True)
    shard1 = ds.shard(num_shards=2, index=1, contiguous=True)
    assert len(shard0) == 50
    assert len(shard1) == 50
    assert shard0[0]["id"] == 0
    assert shard0[-1]["id"] == 49
    assert shard1[0]["id"] == 50
    assert shard1[-1]["id"] == 99

    # interleaved shard
    i_shard0 = ds.shard(num_shards=4, index=0, contiguous=False)
    i_shard1 = ds.shard(num_shards=4, index=1, contiguous=False)
    assert len(i_shard0) == 25
    assert len(i_shard1) == 25
    assert i_shard0[0]["id"] == 0
    assert i_shard0[1]["id"] == 4
    assert i_shard1[0]["id"] == 1
    assert i_shard1[1]["id"] == 5


def test_dataset_pytorch_multi_worker_dataloader(sample_dataset: IndexedParquetDataset) -> None:
    """Verify IndexedParquetDataset runs seamlessly with multi-worker PyTorch DataLoader across fork."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        pytest.skip("PyTorch not installed")

    loader = DataLoader(sample_dataset, batch_size=10, num_workers=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 10
    first_batch = batches[0]
    assert len(first_batch["id"]) == 10
    assert first_batch["id"][0].item() == 0



