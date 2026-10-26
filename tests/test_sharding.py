"""Unit and integration tests for multi-device/distributed sharding support in parquet-dataset-loader."""

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from parquet_dataset_loader.dataloader import DataLoader, DistributedSampler
from parquet_dataset_loader.dataset import IndexedParquetDataset, ParquetDatasetDict
from parquet_dataset_loader.api import load_dataset


@pytest.fixture
def sample_dataset(tmp_path):
    table = pa.Table.from_pydict({
        "id": list(range(20)),
        "val": [f"item_{i}" for i in range(20)],
    })
    pq_path = tmp_path / "test_data.parquet"
    pq.write_table(table, pq_path, row_group_size=5)
    return load_dataset(str(pq_path), split="train", streaming=False)


def test_distributed_sampler_disjoint_partitions():
    """Verify that DistributedSampler produces disjoint subsets covering the dataset."""
    data = list(range(100))
    sampler_0 = DistributedSampler(data, num_replicas=4, rank=0, shuffle=False)
    sampler_1 = DistributedSampler(data, num_replicas=4, rank=1, shuffle=False)
    sampler_2 = DistributedSampler(data, num_replicas=4, rank=2, shuffle=False)
    sampler_3 = DistributedSampler(data, num_replicas=4, rank=3, shuffle=False)

    indices_0 = list(sampler_0)
    indices_1 = list(sampler_1)
    indices_2 = list(sampler_2)
    indices_3 = list(sampler_3)

    assert len(indices_0) == 25
    assert len(indices_1) == 25
    assert len(indices_2) == 25
    assert len(indices_3) == 25

    # Check mutual exclusivity
    all_indices = set(indices_0) | set(indices_1) | set(indices_2) | set(indices_3)
    assert len(all_indices) == 100
    assert len(indices_0) + len(indices_1) + len(indices_2) + len(indices_3) == 100


def test_distributed_sampler_with_shuffle_and_set_epoch():
    """Verify deterministic yet epoch-varying distributed shuffling."""
    data = list(range(50))
    s0 = DistributedSampler(data, num_shards=2, shard_id=0, shuffle=True, seed=123)
    s1 = DistributedSampler(data, num_shards=2, shard_id=1, shuffle=True, seed=123)

    epoch0_0 = list(s0)
    epoch0_1 = list(s1)

    # Disjoint in epoch 0
    assert set(epoch0_0).isdisjoint(set(epoch0_1))
    assert len(set(epoch0_0) | set(epoch0_1)) == 50

    # Advance epoch
    s0.set_epoch(1)
    s1.set_epoch(1)
    epoch1_0 = list(s0)
    epoch1_1 = list(s1)

    # Disjoint in epoch 1
    assert set(epoch1_0).isdisjoint(set(epoch1_1))
    # Order changed from epoch 0
    assert epoch0_0 != epoch1_0


def test_distributed_sampler_contiguous():
    """Verify contiguous chunk sharding."""
    data = list(range(20))
    s0 = DistributedSampler(data, num_replicas=2, rank=0, shuffle=False, contiguous=True)
    s1 = DistributedSampler(data, num_replicas=2, rank=1, shuffle=False, contiguous=True)

    assert list(s0) == list(range(0, 10))
    assert list(s1) == list(range(10, 20))


def test_distributed_sampler_invalid_args():
    data = list(range(10))
    with pytest.raises(ValueError, match="positive"):
        DistributedSampler(data, num_replicas=0, rank=0)

    with pytest.raises(ValueError, match="range"):
        DistributedSampler(data, num_replicas=2, rank=2)


def test_dataloader_sharding_different_batches(sample_dataset):
    """Verify DataLoader with num_shards and shard_id yields distinct batches across ranks."""
    loader_rank0 = DataLoader(
        sample_dataset,
        batch_size=4,
        num_shards=2,
        shard_id=0,
        shuffle=False,
        num_workers=0,
    )
    loader_rank1 = DataLoader(
        sample_dataset,
        batch_size=4,
        num_shards=2,
        shard_id=1,
        shuffle=False,
        num_workers=0,
    )

    batches_0 = list(loader_rank0)
    batches_1 = list(loader_rank1)

    ids_0 = [item for b in batches_0 for item in (b["id"].tolist() if hasattr(b["id"], "tolist") else list(b["id"]))]
    ids_1 = [item for b in batches_1 for item in (b["id"].tolist() if hasattr(b["id"], "tolist") else list(b["id"]))]

    # Verify each rank received exactly half the samples
    assert len(ids_0) == 10
    assert len(ids_1) == 10

    # Verify no overlap between devices
    assert set(ids_0).isdisjoint(set(ids_1))
    assert set(ids_0) | set(ids_1) == set(range(20))


def test_dataloader_set_epoch_updates_distributed_sampler(sample_dataset):
    """Verify DataLoader.set_epoch properly propagates epoch to DistributedSampler."""
    loader0 = DataLoader(
        sample_dataset,
        batch_size=2,
        num_shards=2,
        shard_id=0,
        shuffle=True,
        seed=42,
        num_workers=0,
    )
    assert loader0.sampler is not None
    assert isinstance(loader0.sampler, DistributedSampler)

    loader0.set_epoch(5)
    assert loader0.epoch == 5
    assert loader0.sampler.epoch == 5


def test_parquet_dataset_dict_shard(tmp_path):
    """Verify ParquetDatasetDict.shard correctly shards each split."""
    table_train = pa.Table.from_pydict({"id": list(range(30))})
    table_val = pa.Table.from_pydict({"id": list(range(10))})

    p_train = tmp_path / "train.parquet"
    p_val = tmp_path / "val.parquet"
    pq.write_table(table_train, p_train)
    pq.write_table(table_val, p_val)

    ds_train = load_dataset(str(p_train), split="train")
    ds_val = load_dataset(str(p_val), split="val")
    ds_dict = ParquetDatasetDict({"train": ds_train, "val": ds_val})

    sharded = ds_dict.shard(num_shards=2, index=0)
    assert isinstance(sharded, ParquetDatasetDict)
    assert len(sharded["train"]) == 15
    assert len(sharded["val"]) == 5


def test_load_dataset_sharding_parameter(tmp_path):
    """Verify load_dataset with num_shards and shard_id automatically shards."""
    table = pa.Table.from_pydict({"val": list(range(40))})
    p = tmp_path / "test.parquet"
    pq.write_table(table, p)

    shard0 = load_dataset(str(p), split="train", num_shards=4, shard_id=0)
    shard1 = load_dataset(str(p), split="train", num_shards=4, shard_id=1)

    assert len(shard0) == 10
    assert len(shard1) == 10
    assert shard0[0]["val"] == 0
    assert shard1[0]["val"] == 1


def test_dataset_to_dataloader_with_sharding(sample_dataset):
    """Verify IndexedParquetDataset.to_dataloader forwards sharding arguments."""
    loader = sample_dataset.to_dataloader(batch_size=2, num_shards=2, shard_id=1, num_workers=0)
    assert loader.num_shards == 2
    assert loader.shard_id == 1
    assert len(loader) == 5  # 10 samples / 2 per batch
