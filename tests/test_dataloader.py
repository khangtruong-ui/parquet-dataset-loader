"""Unit and integration tests for high-performance, bug-free DataLoader API.

Tests cover:
1. Default num_workers inference (CPU count and -1).
2. Basic batching, shapes, and drop_last behavior.
3. Shuffling reproducibility with seed and epoch variation.
4. BlockShuffledSampler integration for cache locality.
5. Robust default_collate handling of strings, bytes, None, and variable lengths.
6. Thread vs process worker execution and clean early exit without hanging.
7. Worker exception propagation and clean failure handling.
8. Speed benchmarking (loader.benchmark) and metrics calculation.
9. Debug-mode bottleneck profiling, diagnosis, and recommendations (debug=True only).
10. Early health warnings after several batches (latency spikes, starvation, empty batches).
11. Dataset and DatasetDict .to_dataloader() integration.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, List

import pytest

import parquet_dataset_loader as pdl
from parquet_dataset_loader.dataloader import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    benchmark_dataloader,
    default_collate,
)
from parquet_dataset_loader.dataset import BlockShuffledSampler, IndexedParquetDataset, ParquetDatasetDict
from parquet_dataset_loader.index import build_metadata_index
from parquet_dataset_loader.reader import RowGroupReader
from tests.conftest import create_sample_parquet_file


@pytest.fixture
def sample_indexed_dataset(temp_dir: str) -> IndexedParquetDataset:
    f1 = os.path.join(temp_dir, "train-00000.parquet")
    f2 = os.path.join(temp_dir, "train-00001.parquet")
    create_sample_parquet_file(f1, num_rows=50, row_group_size=25, start_id=0)
    create_sample_parquet_file(f2, num_rows=50, row_group_size=25, start_id=50)

    index = build_metadata_index([f1, f2], split_name="train", use_cache=False)
    reader = RowGroupReader()
    return IndexedParquetDataset(index=index, reader=reader, split="train")


def test_dataloader_default_num_workers(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify num_workers defaults to CPU count and -1 infers CPU count."""
    cpu_count = os.cpu_count() or 1

    # Default is None -> cpu_count
    l_default = DataLoader(sample_indexed_dataset, batch_size=10)
    assert l_default.num_workers == cpu_count

    # Explicit -1 -> cpu_count
    l_inferred = DataLoader(sample_indexed_dataset, batch_size=10, num_workers=-1)
    assert l_inferred.num_workers == cpu_count

    # Explicit 0 -> 0 (in-process)
    l_zero = DataLoader(sample_indexed_dataset, batch_size=10, num_workers=0)
    assert l_zero.num_workers == 0

    # Explicit positive number
    l_custom = DataLoader(sample_indexed_dataset, batch_size=10, num_workers=3)
    assert l_custom.num_workers == 3

    # Invalid negative number raises ValueError
    with pytest.raises(ValueError, match="num_workers must be >= -1"):
        DataLoader(sample_indexed_dataset, batch_size=10, num_workers=-2)


def test_dataloader_basic_iteration(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify basic batching, length, and item retrieval."""
    ds = sample_indexed_dataset
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    # 100 rows / 16 = 6 full batches + 1 batch of 4 rows = 7 batches
    assert len(loader) == 7

    batches = list(loader)
    assert len(batches) == 7

    # Check first batch
    b0 = batches[0]
    assert "id" in b0 and "text" in b0 and "value" in b0 and "payload" in b0
    assert len(b0["id"]) == 16
    assert b0["id"][0] == 0
    assert b0["id"][-1] == 15

    # Check last batch (4 items)
    b_last = batches[-1]
    assert len(b_last["id"]) == 4
    assert b_last["id"][-1] == 99


def test_dataloader_drop_last(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify drop_last drops the final incomplete batch."""
    ds = sample_indexed_dataset
    loader = DataLoader(ds, batch_size=16, shuffle=False, drop_last=True, num_workers=2)

    # 100 // 16 = 6 batches
    assert len(loader) == 6
    batches = list(loader)
    assert len(batches) == 6
    for b in batches:
        assert len(b["id"]) == 16


def test_dataloader_single_process(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify num_workers=0 runs synchronously in the main thread with zero worker overhead."""
    ds = sample_indexed_dataset
    loader = DataLoader(ds, batch_size=20, num_workers=0, shuffle=False)

    batches = list(loader)
    assert len(batches) == 5
    for i, b in enumerate(batches):
        assert len(b["id"]) == 20
        assert b["id"][0] == i * 20


def test_dataloader_shuffling_and_seed(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify shuffling permutes items deterministically with seed and across epochs."""
    ds = sample_indexed_dataset

    # Run with seed 42
    loader1 = DataLoader(ds, batch_size=10, shuffle=True, seed=42, num_workers=0)
    batches1 = list(loader1)
    ids1 = [int(val) for b in batches1 for val in b["id"]]

    assert len(ids1) == 100
    assert set(ids1) == set(range(100))
    # Must not be sequential
    assert ids1 != list(range(100))

    # Same seed reproduces identical order
    loader2 = DataLoader(ds, batch_size=10, shuffle=True, seed=42, num_workers=0)
    batches2 = list(loader2)
    ids2 = [int(val) for b in batches2 for val in b["id"]]
    assert ids1 == ids2

    # Setting epoch changes permutation deterministically
    loader2.set_epoch(1)
    batches_ep1 = list(loader2)
    ids_ep1 = [int(val) for b in batches_ep1 for val in b["id"]]
    assert ids_ep1 != ids1
    assert set(ids_ep1) == set(range(100))


def test_dataloader_block_shuffled_sampler(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify BlockShuffledSampler integration preserves cache locality."""
    ds = sample_indexed_dataset
    sampler = BlockShuffledSampler(ds, window_blocks=1, seed=42, shuffle=True)
    loader = DataLoader(ds, batch_size=25, sampler=sampler, num_workers=2)

    batches = list(loader)
    assert len(batches) == 4
    total_samples = sum(len(b["id"]) for b in batches)
    assert total_samples == 100


def test_default_collate_handling() -> None:
    """Verify default_collate is bug-free and handles complex edge cases."""
    # 1. Strings and bytes
    batch_str = [{"text": "hello", "data": b"abc"}, {"text": "world", "data": b"xyz"}]
    collated_str = default_collate(batch_str)
    assert collated_str["text"] == ["hello", "world"]
    assert collated_str["data"] == [b"abc", b"xyz"]

    # 2. None values (which standard torch collate crashes on with TypeError)
    batch_none = [{"id": 1, "mask": None}, {"id": 2, "mask": None}]
    collated_none = default_collate(batch_none)
    assert collated_none["mask"] == [None, None]

    # 3. Variable-length sequences (which standard torch collate crashes on)
    batch_var = [{"tokens": [1, 2]}, {"tokens": [3, 4, 5]}]
    collated_var = default_collate(batch_var)
    assert collated_var["tokens"] == [[1, 2], [3, 4, 5]]

    # 4. Dict of lists (pre-sliced batch format)
    pre_sliced = {"id": [0, 1, 2], "val": [1.0, 2.0, 3.0], "tag": ["a", "b", "c"]}
    collated_slice = default_collate(pre_sliced)
    assert len(collated_slice["id"]) == 3
    assert collated_slice["tag"] == ["a", "b", "c"]


def test_dataloader_thread_vs_process_workers(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify both thread and process worker modes execute accurately."""
    ds = sample_indexed_dataset

    # Thread mode
    l_thread = DataLoader(ds, batch_size=25, num_workers=2, worker_type="thread")
    b_thread = list(l_thread)
    assert len(b_thread) == 4

    # Process mode
    l_proc = DataLoader(ds, batch_size=25, num_workers=2, worker_type="process")
    b_proc = list(l_proc)
    assert len(b_proc) == 4


def test_dataloader_early_break_no_hang(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify breaking early from the iteration loop shuts down workers without hanging."""
    ds = sample_indexed_dataset
    loader = DataLoader(ds, batch_size=5, num_workers=4, worker_type="thread")

    t0 = time.perf_counter()
    count = 0
    for _ in loader:
        count += 1
        if count >= 3:
            break
    t1 = time.perf_counter()

    assert count == 3
    # Early break must terminate cleanly and immediately (< 1.5s)
    assert (t1 - t0) < 1.5


def test_dataloader_worker_exception_propagation() -> None:
    """Verify exceptions inside workers are safely propagated to main thread without hanging."""
    class FailingDataset:
        def __len__(self) -> int:
            return 20

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            if idx == 5:
                raise ValueError("Intentional worker data corrupt error")
            return {"id": idx}

    loader = DataLoader(FailingDataset(), batch_size=2, num_workers=2, timeout=5.0)

    with pytest.raises(RuntimeError, match="Intentional worker data corrupt error"):
        list(loader)


def test_dataloader_benchmark(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify speed benchmarking measures throughput, latencies, and statistics."""
    ds = sample_indexed_dataset
    loader = DataLoader(ds, batch_size=10, num_workers=2)

    res = loader.benchmark(num_batches=5, warmup=1, show_summary=False)
    assert res["total_batches"] == 5
    assert res["total_samples"] == 50
    assert res["batches_per_sec"] > 0
    assert res["samples_per_sec"] > 0
    assert "mean_batch_time_ms" in res
    assert "min_batch_time_ms" in res
    assert "max_batch_time_ms" in res

    # Standalone helper function
    res_helper = benchmark_dataloader(loader, num_batches=3, warmup=0, show_summary=False)
    assert res_helper["total_batches"] == 3


def test_dataloader_bottleneck_profiling_debug_mode(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify bottleneck viewing is disabled in production and active in debug mode."""
    ds = sample_indexed_dataset

    # 1. Debug disabled (default): zero profiling records
    loader_prod = DataLoader(ds, batch_size=10, num_workers=2, debug=False)
    _ = list(loader_prod)
    report_prod = loader_prod.get_bottleneck_report()
    assert report_prod["debug"] is False
    view_prod = loader_prod.view_bottleneck()
    assert view_prod["debug"] is False

    # 2. Debug enabled: profiles batch components and diagnoses bottleneck
    loader_debug = DataLoader(ds, batch_size=10, num_workers=2, debug=True)
    _ = list(loader_debug)
    report_debug = loader_debug.get_bottleneck_report()
    assert report_debug["debug"] is True
    assert report_debug["total_batches_profiled"] == 10
    assert "fetch_percentage" in report_debug
    assert "collate_percentage" in report_debug
    assert "queue_wait_percentage" in report_debug
    assert "primary_bottleneck" in report_debug
    assert len(report_debug["recommendations"]) > 0

    view_debug = loader_debug.view_bottleneck()
    assert view_debug["primary_bottleneck"] == report_debug["primary_bottleneck"]


def test_dataloader_warnings_after_several_batches() -> None:
    """Verify actionable warnings are emitted after several batches for anomalies."""
    # 1. Empty batch warning
    empty_ds = [{}, {}, {}, {}, {}, {}]
    loader_empty = DataLoader(empty_ds, batch_size=1, num_workers=0)
    with warnings.catch_warnings(record=True) as recorded_empty:
        warnings.simplefilter("always")
        list(loader_empty)
        assert any("empty or contains empty columns" in str(w.message) for w in recorded_empty)

    # 2. Latency spike warning after several batches
    class StallingDataset:
        def __len__(self) -> int:
            return 10

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            if idx == 6:
                time.sleep(0.25)
            return {"id": idx}

    loader_stall = DataLoader(StallingDataset(), batch_size=1, num_workers=0)
    with warnings.catch_warnings(record=True) as recorded_stall:
        warnings.simplefilter("always")
        list(loader_stall)
        assert any("slower than recent average" in str(w.message) for w in recorded_stall)


def test_dataset_to_dataloader_and_dataset_dict(temp_dir: str) -> None:
    """Verify IndexedParquetDataset.to_dataloader and ParquetDatasetDict.to_dataloader."""
    f_train = os.path.join(temp_dir, "train.parquet")
    f_val = os.path.join(temp_dir, "validation.parquet")
    create_sample_parquet_file(f_train, num_rows=40, row_group_size=20, start_id=0)
    create_sample_parquet_file(f_val, num_rows=20, row_group_size=10, start_id=100)

    idx_train = build_metadata_index([f_train], split_name="train", use_cache=False)
    idx_val = build_metadata_index([f_val], split_name="validation", use_cache=False)
    reader = RowGroupReader()

    ds_train = IndexedParquetDataset(idx_train, reader, split="train")
    ds_val = IndexedParquetDataset(idx_val, reader, split="validation")

    # 1. Single dataset .to_dataloader()
    loader = ds_train.to_dataloader(batch_size=10, shuffle=False)
    assert isinstance(loader, DataLoader)
    assert len(loader) == 4

    # 2. ParquetDatasetDict .to_dataloader(split="train")
    pdd = ParquetDatasetDict({"train": ds_train, "validation": ds_val})
    loader_train = pdd.to_dataloader(split="train", batch_size=10)
    assert isinstance(loader_train, DataLoader)
    assert len(loader_train) == 4

    # 3. ParquetDatasetDict .to_dataloader() -> dict of DataLoaders
    all_loaders = pdd.to_dataloader(batch_size=10)
    assert isinstance(all_loaders, dict)
    assert "train" in all_loaders and "validation" in all_loaders
    assert len(all_loaders["train"]) == 4
    assert len(all_loaders["validation"]) == 2


def test_dataloader_context_manager(sample_indexed_dataset: IndexedParquetDataset) -> None:
    """Verify DataLoader context manager automatically closes resources."""
    ds = sample_indexed_dataset
    with DataLoader(ds, batch_size=10, num_workers=2) as loader:
        batches = list(loader)
        assert len(batches) == 10

    assert loader._is_closed is True
    with pytest.raises(RuntimeError, match="Cannot iterate over a closed DataLoader"):
        list(loader)
