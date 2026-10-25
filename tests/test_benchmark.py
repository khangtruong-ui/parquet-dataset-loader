import pytest
from unittest.mock import patch, MagicMock

from parquet_dataset_loader.benchmark import parse_args, run_benchmark

def test_parse_args_defaults():
    args = parse_args([])
    assert args.path == "KhangTruong/COCO-inpainted"
    assert args.split == "train"
    assert args.columns == ["mask"]
    assert args.max_cached_row_groups == 2
    assert args.target_index == 50000
    assert args.runs == 1
    assert args.warmup == 0

def test_parse_args_custom():
    args = parse_args(["--path", "my-dataset", "--split", "test", "--columns", "a", "b", "--target-index", "100", "--runs", "5", "--warmup", "2"])
    assert args.path == "my-dataset"
    assert args.split == "test"
    assert args.columns == ["a", "b"]
    assert args.target_index == 100
    assert args.runs == 5
    assert args.warmup == 2

@patch("parquet_dataset_loader.benchmark.pdl.load_dataset")
def test_run_benchmark(mock_load_dataset, capsys):
    # Mock dataset
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 1000
    mock_ds.column_names = ["mask"]
    mock_ds.__getitem__.return_value = {"mask": b"data"}
    mock_ds.iter_from.return_value = [{"mask": b"d1"}, {"mask": b"d2"}, {"mask": b"d3"}]
    mock_load_dataset.return_value = mock_ds

    args = parse_args(["--path", "dummy", "--target-index", "500"])
    exit_code = run_benchmark(args)

    assert exit_code == 0
    mock_load_dataset.assert_called_once_with(
        path="dummy",
        split="train",
        streaming=True,
        columns=["mask"],
        max_cached_row_groups=2,
    )

    # Assert getitem and slicing were called correctly
    mock_ds.__getitem__.assert_any_call(0)
    mock_ds.__getitem__.assert_any_call(500)
    mock_ds.__getitem__.assert_any_call(501)
    mock_ds.__getitem__.assert_any_call(slice(500, 506))

    # Check output
    captured = capsys.readouterr()
    assert "Benchmarking Dataset: dummy" in captured.out
    assert "Total dataset length: 1,000 rows" in captured.out
    assert "Fetch index 0" in captured.out
    assert "Seek to index 500" in captured.out
    assert "Stream 3 rows" in captured.out


@patch("parquet_dataset_loader.benchmark.pdl.DataLoader")
@patch("parquet_dataset_loader.benchmark.pdl.load_dataset")
def test_run_benchmark_with_dataloader(mock_load_dataset, mock_dataloader_cls, capsys):
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 500
    mock_ds.column_names = ["mask"]
    mock_ds.__getitem__.return_value = {"mask": b"data"}
    mock_ds.iter_from.return_value = [{"mask": b"d1"}, {"mask": b"d2"}, {"mask": b"d3"}]
    mock_load_dataset.return_value = mock_ds

    mock_loader = MagicMock()
    mock_loader.benchmark.return_value = {"total_batches": 10, "batches_per_sec": 100.0}
    mock_dataloader_cls.return_value = mock_loader

    args = parse_args(["--path", "dummy", "--dataloader", "--batch-size", "16", "--num-workers", "4", "--debug"])
    assert args.dataloader is True
    assert args.batch_size == 16
    assert args.num_workers == 4
    assert args.debug is True

    exit_code = run_benchmark(args)
    assert exit_code == 0
    mock_dataloader_cls.assert_called_once_with(mock_ds, batch_size=16, num_workers=4, debug=True)
    mock_loader.benchmark.assert_called_once()
    mock_loader.view_bottleneck.assert_called_once()
