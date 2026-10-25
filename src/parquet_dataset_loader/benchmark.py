"""Benchmark CLI tool for parquet_dataset_loader.

Allows evaluating the performance of streaming dataset loading.
"""

import argparse
import sys
import time
import statistics

import parquet_dataset_loader as pdl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="pdl-benchmark",
        description="Benchmark streaming dataset load times and random access scientifically.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="KhangTruong/COCO-inpainted",
        help="HuggingFace dataset path or local directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to benchmark.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="*",
        default=["mask"],
        help="List of columns to load. E.g., --columns mask image",
    )
    parser.add_argument(
        "--max-cached-row-groups",
        type=int,
        default=2,
        help="Number of row groups to cache in memory.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=50000,
        help="The index to jump to for random access benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of measurement runs for each operation.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup runs before measurement begins.",
    )
    parser.add_argument(
        "--dataloader",
        action="store_true",
        default=False,
        help="Benchmark DataLoader throughput and bottleneck diagnostics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DataLoader benchmark.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="Number of DataLoader workers (-1 or None infers CPU count).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode to profile and view DataLoader bottlenecks.",
    )
    return parser.parse_args(argv)


def measure_operation(name, func, runs=1, warmup=0, scale_to_ms=False):
    """Measures execution time of a callable multiple times and reports stats."""

    # Warmup phase
    for _ in range(warmup):
        try:
            func()
        except Exception as e:
            print(f"Failed during warmup of '{name}': {e}")
            return

    times = []
    # Measurement phase
    for _ in range(runs):
        try:
            t0 = time.perf_counter()
            func()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        except Exception as e:
            print(f"Failed to measure '{name}': {e}")
            return

    if not times:
        return

    # Scaling to ms if requested
    factor = 1000.0 if scale_to_ms else 1.0
    unit = "ms" if scale_to_ms else "s"

    times = [t * factor for t in times]

    if len(times) == 1:
        print(f"{name}: {times[0]:.3f} {unit}")
    else:
        mean_t = statistics.mean(times)
        min_t = min(times)
        max_t = max(times)
        if len(times) > 1:
            stdev = statistics.stdev(times)
            print(f"{name}: Mean: {mean_t:.3f} {unit} | Min: {min_t:.3f} {unit} | Max: {max_t:.3f} {unit} | Stdev: {stdev:.3f} {unit}")
        else:
            print(f"{name}: {times[0]:.3f} {unit}")


def run_benchmark(args):
    print("=" * 70)
    print(f"Benchmarking Dataset: {args.path} (split: {args.split})")
    print(f"Runs: {args.runs} | Warmup: {args.warmup}")
    print("=" * 70)

    # Note: We only run initialization once, since repeated load_dataset
    # creates multiple instances and we want to benchmark the operations on *one* instance.
    t0 = time.perf_counter()
    ds = pdl.load_dataset(
        path=args.path,
        split=args.split,
        streaming=True,
        columns=args.columns if args.columns else None,
        max_cached_row_groups=args.max_cached_row_groups,
    )
    t1 = time.perf_counter()
    print(f"Dataset initialized in: {t1 - t0:.3f} seconds (Note: measured once)")

    # Avoid evaluating length if len(ds) raises an exception
    try:
        ds_len = len(ds)
        print(f"Total dataset length: {ds_len:,} rows")
    except Exception as e:
        print(f"Total dataset length: Unknown ({e})")
        ds_len = -1

    print(f"Available columns: {ds.column_names}")
    print()

    if ds_len == 0:
        print("Dataset is empty. Skipping further benchmarks.")
        return 0

    # 1. First index [0]
    def get_first():
        return ds[0]
    measure_operation("Fetch index 0", get_first, runs=args.runs, warmup=args.warmup, scale_to_ms=False)

    # 2. Direct jump
    target_idx = args.target_index
    if ds_len > 0 and target_idx >= ds_len:
        target_idx = max(0, ds_len // 2)

    def jump_mid():
        return ds[target_idx]
    measure_operation(f"Seek to index {target_idx}", jump_mid, runs=args.runs, warmup=args.warmup, scale_to_ms=False)

    # 3. Consecutive read
    next_idx = target_idx + 1
    if ds_len < 0 or next_idx < ds_len:
        def read_next():
            return ds[next_idx]
        measure_operation(f"Read consecutive sample {next_idx} (cache hit)", read_next, runs=args.runs, warmup=args.warmup, scale_to_ms=True)

    # 4. Slicing
    slice_end = next_idx + 5
    if ds_len < 0 or slice_end <= ds_len:
        def slice_data():
            return ds[target_idx:slice_end]
        measure_operation(f"Slice rows [{target_idx}:{slice_end}]", slice_data, runs=args.runs, warmup=args.warmup, scale_to_ms=True)

    # 5. Streaming
    def stream_data():
        count = 0
        for _ in ds.iter_from(target_idx):
            count += 1
            if count >= 3:
                break
    measure_operation("Stream 3 rows", stream_data, runs=args.runs, warmup=args.warmup, scale_to_ms=True)

    # 6. Optional DataLoader Benchmark
    if getattr(args, "dataloader", False):
        print("\n" + "=" * 70)
        print("Running DataLoader Speed Benchmark & Bottleneck Check...")
        print("=" * 70)
        loader = pdl.DataLoader(
            ds,
            batch_size=getattr(args, "batch_size", 32),
            num_workers=getattr(args, "num_workers", -1),
            debug=getattr(args, "debug", False),
        )
        loader.benchmark(
            num_batches=max(10, args.runs * 10),
            warmup=max(1, args.warmup),
            show_summary=True,
        )
        if getattr(args, "debug", False):
            loader.view_bottleneck()

    print("=" * 70)
    print("Benchmark completed.")
    print("=" * 70)
    return 0


def main(argv=None):
    args = parse_args(argv)
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
