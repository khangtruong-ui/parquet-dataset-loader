"""Benchmark CLI tool for parquet_dataset_loader.

Allows evaluating the performance of streaming dataset loading.
"""

import argparse
import sys
import time

import parquet_dataset_loader as pdl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="pdl-benchmark",
        description="Benchmark streaming dataset load times and random access.",
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
    return parser.parse_args(argv)


def run_benchmark(args):
    print("=" * 70)
    print(f"Benchmarking Dataset: {args.path} (split: {args.split})")
    print("=" * 70)

    # 1. Initialization
    t0 = time.time()
    ds = pdl.load_dataset(
        path=args.path,
        split=args.split,
        streaming=True,
        columns=args.columns if args.columns else None,
        max_cached_row_groups=args.max_cached_row_groups,
    )
    t1 = time.time()

    print(f"Dataset initialized in: {t1 - t0:.2f} seconds")

    # Avoid evaluating length if len(ds) raises an exception for whatever reason,
    # though pdl should support len instantly.
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

    # 2. First index
    t_start = time.time()
    try:
        sample_0 = ds[0]
        t_end = time.time()
        print(f"Fetched index 0 in {t_end - t_start:.2f} seconds")
    except Exception as e:
        print(f"Failed to fetch index 0: {e}")

    print()

    # 3. Direct jump
    target_idx = args.target_index
    if ds_len > 0 and target_idx >= ds_len:
        target_idx = max(0, ds_len // 2)

    print(f"Seeking to index {target_idx}...")
    t_start = time.time()
    try:
        sample_mid = ds[target_idx]
        t_end = time.time()
        print(f"Fetched index {target_idx} in {t_end - t_start:.2f} seconds!")
    except Exception as e:
        print(f"Failed to fetch index {target_idx}: {e}")

    print()

    # 4. Consecutive read
    next_idx = target_idx + 1
    if ds_len < 0 or next_idx < ds_len:
        print(f"Reading consecutive sample {next_idx} from in-memory row group cache...")
        t_start = time.time()
        try:
            sample_next = ds[next_idx]
            t_end = time.time()
            print(f"Fetched index {next_idx} in {(t_end - t_start)*1000:.3f} ms (cache hit expected)")
        except Exception as e:
            print(f"Failed to fetch index {next_idx}: {e}")
        print()

    # 5. Slicing
    slice_end = next_idx + 5
    if ds_len < 0 or slice_end <= ds_len:
        print(f"Slicing rows [{target_idx}:{slice_end}]...")
        t_start = time.time()
        try:
            batch = ds[target_idx:slice_end]
            t_end = time.time()

            # just get length of first column value
            if isinstance(batch, dict) and len(batch) > 0:
                first_col = list(batch.keys())[0]
                slice_len = len(batch[first_col])
            else:
                slice_len = len(batch)

            print(f"Sliced batch of {slice_len} items in {(t_end - t_start)*1000:.3f} ms")
        except Exception as e:
            print(f"Failed to slice: {e}")
        print()

    # 6. Streaming
    print("Streaming 3 rows...")
    try:
        t_start = time.time()
        count = 0
        for idx, item in enumerate(ds.iter_from(target_idx)):
            count += 1
            if count >= 3:
                break
        t_end = time.time()
        print(f"Streamed {count} rows in {(t_end - t_start)*1000:.3f} ms")
    except Exception as e:
        print(f"Failed to stream: {e}")

    print("=" * 70)
    print("Benchmark completed.")
    print("=" * 70)
    return 0


def main(argv=None):
    args = parse_args(argv)
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
