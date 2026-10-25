"""Example: High-Performance DataLoader Usage, Benchmarking, and Bottleneck Analysis.

Demonstrates using parquet_dataset_loader's bug-free DataLoader API:
1. Multi-worker prefetching with automatic CPU inference (num_workers=-1).
2. Drop-in compatibility with PyTorch training loops.
3. Speed benchmarking via loader.benchmark().
4. Debug-mode bottleneck profiling via loader.view_bottleneck().
"""

from typing import Any, Dict, List
import parquet_dataset_loader as pdl


def main() -> None:
    print("=" * 72)
    print("parquet-dataset-loader: DataLoader Example")
    print("=" * 72)

    print("\n1. Initializing streaming dataset...")
    ds = pdl.load_dataset(
        path="KhangTruong/COCO-inpainted",
        split="train",
        streaming=True,
        columns=["mask"],
        max_cached_row_groups=2,
    )
    print(f"Dataset ready: {len(ds):,} rows")

    # 2. Create DataLoader with CPU-inferred workers and debug profiling
    print("\n2. Initializing DataLoader with num_workers=-1 (auto CPU count) and debug=True...")
    loader = ds.to_dataloader(
        batch_size=16,
        shuffle=False,
        num_workers=-1,  # Infers os.cpu_count()
        prefetch_factor=2,
        debug=True,       # Enable bottleneck diagnostics
    )
    print(f"DataLoader created: {len(loader):,} batches, {loader.num_workers} workers")

    # 3. Iterate 3 mini-batches
    print("\n3. Iterating first 3 mini-batches:")
    for batch_idx, batch in enumerate(loader):
        paths = [item["path"] for item in batch["mask"]]
        print(f"  Batch {batch_idx + 1}: {len(paths)} samples -> first item: {paths[0]}")
        if batch_idx >= 2:
            break

    # 4. View Bottleneck Analysis (debug mode)
    print("\n4. Inspecting Bottleneck Profile:")
    loader.view_bottleneck()

    # 5. Run Speed Benchmark
    print("\n5. Running DataLoader Speed Benchmark (5 batches, 1 warmup):")
    loader.benchmark(num_batches=5, warmup=1, show_summary=True)

    print("\nDataLoader example completed successfully!")


if __name__ == "__main__":
    main()
