"""Example: Custom batch iteration and DataLoader usage.

Demonstrates using IndexedParquetDataset with batch iteration and PyTorch DataLoader
patterns for distributed or single-node model training.
"""

from typing import Any, Dict, Iterator, List
import parquet_dataset_loader as pdl


def batch_generator(dataset: pdl.IndexedParquetDataset, batch_size: int = 16) -> Iterator[Dict[str, List[Any]]]:
    """Generate mini-batches directly from an IndexedParquetDataset."""
    total_len = len(dataset)
    for i in range(0, total_len, batch_size):
        end = min(i + batch_size, total_len)
        yield dataset[i:end]


def main() -> None:
    print("Initializing streaming dataset...")
    ds = pdl.load_dataset(
        path="KhangTruong/COCO-inpainted",
        split="train",
        streaming=True,
        columns=["mask"],
        max_cached_row_groups=2,
    )

    print(f"Dataset ready: {len(ds):,} rows")

    # Iterate 3 mini-batches of size 8
    print("\nIterating mini-batches:")
    for batch_idx, batch in enumerate(batch_generator(ds, batch_size=8)):
        paths = [item["path"] for item in batch["mask"]]
        print(f"Batch {batch_idx + 1}: {len(paths)} items -> sample: {paths[0]}")
        if batch_idx >= 2:
            break

    print("\nBatch iteration completed successfully!")


if __name__ == "__main__":
    main()
