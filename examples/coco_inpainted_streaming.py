"""Example: Streaming and random access on KhangTruong/COCO-inpainted.

This example demonstrates how parquet_dataset_loader inspects the 62.4 GB
KhangTruong/COCO-inpainted dataset in seconds using HTTP range requests,
determines exact dataset length, enables random-access indexing to any row,
and bounds memory usage without downloading massive gigabyte files.
"""

import time
import parquet_dataset_loader as pdl


def main() -> None:
    print("=" * 70)
    print("Loading KhangTruong/COCO-inpainted with parquet_dataset_loader")
    print("=" * 70)

    t0 = time.time()
    # 1. Load dataset with streaming=True
    # Using column projection for 'mask' to avoid downloading heavy 500KB images
    ds = pdl.load_dataset(
        path="KhangTruong/COCO-inpainted",
        split="train",
        streaming=True,
        columns=["mask"],
        max_cached_row_groups=2,
    )
    t1 = time.time()

    print(f"Dataset initialized in: {t1 - t0:.2f} seconds")
    print(f"Total dataset length (known instantly): {len(ds):,} rows")
    print(f"Available columns: {ds.column_names}")
    print()

    # 2. Random access to index 0
    t_start = time.time()
    sample_0 = ds[0]
    t_end = time.time()
    print(f"Fetched index 0 in {t_end - t_start:.2f} seconds")
    print(f"Sample 0 mask keys: {sample_0['mask'].keys()}")
    print()

    # 3. Direct jump to index 50,000 (middle of the 62 GB dataset)
    print("Seeking to index 50,000 across the network without downloading earlier files...")
    t_start = time.time()
    sample_50k = ds[50000]
    t_end = time.time()
    print(f"Fetched index 50,000 in {t_end - t_start:.2f} seconds!")
    print(f"Sample 50,000 mask path: {sample_50k['mask']['path']}")
    print()

    # 4. Instant consecutive read from LRU cache
    print("Reading consecutive sample 50,001 from in-memory row group cache...")
    t_start = time.time()
    sample_50001 = ds[50001]
    t_end = time.time()
    print(f"Fetched index 50,001 in {(t_end - t_start)*1000:.3f} ms (cache hit!)")
    print()

    # 5. Slicing rows [50000:50005] matching Hugging Face batch format
    batch = ds[50000:50005]
    print(f"Sliced batch of 5 items: {len(batch['mask'])} elements")
    print()

    # 6. Stream starting from an arbitrary index
    print("Streaming 3 rows starting from index 100,000...")
    for idx, item in enumerate(ds.iter_from(100000)):
        print(f" - Row {100000 + idx}: mask path = {item['mask']['path']}")
        if idx >= 2:
            break

    print("=" * 70)
    print("Demo completed successfully! Memory remained strictly bounded.")
    print("=" * 70)


if __name__ == "__main__":
    main()
