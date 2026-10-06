"""Example: Disk caching and local loading.

Demonstrates:
1. Streaming with disk caching enabled (fetched row groups cached as Feather for 0ms re-reads).
2. Saving datasets to disk with save_to_disk().
3. Loading datasets from disk with load_from_disk().
"""

import tempfile
import parquet_dataset_loader as pdl


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Working in temporary cache directory: {temp_dir}")

        # 1. Stream with disk_cache=True
        print("\n1. Initializing streaming with disk_cache=True...")
        ds = pdl.load_dataset(
            path="KhangTruong/COCO-inpainted",
            split="train",
            streaming=True,
            columns=["mask"],
            cache_dir=temp_dir,
            disk_cache=True,
            max_cached_row_groups=2,
        )

        # Access a small slice
        sub = ds.take(10)
        print(f"Read 10 samples. Local cache now stores row group 0.")

        # 2. Save subset to disk
        save_path = f"{temp_dir}/saved_subset"
        print(f"\n2. Saving subset to disk at {save_path}...")
        sub.save_to_disk(save_path)

        # 3. Reload from disk
        print("\n3. Loading from disk with load_from_disk()...")
        reloaded = pdl.load_from_disk(save_path)
        print(f"Reloaded dataset has {len(reloaded)} rows.")
        print(f"First sample mask path: {reloaded[0]['mask']['path']}")

    print("\nDisk caching example completed successfully!")


if __name__ == "__main__":
    main()
