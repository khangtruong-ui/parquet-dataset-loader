"""Example: Streaming + Save-to-Disk at the same time without blocking.

This example demonstrates how you can start streaming immediately (step 0 starts
in milliseconds without waiting for a 62 GB download), while progressively
persisting accessed row groups to local disk in the background.
"""

import tempfile
import time
import parquet_dataset_loader as pdl


def main() -> None:
    print("=" * 70)
    print("Streaming + Save to Disk Simultaneously (Zero Initial Blocking)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = f"{temp_dir}/coco_local_archive"
        print(f"Target save path: {save_path}")

        # 1. Initialize with streaming=True and save_to_disk
        # Starts instantly! Zero blocking wait time.
        t0 = time.time()
        ds = pdl.load_dataset(
            path="KhangTruong/COCO-inpainted",
            split="train",
            streaming=True,
            save_to_disk=save_path,
            columns=["mask"],
            max_cached_row_groups=2,
        )
        t1 = time.time()

        print(f"Dataset initialized in: {t1 - t0:.2f} seconds")
        print(f"Total dataset length: {len(ds):,} rows")
        print(f"Initial disk save progress: {ds.save_progress * 100:.1f}%")
        print()

        # 2. Access first row immediately (streaming over HTTP, saving to disk simultaneously)
        print("Reading row 0 immediately (unblocked)...")
        t_start = time.time()
        row_0 = ds[0]
        t_end = time.time()
        print(f"Row 0 fetched in {t_end - t_start:.2f}s! Mask path: {row_0['mask']['path']}")
        print(f"Disk save progress after row 0: {ds.save_progress * 100:.2f}%")
        print()

        # 3. Read consecutive sample 50 from the same row group (0ms cache hit)
        t_start = time.time()
        row_50 = ds[50]
        t_end = time.time()
        print(f"Row 50 fetched in {(t_end - t_start)*1000:.3f}ms (local hit)!")
        print()

        # 4. Stream a slice of 10 rows and verify they are saved to disk
        print("Streaming 10 rows...")
        for i in range(10):
            _ = ds[i]

        print(f"Row groups saved so far: {ds.progressive_saver.saved_count}")
        print(f"Save progress: {ds.save_progress * 100:.2f}%")
        print()

        # 5. Finalize manifest
        ds.progressive_saver.finalize()
        print("Progressive archive manifest finalized on disk.")

        # 6. Load saved dataset offline with load_from_disk
        print("Reloading saved samples from disk offline with load_from_disk()...")
        reloaded = pdl.load_from_disk(save_path)
        print(f"Successfully reloaded {len(reloaded)} rows directly from disk!")
        print(f"Reloaded row 0 mask path: {reloaded[0]['mask']['path']}")

    print("=" * 70)
    print("Demo completed successfully! Zero wait time, immediate streaming.")
    print("=" * 70)


if __name__ == "__main__":
    main()
