"""Example demonstrating multi-dataset management, stream-from-index prefetching, and disk resumption.

Run with:
    python examples/manager_and_resume_example.py
"""

import os
import shutil
import tempfile
import time
import pyarrow as pa
import pyarrow.parquet as pq

import parquet_dataset_loader as pdl


def create_mock_remote_dataset(base_dir: str, num_files: int = 4, rows_per_file: int = 10) -> str:
    """Create a multi-chunk mock dataset simulating a remote repository."""
    os.makedirs(base_dir, exist_ok=True)
    for i in range(num_files):
        ids = list(range(i * rows_per_file, (i + 1) * rows_per_file))
        tbl = pa.Table.from_arrays(
            [
                pa.array(ids, type=pa.int64()),
                pa.array([f"text_{x}" for x in ids], type=pa.string()),
                pa.array([float(x) * 2.0 for x in ids], type=pa.float64()),
            ],
            names=["id", "text", "score"],
        )
        fpath = os.path.join(base_dir, f"train_part_{i:03d}.parquet")
        pq.write_table(tbl, fpath, row_group_size=rows_per_file)
    return base_dir


def main() -> None:
    temp_dir = tempfile.mkdtemp(prefix="pdl_demo_")
    remote_src = os.path.join(temp_dir, "simulated_remote")
    save_path_1 = os.path.join(temp_dir, "local_cache_1")
    save_path_2 = os.path.join(temp_dir, "local_cache_2")

    try:
        print("=" * 70)
        print("1. Creating Mock Multi-File Dataset (40 rows across 4 row groups)...")
        create_mock_remote_dataset(remote_src, num_files=4, rows_per_file=10)
        print("Mock dataset ready at:", remote_src)

        print("\n" + "=" * 70)
        print("2. Multiple Dataset Instance Management (ds1, ds2, ds3)")
        print("=" * 70)

        # Create two managed dataset instances simultaneously
        ds1 = pdl.load_dataset(
            remote_src,
            split="train",
            dataset_id="model_a_train",
            columns=["id", "text"],
        )
        ds2 = pdl.load_dataset(
            remote_src,
            split="train",
            dataset_id="model_b_train",
            columns=["id", "score"],
            shuffle=True,
            seed=42,
        )

        print("Active datasets in registry:")
        active = pdl.list_active_datasets()
        for ds_id, status in active.items():
            print(f"  - [{ds_id}] rows={status['num_rows']}, columns={status['columns']}, shuffled={status['is_shuffled']}")

        # Scoped dataset management context
        print("\nEntering managed_datasets() context block...")
        with pdl.managed_datasets() as mgr:
            ds_scoped = pdl.load_dataset(remote_src, split="train", dataset_id="scoped_eval")
            print(f"  Inside context: active datasets = {len(mgr)} ({list(mgr.list_datasets().keys())})")
            _ = ds_scoped[0]

        print(f"Exited context: active datasets = {len(pdl.get_dataset_manager())}")
        print(f"ds_scoped.is_closed = {ds_scoped.is_closed}")

        # Close ds1 and ds2
        ds1.close()
        ds2.close()
        print(f"Closed ds1 & ds2. Active remaining: {len(pdl.get_dataset_manager())}")

        print("\n" + "=" * 70)
        print("3. Stream-from-Index + Background Downloading (Forward Prefetching)")
        print("=" * 70)
        # Starting stream from row index 20 (row group 2 of 4)
        # BackgroundDownloader downloads row group 2 & 3 FIRST, then wraps around to 0 & 1!
        print("Loading dataset streaming from row index 20 with background downloading...")
        ds_stream = pdl.load_dataset(
            remote_src,
            split="train",
            streaming=True,
            start_index=20,
            save_to_disk=save_path_1,
            background_download=True,
            dataset_id="bg_streamer",
        )
        print(f"Initial row group save progress: {ds_stream.save_progress * 100:.1f}%")
        print(f"First row read immediately from index 20: {ds_stream[0]}")

        # Wait briefly for background prefetcher to download
        if ds_stream.background_downloader:
            ds_stream.background_downloader.join(timeout=3.0)

        print(f"Final row group save progress: {ds_stream.save_progress * 100:.1f}%")
        print(f"Is fully saved to disk: {ds_stream.is_fully_saved}")
        ds_stream.close()

        print("\n" + "=" * 70)
        print("4. Resuming Incomplete Datasets on Disk")
        print("=" * 70)
        # Simulate an interrupted streaming save: only fetch first row group (rows 0..9)
        print("Simulating interrupted download (saving 1 of 4 row groups)...")
        interrupted_ds = pdl.load_dataset(
            remote_src,
            split="train",
            streaming=True,
            save_to_disk=save_path_2,
            background_download=False,
        )
        _ = interrupted_ds[0]  # triggers row group 0 save
        interrupted_ds.close()
        print(f"Interrupted save: 1 row group written to {save_path_2}")

        # Resume the dataset seamlessly via resume_dataset()
        print("Resuming incomplete dataset using resume_dataset()...")
        resumed_ds = pdl.resume_dataset(
            dataset_path=save_path_2,
            background_download=False,
        )
        print(f"Resumed dataset total rows: {len(resumed_ds)}")
        # Read from the previously unsaved portion (row index 35)
        print(f"Accessing previously missing row 35: {resumed_ds[35]}")
        resumed_ds.close()

        # Offline reload now works completely without network
        print("Offline load_from_disk() verification:")
        offline_ds = pdl.load_from_disk(save_path_2)
        print(f"Offline dataset total rows: {len(offline_ds)}, first row: {offline_ds[0]}")
        offline_ds.close()

        print("\n" + "=" * 70)
        print("All multi-dataset lifecycle and disk resume demonstrations completed successfully!")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
