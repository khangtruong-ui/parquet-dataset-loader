"""Example demonstrating streaming from index and shuffling with seed.

Demonstrates:
1. Shuffling with seed in both streaming and non-streaming modes.
2. Direct parameter usage in load_dataset(..., shuffle=True, seed=42).
3. Streaming from an arbitrary start index via iter_from(start_index) or start_index parameter.
4. Combining reproducible shuffle with seeking to arbitrary resume indices.
5. Streaming buffer shuffle for bounded memory iteration over remote datasets.
"""

import os
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import parquet_dataset_loader as pdl


def create_sample_parquet(path: str, start_id: int, num_rows: int = 50, row_group_size: int = 25):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ids = list(range(start_id, start_id + num_rows))
    table = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array([f"sample_{i}" for i in ids], type=pa.string()),
        ],
        names=["id", "text"],
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def main():
    # 1. Create a local sample dataset with 100 rows across 2 files
    temp_dir = tempfile.mkdtemp()
    f1 = f"{temp_dir}/train-00000.parquet"
    f2 = f"{temp_dir}/train-00001.parquet"
    create_sample_parquet(f1, start_id=0, num_rows=50, row_group_size=25)
    create_sample_parquet(f2, start_id=50, num_rows=50, row_group_size=25)

    print("--- 1. Reproducible Shuffling with Seed (Streaming Mode) ---")
    ds_stream = pdl.load_dataset(
        temp_dir,
        split="train",
        streaming=True,
        shuffle=True,
        seed=42,
    )
    print(f"Loaded dataset rows: {len(ds_stream)}")
    sample_ids = [ds_stream[i]["id"] for i in range(5)]
    print(f"First 5 samples with seed=42: {sample_ids}")

    # Reload with identical seed=42 to demonstrate reproducibility
    ds_stream_again = pdl.load_dataset(
        temp_dir,
        split="train",
        streaming=True,
        shuffle=True,
        seed=42,
    )
    sample_ids_again = [ds_stream_again[i]["id"] for i in range(5)]
    print(f"First 5 samples again:        {sample_ids_again}")
    assert sample_ids == sample_ids_again
    print("✓ Shuffling is 100% reproducible!")

    print("\n--- 2. Shuffling with Seed (Non-Streaming Mode) ---")
    ds_non_stream = pdl.load_dataset(
        temp_dir,
        split="train",
        streaming=False,
        shuffle=True,
        seed=42,
    )
    non_stream_ids = [ds_non_stream[i]["id"] for i in range(5)]
    print(f"First 5 non-streaming samples with seed=42: {non_stream_ids}")
    assert non_stream_ids == sample_ids
    print("✓ Non-streaming produces the exact same shuffled order!")

    print("\n--- 3. Streaming from Index (Resume / Jump) ---")
    # Stream starting from index 80
    print("Streaming rows starting at index 80:")
    for row in ds_stream.iter_from(start_index=80):
        print(f"  Row ID: {row['id']}")

    print("\n--- 4. Direct load_dataset with start_index ---")
    ds_from_80 = pdl.load_dataset(
        temp_dir,
        split="train",
        streaming=True,
        shuffle=True,
        seed=42,
        start_index=80,
    )
    print(f"Rows remaining from index 80: {len(ds_from_80)}")
    direct_ids = [r["id"] for r in ds_from_80]
    iter_ids = [r["id"] for r in ds_stream.iter_from(start_index=80)]
    assert direct_ids == iter_ids
    print("✓ load_dataset(..., start_index=80) matches ds.iter_from(80)!")

    print("\n--- 5. Streaming Buffer-Based Shuffle ---")
    # Buffer shuffle maintains a bounded buffer while streaming sequentially
    ds_buf = pdl.load_dataset(temp_dir, split="train", streaming=True)
    buf_stream = list(ds_buf.shuffle(seed=42, buffer_size=10).iter_from(start_index=90))
    print(f"Buffer-shuffled rows from index 90: {[r['id'] for r in buf_stream]}")

    print("\nAll shuffle and stream-from-index demonstrations succeeded!")


if __name__ == "__main__":
    main()
