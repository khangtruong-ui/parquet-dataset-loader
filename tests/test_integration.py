"""Integration test with the live KhangTruong/COCO-inpainted Hugging Face dataset.

This test validates:
1. Fast metadata indexing across 62 remote Parquet files without downloading 62 GB.
2. Exact length calculation (122,184 rows).
3. Column-projected row group fetching (downloading only mask metadata, ~250 KB).
4. Direct random access seeking to index 50,000 without downloading earlier files.
"""

import pytest
from parquet_dataset_loader.api import load_dataset
from parquet_dataset_loader.dataset import IndexedParquetDataset


@pytest.mark.integration
def test_coco_inpainted_indexed_streaming() -> None:
    # Load with streaming=True and column projection to avoid large image downloads
    ds = load_dataset(
        path="KhangTruong/COCO-inpainted",
        split="train",
        streaming=True,
        columns=["mask"],
        max_cached_row_groups=2,
    )

    assert isinstance(ds, IndexedParquetDataset)
    # The train split has 62 files of ~2000 rows each -> 122,184 rows
    assert len(ds) == 122184
    assert ds.num_rows == 122184
    assert ds.column_names == ["mask"]

    # Test reading row 0
    row_0 = ds[0]
    assert isinstance(row_0, dict)
    assert "mask" in row_0
    assert "bytes" in row_0["mask"]

    # Test seeking to index 50,000 without touching files 0-24
    row_50k = ds[50000]
    assert isinstance(row_50k, dict)
    assert "mask" in row_50k

    # Test slice of 2 items
    batch = ds[50000:50002]
    assert len(batch["mask"]) == 2
