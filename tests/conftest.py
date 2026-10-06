"""Pytest test fixtures and sample data generators."""

import os
import shutil
import tempfile
from typing import Generator, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide a temporary directory that is automatically deleted after the test."""
    d = tempfile.mkdtemp(prefix="pdl_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def create_sample_parquet_file(
    file_path: str,
    num_rows: int = 100,
    row_group_size: int = 25,
    start_id: int = 0,
) -> str:
    """Create a sample Parquet file with specified rows and row group size.

    Columns:
    - id: int64
    - text: string
    - value: float64
    - payload: binary
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    ids = list(range(start_id, start_id + num_rows))
    texts = [f"sample_text_{i}" for i in ids]
    values = [float(i) * 1.5 for i in ids]
    payloads = [f"payload_{i}".encode("utf-8") for i in ids]

    table = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(texts, type=pa.string()),
            pa.array(values, type=pa.float64()),
            pa.array(payloads, type=pa.binary()),
        ],
        names=["id", "text", "value", "payload"],
    )

    pq.write_table(table, file_path, row_group_size=row_group_size)
    return file_path


@pytest.fixture
def sample_dataset_dir(temp_dir: str) -> str:
    """Create a multi-file, multi-split sample dataset in temp_dir.

    Directory layout:
    - data/
      - train-00000.parquet (50 rows, 2 row groups of 25)
      - train-00001.parquet (50 rows, 2 row groups of 25)
      - validation-00000.parquet (30 rows, 1 row group of 30)
    Total train rows: 100
    Total validation rows: 30
    """
    data_dir = os.path.join(temp_dir, "data")
    create_sample_parquet_file(
        os.path.join(data_dir, "train-00000.parquet"),
        num_rows=50,
        row_group_size=25,
        start_id=0,
    )
    create_sample_parquet_file(
        os.path.join(data_dir, "train-00001.parquet"),
        num_rows=50,
        row_group_size=25,
        start_id=50,
    )
    create_sample_parquet_file(
        os.path.join(data_dir, "validation-00000.parquet"),
        num_rows=30,
        row_group_size=30,
        start_id=1000,
    )
    return data_dir
