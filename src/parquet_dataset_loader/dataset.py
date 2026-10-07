"""Dataset abstractions for indexed Parquet streaming and Hugging Face compatibility.

This module provides:
1. IndexedParquetDataset: A sequence-like, PyTorch DataLoader-compatible dataset
   supporting O(log M) random-access indexing, instant slice retrieval, and
   bounded memory streaming.
2. ParquetDatasetDict: A multi-split dictionary container mirroring Hugging Face's
   DatasetDict.
"""

from __future__ import annotations

import collections.abc
import os
import random
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from parquet_dataset_loader.exceptions import (
    IndexOutOfBoundsError,
    ParquetDatasetError,
)
from parquet_dataset_loader.index import MetadataIndex, RowGroupInfo
from parquet_dataset_loader.reader import RowGroupReader


class IndexedParquetDataset(collections.abc.Sequence):
    """An indexed, random-access Parquet dataset designed for streaming.

    Unlike standard streaming datasets which require sequential scanning,
    IndexedParquetDataset uses a precomputed metadata index to seek directly
    to any row index or row range using HTTP range requests without downloading
    full files.

    Features:
    - O(log M) random access via integer indexing: ds[50000].
    - Bounded memory footprint via an LRU row-group cache.
    - Slicing and batch indexing matching Hugging Face Dataset format.
    - Instant seeking to arbitrary positions: ds.iter_from(start_index) or ds.stream(start_index).
    - Reproducible shuffling with seed: ds.shuffle(seed=42) across both streaming and non-streaming.
    - Streaming buffer shuffle: ds.shuffle(seed=42, buffer_size=1000).
    - Column projection: ds.select_columns(["col1"]).
    - Zero-copy view slicing: ds.take(n), ds.skip(n).
    - PyTorch Dataset compatible (implements __len__ and __getitem__).
    - Conversion to Arrow, Pandas, and Hugging Face Dataset.

    Attributes:
        index: The MetadataIndex for this dataset split.
        reader: The RowGroupReader managing I/O and caching.
        columns: Optional list of projected column names.
        offset: Global row start offset for view slices.
        length: Number of rows in this view.
        indices: Explicit list of global row indices when shuffled or subsetted.
        buffer_size: Optional buffer size for streaming buffer shuffle.
        seed: Optional seed for reproducible shuffle operations.
    """

    def __init__(
        self,
        index: MetadataIndex,
        reader: RowGroupReader,
        columns: Optional[Sequence[str]] = None,
        offset: int = 0,
        length: Optional[int] = None,
        split: Optional[str] = None,
        background_downloader: Optional[Any] = None,
        indices: Optional[Sequence[int]] = None,
        buffer_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.index = index
        self.reader = reader
        self.columns = list(columns) if columns is not None else None
        self.background_downloader = background_downloader
        self.buffer_size = buffer_size
        self.seed = seed

        if indices is not None:
            self.indices: Optional[List[int]] = list(indices)
            self.offset = 0
            self._length = len(self.indices)
        else:
            self.indices = None
            self.offset = max(0, offset)
            max_len = max(0, self.index.total_rows - self.offset)
            self._length = max_len if length is None else max(0, min(length, max_len))

        self.split = split or self.index.split_name

        # Schema projection
        if self.columns is not None:
            self._schema = pa.schema([self.index.schema.field(c) for c in self.columns])
            self.column_names = list(self.columns)
        else:
            self._schema = self.index.schema
            self.column_names = list(self.index.column_names)

        self.features = self.index.features_info

    @property
    def schema(self) -> pa.Schema:
        """PyArrow schema for this dataset."""
        return self._schema

    @property
    def num_rows(self) -> int:
        """Total number of rows in this dataset view."""
        return self._length

    def __len__(self) -> int:
        return self._length

    def _get_single_row(self, idx: int) -> Dict[str, Any]:
        """Fetch a single row by relative index as a Python dictionary."""
        global_idx = self.indices[idx] if self.indices is not None else (self.offset + idx)
        rg_info, local_row = self.index.locate_row(global_idx)

        table = self.reader.read_row_group(
            file_url=rg_info.file_url,
            rg_index=rg_info.rg_index,
            columns=self.columns,
            file_index=rg_info.file_index,
        )

        # Slice 1 row and extract as dictionary
        row_slice = table.slice(local_row, 1)
        pylist = row_slice.to_pylist()
        if not pylist:
            raise ParquetDatasetError(
                f"Failed to extract row at offset {local_row} from row group {rg_info.rg_index}"
            )
        return pylist[0]

    def _get_slice(self, s: slice) -> Dict[str, List[Any]]:
        """Fetch a slice of rows matching Hugging Face batch format {col: [vals]}."""
        start, stop, step = s.indices(self._length)
        indices = list(range(start, stop, step))
        return self._get_indices(indices)

    def _get_indices(self, indices: Sequence[int]) -> Dict[str, List[Any]]:
        """Fetch multiple rows by index list matching Hugging Face batch format."""
        batch: Dict[str, List[Any]] = {col: [] for col in self.column_names}
        for idx in indices:
            row = self._get_single_row(idx)
            for col in self.column_names:
                batch[col].append(row.get(col))
        return batch

    def __getitem__(
        self, key: Union[int, slice, Sequence[int]]
    ) -> Union[Dict[str, Any], Dict[str, List[Any]]]:
        """Retrieve a single sample, slice, or batch of samples.

        - ds[i]: Returns a dictionary of {column_name: value}.
        - ds[start:stop]: Returns a dictionary of {column_name: [values]}, matching HF datasets.
        - ds[[i, j, k]]: Returns a dictionary of {column_name: [values]}.

        Args:
            key: Integer index, slice, or sequence of indices.

        Returns:
            A sample dictionary or batch dictionary.
        """
        if isinstance(key, int):
            if key < 0:
                key += self._length
            if key < 0 or key >= self._length:
                raise IndexOutOfBoundsError(key, self._length)
            return self._get_single_row(key)

        if isinstance(key, slice):
            return self._get_slice(key)

        if isinstance(key, (list, tuple)):
            norm_indices = []
            for i in key:
                if i < 0:
                    i += self._length
                if i < 0 or i >= self._length:
                    raise IndexOutOfBoundsError(i, self._length)
                norm_indices.append(i)
            return self._get_indices(norm_indices)

        raise TypeError(f"Invalid key type: {type(key)}. Expected int, slice, or sequence.")

    def shuffle(
        self,
        seed: Optional[int] = None,
        buffer_size: Optional[int] = None,
    ) -> IndexedParquetDataset:
        """Randomly shuffle the dataset with an optional reproducible seed.

        Supports both index permutation shuffle (default, buffer_size=None)
        and streaming buffer-based shuffle (buffer_size=N).

        In index permutation mode, all downstream operations (indexing, slicing,
        iter_from, PyTorch DataLoader) reflect the shuffled order instantly.
        Running with the same seed guarantees identical reproducible ordering.

        Args:
            seed: Optional integer seed for deterministic pseudo-random shuffling.
            buffer_size: Optional integer buffer size for streaming buffer shuffle.
                If provided, sequential streaming maintains a buffer of this size,
                yielding items randomly from the buffer.

        Returns:
            A new shuffled IndexedParquetDataset instance.
        """
        if buffer_size is not None and buffer_size > 0:
            return IndexedParquetDataset(
                index=self.index,
                reader=self.reader,
                columns=self.columns,
                offset=self.offset,
                length=self._length,
                split=self.split,
                background_downloader=self.background_downloader,
                indices=self.indices,
                buffer_size=buffer_size,
                seed=seed,
            )

        base_indices = (
            list(self.indices)
            if self.indices is not None
            else [self.offset + i for i in range(self._length)]
        )
        rng = random.Random(seed)
        shuffled_indices = list(base_indices)
        rng.shuffle(shuffled_indices)

        return IndexedParquetDataset(
            index=self.index,
            reader=self.reader,
            columns=self.columns,
            indices=shuffled_indices,
            split=self.split,
            background_downloader=self.background_downloader,
            seed=seed,
        )

    def iter_from(
        self,
        start_index: int = 0,
        buffer_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream dataset rows sequentially starting from a specific index.

        Efficiently jumps directly to start_index without downloading or scanning
        any prior row groups. Keeps memory bounded via LRU caching.
        If the dataset is shuffled, yields rows according to the shuffled order.

        Args:
            start_index: Starting index in this dataset view (0-indexed).
            buffer_size: Optional override for buffer-based shuffle size.
            seed: Optional override for shuffle seed.

        Yields:
            Row dictionaries {column: value}.
        """
        if start_index < 0:
            start_index += self._length
        start_index = max(0, min(start_index, self._length))

        eff_buf_size = buffer_size if buffer_size is not None else self.buffer_size
        eff_seed = seed if seed is not None else self.seed

        if eff_buf_size is not None and eff_buf_size > 0:
            rng = random.Random(eff_seed)
            buf: List[Dict[str, Any]] = []
            buf_limit = max(1, eff_buf_size)

            for i in range(start_index, self._length):
                row = self._get_single_row(i)
                if len(buf) < buf_limit:
                    buf.append(row)
                else:
                    evict_idx = rng.randint(0, buf_limit - 1)
                    yield buf[evict_idx]
                    buf[evict_idx] = row

            rng.shuffle(buf)
            for row in buf:
                yield row
        else:
            for i in range(start_index, self._length):
                yield self._get_single_row(i)

    def stream_from(
        self,
        start_index: int = 0,
        buffer_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream dataset rows sequentially starting from a specific index.

        Alias for iter_from().
        """
        return self.iter_from(start_index=start_index, buffer_size=buffer_size, seed=seed)

    def stream(
        self,
        start_index: Optional[int] = None,
        from_index: Optional[int] = None,
        buffer_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream dataset rows sequentially starting from an optional index.

        Args:
            start_index: Starting index in this dataset view (0-indexed).
            from_index: Alias for start_index.
            buffer_size: Optional buffer size for streaming buffer shuffle.
            seed: Optional seed for shuffle.

        Yields:
            Row dictionaries {column: value}.
        """
        idx = 0
        if start_index is not None:
            idx = start_index
        elif from_index is not None:
            idx = from_index
        return self.iter_from(start_index=idx, buffer_size=buffer_size, seed=seed)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all rows of the dataset sequentially."""
        return self.iter_from(0)

    def select_columns(self, columns: Sequence[str]) -> IndexedParquetDataset:
        """Create a new dataset view projected to a subset of columns.

        Only the specified columns will be transferred over the network when
        reading row groups, drastically saving bandwidth and memory.

        Args:
            columns: Sequence of column names to keep.

        Returns:
            A new IndexedParquetDataset instance.
        """
        for col in columns:
            if col not in self.index.column_names:
                raise ValueError(
                    f"Column '{col}' not in dataset columns: {self.index.column_names}"
                )

        return IndexedParquetDataset(
            index=self.index,
            reader=self.reader,
            columns=columns,
            offset=self.offset,
            length=self._length,
            split=self.split,
            background_downloader=self.background_downloader,
            indices=self.indices,
            buffer_size=self.buffer_size,
            seed=self.seed,
        )

    def take(self, n: int) -> IndexedParquetDataset:
        """Create a zero-copy view of the first n rows.

        Args:
            n: Number of rows to take.

        Returns:
            A new IndexedParquetDataset view.
        """
        n = max(0, min(n, self._length))
        if self.indices is not None:
            return IndexedParquetDataset(
                index=self.index,
                reader=self.reader,
                columns=self.columns,
                indices=self.indices[:n],
                split=self.split,
                background_downloader=self.background_downloader,
                buffer_size=self.buffer_size,
                seed=self.seed,
            )
        return IndexedParquetDataset(
            index=self.index,
            reader=self.reader,
            columns=self.columns,
            offset=self.offset,
            length=n,
            split=self.split,
            background_downloader=self.background_downloader,
            buffer_size=self.buffer_size,
            seed=self.seed,
        )

    def skip(self, n: int) -> IndexedParquetDataset:
        """Create a zero-copy view skipping the first n rows.

        Args:
            n: Number of rows to skip.

        Returns:
            A new IndexedParquetDataset view.
        """
        n = max(0, min(n, self._length))
        if self.indices is not None:
            return IndexedParquetDataset(
                index=self.index,
                reader=self.reader,
                columns=self.columns,
                indices=self.indices[n:],
                split=self.split,
                background_downloader=self.background_downloader,
                buffer_size=self.buffer_size,
                seed=self.seed,
            )
        return IndexedParquetDataset(
            index=self.index,
            reader=self.reader,
            columns=self.columns,
            offset=self.offset + n,
            length=self._length - n,
            split=self.split,
            background_downloader=self.background_downloader,
            buffer_size=self.buffer_size,
            seed=self.seed,
        )

    def slice(self, start: int, length: int) -> IndexedParquetDataset:
        """Create a zero-copy dataset view starting at start with length rows."""
        start = max(0, min(start, self._length))
        length = max(0, min(length, self._length - start))
        if self.indices is not None:
            return IndexedParquetDataset(
                index=self.index,
                reader=self.reader,
                columns=self.columns,
                indices=self.indices[start : start + length],
                split=self.split,
                background_downloader=self.background_downloader,
                buffer_size=self.buffer_size,
                seed=self.seed,
            )
        return IndexedParquetDataset(
            index=self.index,
            reader=self.reader,
            columns=self.columns,
            offset=self.offset + start,
            length=length,
            split=self.split,
            background_downloader=self.background_downloader,
            buffer_size=self.buffer_size,
            seed=self.seed,
        )

    def to_arrow(self, batch_size: int = 1000) -> pa.Table:
        """Materialize this dataset view as an in-memory PyArrow Table.

        Warning: Only use for datasets or slices that comfortably fit in RAM.

        Args:
            batch_size: Read batch size for building the table.

        Returns:
            PyArrow Table containing all rows in this view.
        """
        if self._length == 0:
            return pa.Table.from_batches([], schema=self._schema)

        if self.indices is None:
            batches: List[pa.RecordBatch] = []
            global_start = self.offset
            global_stop = self.offset + self._length

            ranges = self.index.locate_range(global_start, global_stop)
            for rg_info, local_start, local_stop in ranges:
                table = self.reader.read_row_group(
                    file_url=rg_info.file_url,
                    rg_index=rg_info.rg_index,
                    columns=self.columns,
                    file_index=rg_info.file_index,
                )
                sub_table = table.slice(local_start, local_stop - local_start)
                batches.extend(sub_table.to_batches())

            if not batches:
                return pa.Table.from_batches([], schema=self._schema)
            return pa.Table.from_batches(batches, schema=self._schema)

        # When indices is present (shuffled or custom index view)
        # Group indices by row group to avoid repeated I/O / decodes
        rg_map: Dict[Tuple[str, int, int], List[Tuple[int, int]]] = {}
        for pos, g_idx in enumerate(self.indices):
            rg_info, local_row = self.index.locate_row(g_idx)
            key = (rg_info.file_url, rg_info.rg_index, rg_info.file_index)
            if key not in rg_map:
                rg_map[key] = []
            rg_map[key].append((pos, local_row))

        extracted_rows: List[Optional[Dict[str, Any]]] = [None] * self._length
        for (file_url, rg_index, file_index), pos_local_list in rg_map.items():
            rg_table = self.reader.read_row_group(
                file_url=file_url,
                rg_index=rg_index,
                columns=self.columns,
                file_index=file_index,
            )
            for pos, local_row in pos_local_list:
                row_slice = rg_table.slice(local_row, 1)
                extracted_rows[pos] = row_slice.to_pylist()[0]

        return pa.Table.from_pylist(extracted_rows, schema=self._schema)

    @property
    def progressive_saver(self) -> Optional[Any]:
        """Access the ProgressiveDiskSaver instance if configured."""
        return self.reader.progressive_saver

    @property
    def save_progress(self) -> float:
        """Fraction of total row groups saved to disk (0.0 to 1.0)."""
        if self.progressive_saver is not None:
            return self.progressive_saver.save_progress
        return 0.0

    @property
    def is_fully_saved(self) -> bool:
        """Whether all row groups in this dataset are saved to disk."""
        if self.progressive_saver is not None:
            return self.progressive_saver.is_complete
        return False

    def stream_and_save(self, show_progress: bool = True) -> IndexedParquetDataset:
        """Stream through the entire dataset, saving all row groups to disk.

        Enables users to stream data with zero initial blocking while simultaneously
        writing the entire dataset to disk.

        Args:
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            self
        """
        from tqdm import tqdm

        pbar = (
            tqdm(total=len(self), desc=f"Streaming & saving {self.split}", unit="row")
            if show_progress
            else None
        )

        for rg_info in self.index.row_groups:
            self.reader.read_row_group(
                file_url=rg_info.file_url,
                rg_index=rg_info.rg_index,
                columns=self.columns,
                file_index=rg_info.file_index,
            )
            if pbar:
                pbar.update(rg_info.num_rows)

        if pbar:
            pbar.close()

        if self.progressive_saver is not None:
            self.progressive_saver.finalize(self.index)

        return self

    def close(self) -> None:
        """Stop any background downloader and close open reader handles."""
        if self.background_downloader is not None:
            self.background_downloader.stop()
        self.reader.close()

    def to_pandas(self) -> Any:
        """Materialize this dataset view as a pandas DataFrame."""
        return self.to_arrow().to_pandas()

    def to_hf_dataset(self) -> Any:
        """Convert this dataset view to a Hugging Face datasets.Dataset object."""
        import datasets
        table = self.to_arrow()
        return datasets.Dataset(table)

    def save_to_disk(self, target_dir: str) -> None:
        """Save this dataset to a local directory in Arrow IPC format.

        Compatible with load_from_disk.

        Args:
            target_dir: Local destination directory.
        """
        os.makedirs(target_dir, exist_ok=True)
        table = self.to_arrow()
        data_file = os.path.join(target_dir, f"{self.split}.feather")
        feather.write_feather(table, data_file, compression="zstd")

        # Save metadata info
        state = {
            "split": self.split,
            "num_rows": len(self),
            "columns": self.column_names,
            "data_file": os.path.basename(data_file),
            "format": "feather",
        }
        import json
        with open(os.path.join(target_dir, f"{self.split}_info.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def __repr__(self) -> str:
        col_repr = ", ".join(self.column_names[:5])
        if len(self.column_names) > 5:
            col_repr += f", ... (+{len(self.column_names) - 5} more)"
        extra = ""
        if self.indices is not None:
            extra = ",\n  shuffled: True"
        return (
            f"IndexedParquetDataset(\n"
            f"  split: '{self.split}',\n"
            f"  num_rows: {self._length:,},\n"
            f"  columns: [{col_repr}]{extra}\n"
            f")"
        )


class ParquetDatasetDict(dict):
    """Dictionary container mapping split names to IndexedParquetDataset instances.

    Mirrors Hugging Face's DatasetDict behavior and interfaces.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def select_columns(self, columns: Sequence[str]) -> ParquetDatasetDict:
        """Apply column projection to all splits in the dictionary."""
        return ParquetDatasetDict(
            {split: ds.select_columns(columns) for split, ds in self.items()}
        )

    def shuffle(
        self,
        seed: Optional[int] = None,
        buffer_size: Optional[int] = None,
    ) -> ParquetDatasetDict:
        """Apply shuffle with seed to all splits in the dictionary.

        Args:
            seed: Optional integer seed for reproducible shuffling.
            buffer_size: Optional buffer size for streaming buffer shuffle.

        Returns:
            A new ParquetDatasetDict containing shuffled datasets.
        """
        return ParquetDatasetDict(
            {split: ds.shuffle(seed=seed, buffer_size=buffer_size) for split, ds in self.items()}
        )

    def skip(self, n: int) -> ParquetDatasetDict:
        """Skip the first n rows in each split."""
        return ParquetDatasetDict({split: ds.skip(n) for split, ds in self.items()})

    def take(self, n: int) -> ParquetDatasetDict:
        """Take the first n rows in each split."""
        return ParquetDatasetDict({split: ds.take(n) for split, ds in self.items()})

    def slice(self, start: int, length: int) -> ParquetDatasetDict:
        """Slice start to start + length rows in each split."""
        return ParquetDatasetDict({split: ds.slice(start, length) for split, ds in self.items()})

    def to_hf_dataset(self) -> Any:
        """Convert all splits to a Hugging Face DatasetDict."""
        import datasets
        return datasets.DatasetDict(
            {split: ds.to_hf_dataset() for split, ds in self.items()}
        )

    def save_to_disk(self, target_dir: str) -> None:
        """Save all splits to a local directory."""
        os.makedirs(target_dir, exist_ok=True)
        for split, ds in self.items():
            split_dir = os.path.join(target_dir, split)
            ds.save_to_disk(split_dir)

    def __repr__(self) -> str:
        splits_str = ",\n".join(
            f"  '{k}': IndexedParquetDataset(num_rows={len(v):,})"
            for k, v in self.items()
        )
        return f"ParquetDatasetDict({{\n{splits_str}\n}})"
