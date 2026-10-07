"""High-level user-facing API matching Hugging Face datasets conventions.

This module exposes:
- load_dataset(): Universal loader for remote Hugging Face and local Parquet datasets.
- load_from_disk(): Re-loader for datasets previously saved via save_to_disk().
"""

from __future__ import annotations

import os
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from parquet_dataset_loader.cache import DiskCache, RowGroupMemoryCache
from parquet_dataset_loader.dataset import IndexedParquetDataset, ParquetDatasetDict
from parquet_dataset_loader.exceptions import (
    DatasetNotFoundError,
    ParquetDatasetError,
    SplitNotFoundError,
)
from parquet_dataset_loader.hf_resolver import (
    infer_split_name,
    parse_split_slice,
    resolve_hf_token,
    resolve_parquet_dataset,
)
from parquet_dataset_loader.index import build_metadata_index
from parquet_dataset_loader.progressive import BackgroundDownloader, ProgressiveDiskSaver
from parquet_dataset_loader.reader import RowGroupReader, download_parquet_files

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/parquet_dataset_loader")


def load_dataset(
    path: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]],
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    save_to_disk: Optional[Union[bool, str]] = None,
    background_download: bool = False,
    columns: Optional[Sequence[str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    max_cached_row_groups: int = 2,
    disk_cache: bool = False,
    max_workers: int = 16,
    show_progress: bool = False,
    shuffle: Optional[bool] = None,
    seed: Optional[int] = None,
    buffer_size: Optional[int] = None,
    start_index: Optional[int] = None,
    from_index: Optional[int] = None,
    **kwargs: Any,
) -> Union[IndexedParquetDataset, ParquetDatasetDict]:
    """Load a Parquet dataset from Hugging Face Hub, local files, or remote URLs.

    Provides a drop-in replacement for `datasets.load_dataset` with high-performance
    metadata indexing, random-access streaming, reproducible shuffle with seed,
    streaming from arbitrary index, and simultaneous progressive disk persistence.

    Args:
        path: Hugging Face repo ID (e.g. 'KhangTruong/COCO-inpainted'), local path,
            directory, direct Parquet URL, or list/mapping of files.
        name: Optional dataset configuration name.
        data_dir: Subdirectory within the dataset repository.
        data_files: Specific file or files override (glob, list, or dict).
        split: Specific split to load (e.g. 'train', 'validation', or sliced
            like 'train[:1000]'), or None to load all splits.
        cache_dir: Directory for caching metadata indices, row groups, or full files.
            Defaults to ~/.cache/parquet_dataset_loader.
        streaming: If True, uses random-access streaming without downloading full
            parquet files to disk. If False, downloads files to disk and opens them locally.
        save_to_disk: If provided (path string or True), streams immediately with zero
            initial wait while progressively saving fetched row groups to disk simultaneously.
        background_download: If True and save_to_disk is enabled, downloads remaining
            row groups in a background worker thread while foreground streams unblocked.
        columns: Optional column projection list. Only these columns will be transferred
            or decoded, dramatically saving bandwidth and memory.
        token: Hugging Face auth token, or True to use stored credentials.
        revision: Specific git revision, branch, or tag (default 'main').
        max_cached_row_groups: Number of decoded row groups to retain in RAM simultaneously.
        disk_cache: Whether to cache fetched row groups to local disk during streaming.
        max_workers: Concurrency level for metadata indexing or file downloads.
        show_progress: Whether to show progress bars.
        shuffle: If True (or if seed is provided without shuffle=False), shuffles the
            dataset. Supported with both streaming and non-streaming modes.
        seed: Optional integer seed for reproducible shuffling.
        buffer_size: Optional buffer size for streaming buffer-based shuffle.
        start_index: Optional index to stream from (0-indexed). Resumes or starts
            streaming from this index. Supported in both streaming and non-streaming.
        from_index: Alias for start_index.
        **kwargs: Additional parameters for forward compatibility.

    Returns:
        An IndexedParquetDataset if a single split was requested, or a
        ParquetDatasetDict if split was None.

    Raises:
        DatasetNotFoundError: If the repository or files cannot be located.
        SplitNotFoundError: If the requested split does not exist.
        IndexOutOfBoundsError: If slice indices exceed the split bounds.
    """
    resolved_cache_dir = os.path.abspath(os.path.expanduser(cache_dir or DEFAULT_CACHE_DIR))
    os.makedirs(resolved_cache_dir, exist_ok=True)

    # Resolve HF authentication token (from explicit token, HF_TOKEN env var, or local login cache)
    auth_token = resolve_hf_token(token)
    token_for_children = auth_token if auth_token else False

    # Parse potential split slicing (e.g. 'train[:1000]')
    base_split, slice_obj = parse_split_slice(split)

    # Resolve all files for each split
    splits_map = resolve_parquet_dataset(
        path=path,
        name=name,
        split=base_split,
        data_files=data_files,
        token=token_for_children,
        revision=revision,
    )

    if base_split and base_split not in splits_map:
        raise SplitNotFoundError(base_split, list(splits_map.keys()))

    target_splits = [base_split] if base_split else list(splits_map.keys())

    # If save_to_disk is requested, automatically switch to streaming=True
    # so we never block upfront downloading full files!
    if save_to_disk is not None:
        streaming = True

    # Mode 1: Non-streaming (download full parquet files to disk if remote)
    if not streaming:
        downloaded_splits_map: Dict[str, List[str]] = {}
        for s in target_splits:
            urls = splits_map[s]
            remote_urls = [u for u in urls if u.startswith(("http://", "https://"))]
            if remote_urls:
                safe_repo_name = (
                    str(path).replace("/", "_").replace(":", "_") if isinstance(path, str) else "dataset"
                )
                split_dl_dir = os.path.join(resolved_cache_dir, "downloads", safe_repo_name, s)
                local_files = download_parquet_files(
                    urls=urls,
                    target_dir=split_dl_dir,
                    token=token_for_children,
                    max_workers=max_workers,
                    show_progress=show_progress,
                )
                downloaded_splits_map[s] = local_files
            else:
                downloaded_splits_map[s] = urls
        splits_map = downloaded_splits_map

    # Determine whether shuffle and start_index are requested
    should_shuffle = False
    if shuffle is True:
        should_shuffle = True
    elif shuffle is None and seed is not None:
        should_shuffle = True

    eff_start_index = start_index if start_index is not None else from_index

    # Build datasets for target splits
    datasets: Dict[str, IndexedParquetDataset] = {}

    for s in target_splits:
        file_list = splits_map[s]

        # Build / retrieve cached metadata index
        index = build_metadata_index(
            files=file_list,
            split_name=s,
            cache_dir=resolved_cache_dir,
            max_workers=max_workers,
            token=token_for_children,
        )

        mem_cache = RowGroupMemoryCache(max_entries=max_cached_row_groups)
        d_cache = DiskCache(resolved_cache_dir) if disk_cache else None

        # Setup progressive save-to-disk if requested
        prog_saver: Optional[ProgressiveDiskSaver] = None
        if save_to_disk is not None:
            if isinstance(save_to_disk, str):
                base_save_dir = os.path.abspath(os.path.expanduser(save_to_disk))
            else:
                safe_repo_name = (
                    str(path).replace("/", "_").replace(":", "_") if isinstance(path, str) else "dataset"
                )
                base_save_dir = os.path.join(resolved_cache_dir, "saved", safe_repo_name)

            split_save_dir = (
                os.path.join(base_save_dir, s) if base_split is None else base_save_dir
            )
            prog_saver = ProgressiveDiskSaver(
                target_dir=split_save_dir,
                split_name=s,
                total_row_groups=len(index.row_groups),
                schema=index.schema,
            )

        reader = RowGroupReader(
            memory_cache=mem_cache,
            disk_cache=d_cache,
            progressive_saver=prog_saver,
            token=token_for_children,
        )

        bg_downloader: Optional[BackgroundDownloader] = None
        if background_download and prog_saver is not None:
            bg_downloader = BackgroundDownloader(
                reader=reader,
                row_groups=index.row_groups,
                progressive_saver=prog_saver,
                columns=columns,
            )
            bg_downloader.start()

        ds = IndexedParquetDataset(
            index=index,
            reader=reader,
            columns=columns,
            split=s,
            background_downloader=bg_downloader,
        )

        # Apply split slice if requested (e.g. 'train[:1000]')
        if slice_obj is not None:
            start = slice_obj.start or 0
            stop = slice_obj.stop if slice_obj.stop is not None else len(ds)
            length = max(0, stop - start)
            ds = ds.slice(start, length)

        # Apply shuffle with seed if requested
        if should_shuffle:
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

        # Apply stream start index if requested
        if eff_start_index is not None and eff_start_index > 0:
            ds = ds.skip(eff_start_index)

        datasets[s] = ds

    if base_split is not None:
        return datasets[base_split]

    return ParquetDatasetDict(datasets)


def load_from_disk(
    dataset_path: str,
    columns: Optional[Sequence[str]] = None,
    shuffle: Optional[bool] = None,
    seed: Optional[int] = None,
    buffer_size: Optional[int] = None,
    start_index: Optional[int] = None,
    from_index: Optional[int] = None,
    **kwargs: Any,
) -> Union[IndexedParquetDataset, ParquetDatasetDict]:
    """Load a dataset previously saved to disk via save_to_disk().

    Args:
        dataset_path: Path to the directory containing saved dataset files.
        columns: Optional column projection list.
        shuffle: Optional shuffle flag.
        seed: Optional integer seed for reproducible shuffling.
        buffer_size: Optional buffer size for streaming buffer shuffle.
        start_index: Optional row index to stream from.
        from_index: Alias for start_index.
        **kwargs: Additional parameters passed to load_dataset.

    Returns:
        IndexedParquetDataset or ParquetDatasetDict.
    """
    abs_path = os.path.abspath(os.path.expanduser(dataset_path))
    if not os.path.exists(abs_path):
        raise DatasetNotFoundError(dataset_path, "Local path does not exist.")

    # Check if single split or multiple splits
    entries = os.listdir(abs_path)
    subdirs = [e for e in entries if os.path.isdir(os.path.join(abs_path, e))]

    if subdirs and any(os.path.exists(os.path.join(abs_path, d, f"{d}.feather")) for d in subdirs):
        # Multiple splits
        splits_dict: Dict[str, IndexedParquetDataset] = {}
        for d in sorted(subdirs):
            split_dir = os.path.join(abs_path, d)
            splits_dict[d] = load_from_disk(
                split_dir,
                columns=columns,
                shuffle=shuffle,
                seed=seed,
                buffer_size=buffer_size,
                start_index=start_index,
                from_index=from_index,
                **kwargs,
            )  # type: ignore
        return ParquetDatasetDict(splits_dict)

    # Check if this is a progressive save directory with row_groups/
    rg_dir = os.path.join(abs_path, "row_groups")
    if os.path.exists(rg_dir):
        rg_files = []
        for root, _, files in os.walk(rg_dir):
            for f in sorted(files):
                if f.endswith(".feather"):
                    rg_files.append(os.path.join(root, f))
        rg_files.sort()
        if rg_files:
            batches = []
            schema = None
            for rgf in rg_files:
                tbl = feather.read_table(
                    rgf, columns=list(columns) if columns else None, memory_map=True
                )
                if schema is None:
                    schema = tbl.schema
                batches.extend(tbl.to_batches())
            if not batches:
                table = pa.Table.from_batches([], schema=schema or pa.schema([]))
            else:
                table = pa.Table.from_batches(batches, schema=schema)
            manifest_files = [f for f in entries if f.endswith("_manifest.json")]
            split_name = (
                manifest_files[0].replace("_manifest.json", "")
                if manifest_files
                else "train"
            )
            tmp_parquet = os.path.join(abs_path, f"{split_name}.parquet")
            pq.write_table(table, tmp_parquet, compression="snappy")
            return load_dataset(
                path=tmp_parquet,
                split=split_name,
                streaming=False,
                columns=columns,
                shuffle=shuffle,
                seed=seed,
                buffer_size=buffer_size,
                start_index=start_index,
                from_index=from_index,
                **kwargs,
            )

    # Single split
    # Look for .feather or .parquet files
    data_files = [f for f in entries if f.endswith((".feather", ".parquet", ".pq"))]
    if not data_files:
        raise DatasetNotFoundError(dataset_path, "No data files found in saved directory.")

    first_file = os.path.join(abs_path, data_files[0])
    split_name = infer_split_name(first_file)

    if first_file.endswith(".feather"):
        # Load via feather
        table = feather.read_table(first_file, columns=list(columns) if columns else None, memory_map=True)
        # Create a local in-memory/feather IndexedParquetDataset or return HF dataset
        # To maintain exact interface, write out a fast parquet or construct index:
        tmp_parquet = os.path.join(abs_path, f"{split_name}.parquet")
        if not os.path.exists(tmp_parquet):
            pq.write_table(table, tmp_parquet, compression="snappy")
        first_file = tmp_parquet

    return load_dataset(
        path=first_file,
        split=split_name,
        streaming=False,
        columns=columns,
        shuffle=shuffle,
        seed=seed,
        buffer_size=buffer_size,
        start_index=start_index,
        from_index=from_index,
        **kwargs,
    )
