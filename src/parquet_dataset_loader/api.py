"""High-level user-facing API matching Hugging Face datasets conventions.

This module exposes:
- load_dataset(): Universal loader for remote Hugging Face and local Parquet datasets.
- load_from_disk(): Re-loader for datasets previously saved via save_to_disk().
"""

from __future__ import annotations

import json
import logging
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
from parquet_dataset_loader.manager import (
    DatasetManager,
    close_all_datasets,
    get_dataset_manager,
    list_active_datasets,
    managed_datasets,
)
from parquet_dataset_loader.progressive import BackgroundDownloader, ProgressiveDiskSaver
from parquet_dataset_loader.reader import RowGroupReader, download_parquet_files

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.environ.get(
    "PARQUET_DATASET_LOADER_CACHE",
    os.path.expanduser("~/.cache/parquet_dataset_loader"),
)
DEFAULT_SAVE_DIR = os.environ.get(
    "PARQUET_DATASET_LOADER_SAVE_DIR",
    os.path.join(DEFAULT_CACHE_DIR, "saved"),
)


def load_dataset(
    path: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]],
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    save_to_disk: Union[bool, str] = False,
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
    dataset_id: Optional[str] = None,
    manage: bool = True,
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
            Defaults to False (disabled when streaming=True). If True, saves to the default
            cache directory (~/.cache/parquet_dataset_loader/saved).
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

    # Auto-discover existing local downloaded Parquet files in cache directory
    safe_repo_name = (
        str(path).replace("/", "_").replace(":", "_") if isinstance(path, str) else "dataset"
    )
    for s in target_splits:
        urls = splits_map[s]
        remote_urls = [u for u in urls if u.startswith(("http://", "https://"))]
        if remote_urls:
            split_dl_dir = os.path.join(resolved_cache_dir, "downloads", safe_repo_name, s)
            if os.path.isdir(split_dl_dir):
                local_candidates = [
                    os.path.join(split_dl_dir, f)
                    for f in os.listdir(split_dl_dir)
                    if f.endswith((".parquet", ".pq"))
                ]
                if len(local_candidates) >= len(urls) and len(local_candidates) > 0:
                    logger.info(
                        "Found %d cached local Parquet files in '%s' for split '%s'. "
                        "Using local files for instant zero-network access.",
                        len(local_candidates),
                        split_dl_dir,
                        s,
                    )
                    splits_map[s] = sorted(local_candidates)

    # Determine whether progressive disk persistence is requested.
    # Default is False when streaming=True.
    should_save_to_disk = bool(save_to_disk)
    if should_save_to_disk:
        streaming = True

    # Mode 1: Non-streaming (download full parquet files to disk if remote)
    if not streaming:
        downloaded_splits_map: Dict[str, List[str]] = {}
        for s in target_splits:
            urls = splits_map[s]
            remote_urls = [u for u in urls if u.startswith(("http://", "https://"))]
            if remote_urls:
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
        if should_save_to_disk:
            if isinstance(save_to_disk, str):
                base_save_dir = os.path.abspath(os.path.expanduser(save_to_disk))
            else:
                safe_repo_name = (
                    str(path).replace("/", "_").replace(":", "_") if isinstance(path, str) else "dataset"
                )
                base_save_dir = os.path.join(DEFAULT_SAVE_DIR, safe_repo_name)

            if base_split is None:
                os.makedirs(base_save_dir, exist_ok=True)
                dict_meta_file = os.path.join(base_save_dir, "dataset_dict.json")
                try:
                    with open(dict_meta_file, "w", encoding="utf-8") as f:
                        json.dump({"splits": target_splits}, f, indent=2)
                except Exception:
                    pass

            split_save_dir = (
                os.path.join(base_save_dir, s) if base_split is None else base_save_dir
            )
            prog_saver = ProgressiveDiskSaver(
                target_dir=split_save_dir,
                split_name=s,
                total_row_groups=len(index.row_groups),
                schema=index.schema,
                source_path=path if isinstance(path, str) else None,
            )

        reader = RowGroupReader(
            memory_cache=mem_cache,
            disk_cache=d_cache,
            progressive_saver=prog_saver,
            token=token_for_children,
        )

        bg_downloader: Optional[BackgroundDownloader] = None
        if background_download and prog_saver is not None:
            start_rg_idx = 0
            if eff_start_index is not None and eff_start_index > 0:
                target_row = min(eff_start_index, max(0, index.total_rows - 1))
                rg_info, _ = index.locate_row(target_row)
                for i_rg, rg_item in enumerate(index.row_groups):
                    if rg_item.file_index == rg_info.file_index and rg_item.rg_index == rg_info.rg_index:
                        start_rg_idx = i_rg
                        break

            bg_downloader = BackgroundDownloader(
                reader=reader,
                row_groups=index.row_groups,
                progressive_saver=prog_saver,
                columns=columns,
                start_rg_index=start_rg_idx,
            )
            bg_downloader.start()

        ds_instance_id = f"{dataset_id}_{s}" if dataset_id and base_split is None else dataset_id
        ds = IndexedParquetDataset(
            index=index,
            reader=reader,
            columns=columns,
            split=s,
            background_downloader=bg_downloader,
            dataset_id=ds_instance_id,
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
        result_ds: Union[IndexedParquetDataset, ParquetDatasetDict] = datasets[base_split]
    else:
        result_ds = ParquetDatasetDict(datasets, dataset_id=dataset_id)

    if manage:
        get_dataset_manager().register(result_ds, dataset_id=dataset_id)

    return result_ds


def load_from_disk(
    dataset_path: Optional[str] = None,
    split: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    shuffle: Optional[bool] = None,
    seed: Optional[int] = None,
    buffer_size: Optional[int] = None,
    start_index: Optional[int] = None,
    from_index: Optional[int] = None,
    resume: bool = True,
    allow_incomplete: bool = True,
    background_download: bool = False,
    token: Optional[Union[bool, str]] = None,
    dataset_id: Optional[str] = None,
    manage: bool = True,
    **kwargs: Any,
) -> Union[IndexedParquetDataset, ParquetDatasetDict]:
    """Load a dataset previously saved to disk via save_to_disk().

    Supports resuming incomplete downloads and progressive saves, either by
    reconnecting to the original source or loading the available local rows.
    Handles both ParquetDatasetDict (multi-split) and IndexedParquetDataset (single split).

    Args:
        dataset_path: Path to the directory containing saved dataset files, or None
            to load from the default save directory (~/.cache/parquet_dataset_loader/saved).
        split: Optional split name to load if dataset_path contains multiple splits.
        columns: Optional column projection list.
        shuffle: Optional shuffle flag.
        seed: Optional integer seed for reproducible shuffling.
        buffer_size: Optional buffer size for streaming buffer shuffle.
        start_index: Optional row index to stream from.
        from_index: Alias for start_index.
        resume: If True and the dataset on disk is incomplete with a recorded source,
            automatically resumes streaming and completing missing row groups.
        allow_incomplete: If True, allows loading available local rows even if incomplete.
        background_download: If resuming, whether to download remaining rows in background.
        token: Optional auth token.
        dataset_id: Optional unique identifier for DatasetManager.
        manage: Whether to register with DatasetManager.
        **kwargs: Additional parameters passed to load_dataset.

    Returns:
        IndexedParquetDataset or ParquetDatasetDict.
    """
    if dataset_path is None:
        if not os.path.exists(DEFAULT_SAVE_DIR):
            raise DatasetNotFoundError(
                DEFAULT_SAVE_DIR,
                "No default save directory found. Please specify dataset_path explicitly.",
            )
        candidates = [
            os.path.join(DEFAULT_SAVE_DIR, d)
            for d in os.listdir(DEFAULT_SAVE_DIR)
            if os.path.isdir(os.path.join(DEFAULT_SAVE_DIR, d)) and not d.startswith(".")
        ]
        if not candidates:
            raise DatasetNotFoundError(
                DEFAULT_SAVE_DIR,
                "No saved datasets found in default save directory.",
            )
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        abs_path = os.path.abspath(candidates[0])
    else:
        abs_path = os.path.abspath(os.path.expanduser(dataset_path))

    if not os.path.exists(abs_path):
        raise DatasetNotFoundError(dataset_path or abs_path, "Local path does not exist.")

    entries = os.listdir(abs_path)

    # 1. Check if dataset_dict.json exists in abs_path (standard DatasetDict format)
    dict_manifest_file = os.path.join(abs_path, "dataset_dict.json")
    if os.path.exists(dict_manifest_file):
        try:
            with open(dict_manifest_file, "r", encoding="utf-8") as f:
                dmeta = json.load(f)
            splits_list = dmeta.get("splits", [])
        except Exception:
            splits_list = []

        if splits_list:
            if split is not None:
                if split in splits_list:
                    return load_from_disk(
                        os.path.join(abs_path, split),
                        split=split,
                        columns=columns,
                        shuffle=shuffle,
                        seed=seed,
                        buffer_size=buffer_size,
                        start_index=start_index,
                        from_index=from_index,
                        resume=resume,
                        allow_incomplete=allow_incomplete,
                        background_download=background_download,
                        token=token,
                        dataset_id=dataset_id,
                        manage=manage,
                        **kwargs,
                    )
                raise SplitNotFoundError(split, splits_list)

            splits_dict: Dict[str, IndexedParquetDataset] = {}
            for s in splits_list:
                s_dir = os.path.join(abs_path, s)
                splits_dict[s] = load_from_disk(
                    s_dir,
                    split=s,
                    columns=columns,
                    shuffle=shuffle,
                    seed=seed,
                    buffer_size=buffer_size,
                    start_index=start_index,
                    from_index=from_index,
                    resume=resume,
                    allow_incomplete=allow_incomplete,
                    background_download=background_download,
                    token=token,
                    manage=False,
                    **kwargs,
                )  # type: ignore
            dict_res = ParquetDatasetDict(splits_dict, dataset_id=dataset_id)
            if manage:
                get_dataset_manager().register(dict_res, dataset_id=dataset_id)
            return dict_res

    # 2. Check for multi-split subdirectories if no dataset_dict.json
    subdirs = [
        e
        for e in sorted(entries)
        if os.path.isdir(os.path.join(abs_path, e)) and not e.startswith((".", "__"))
    ]

    def _is_split_dir(d_path: str) -> bool:
        if not os.path.isdir(d_path):
            return False
        d_entries = os.listdir(d_path)
        return (
            os.path.exists(os.path.join(d_path, "row_groups"))
            or any(f.endswith((".parquet", ".pq", ".feather", ".arrow")) for f in d_entries)
            or any(f.endswith(("_manifest.json", "_info.json", "state.json")) for f in d_entries)
        )

    split_subdirs = [d for d in subdirs if _is_split_dir(os.path.join(abs_path, d))]
    has_root_data = (
        os.path.exists(os.path.join(abs_path, "row_groups"))
        or any(f.endswith((".parquet", ".pq", ".feather", ".arrow")) for f in entries)
    )

    if split_subdirs and not has_root_data:
        if split is not None:
            if split in split_subdirs:
                return load_from_disk(
                    os.path.join(abs_path, split),
                    split=split,
                    columns=columns,
                    shuffle=shuffle,
                    seed=seed,
                    buffer_size=buffer_size,
                    start_index=start_index,
                    from_index=from_index,
                    resume=resume,
                    allow_incomplete=allow_incomplete,
                    background_download=background_download,
                    token=token,
                    dataset_id=dataset_id,
                    manage=manage,
                    **kwargs,
                )
            raise SplitNotFoundError(split, split_subdirs)

        splits_dict = {}
        for d in split_subdirs:
            split_dir = os.path.join(abs_path, d)
            splits_dict[d] = load_from_disk(
                split_dir,
                split=d,
                columns=columns,
                shuffle=shuffle,
                seed=seed,
                buffer_size=buffer_size,
                start_index=start_index,
                from_index=from_index,
                resume=resume,
                allow_incomplete=allow_incomplete,
                background_download=background_download,
                token=token,
                manage=False,
                **kwargs,
            )  # type: ignore
        dict_res = ParquetDatasetDict(splits_dict, dataset_id=dataset_id)
        if manage:
            get_dataset_manager().register(dict_res, dataset_id=dataset_id)
        return dict_res

    # 3. Single split loading
    eff_split = split
    if not eff_split:
        manifest_files = [f for f in entries if f.endswith("_manifest.json")]
        if manifest_files:
            eff_split = manifest_files[0].replace("_manifest.json", "")
    if not eff_split:
        info_files = [f for f in entries if f.endswith("_info.json") and not f.startswith("dataset_")]
        if info_files:
            try:
                with open(os.path.join(abs_path, info_files[0]), "r", encoding="utf-8") as f:
                    imeta = json.load(f)
                eff_split = imeta.get("split")
            except Exception:
                pass
    if not eff_split:
        base_name = os.path.basename(abs_path)
        if base_name and base_name not in ("saved", "downloads", "data"):
            eff_split = base_name

    # Check progressive save directory with row_groups/
    rg_dir = os.path.join(abs_path, "row_groups")
    if os.path.exists(rg_dir):
        manifest_files = [f for f in entries if f.endswith("_manifest.json")]
        split_name = eff_split or (
            manifest_files[0].replace("_manifest.json", "")
            if manifest_files
            else "train"
        )
        is_complete = False
        source_path = None
        if manifest_files:
            try:
                with open(os.path.join(abs_path, manifest_files[0]), "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                is_complete = mdata.get("is_complete", False)
                source_path = mdata.get("source_path")
            except Exception:
                pass

        if not is_complete and resume and source_path:
            logger.info("Resuming incomplete dataset from source '%s' to '%s'", source_path, abs_path)
            return load_dataset(
                path=source_path,
                split=split_name,
                streaming=True,
                save_to_disk=abs_path,
                columns=columns,
                shuffle=shuffle,
                seed=seed,
                buffer_size=buffer_size,
                start_index=start_index,
                from_index=from_index,
                background_download=background_download,
                token=token,
                dataset_id=dataset_id,
                manage=manage,
                **kwargs,
            )

        if not is_complete and not allow_incomplete:
            raise ParquetDatasetError(
                f"Incomplete progressive dataset at '{dataset_path or abs_path}' has not finished saving. "
                "Set resume=True to finish downloading from source, or allow_incomplete=True to load partial rows."
            )

        rg_files = []
        for root, _, files in os.walk(rg_dir):
            for f in sorted(files):
                if f.endswith(".feather") and ".tmp." not in f:
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
                dataset_id=dataset_id,
                manage=manage,
                **kwargs,
            )

    # Check for direct parquet files
    parquet_files = [f for f in entries if f.endswith((".parquet", ".pq"))]
    split_name = eff_split or (
        infer_split_name(parquet_files[0]) if parquet_files else "train"
    )

    if parquet_files:
        if f"{split_name}.parquet" in parquet_files:
            target_path = os.path.join(abs_path, f"{split_name}.parquet")
        elif len(parquet_files) == 1:
            target_path = os.path.join(abs_path, parquet_files[0])
        else:
            target_path = [os.path.join(abs_path, f) for f in sorted(parquet_files)]

        return load_dataset(
            path=target_path,
            split=split_name,
            streaming=False,
            columns=columns,
            shuffle=shuffle,
            seed=seed,
            buffer_size=buffer_size,
            start_index=start_index,
            from_index=from_index,
            dataset_id=dataset_id,
            manage=manage,
            **kwargs,
        )

    # Check for feather files
    feather_files = [f for f in entries if f.endswith(".feather")]
    if feather_files:
        target_feather = (
            os.path.join(abs_path, f"{split_name}.feather")
            if f"{split_name}.feather" in feather_files
            else os.path.join(abs_path, feather_files[0])
        )
        table = feather.read_table(
            target_feather, columns=list(columns) if columns else None, memory_map=True
        )
        tmp_parquet = os.path.join(abs_path, f"{split_name}.parquet")
        if not os.path.exists(tmp_parquet):
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
            dataset_id=dataset_id,
            manage=manage,
            **kwargs,
        )

    # Check for Arrow IPC files (.arrow)
    arrow_files = [f for f in entries if f.endswith(".arrow")]
    if arrow_files:
        target_arrow = os.path.join(abs_path, arrow_files[0])
        try:
            with pa.ipc.open_file(target_arrow) as reader:
                table = reader.read_all()
        except Exception:
            with pa.ipc.open_stream(target_arrow) as reader:
                table = reader.read_all()
        tmp_parquet = os.path.join(abs_path, f"{split_name}.parquet")
        if not os.path.exists(tmp_parquet):
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
            dataset_id=dataset_id,
            manage=manage,
            **kwargs,
        )

    raise DatasetNotFoundError(
        dataset_path or abs_path, "No data files found in saved directory."
    )


def resume_dataset(
    dataset_path: str,
    source_path: Optional[str] = None,
    split: Optional[str] = None,
    background_download: bool = True,
    columns: Optional[Sequence[str]] = None,
    token: Optional[Union[bool, str]] = None,
    dataset_id: Optional[str] = None,
    manage: bool = True,
    **kwargs: Any,
) -> Union[IndexedParquetDataset, ParquetDatasetDict]:
    """Resume saving/downloading an incomplete dataset on disk.

    Reads the saved manifest to identify the source and missing row groups,
    then unblocks streaming immediately while downloading remaining data in background.

    Args:
        dataset_path: Path to the local incomplete dataset directory.
        source_path: Optional source repo ID or URL override.
        split: Optional split name override.
        background_download: Whether to prefetch remaining row groups in background.
        columns: Optional column projection list.
        token: Optional auth token.
        dataset_id: Optional dataset instance ID.
        manage: Whether to register with DatasetManager.
        **kwargs: Additional parameters passed to load_dataset.

    Returns:
        IndexedParquetDataset or ParquetDatasetDict resuming download and streaming.
    """
    abs_path = os.path.abspath(os.path.expanduser(dataset_path))
    if not os.path.exists(abs_path):
        raise DatasetNotFoundError(dataset_path, "Directory does not exist.")

    entries = os.listdir(abs_path)

    # Check for multi-split dataset_dict.json
    dict_json = os.path.join(abs_path, "dataset_dict.json")
    if os.path.exists(dict_json):
        try:
            with open(dict_json, "r", encoding="utf-8") as f:
                dmeta = json.load(f)
            splits_list = dmeta.get("splits", [])
        except Exception:
            splits_list = []

        if splits_list:
            if split is not None:
                if split in splits_list:
                    return resume_dataset(
                        os.path.join(abs_path, split),
                        source_path=source_path,
                        split=split,
                        background_download=background_download,
                        columns=columns,
                        token=token,
                        dataset_id=dataset_id,
                        manage=manage,
                        **kwargs,
                    )
                raise SplitNotFoundError(split, splits_list)

            splits_dict: Dict[str, IndexedParquetDataset] = {}
            for s in splits_list:
                splits_dict[s] = resume_dataset(
                    os.path.join(abs_path, s),
                    source_path=source_path,
                    split=s,
                    background_download=background_download,
                    columns=columns,
                    token=token,
                    manage=False,
                    **kwargs,
                )  # type: ignore
            dict_res = ParquetDatasetDict(splits_dict, dataset_id=dataset_id)
            if manage:
                get_dataset_manager().register(dict_res, dataset_id=dataset_id)
            return dict_res

    # Single split
    manifest_files = [f for f in entries if f.endswith("_manifest.json")]
    detected_split = split
    detected_source = source_path

    if manifest_files:
        if not detected_split:
            detected_split = manifest_files[0].replace("_manifest.json", "")
        if not detected_source:
            try:
                with open(os.path.join(abs_path, manifest_files[0]), "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                detected_source = mdata.get("source_path")
            except Exception:
                pass

    if not detected_source:
        raise ParquetDatasetError(
            f"Cannot resume dataset at '{dataset_path}': no source_path found in manifest. "
            "Please provide source_path explicitly."
        )

    return load_dataset(
        path=detected_source,
        split=detected_split,
        streaming=True,
        save_to_disk=abs_path,
        background_download=background_download,
        columns=columns,
        token=token,
        dataset_id=dataset_id,
        manage=manage,
        **kwargs,
    )
