"""Hugging Face dataset resolution and file locator utilities.

This module resolves dataset identifiers (Hugging Face repository IDs, local paths,
remote URLs, glob patterns, or data_files mappings) into a structured mapping of
split names to concrete file paths or download/stream URLs.
"""

import glob
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from parquet_dataset_loader.exceptions import (
    DatasetNotFoundError,
    ParquetDatasetError,
    SplitNotFoundError,
)


def resolve_hf_token(token: Optional[Union[bool, str]] = None) -> Optional[str]:
    """Resolve a Hugging Face authentication token from arguments, env vars, or local cache.

    Resolution order:
    1. If `token is False`: returns None (explicitly disables authentication).
    2. If `isinstance(token, str)` and non-empty: returns `token.strip()`.
    3. If `token is None` or `token is True`:
       a. `HF_TOKEN` environment variable.
       b. `HUGGING_FACE_HUB_TOKEN` environment variable.
       c. `huggingface_hub.get_token()` (which checks local `~/.cache/huggingface/token`,
          Colab secrets, etc.).

    Args:
        token: Explicit token string, bool, or None.

    Returns:
        Cleaned Bearer token string, or None if no token is available or disabled.
    """
    if token is False:
        return None

    if isinstance(token, str):
        cleaned = token.strip()
        return cleaned if cleaned else None

    # Check HF_TOKEN then HUGGING_FACE_HUB_TOKEN
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_var)
        if val:
            cleaned = val.strip()
            if cleaned:
                return cleaned

    # Fall back to huggingface_hub stored token
    try:
        stored = get_token()
        if stored:
            cleaned = stored.strip()
            if cleaned:
                return cleaned
    except Exception:
        pass

    return None


def create_retry_session(
    retries: int = 4,
    backoff_factor: float = 0.5,
    status_forcelist: Sequence[int] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Create a requests.Session with exponential backoff retries.

    Args:
        retries: Total number of retry attempts.
        backoff_factor: Multiplier for exponential sleep between retries.
        status_forcelist: HTTP status codes that trigger a retry.

    Returns:
        A configured requests.Session instance.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def parse_split_slice(split: Optional[str]) -> Tuple[Optional[str], Optional[slice]]:
    """Parse a Hugging Face split string with optional slicing syntax.

    Examples:
        - "train" -> ("train", None)
        - "train[:1000]" -> ("train", slice(0, 1000, None))
        - "train[100:200]" -> ("train", slice(100, 200, None))
        - "validation[-50:]" -> ("validation", slice(-50, None, None))
        - "train[::2]" -> ("train", slice(None, None, 2))
        - None -> (None, None)

    Args:
        split: Split specification string or None.

    Returns:
        A tuple of (base_split_name, slice_object).
    """
    if split is None:
        return None, None

    pattern = r"^([a-zA-Z0-9_\-]+)(?:\[([0-9\-:]+)\])?$"
    match = re.match(pattern, split.strip())
    if not match:
        return split.strip(), None

    base_split = match.group(1)
    slice_expr = match.group(2)

    if slice_expr is None:
        return base_split, None

    parts = slice_expr.split(":")
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    if len(parts) >= 1 and parts[0] != "":
        start = int(parts[0])
    if len(parts) >= 2 and parts[1] != "":
        stop = int(parts[1])
    if len(parts) >= 3 and parts[2] != "":
        step = int(parts[2])

    if len(parts) == 1:
        # e.g., split="train[10]" -> slice(10, 11)
        idx = int(parts[0])
        return base_split, slice(idx, idx + 1)

    return base_split, slice(start, stop, step)


def infer_split_name(filename_or_path: str) -> str:
    """Infer the dataset split name from a filename or path.

    Rules applied in priority order:
    1. If filename contains 'train' (case-insensitive) -> 'train'
    2. If filename contains 'val', 'valid', 'validation', 'dev', 'eval' -> 'validation'
    3. If filename contains 'test' -> 'test'
    4. Otherwise -> 'train' (default standard split)

    Args:
        filename_or_path: The file path, URL, or filename to inspect.

    Returns:
        The inferred split name (e.g. 'train', 'validation', 'test').
    """
    basename = os.path.basename(filename_or_path).lower()
    # Check whole word or delimiter-separated tokens
    tokens = re.split(r"[^a-z0-9]", basename)

    if any(t in tokens for t in ["val", "valid", "validation", "dev", "eval"]):
        return "validation"
    if any(t in tokens for t in ["test", "testing"]):
        return "test"
    if any(t in tokens for t in ["train", "training"]):
        return "train"

    # Substring checks if token match didn't catch it
    if "val" in basename or "dev" in basename:
        return "validation"
    if "test" in basename:
        return "test"
    if "train" in basename:
        return "train"

    return "train"


def resolve_local_files(
    path: str,
) -> Dict[str, List[str]]:
    """Resolve a local file or directory into split mappings.

    Args:
        path: Path to a local Parquet file or a directory containing Parquet files.

    Returns:
        Mapping of split name to sorted list of absolute file paths.

    Raises:
        DatasetNotFoundError: If the path does not exist or contains no Parquet files.
    """
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(abs_path):
        raise DatasetNotFoundError(path, "Local file or directory does not exist.")

    if os.path.isfile(abs_path):
        if not abs_path.endswith((".parquet", ".pq")):
            raise DatasetNotFoundError(
                path, "Local file exists but is not a Parquet file (.parquet or .pq)."
            )
        split = infer_split_name(abs_path)
        return {split: [abs_path]}

    # It is a directory
    parquet_files: List[str] = []
    for root, _, files in os.walk(abs_path):
        for f in files:
            if f.endswith((".parquet", ".pq")):
                parquet_files.append(os.path.join(root, f))

    if not parquet_files:
        raise DatasetNotFoundError(
            path, f"No Parquet files found in directory '{abs_path}'."
        )

    splits: Dict[str, List[str]] = {}
    for f in sorted(parquet_files):
        s = infer_split_name(f)
        splits.setdefault(s, []).append(f)

    return splits


def resolve_hf_hub_dataset(
    repo_id: str,
    name: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, List[str]]:
    """Resolve a Hugging Face Hub dataset repository into split file URLs.

    First queries the Hugging Face /parquet API endpoint to obtain the exact split
    manifest. If that endpoint returns 404 or an error, falls back to inspecting
    repository files via HfApi / HfFileSystem and categorizing them into splits.

    Args:
        repo_id: The Hugging Face repo identifier (e.g. 'KhangTruong/COCO-inpainted').
        name: Configuration name (default is 'default' or first available config).
        revision: Git revision/branch/tag (default 'main').
        token: Hugging Face authentication token, or True to use cached token.
        session: Optional preconfigured requests.Session.

    Returns:
        Mapping of split name to list of Parquet file URLs.

    Raises:
        DatasetNotFoundError: If the repository does not exist or has no Parquet files.
    """
    if revision is None:
        revision = "main"

    auth_token = resolve_hf_token(token)

    headers: Dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    sess = session or create_retry_session()

    # Step 1: Try Hugging Face /parquet API endpoint
    parquet_api_url = f"https://huggingface.co/api/datasets/{repo_id}/parquet"
    try:
        resp = sess.get(parquet_api_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                config_name = name or ("default" if "default" in data else next(iter(data.keys()), None))
                if config_name and config_name in data:
                    split_map = data[config_name]
                    if isinstance(split_map, dict) and split_map:
                        return {s: list(urls) for s, urls in split_map.items() if urls}
    except Exception:
        # Fall back gracefully to repository listing
        pass

    # Step 2: Fallback to HfApi list_repo_files
    try:
        api = HfApi(token=auth_token)
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    except Exception as e:
        raise DatasetNotFoundError(
            repo_id, f"Could not access Hugging Face repository '{repo_id}': {e}"
        ) from e

    parquet_files = [f for f in repo_files if f.endswith((".parquet", ".pq"))]
    if not parquet_files:
        raise DatasetNotFoundError(
            repo_id, f"No Parquet files found in Hugging Face repository '{repo_id}'."
        )

    splits: Dict[str, List[str]] = {}
    for f in sorted(parquet_files):
        s = infer_split_name(f)
        url = hf_hub_url(repo_id=repo_id, filename=f, revision=revision, repo_type="dataset")
        splits.setdefault(s, []).append(url)

    return splits


def resolve_parquet_dataset(
    path: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]],
    name: Optional[str] = None,
    split: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, List[str]]:
    """High-level resolver for all supported dataset source specifications.

    Resolves:
    1. Direct file paths / directories (local).
    2. Explicit data_files (string glob, list, or mapping).
    3. Remote Hugging Face Hub repositories (e.g. 'KhangTruong/COCO-inpainted').
    4. HTTP/HTTPS URLs pointing directly to Parquet files.

    Args:
        path: Path or Hugging Face dataset identifier.
        name: Dataset configuration name.
        split: Specific split to filter by, or None for all splits.
        data_files: Explicit file specification override.
        token: Hugging Face authentication token.
        revision: Repository git revision.
        session: Optional HTTP session with retries.

    Returns:
        Mapping of split name to list of resolved file paths or URLs.

    Raises:
        DatasetNotFoundError: If the dataset cannot be found.
        SplitNotFoundError: If the requested split is not present.
    """
    splits: Dict[str, List[str]] = {}

    # Case A: data_files parameter is provided
    if data_files is not None:
        if isinstance(data_files, str):
            expanded = glob.glob(os.path.expanduser(data_files))
            files = sorted(expanded) if expanded else [data_files]
            splits = {infer_split_name(f): [f] for f in files}
            # Merge files into proper split lists
            merged: Dict[str, List[str]] = {}
            for f in files:
                merged.setdefault(infer_split_name(f), []).append(f)
            splits = merged
        elif isinstance(data_files, Mapping):
            for s, f_spec in data_files.items():
                if isinstance(f_spec, str):
                    exp = glob.glob(os.path.expanduser(f_spec))
                    splits[s] = sorted(exp) if exp else [f_spec]
                else:
                    file_list = []
                    for item in f_spec:
                        exp = glob.glob(os.path.expanduser(item))
                        file_list.extend(sorted(exp) if exp else [item])
                    splits[s] = file_list
        elif isinstance(data_files, Sequence):
            merged = {}
            for item in data_files:
                exp = glob.glob(os.path.expanduser(item))
                files = sorted(exp) if exp else [item]
                for f in files:
                    merged.setdefault(infer_split_name(f), []).append(f)
            splits = merged

    # Case B: path is a local file or directory that exists
    elif isinstance(path, str) and os.path.exists(os.path.expanduser(path)):
        splits = resolve_local_files(path)

    # Case C: path is a list of URLs or files
    elif isinstance(path, (list, tuple)):
        merged = {}
        for item in path:
            merged.setdefault(infer_split_name(item), []).append(str(item))
        splits = merged

    # Case D: path is a Hugging Face repository identifier
    elif isinstance(path, str):
        splits = resolve_hf_hub_dataset(
            repo_id=path,
            name=name,
            revision=revision,
            token=resolve_hf_token(token),
            session=session,
        )
    else:
        raise ParquetDatasetError(f"Unsupported path specification: {type(path)}")

    if not splits:
        raise DatasetNotFoundError(str(path), "No Parquet files could be resolved.")

    # Filter by requested split if specified
    if split is not None:
        base_split, _ = parse_split_slice(split)
        if base_split and base_split not in splits:
            raise SplitNotFoundError(base_split, list(splits.keys()))

    return splits
