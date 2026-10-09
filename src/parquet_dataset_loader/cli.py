"""CLI tools for parquet_dataset_loader lifecycle management."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

logger = logging.getLogger("parquet_dataset_loader.cli")


def cleanup_stale_cache_files(cache_dir: Optional[str] = None) -> int:
    """Find and remove stale temporary files (*.tmp.*) from cache directories."""
    from parquet_dataset_loader.api import DEFAULT_CACHE_DIR

    target_dir = os.path.abspath(os.path.expanduser(cache_dir or DEFAULT_CACHE_DIR))
    removed_count = 0
    if not os.path.exists(target_dir):
        return 0

    for root, _, files in os.walk(target_dir):
        for fname in files:
            if ".tmp." in fname or fname.endswith(".tmp"):
                fpath = os.path.join(root, fname)
                try:
                    os.remove(fpath)
                    removed_count += 1
                except OSError as e:
                    logger.debug("Could not remove temp file %s: %s", fpath, e)
    return removed_count


def cli_kill(argv: Optional[List[str]] = None) -> int:
    """CLI command to stop all parquet-dataset-loader background tasks and clean up."""
    parser = argparse.ArgumentParser(
        prog="pdl-kill",
        description="Stop all background downloaders and clean temporary cache files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional custom cache directory to clean.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output.",
    )
    args = parser.parse_args(argv)

    from parquet_dataset_loader.manager import cleanup_background_tasks

    cleaned_tasks = cleanup_background_tasks()
    removed_files = cleanup_stale_cache_files(args.cache_dir)

    if not args.quiet:
        print(f"✅ Cleaned up {cleaned_tasks} active dataset task(s).")
        if removed_files > 0:
            print(f"🧹 Removed {removed_files} stale temporary cache file(s).")

    return 0


if __name__ == "__main__":
    sys.exit(cli_kill())
