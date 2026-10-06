"""Progressive streaming and simultaneous disk persistence.

This module enables datasets to be streamed immediately with zero initial blocking,
while progressively saving fetched row groups to local disk in the background or as
they are consumed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pyarrow as pa
import pyarrow.feather as feather

if TYPE_CHECKING:
    from parquet_dataset_loader.index import MetadataIndex, RowGroupInfo
    from parquet_dataset_loader.reader import RowGroupReader

logger = logging.getLogger(__name__)


class ProgressiveDiskSaver:
    """Manages progressive on-disk saving of streamed row groups.

    Allows callers to stream datasets immediately without waiting for a full download,
    while automatically persisting each accessed row group as a memory-mapped
    Feather file on local disk.

    Attributes:
        target_dir: Local directory where the dataset is progressively stored.
        split_name: Split identifier (e.g. 'train').
        total_row_groups: Total row group count for this split.
    """

    def __init__(
        self,
        target_dir: str,
        split_name: str = "train",
        total_row_groups: int = 0,
        schema: Optional[pa.Schema] = None,
    ) -> None:
        self.target_dir = os.path.abspath(os.path.expanduser(target_dir))
        self.split_name = split_name
        self.total_row_groups = total_row_groups
        self.schema = schema

        self.rg_dir = os.path.join(self.target_dir, "row_groups", split_name)
        os.makedirs(self.rg_dir, exist_ok=True)

        self.manifest_file = os.path.join(
            self.target_dir, f"{split_name}_manifest.json"
        )
        self._lock = threading.RLock()
        self._saved_rgs: Set[Tuple[int, int]] = set()

        self._load_manifest()

    def _rg_filename(self, file_idx: int, rg_idx: int) -> str:
        return os.path.join(self.rg_dir, f"rg_{file_idx:05d}_{rg_idx:05d}.feather")

    def _load_manifest(self) -> None:
        """Load the list of previously saved row groups from disk."""
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                saved = data.get("saved_row_groups", [])
                self._saved_rgs = {tuple(item) for item in saved}  # type: ignore
            except Exception as e:
                logger.warning("Could not read manifest file %s: %s", self.manifest_file, e)

    def _save_manifest(self) -> None:
        """Atomically persist the manifest of saved row groups."""
        tmp_file = f"{self.manifest_file}.tmp.{os.getpid()}_{threading.get_ident()}_{time.time_ns()}"
        payload = {
            "split_name": self.split_name,
            "total_row_groups": self.total_row_groups,
            "saved_count": len(self._saved_rgs),
            "is_complete": self.is_complete,
            "saved_row_groups": sorted(list(self._saved_rgs)),
        }
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_file, self.manifest_file)
        except Exception as e:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
            logger.warning("Failed saving manifest: %s", e)

    def is_saved(self, file_idx: int, rg_idx: int) -> bool:
        """Check if a specific row group has been saved to disk."""
        with self._lock:
            return (file_idx, rg_idx) in self._saved_rgs

    def get(
        self,
        file_idx: int,
        rg_idx: int,
        columns: Optional[Sequence[str]] = None,
    ) -> Optional[pa.Table]:
        """Read a saved row group from disk via memory mapping."""
        if not self.is_saved(file_idx, rg_idx):
            return None

        filepath = self._rg_filename(file_idx, rg_idx)
        if not os.path.exists(filepath):
            with self._lock:
                self._saved_rgs.discard((file_idx, rg_idx))
            return None

        try:
            return feather.read_table(
                filepath,
                columns=list(columns) if columns else None,
                memory_map=True,
            )
        except Exception as e:
            logger.warning("Error reading cached row group %s: %s", filepath, e)
            return None

    def save(
        self,
        file_idx: int,
        rg_idx: int,
        table: pa.Table,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        """Save a decoded row group table to disk."""
        with self._lock:
            if (file_idx, rg_idx) in self._saved_rgs:
                return

            filepath = self._rg_filename(file_idx, rg_idx)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            tmp_file = f"{filepath}.tmp.{os.getpid()}_{threading.get_ident()}_{time.time_ns()}"

            try:
                feather.write_feather(table, tmp_file, compression="zstd")
                os.replace(tmp_file, filepath)
                self._saved_rgs.add((file_idx, rg_idx))
                self._save_manifest()
            except Exception as e:
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except OSError:
                        pass
                logger.warning("Failed writing row group to disk: %s", e)

    @property
    def saved_count(self) -> int:
        """Number of row groups currently persisted to disk."""
        with self._lock:
            return len(self._saved_rgs)

    @property
    def save_progress(self) -> float:
        """Fraction of total row groups saved (0.0 to 1.0)."""
        if self.total_row_groups <= 0:
            return 1.0
        with self._lock:
            return len(self._saved_rgs) / self.total_row_groups

    @property
    def is_complete(self) -> bool:
        """Whether all row groups for this split have been saved."""
        if self.total_row_groups <= 0:
            return True
        with self._lock:
            return len(self._saved_rgs) >= self.total_row_groups

    def finalize(self, index: Optional[MetadataIndex] = None) -> str:
        """Write final dataset metadata info enabling offline load_from_disk()."""
        info_file = os.path.join(self.target_dir, f"{self.split_name}_info.json")
        info: Dict[str, Any] = {
            "split": self.split_name,
            "saved_row_groups": len(self._saved_rgs),
            "total_row_groups": self.total_row_groups,
            "is_complete": self.is_complete,
            "format": "feather_row_groups",
        }
        if index is not None:
            info["total_rows"] = index.total_rows
            info["columns"] = index.column_names

        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        return info_file


class BackgroundDownloader:
    """Asynchronous worker that downloads remaining row groups in the background.

    Runs in a background daemon thread while foreground operations stream data
    without blocking.
    """

    def __init__(
        self,
        reader: RowGroupReader,
        row_groups: Sequence[RowGroupInfo],
        progressive_saver: ProgressiveDiskSaver,
        columns: Optional[Sequence[str]] = None,
        delay_between_requests: float = 0.02,
    ) -> None:
        self.reader = reader
        self.row_groups = row_groups
        self.progressive_saver = progressive_saver
        self.columns = columns
        self.delay_between_requests = delay_between_requests

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background downloader thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="pdl-background-downloader",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        """Background worker loop."""
        for rg in self.row_groups:
            if self._stop_event.is_set():
                break

            # If already saved by foreground thread or earlier run, skip
            if self.progressive_saver.is_saved(rg.file_index, rg.rg_index):
                continue

            try:
                # Reading triggers reader caching and progressive saving automatically
                self.reader.read_row_group(
                    file_url=rg.file_url,
                    rg_index=rg.rg_index,
                    columns=self.columns,
                    file_index=rg.file_index,
                )
            except Exception as e:
                logger.debug(
                    "Background download failed for row group (%d, %d): %s",
                    rg.file_index,
                    rg.rg_index,
                    e,
                )

            if self.delay_between_requests > 0:
                time.sleep(self.delay_between_requests)

    def stop(self) -> None:
        """Signal the background worker to stop."""
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the background worker thread to finish."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        """Check if background worker is currently active."""
        return self._thread is not None and self._thread.is_alive()
