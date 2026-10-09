"""Dataset manager and lifecycle registry for multiple dataset instances.

This module provides:
- DatasetManager: Central registry and resource manager for active dataset instances.
- get_dataset_manager(): Access the global default DatasetManager singleton.
- list_active_datasets(): Introspect all currently active dataset instances.
- close_all_datasets(): Cleanly close and release all open datasets.
- managed_datasets(): Context manager for scoped dataset instance lifetimes.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from parquet_dataset_loader.dataset import IndexedParquetDataset, ParquetDatasetDict

logger = logging.getLogger(__name__)


class DatasetManager:
    """Central registry and lifecycle manager for active dataset instances.

    Tracks active IndexedParquetDataset and ParquetDatasetDict instances,
    enables concurrent multi-dataset monitoring, coordinated resource closing,
    and thread-safe instance lifecycle management.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._datasets: Dict[str, Union[IndexedParquetDataset, ParquetDatasetDict]] = {}
        self._counter: int = 0

    def register(
        self,
        dataset: Union[IndexedParquetDataset, ParquetDatasetDict],
        dataset_id: Optional[str] = None,
    ) -> str:
        """Register a dataset instance with the manager.

        Args:
            dataset: The dataset or dataset dict instance to manage.
            dataset_id: Optional unique identifier. If not provided, an
                automatic ID (e.g. 'dataset_1') is assigned.

        Returns:
            The assigned dataset ID string.
        """
        with self._lock:
            if dataset_id is None:
                self._counter += 1
                base_id = f"dataset_{self._counter}"
            else:
                base_id = str(dataset_id)

            resolved_id = base_id
            suffix = 1
            while resolved_id in self._datasets:
                resolved_id = f"{base_id}_{suffix}"
                suffix += 1

            self._datasets[resolved_id] = dataset

            # Store the registered ID on the instance
            if hasattr(dataset, "dataset_id"):
                try:
                    dataset.dataset_id = resolved_id
                except AttributeError:
                    pass
            if hasattr(dataset, "_dataset_id"):
                dataset._dataset_id = resolved_id

            return resolved_id

    def unregister(
        self, dataset_id: str
    ) -> Optional[Union[IndexedParquetDataset, ParquetDatasetDict]]:
        """Unregister a dataset from the manager without necessarily closing it.

        Args:
            dataset_id: ID of the dataset to remove from registry.

        Returns:
            The unregistered dataset instance, or None if not found.
        """
        with self._lock:
            return self._datasets.pop(dataset_id, None)

    def get(
        self, dataset_id: str
    ) -> Optional[Union[IndexedParquetDataset, ParquetDatasetDict]]:
        """Retrieve an active dataset by its ID.

        Args:
            dataset_id: Unique ID of the dataset.

        Returns:
            The dataset instance, or None if not found.
        """
        with self._lock:
            return self._datasets.get(dataset_id)

    def list_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Return status dictionaries for all currently managed dataset instances.

        Returns:
            Dictionary mapping dataset_id to its status dictionary.
        """
        with self._lock:
            result: Dict[str, Dict[str, Any]] = {}
            for ds_id, ds in self._datasets.items():
                if hasattr(ds, "status"):
                    result[ds_id] = ds.status
                else:
                    result[ds_id] = {
                        "dataset_id": ds_id,
                        "type": type(ds).__name__,
                        "len": len(ds) if hasattr(ds, "__len__") else None,
                    }
            return result

    def close(self, dataset_id: str) -> bool:
        """Close and unregister a specific dataset instance.

        Stops any associated background downloader threads, closes open file
        handles, clears memory caches, and unregisters the instance.

        Args:
            dataset_id: Unique ID of the dataset to close.

        Returns:
            True if dataset was found and closed, False otherwise.
        """
        with self._lock:
            ds = self._datasets.pop(dataset_id, None)
            if ds is not None:
                if hasattr(ds, "close"):
                    try:
                        ds.close()
                    except Exception as e:
                        logger.warning("Error closing dataset %s: %s", dataset_id, e)
                return True
            return False

    def close_all(self) -> int:
        """Close and unregister all managed dataset instances.

        Returns:
            Count of datasets that were closed.
        """
        with self._lock:
            count = len(self._datasets)
            for ds_id, ds in list(self._datasets.items()):
                try:
                    if hasattr(ds, "close"):
                        ds.close()
                except Exception as e:
                    logger.warning("Error closing dataset %s: %s", ds_id, e)
            self._datasets.clear()
            return count

    def stop_all_background_tasks(self) -> int:
        """Stop all background downloaders across all registered datasets."""
        with self._lock:
            count = 0
            for ds in self._datasets.values():
                if hasattr(ds, "stop_background_download"):
                    try:
                        ds.stop_background_download()
                        count += 1
                    except Exception as e:
                        logger.warning("Error stopping background download: %s", e)
            return count

    @property
    def active_count(self) -> int:
        """Number of active datasets currently managed."""
        with self._lock:
            return len(self._datasets)

    def __len__(self) -> int:
        return self.active_count

    def __contains__(self, dataset_id: str) -> bool:
        with self._lock:
            return dataset_id in self._datasets

    def __repr__(self) -> str:
        with self._lock:
            return f"DatasetManager(active_datasets={len(self._datasets)})"

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()



# Global singleton instance
_GLOBAL_DATASET_MANAGER: Optional[DatasetManager] = None
_GLOBAL_MANAGER_LOCK = threading.Lock()


def get_dataset_manager() -> DatasetManager:
    """Retrieve the global default DatasetManager singleton."""
    global _GLOBAL_DATASET_MANAGER
    if _GLOBAL_DATASET_MANAGER is None:
        with _GLOBAL_MANAGER_LOCK:
            if _GLOBAL_DATASET_MANAGER is None:
                _GLOBAL_DATASET_MANAGER = DatasetManager()
    return _GLOBAL_DATASET_MANAGER


def list_active_datasets() -> Dict[str, Dict[str, Any]]:
    """List status metadata for all currently managed dataset instances."""
    return get_dataset_manager().list_datasets()


def close_all_datasets() -> int:
    """Close all currently active managed datasets across the process."""
    return get_dataset_manager().close_all()


def stop_all_background_tasks() -> int:
    """Stop all background downloaders across all registered datasets."""
    return get_dataset_manager().stop_all_background_tasks()


def cleanup_background_tasks() -> int:
    """Stop all background downloads, close all active datasets, and release resources."""
    mgr = get_dataset_manager()
    bg_count = mgr.stop_all_background_tasks()
    ds_count = mgr.close_all()
    return bg_count + ds_count


@contextlib.contextmanager
def managed_datasets(manager: Optional[DatasetManager] = None) -> Iterator[DatasetManager]:
    """Context manager ensuring all datasets created inside are closed on exit.

    Example:
        ```python
        with pdl.managed_datasets() as mgr:
            ds1 = pdl.load_dataset("repo1", streaming=True)
            ds2 = pdl.load_dataset("repo2", streaming=True)
            # do work
        # Both ds1 and ds2 are automatically closed here!
        ```
    """
    mgr = manager or get_dataset_manager()
    initial_ids = set(mgr._datasets.keys())
    try:
        yield mgr
    finally:
        # Close any datasets that were registered during this block
        with mgr._lock:
            current_ids = list(mgr._datasets.keys())
            for ds_id in current_ids:
                if ds_id not in initial_ids:
                    mgr.close(ds_id)
