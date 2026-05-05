from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_extract_tar(tar_path: Path, extract_dir: Path) -> None:
    """
    Extract a tar archive without allowing path traversal.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r:*") as tar:
        for member in tar.getmembers():
            member_path = (extract_dir / member.name).resolve()
            if not str(member_path).startswith(str(extract_dir.resolve())):
                raise ValueError(f"Unsafe tar member path: {member.name}")
        tar.extractall(path=extract_dir)


def _iter_tar_paths(search_roots: Iterable[Path]) -> List[Path]:
    tar_paths: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        tar_paths.extend([p for p in root.rglob("*.tar") if p.is_file()])
    # Deterministic ordering for reproducibility.
    return sorted(set(tar_paths), key=lambda p: str(p))


def _collect_data_files(data_dirs: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for d in data_dirs:
        if not d.exists():
            continue
        files.extend([p for p in d.rglob("*.parquet") if p.is_file()])
        files.extend([p for p in d.rglob("*.csv") if p.is_file()])
    return sorted(set(files), key=lambda p: str(p))


def _load_files(files: List[Path]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for p in files:
        if p.suffix.lower() == ".parquet":
            dfs.append(pd.read_parquet(p))
        elif p.suffix.lower() == ".csv":
            dfs.append(pd.read_csv(p))
        else:
            raise ValueError(f"Unsupported file type: {p}")

    if not dfs:
        raise FileNotFoundError("No parquet/csv files found after extraction.")

    df = pd.concat(dfs, ignore_index=True)
    # If a `timestamp` column exists, keep the last record per timestamp to
    # avoid duplicates introduced by multi-part archives.
    if "timestamp" in df.columns:
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df


def load_real_dataset(
    *,
    project_root: Path,
    dataset_path: Path,
    extract_dir: Optional[Path] = None,
    extract_tars: bool = True,
) -> pd.DataFrame:
    """
    Load real reliability telemetry from parquet/csv files.

    - If `.tar` archives exist and `extract_dir` doesn't already contain data, extract them.
    - Load all parquet/csv files and concatenate into a single DataFrame.
    """
    dataset_path = dataset_path.resolve()
    project_root = project_root.resolve()
    extract_dir = (extract_dir or dataset_path / "extracted").resolve()

    if extract_tars:
        data_already_present = any(extract_dir.rglob("*.parquet")) or any(extract_dir.rglob("*.csv"))
        if not data_already_present:
            tar_paths = _iter_tar_paths([project_root, dataset_path])
            if tar_paths:
                logger.info("Extracting %d tar archives into %s", len(tar_paths), extract_dir)
                for tar_path in tar_paths:
                    _safe_extract_tar(tar_path, extract_dir)
            else:
                logger.info("No tar archives found under %s (and %s)", project_root, dataset_path)

    # Prefer extracted files, but fall back to dataset_path in case nothing was extracted.
    data_files = _collect_data_files([extract_dir, dataset_path])
    logger.info("Loading %d real data files", len(data_files))
    return _load_files(data_files)

