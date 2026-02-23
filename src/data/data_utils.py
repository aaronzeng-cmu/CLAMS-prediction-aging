"""
Cluster data access layer for the CLAMS aging EEG dataset.

Data layout on ORCD:
    /orcd/data/ldlewis/001/om2/shared/aging/{subject}/run{N}/

Override base directory via the AGING_DATA_DIR environment variable or
pass ``data_dir`` explicitly to any function.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_BASE = os.environ.get(
    'AGING_DATA_DIR',
    '/orcd/data/ldlewis/001/om2/shared/aging',
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_data_dir(data_dir: Optional[str]) -> str:
    return data_dir if data_dir is not None else DATA_BASE


def _find_eeg_file(run_dir: Path) -> Optional[Path]:
    """Return the first recognised EEG file in *run_dir*, or None."""
    for pattern in ('*.mat', '*.h5', '*.hdf5', '*.npz'):
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_eeg(
    subject: str,
    run_num: int,
    channel: int = 16,
    target_fs: int = 200,
    return_header: bool = False,
    data_dir: Optional[str] = None,
) -> np.ndarray | Tuple[np.ndarray, Dict]:
    """Load and resample a single-channel EEG recording.

    Discovers the file at ``DATA_BASE/{subject}/run{run_num}/`` and tries
    HDF5/MAT (h5py), then NPZ.  Extracts *channel* (default 16 = Fpz,
    0-indexed).  Handles both ``[samples × channels]`` and
    ``[channels × samples]`` layouts.

    Resamples from the native 500 Hz to *target_fs* (200 Hz) using
    ``scipy.signal.resample_poly(data, 2, 5)`` — an exact integer-ratio
    polyphase filter with no aliasing artefacts.

    Args:
        subject:       Subject directory name.
        run_num:       Run number (used to build ``run{N}`` directory).
        channel:       EEG channel index (0-indexed).  Default 16 = Fpz.
        target_fs:     Target sample rate after resampling.  Default 200 Hz.
        return_header: If True, also return a metadata dict.
        data_dir:      Override base data directory.

    Returns:
        1-D float64 array of shape ``(T,)``.
        If *return_header* is True, returns ``(array, header_dict)``.
    """
    from scipy.signal import resample_poly

    base = Path(_resolve_data_dir(data_dir))
    run_dir = base / subject / f'run{run_num}'

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    eeg_file = _find_eeg_file(run_dir)
    if eeg_file is None:
        raise FileNotFoundError(f"No EEG file found in {run_dir}")

    ext = eeg_file.suffix.lower()
    source_fs = 500  # assumed native sample rate

    if ext in ('.mat', '.h5', '.hdf5'):
        import h5py
        with h5py.File(str(eeg_file), 'r') as f:
            # Try common keys; fall back to first numeric dataset
            for key in ('EEG', 'eeg', 'data', 'Data'):
                if key in f:
                    arr = f[key][()]
                    break
            else:
                for key in f.keys():
                    try:
                        arr = f[key][()]
                        if arr.ndim >= 1:
                            break
                    except Exception:
                        continue
                else:
                    raise KeyError(f"No usable EEG dataset found in {eeg_file}")

        arr = np.array(arr, dtype=np.float64)

    elif ext == '.npz':
        data = np.load(str(eeg_file))
        for key in ('EEG', 'eeg', 'data', 'Data'):
            if key in data:
                arr = data[key].astype(np.float64)
                break
        else:
            # Use first array
            key = list(data.keys())[0]
            arr = data[key].astype(np.float64)

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Normalise shape: want [channels × samples]
    if arr.ndim == 1:
        # Single-channel recording — no channel axis
        eeg_1d = arr
    elif arr.ndim == 2:
        n_rows, n_cols = arr.shape
        if n_rows < n_cols:
            # [channels × samples]
            eeg_1d = arr[channel, :]
        else:
            # [samples × channels]
            eeg_1d = arr[:, channel]
    else:
        raise ValueError(f"Unexpected EEG array shape: {arr.shape}")

    # Resample 500 → 200 Hz (up=2, down=5)
    if source_fs != target_fs:
        # Compute integer up/down ratio
        from math import gcd
        g = gcd(target_fs, source_fs)
        up = target_fs // g
        down = source_fs // g
        eeg_1d = resample_poly(eeg_1d, up, down).astype(np.float64)

    if return_header:
        header = {
            'EEG_fs': source_fs,
            'target_fs': target_fs,
            'subject': subject,
            'run': run_num,
            'channel': channel,
            'file': str(eeg_file),
        }
        return eeg_1d, header

    return eeg_1d


def find_subjects(data_dir: Optional[str] = None) -> List[str]:
    """List all subject directories under *data_dir*.

    Excludes hidden directories (names starting with '.') and
    non-directory entries.

    Args:
        data_dir: Override base data directory.

    Returns:
        Sorted list of subject directory names.
    """
    base = Path(_resolve_data_dir(data_dir))
    if not base.exists():
        return []
    return sorted(
        p.name for p in base.iterdir()
        if p.is_dir() and not p.name.startswith('.')
    )


def find_runs(subject: str, data_dir: Optional[str] = None) -> List[int]:
    """List run numbers for *subject*.

    Scans for directories named ``run{N}`` where N is a positive integer.

    Args:
        subject:  Subject directory name.
        data_dir: Override base data directory.

    Returns:
        Sorted list of integer run numbers.
    """
    base = Path(_resolve_data_dir(data_dir))
    subj_dir = base / subject
    if not subj_dir.exists():
        return []

    runs: List[int] = []
    for p in subj_dir.iterdir():
        if p.is_dir() and p.name.startswith('run'):
            suffix = p.name[3:]
            if suffix.isdigit():
                runs.append(int(suffix))
    return sorted(runs)


def load_sleep_scores(
    subject: str,
    run_num: int,
    data_dir: Optional[str] = None,
):
    """Load sleep-stage scores for one run, if present.

    Tries CSV files in ``DATA_BASE/{subject}/run{run_num}/`` with columns
    ``start_time``, ``end_time``, ``sleep_stage``.

    Returns:
        ``pandas.DataFrame`` on success, or ``None`` if no CSV found or
        ``pandas`` is unavailable.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    base = Path(_resolve_data_dir(data_dir))
    run_dir = base / subject / f'run{run_num}'
    if not run_dir.exists():
        return None

    for csv_path in sorted(run_dir.glob('*.csv')):
        try:
            df = pd.read_csv(csv_path)
            # Accept if expected columns (case-insensitive) are present
            cols_lower = [c.lower() for c in df.columns]
            if 'sleep_stage' in cols_lower:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception:
            continue

    return None


def list_eeg_records(
    subjects: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
) -> List[Tuple[str, int]]:
    """Return all ``(subject, run_num)`` pairs available in *data_dir*.

    Args:
        subjects: If provided, restrict to this list of subjects.
        data_dir: Override base data directory.

    Returns:
        List of ``(subject, run_num)`` tuples, sorted by subject then run.
    """
    if subjects is None:
        subjects = find_subjects(data_dir)

    records: List[Tuple[str, int]] = []
    for subj in subjects:
        for run in find_runs(subj, data_dir):
            records.append((subj, run))
    return records
