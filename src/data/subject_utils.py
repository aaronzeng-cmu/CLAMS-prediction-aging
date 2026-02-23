"""
Subject manifest utilities for the CLAMS aging dataset.

Reads ``subject_manifest.csv`` (also tries ``subjects.csv`` and
``participants.csv``) from the data base directory.

Expected columns (case-insensitive):
    SubjectID (or subject_id / Subject), Age, Usable
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from .data_utils import DATA_BASE, _resolve_data_dir, find_subjects

logger = logging.getLogger(__name__)

_MANIFEST_NAMES = ('subject_manifest.csv', 'subjects.csv', 'participants.csv')

_AGE_COL_ALIASES = ('age',)
_ID_COL_ALIASES = ('subjectid', 'subject_id', 'subject', 'participant_id',
                   'participant', 'id')
_USABLE_COL_ALIASES = ('usable',)


def _load_manifest(data_dir: Optional[str] = None):
    """Try to load a subject manifest CSV.  Returns a DataFrame or None."""
    try:
        import pandas as pd
    except ImportError:
        return None

    base = Path(_resolve_data_dir(data_dir))
    for name in _MANIFEST_NAMES:
        path = base / name
        if path.exists():
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip().lower() for c in df.columns]
                return df
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", path, exc)
    return None


def _find_col(df_cols, aliases):
    """Return the first column name (lower-cased) that matches an alias."""
    for col in df_cols:
        if col in aliases:
            return col
    return None


def get_subjects_by_age_group(
    group: str,
    usable_only: bool = True,
    data_dir: Optional[str] = None,
) -> List[str]:
    """Return subjects belonging to an age group.

    Args:
        group:       ``'aging'`` (Age ≥ 60), ``'young'`` (Age < 60), or
                     ``'all'`` (no age filter).
        usable_only: If True, restrict to rows where ``Usable == 1``.
        data_dir:    Override base data directory.

    Returns:
        Sorted list of subject ID strings.

    Notes:
        Falls back to ``find_subjects()`` with a warning if the manifest CSV
        is absent or cannot be parsed.
    """
    if group not in ('aging', 'young', 'all'):
        raise ValueError(f"group must be 'aging', 'young', or 'all', got {group!r}")

    df = _load_manifest(data_dir)

    if df is None:
        logger.warning(
            "Subject manifest not found in %s; returning all subjects from filesystem.",
            _resolve_data_dir(data_dir),
        )
        return find_subjects(data_dir)

    id_col = _find_col(df.columns, _ID_COL_ALIASES)
    age_col = _find_col(df.columns, _AGE_COL_ALIASES)
    usable_col = _find_col(df.columns, _USABLE_COL_ALIASES)

    if id_col is None:
        logger.warning(
            "Manifest has no recognisable subject-ID column "
            "(tried %s); returning all subjects from filesystem.",
            _ID_COL_ALIASES,
        )
        return find_subjects(data_dir)

    mask = [True] * len(df)

    if usable_only and usable_col is not None:
        import numpy as np
        mask = df[usable_col].astype(float) == 1

    if group != 'all' and age_col is not None:
        import numpy as np
        ages = df[age_col].astype(float)
        if group == 'aging':
            age_mask = ages >= 60
        else:
            age_mask = ages < 60
        mask = mask & age_mask
    elif group != 'all':
        logger.warning(
            "Manifest has no 'Age' column; ignoring age filter (returning all usable subjects)."
        )

    subjects = df.loc[mask, id_col].astype(str).tolist()
    return sorted(subjects)


def get_all_usable_subjects(data_dir: Optional[str] = None) -> List[str]:
    """Return all subjects where ``Usable == 1``.

    Falls back to ``find_subjects()`` if the manifest is absent.

    Args:
        data_dir: Override base data directory.

    Returns:
        Sorted list of subject ID strings.
    """
    return get_subjects_by_age_group('all', usable_only=True, data_dir=data_dir)
