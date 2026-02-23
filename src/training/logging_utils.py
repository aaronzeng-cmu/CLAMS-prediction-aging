"""
Structured logging helpers for CLAMS training runs.

Provides:
    setup_logger   — console + file logger, SLURM-job-ID-aware
    log_config     — dump all config fields and subject splits
    log_epoch      — one-liner epoch summary with optional "NEW BEST" flag
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from .config import TrainingConfig


def setup_logger(run_dir: str, name: str = 'train') -> logging.Logger:
    """Create (or retrieve) a named logger that writes to console and file.

    The log file is placed at ``run_dir/logs/train.log``.  When running
    under SLURM (``SLURM_JOB_ID`` env var is set) the filename is prefixed
    with the job ID: ``run_dir/logs/{jobid}_train.log``.

    The function is idempotent — calling it twice with the same *name* will
    not add duplicate handlers.

    Args:
        run_dir: Root directory of the current training run.
        name:    Logger name (default ``'train'``).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    log = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if log.handlers:
        return log

    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # File handler — DEBUG and above
    log_dir = Path(run_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    job_id = os.environ.get('SLURM_JOB_ID', '')
    log_filename = f'{job_id}_train.log' if job_id else 'train.log'
    log_path = log_dir / log_filename

    fh = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    log.info("Logger initialised  →  %s", log_path)
    return log


def log_config(
    logger: logging.Logger,
    config: TrainingConfig,
    subjects_train: List[str],
    subjects_val: List[str],
    subjects_test: List[str],
) -> None:
    """Log all TrainingConfig fields and the subject splits.

    Args:
        logger:         Logger returned by :func:`setup_logger`.
        config:         Active training configuration.
        subjects_train: List of training subject IDs.
        subjects_val:   List of validation subject IDs.
        subjects_test:  List of test subject IDs.
    """
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)

    # Top-level fields
    for field_name in vars(config):
        val = getattr(config, field_name)
        # Pretty-print nested configs
        if hasattr(val, '__dataclass_fields__'):
            logger.info("  %s:", field_name)
            for sub_field in vars(val):
                logger.info("    %s = %s", sub_field, getattr(val, sub_field))
        else:
            logger.info("  %s = %s", field_name, val)

    logger.info("-" * 60)
    logger.info("Subject splits:")
    logger.info("  train (%d): %s", len(subjects_train), subjects_train)
    logger.info("  val   (%d): %s", len(subjects_val), subjects_val)
    logger.info("  test  (%d): %s", len(subjects_test), subjects_test)
    logger.info("=" * 60)


def log_epoch(
    logger: logging.Logger,
    epoch: int,
    max_epochs: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    elapsed: float,
    is_best: bool = False,
) -> None:
    """Log a one-liner summary for a completed training epoch.

    Args:
        logger:     Logger instance.
        epoch:      Current epoch number (1-indexed).
        max_epochs: Total number of epochs scheduled.
        train_loss: Mean training loss for this epoch.
        val_loss:   Mean validation loss for this epoch.
        lr:         Current learning rate.
        elapsed:    Wall-clock time for this epoch (seconds).
        is_best:    If True, appends ``*** NEW BEST ***`` to the line.
    """
    msg = (
        f"epoch {epoch:03d}/{max_epochs}  "
        f"train={train_loss:.4f}  val={val_loss:.4f}  "
        f"lr={lr:.2e}  ({elapsed:.1f}s)"
    )
    if is_best:
        msg += "  *** NEW BEST ***"
    logger.info(msg)
