"""
Debug and evaluation figure generation for CLAMS training runs.

All figures are saved as 150-dpi PNG files and rendered headlessly
(no display required — safe for SLURM cluster nodes).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')   # Must be before any other matplotlib import

import matplotlib.pyplot as plt
import numpy as np

_DPI = 150


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Feature sample
# ---------------------------------------------------------------------------

def save_feature_sample(
    eeg: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    fs: int,
    shift_ms: int,
    title: str,
    out_path: str,
    window_sec: float = 30.0,
) -> None:
    """Save a 4-row diagnostic figure showing a 30-second data window.

    Rows:
        1. Raw EEG (row 0 of *features*)
        2. First-difference EEG (row 1 of *features*)
        3. Bandpass-filtered EEG (row 2 of *features*)
        4. cos and sin labels (NaN region shaded in red)

    Args:
        eeg:        1-D raw EEG array (T,).  Used only for its length.
        features:   Feature matrix (C, T).
        labels:     Label matrix (2, T) — [cos, sin] rows.
        fs:         Sample rate (Hz).
        shift_ms:   Label shift in milliseconds (for annotation).
        title:      Figure suptitle string.
        out_path:   Output PNG path.
        window_sec: Window length to display (seconds).  Default 30 s.
    """
    _ensure_dir(out_path)
    T = features.shape[1]
    n_samples = min(int(window_sec * fs), T)
    t = np.arange(n_samples) / fs  # time axis (s)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=11)

    # Row 0 — raw EEG
    ax = axes[0]
    ax.plot(t, features[0, :n_samples], lw=0.6, color='steelblue')
    ax.set_ylabel('Raw EEG (µV)')
    ax.grid(True, alpha=0.3)

    # Row 1 — first difference
    ax = axes[1]
    ax.plot(t, features[1, :n_samples], lw=0.6, color='darkorange')
    ax.set_ylabel('Diff EEG')
    ax.grid(True, alpha=0.3)

    # Row 2 — bandpass
    ax = axes[2]
    ax.plot(t, features[2, :n_samples], lw=0.6, color='forestgreen')
    ax.set_ylabel('Bandpass (0.4–1.2 Hz)')
    ax.grid(True, alpha=0.3)

    # Row 3 — labels
    ax = axes[3]
    cos_labels = labels[0, :n_samples]
    sin_labels = labels[1, :n_samples]
    ax.plot(t, cos_labels, lw=0.7, color='royalblue', label=f'cos(θ+{shift_ms}ms)')
    ax.plot(t, sin_labels, lw=0.7, color='tomato', label=f'sin(θ+{shift_ms}ms)')
    # Shade NaN regions
    nan_mask = np.isnan(cos_labels) | np.isnan(sin_labels)
    if nan_mask.any():
        ax.fill_between(t, -1.2, 1.2, where=nan_mask,
                        color='salmon', alpha=0.4, label='NaN (boundary)')
    ax.set_ylim(-1.3, 1.3)
    ax.set_ylabel('Label value')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Label distribution
# ---------------------------------------------------------------------------

def save_label_distribution(
    all_cos: np.ndarray,
    all_sin: np.ndarray,
    out_path: str,
) -> None:
    """Save a 2-panel figure showing the distribution of label phases.

    Panel 1: Polar histogram (36 bins = 10° per bin).
    Panel 2: Scatter of (cos, sin) on the unit circle (sub-sampled).

    Args:
        all_cos: Flattened cosine label values (after NaN removal).
        all_sin: Flattened sine label values (after NaN removal).
        out_path: Output PNG path.
    """
    _ensure_dir(out_path)
    phases = np.arctan2(all_sin, all_cos)

    fig = plt.figure(figsize=(10, 4))

    # Panel 1 — polar histogram
    ax1 = fig.add_subplot(121, projection='polar')
    bins = np.linspace(-np.pi, np.pi, 37)
    counts, _ = np.histogram(phases, bins=bins)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    ax1.bar(bin_centres, counts, width=width, color='steelblue',
            edgecolor='white', alpha=0.8)
    ax1.set_title('Phase distribution (polar histogram)', pad=12, fontsize=10)

    # Panel 2 — unit-circle scatter (subsample for speed)
    ax2 = fig.add_subplot(122)
    n = min(len(all_cos), 5000)
    idx = np.random.choice(len(all_cos), n, replace=False)
    ax2.scatter(all_cos[idx], all_sin[idx], s=2, alpha=0.3, color='steelblue')
    theta = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, label='unit circle')
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_xlabel('cos θ')
    ax2.set_ylabel('sin θ')
    ax2.set_title('(cos, sin) scatter', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Training curves
# ---------------------------------------------------------------------------

def save_training_curves(
    metrics_csv_path: str,
    out_path: str,
    title: str = '',
) -> None:
    """Save a 2-panel training/validation loss + LR figure.

    Reads the CSV written by ``train_one_model`` (columns: epoch,
    train_loss, val_loss, lr, elapsed_s).

    Panel 1: train and val loss vs epoch; best epoch shown as dashed line.
    Panel 2: LR vs epoch (log-scale y-axis).

    Args:
        metrics_csv_path: Path to the training-log CSV.
        out_path:         Output PNG path.
        title:            Optional figure suptitle.
    """
    import csv
    _ensure_dir(out_path)

    epochs, train_losses, val_losses, lrs = [], [], [], []
    with open(metrics_csv_path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            lrs.append(float(row['lr']))

    if not epochs:
        return

    best_epoch = epochs[int(np.argmin(val_losses))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if title:
        fig.suptitle(title, fontsize=11)

    # Panel 1 — losses
    ax1.plot(epochs, train_losses, label='train', color='steelblue', lw=1.2)
    ax1.plot(epochs, val_losses, label='val', color='tomato', lw=1.2)
    ax1.axvline(best_epoch, color='grey', linestyle='--', lw=0.8,
                label=f'best epoch {best_epoch}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Circular MSE loss')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Loss curves')

    # Panel 2 — LR
    ax2.plot(epochs, lrs, color='darkorange', lw=1.2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning rate')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_title('Learning rate schedule')

    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Prediction sample
# ---------------------------------------------------------------------------

def save_prediction_sample(
    true_phase: np.ndarray,
    pred_phase: np.ndarray,
    fs: int,
    shift_ms: int,
    mace_val: float,
    plv_val: float,
    out_path: str,
    window_sec: float = 60.0,
) -> None:
    """Save a 2-row figure comparing true and predicted phase over time.

    Row 1: True phase (blue) and predicted phase (orange) vs time.
    Row 2: Absolute circular error in degrees vs time.

    Args:
        true_phase: Ground-truth phase array (T,) in radians.
        pred_phase: Predicted phase array (T,) in radians.
        fs:         Sample rate (Hz).
        shift_ms:   Shift value in milliseconds (for annotation).
        mace_val:   Mean absolute circular error in radians.
        plv_val:    Phase-locking value.
        out_path:   Output PNG path.
        window_sec: Window length to display.  Default 60 s.
    """
    _ensure_dir(out_path)
    T = len(true_phase)
    n_samples = min(int(window_sec * fs), T)
    t = np.arange(n_samples) / fs

    error_deg = np.degrees(
        np.abs(np.arctan2(
            np.sin(pred_phase[:n_samples] - true_phase[:n_samples]),
            np.cos(pred_phase[:n_samples] - true_phase[:n_samples]),
        ))
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(
        f"Prediction sample — shift={shift_ms}ms  "
        f"MACE={np.degrees(mace_val):.1f}°  PLV={plv_val:.3f}",
        fontsize=10,
    )

    ax1.plot(t, np.degrees(true_phase[:n_samples]),
             lw=0.7, color='royalblue', label='true phase')
    ax1.plot(t, np.degrees(pred_phase[:n_samples]),
             lw=0.7, color='darkorange', alpha=0.8, label='predicted phase')
    ax1.set_ylabel('Phase (°)')
    ax1.set_ylim(-200, 200)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, error_deg, lw=0.7, color='tomato')
    ax2.axhline(np.degrees(mace_val), color='grey', linestyle='--', lw=0.8,
                label=f'MACE = {np.degrees(mace_val):.1f}°')
    ax2.set_ylabel('|error| (°)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(0, 185)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Phase error histogram
# ---------------------------------------------------------------------------

def save_phase_error_hist(
    errors_deg: np.ndarray,
    mace_deg: float,
    plv_val: float,
    out_path: str,
) -> None:
    """Save a 2-panel figure showing the distribution of phase errors.

    Panel 1: Histogram of |error| in degrees with cumulative-density overlay.
    Panel 2: Polar histogram of signed error (36 bins).

    Args:
        errors_deg: Array of signed circular errors in degrees (T,).
        mace_deg:   Mean absolute circular error in degrees.
        plv_val:    Phase-locking value.
        out_path:   Output PNG path.
    """
    _ensure_dir(out_path)

    abs_errors = np.abs(errors_deg)

    fig = plt.figure(figsize=(11, 4))
    fig.suptitle(
        f"Phase error distribution — MACE={mace_deg:.1f}°  PLV={plv_val:.3f}",
        fontsize=10,
    )

    # Panel 1 — |error| histogram with cumulative overlay
    ax1 = fig.add_subplot(121)
    n_bins = 36
    counts, bin_edges, _ = ax1.hist(abs_errors, bins=n_bins, range=(0, 180),
                                     color='steelblue', edgecolor='white',
                                     alpha=0.8, density=True, label='density')
    ax1.axvline(mace_deg, color='tomato', linestyle='--', lw=1.2,
                label=f'MACE = {mace_deg:.1f}°')

    # Cumulative on right y-axis
    ax1r = ax1.twinx()
    sorted_err = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax1r.plot(sorted_err, cdf, color='darkorange', lw=1.2, label='CDF')
    ax1r.set_ylabel('Cumulative fraction', color='darkorange')
    ax1r.tick_params(axis='y', labelcolor='darkorange')
    ax1r.set_ylim(0, 1.05)

    ax1.set_xlabel('|Phase error| (°)')
    ax1.set_ylabel('Density')
    ax1.set_xlim(0, 180)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2 — polar histogram of signed error
    ax2 = fig.add_subplot(122, projection='polar')
    bins = np.linspace(-np.pi, np.pi, 37)
    signed_rad = np.deg2rad(errors_deg)
    # Wrap to [-pi, pi]
    signed_rad = np.arctan2(np.sin(signed_rad), np.cos(signed_rad))
    hist_counts, _ = np.histogram(signed_rad, bins=bins)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    ax2.bar(bin_centres, hist_counts, width=width, color='steelblue',
            edgecolor='white', alpha=0.8)
    ax2.set_title('Signed error (polar)', pad=12, fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
