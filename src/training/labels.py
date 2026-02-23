"""
Offline ground-truth phase generation.
No PyTorch dependency — pure NumPy / SciPy.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from .config import LabelConfig


def bandpass_filtfilt(
    signal: np.ndarray,
    fs: int,
    lo: float,
    hi: float,
    order: int,
) -> np.ndarray:
    """Zero-phase bandpass filter using SOS cascade.

    Uses butter(..., output='sos') + sosfiltfilt to avoid BA numerical
    instability at low frequencies (critical for the 0.4 Hz lower edge).

    Args:
        signal: 1-D array of shape (T,).
        fs: Sample rate in Hz.
        lo: Lower cutoff (Hz).
        hi: Upper cutoff (Hz).
        order: Butterworth filter order.

    Returns:
        Filtered signal, same shape as input.
    """
    sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)


def compute_instantaneous_phase(filtered_signal: np.ndarray) -> np.ndarray:
    """Compute analytic signal via Hilbert transform, return phase in [-pi, pi].

    Args:
        filtered_signal: 1-D real-valued array (T,).

    Returns:
        Instantaneous phase in radians, shape (T,).
    """
    analytic = hilbert(filtered_signal)
    return np.angle(analytic)


def generate_labels(
    eeg: np.ndarray,
    fs: int,
    cfg: LabelConfig,
) -> np.ndarray:
    """Generate ground-truth [cos, sin] phase labels with a forward shift.

    The label at time t encodes the phase at t + shift_samples:
        label[t] = [cos(phase(t + shift_samples)), sin(phase(t + shift_samples))]

    The last `shift_samples` entries are set to NaN because no future ground
    truth is available; the dataset loader masks these windows.

    Args:
        eeg: 1-D raw EEG array of shape (T,).
        fs: Sample rate in Hz.
        cfg: LabelConfig with shift_ms and filter parameters.

    Returns:
        Array of shape (2, T): [cos(phase); sin(phase)].
        Last shift_samples columns are NaN.
    """
    shift_samples = round(cfg.shift_ms * fs / 1000)

    # Offline zero-phase bandpass (filtfilt) — separate from online feature filter
    filtered = bandpass_filtfilt(eeg, fs, cfg.label_bp_lo, cfg.label_bp_hi, cfg.label_bp_order)
    phase = compute_instantaneous_phase(filtered)   # (T,) in [-pi, pi]

    T = len(eeg)
    cos_labels = np.full(T, np.nan, dtype=np.float32)
    sin_labels = np.full(T, np.nan, dtype=np.float32)

    if shift_samples < T:
        # label[t] = phase(t + shift_samples)  =>  valid for t in [0, T-shift_samples)
        cos_labels[:T - shift_samples] = np.cos(phase[shift_samples:]).astype(np.float32)
        sin_labels[:T - shift_samples] = np.sin(phase[shift_samples:]).astype(np.float32)

    return np.stack([cos_labels, sin_labels], axis=0)   # (2, T)
