"""
Causal IIR feature extraction.
Mirrors MATLAB filter(b, a, x, zi) exactly using scipy.signal.sosfilt with
persistent state — training-inference consistency.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .config import FeatureConfig


class CausalBandpassFilter:
    """Stateful causal (one-pass) IIR bandpass filter.

    Uses SOS cascade for numerical stability at low frequencies.
    State is zero-initialized (not sosfilt_zi steady-state) because
    EEG has arbitrary DC offset at recording start.

    Mirrors ``[y, z] = filter(b, a, x, z)`` in MATLAB.
    """

    def __init__(self, fs: int, lo: float, hi: float, order: int) -> None:
        self._sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
        self._zi: np.ndarray | None = None   # initialized lazily on first apply()
        self.reset()

    def reset(self) -> None:
        """Zero-initialize filter state (not steady-state)."""
        n_sections = self._sos.shape[0]
        self._zi = np.zeros((n_sections, 2), dtype=np.float64)

    def apply(self, chunk: np.ndarray) -> np.ndarray:
        """Apply causal filter to chunk, updating internal state.

        Args:
            chunk: 1-D array of shape (N,).

        Returns:
            Filtered output, shape (N,).
        """
        out, self._zi = sosfilt(self._sos, chunk.astype(np.float64), zi=self._zi)
        return out.astype(np.float32)


def extract_features(
    chunk_with_prev: np.ndarray,
    bp_filter: CausalBandpassFilter,
    feature_names: tuple,
) -> np.ndarray:
    """Extract causal EEG features from a chunk with one previous sample prepended.

    Args:
        chunk_with_prev: 1-D array of length T+1.
            chunk_with_prev[0] is the sample from the previous chunk (for diff).
            chunk_with_prev[1:] are the T new samples.
        bp_filter: Stateful CausalBandpassFilter (state updated in place).
        feature_names: Ordered tuple of feature names to compute.
            Supported: 'raw', 'diff', 'bandpass', 'so_envelope'.

    Returns:
        Feature array of shape (C, T) where C = len(feature_names).
    """
    raw = chunk_with_prev[1:].astype(np.float32)    # (T,)
    T = len(raw)

    feature_map: dict[str, np.ndarray] = {}

    if 'raw' in feature_names:
        feature_map['raw'] = raw

    if 'diff' in feature_names:
        feature_map['diff'] = np.diff(chunk_with_prev.astype(np.float32))  # (T,)

    if 'bandpass' in feature_names or 'so_envelope' in feature_names:
        bp = bp_filter.apply(raw)
        if 'bandpass' in feature_names:
            feature_map['bandpass'] = bp
        if 'so_envelope' in feature_names:
            # Causal RMS envelope with window=100 samples
            window = 100
            sq = bp ** 2
            # Cumulative sum approach for causal sliding RMS
            envelope = np.zeros(T, dtype=np.float32)
            for i in range(T):
                start = max(0, i - window + 1)
                envelope[i] = np.sqrt(np.mean(sq[start:i + 1]))
            feature_map['so_envelope'] = envelope

    rows = [feature_map[name] for name in feature_names]
    return np.stack(rows, axis=0)   # (C, T)


def build_feature_matrix_offline(
    eeg: np.ndarray,
    cfg: FeatureConfig,
) -> np.ndarray:
    """Build feature matrix for a full recording using a single causal filter pass.

    Training-time function that applies lfilter sequentially across the entire
    recording to produce features consistent with online inference.

    Args:
        eeg: 1-D raw EEG array of shape (T,).
        cfg: FeatureConfig.

    Returns:
        Feature array of shape (C, T).
    """
    bp_filter = CausalBandpassFilter(cfg.fs, cfg.bp_lo, cfg.bp_hi, cfg.bp_order)

    # Prepend a zero as the "previous sample" for the first chunk
    chunk_with_prev = np.concatenate([[0.0], eeg])
    return extract_features(chunk_with_prev, bp_filter, cfg.feature_names)
