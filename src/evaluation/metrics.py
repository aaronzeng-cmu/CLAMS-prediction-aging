"""
Circular phase accuracy metrics.

All functions operate on phase arrays in radians.
"""
from __future__ import annotations

import numpy as np


def mace(pred_phase: np.ndarray, true_phase: np.ndarray) -> float:
    """Mean Absolute Circular Error (MACE) in radians.

    Computes the mean of |arctan2(sin(pred-true), cos(pred-true))|,
    correctly handling the circular wrap-around at ±π.

    Args:
        pred_phase: Predicted phase array (T,) in radians.
        true_phase: Ground-truth phase array (T,) in radians.

    Returns:
        MACE in [0, π].
    """
    diff = pred_phase - true_phase
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return float(np.mean(np.abs(wrapped)))


def plv(pred_phase: np.ndarray, true_phase: np.ndarray) -> float:
    """Phase-Locking Value (PLV).

    PLV = |mean(exp(i * (pred - true)))|

    Args:
        pred_phase: Predicted phase array (T,) in radians.
        true_phase: Ground-truth phase array (T,) in radians.

    Returns:
        PLV in [0, 1]. 1 = perfect locking, 0 = random.
    """
    diff = pred_phase - true_phase
    return float(np.abs(np.mean(np.exp(1j * diff))))


def up_phase_rate(
    pred_phase: np.ndarray,
    window: tuple[float, float] = (-np.pi / 2, np.pi / 2),
) -> float:
    """Fraction of predicted phases falling within an up-phase window.

    The up-phase window defaults to (-π/2, π/2), corresponding to the
    ascending half of the slow oscillation (approaching the positive peak).

    Args:
        pred_phase: Predicted phase array (T,) in radians.
        window: (lo, hi) phase window in radians.

    Returns:
        Fraction in [0, 1].
    """
    lo, hi = window
    in_window = (pred_phase >= lo) & (pred_phase <= hi)
    return float(np.mean(in_window))


def from_cos_sin(cos_pred: np.ndarray, sin_pred: np.ndarray) -> np.ndarray:
    """Convert (cos, sin) representation to phase angle in [-π, π].

    Equivalent to MATLAB ``angle(cos + i*sin)``.

    Args:
        cos_pred: Cosine component array (T,).
        sin_pred: Sine component array (T,).

    Returns:
        Phase angle array (T,) in radians.
    """
    return np.arctan2(sin_pred, cos_pred)
