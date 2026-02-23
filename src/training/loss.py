"""
Loss functions for circular (phase) prediction in (cos, sin) space.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def circular_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss in (cos, sin) space, equivalent to 2*(1 - cos(Δθ)) per element.

    ||[cos a - cos b, sin a - sin b]||^2 = 2 - 2*cos(a - b)

    This is continuous and fully differentiable everywhere — unlike atan2-based
    losses which have discontinuities at ±π.

    Value semantics:
        0.0  = perfect prediction
        2.0  = opposite phase (maximum error)
        1.0  = 90° error

    Args:
        pred:   (B, 2, T) — predicted [cos θ; sin θ] at each timestep.
        target: (B, 2, T) — ground-truth [cos θ; sin θ].
        mask:   Optional (B, T) boolean tensor. True = include in loss.
                If None, all timesteps are included.

    Returns:
        Scalar loss tensor.
    """
    # Element-wise squared error: (B, 2, T)
    sq_err = (pred - target) ** 2

    # Mean over the 2 (cos/sin) dimension: (B, T)
    # This implements ||[cos a - cos b, sin a - sin b]||^2 / 2 = 1 - cos(a-b)
    per_step = sq_err.mean(dim=1)

    if mask is not None:
        # mask: (B, T), True = valid
        per_step = per_step * mask.float()
        n_valid = mask.float().sum().clamp(min=1.0)
        return per_step.sum() / n_valid
    else:
        return per_step.mean()


def unit_norm_regularizer(
    pred: torch.Tensor,
    weight: float = 0.01,
) -> torch.Tensor:
    """Penalize deviation of predicted vectors from unit norm.

    Encourages the network to output vectors on the unit circle, which
    makes the output directly interpretable as [cos θ, sin θ].

    Args:
        pred:   (B, 2, T) — predicted [cos θ; sin θ].
        weight: Regularization strength.

    Returns:
        Scalar regularization loss.
    """
    norms = pred.norm(dim=1)        # (B, T)
    deviation = (norms - 1.0) ** 2  # penalize ||v|| != 1
    return weight * deviation.mean()
