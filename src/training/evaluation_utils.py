"""
Internal helpers for generating evaluation figures during training.
"""
from __future__ import annotations

import numpy as np
import torch


def _run_model_on_features(model, features: np.ndarray, device: str) -> np.ndarray:
    """Run a trained model on a precomputed feature matrix.

    Processes the full recording in a single forward pass (batch_size=1).
    The model is run in eval mode with no gradient tracking.

    Args:
        model:    Trained PhasePredictor (LSTM or TCN).
        features: Feature matrix of shape ``(C, T)``.
        device:   Torch device string.

    Returns:
        Predicted [cos, sin] matrix of shape ``(2, T)``.
    """
    model.eval()
    dev = torch.device(device)
    x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(dev)  # (1, C, T)
    with torch.no_grad():
        out = model(x)   # (1, 2, T)
    return out.squeeze(0).cpu().numpy()   # (2, T)
