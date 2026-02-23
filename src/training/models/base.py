"""Abstract base class for all phase predictor models."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class PhasePredictor(nn.Module, ABC):
    """Base class for causal EEG phase predictors.

    All subclasses must implement a fully causal forward pass:
    output at timestep t may only depend on input at timesteps ≤ t.

    Input/output contract (matches MATLAB predictAndUpdateState):
        x: (B, C, T)  — batch of EEG feature sequences
            C channels: [raw, diff, bandpass] (or augmented set)
        returns: (B, 2, T)  — [cos θ; sin θ] at each timestep
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Causal forward pass.

        Args:
            x: (B, C, T) input feature tensor.

        Returns:
            (B, 2, T) predicted [cos θ, sin θ] at each timestep.
        """
        ...

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
