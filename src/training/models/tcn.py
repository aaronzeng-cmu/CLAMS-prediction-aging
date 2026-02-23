"""Temporal Convolutional Network (TCN) phase predictor.

Causality guaranteed by left-only (causal) padding — future samples
never influence past outputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import PhasePredictor
from ..config import TCNConfig


class CausalConv1d(nn.Module):
    """1-D convolution with left-only (causal) padding.

    Pads (kernel_size - 1) * dilation zeros on the left, zero on the right,
    then applies a standard Conv1d with padding=0.

    This ensures output[:, :, t] depends only on input[:, :, :t+1].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Left-only pad: (left, right) = (self.pad, 0)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual dilated causal convolutional block.

    Structure:
        CausalConv1d -> BN -> ReLU -> Dropout ->
        CausalConv1d -> BN -> ReLU -> Dropout
        + residual (1×1 conv if in_channels != out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.net(x) + residual


class PhaseTCN(PhasePredictor):
    """Dilated causal TCN: (B, C, T) -> (B, 2, T).

    Receptive field (default config):
        (kernel_size - 1) * sum(2^i for i in range(n_blocks)) + 1
        = (3-1) * (1+2+4+8) + 1 = 31 samples = 155 ms at 200 Hz

    Dilation doubles per block: 1, 2, 4, 8, ...
    All convolutions are causally padded — no future leakage.
    """

    def __init__(self, cfg: TCNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers = []
        in_ch = cfg.input_size
        for i, out_ch in enumerate(cfg.num_channels):
            dilation = 2 ** i
            layers.append(
                TCNBlock(in_ch, out_ch, cfg.kernel_size, dilation, cfg.dropout)
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.output_conv = nn.Conv1d(in_ch, cfg.output_size, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, T)

        Returns:
            (B, 2, T) — fully causal.
        """
        out = self.network(x)           # (B, last_ch, T)
        return self.output_conv(out)    # (B, 2, T)
