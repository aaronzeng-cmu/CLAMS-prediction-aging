"""Sequence-to-sequence LSTM phase predictor."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .base import PhasePredictor
from ..config import LSTMConfig


class PhaseLSTM(PhasePredictor):
    """Seq2seq LSTM: (B, C, T) -> (B, 2, T).

    Architecture:
        nn.LSTM (input_size=C, hidden_size=H, num_layers=L, batch_first=True)
        nn.Linear(H, 2)

    Gate ordering is identical to MATLAB lstmLayer:
        PyTorch weight_ih layout: [i|f|g|o]  rows (0:H, H:2H, 2H:3H, 3H:4H)
        MATLAB InputWeights layout: [i|f|g|o] rows — NO REORDERING NEEDED

    Hidden state is zero-initialized per batch during training (not carried
    across batches). For online inference use predictAndUpdateState pattern.
    """

    def __init__(self, cfg: LSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(cfg.hidden_size, cfg.output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, T)

        Returns:
            (B, 2, T) — raw [cos, sin] without activation.
        """
        # (B, C, T) -> (B, T, C) for batch_first LSTM
        x = x.permute(0, 2, 1)

        # h0, c0 default to zeros — no hidden state across batches
        out, _ = self.lstm(x)           # (B, T, H)
        out = self.fc(out)              # (B, T, 2)

        # (B, T, 2) -> (B, 2, T)
        return out.permute(0, 2, 1)
