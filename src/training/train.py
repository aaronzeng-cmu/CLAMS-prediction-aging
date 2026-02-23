"""
Training loop for CLAMS phase predictors.
Supports LSTM and TCN models, early stopping, and per-shift checkpointing.
"""
from __future__ import annotations

import copy
import csv
import os
import time
from dataclasses import replace
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import LabelConfig, TrainingConfig
from .dataset import make_dataloaders
from .export import export_lstm_to_mat, export_tcn_to_onnx
from .loss import circular_mse_loss
from .models import PhasePredictor, build_model


def _set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_model(
    file_paths: List[str],
    config: TrainingConfig,
    shift_ms: int,
    device: str = 'cpu',
) -> PhasePredictor:
    """Train a single model for one shift value.

    Steps:
        1. Override label_config.shift_ms = shift_ms
        2. Build data loaders (subject-level split)
        3. Build model from config
        4. Train with AdamW + CosineAnnealingLR + early stopping
        5. Save checkpoint to models/{model_type}_shift_{shift_ms}ms_checkpoint.pt
        6. Write CSV training log

    Args:
        file_paths: Paths to all EEG subject files.
        config: TrainingConfig (label_config.shift_ms overridden internally).
        shift_ms: Prediction shift in milliseconds.
        device: Torch device string ('cpu', 'cuda', etc.).

    Returns:
        Trained model (best validation checkpoint).
    """
    _set_seed(config.seed)
    dev = torch.device(device)

    # Override shift_ms in label config
    label_cfg = LabelConfig(
        shift_ms=shift_ms,
        label_bp_lo=config.label_config.label_bp_lo,
        label_bp_hi=config.label_config.label_bp_hi,
        label_bp_order=config.label_config.label_bp_order,
    )

    train_loader, val_loader, _ = make_dataloaders(
        file_paths,
        config.feature_config,
        label_cfg,
        config,
    )

    model = build_model(config).to(dev)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=1e-5)

    os.makedirs(config.model_dir, exist_ok=True)
    ckpt_path = os.path.join(
        config.model_dir,
        f"{config.model_type}_shift_{shift_ms}ms_checkpoint.pt",
    )
    log_path = os.path.join(
        config.model_dir,
        f"training_log_{config.model_type}_shift_{shift_ms}ms.csv",
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    log_rows: list[dict] = []

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for X, Y in train_loader:
            X, Y = X.to(dev), Y.to(dev)
            optimizer.zero_grad()
            pred = model(X)                      # (B, 2, T)
            loss = circular_mse_loss(pred, Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / max(train_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(dev), Y.to(dev)
                pred = model(X)
                loss = circular_mse_loss(pred, Y)
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        elapsed = time.time() - t0

        log_rows.append({
            'epoch': epoch,
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss:.6f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            'elapsed_s': f'{elapsed:.1f}',
        })
        print(
            f"[shift={shift_ms}ms | epoch {epoch:03d}/{config.max_epochs}] "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.1e}  ({elapsed:.1f}s)"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save({'state_dict': best_state, 'config': config, 'shift_ms': shift_ms},
                       ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch} (patience={config.patience})")
                break

    # Write CSV log
    if log_rows:
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_all_shifts(
    file_paths: List[str],
    config: TrainingConfig,
    device: str = 'cpu',
) -> Dict[int, PhasePredictor]:
    """Train and export a model for each shift value in config.shift_ms_list.

    Args:
        file_paths: Paths to all EEG subject files.
        config: TrainingConfig.
        device: Torch device string.

    Returns:
        Dictionary mapping shift_ms -> trained model.
    """
    models = {}
    for shift_ms in config.shift_ms_list:
        print(f"\n{'='*60}")
        print(f"Training {config.model_type.upper()} for shift={shift_ms}ms")
        print(f"{'='*60}")
        model = train_one_model(file_paths, config, shift_ms, device=device)
        models[shift_ms] = model

        # Export immediately after training
        output_base = os.path.join(
            config.model_dir,
            f"{config.model_type}_shift_{shift_ms}ms",
        )
        if config.model_type == 'lstm':
            export_lstm_to_mat(model, config.lstm_config, output_base + '.mat')
        elif config.model_type == 'tcn':
            export_tcn_to_onnx(model, config.tcn_config, output_base + '.onnx')

    return models
