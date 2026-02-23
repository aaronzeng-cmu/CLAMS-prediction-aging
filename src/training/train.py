"""
Training loop for CLAMS phase predictors.
Supports LSTM and TCN models, early stopping, and per-shift checkpointing.
"""
from __future__ import annotations

import copy
import csv
import logging
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional

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
    logger: Optional[logging.Logger] = None,
    run_dir: Optional[str] = None,
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
        logger: Optional logger (falls back to print if None).
        run_dir: Optional run directory for saving figures alongside models.

    Returns:
        Trained model (best validation checkpoint).
    """
    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
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

    # Model directory: prefer run_dir/models/ over config.model_dir
    model_dir = os.path.join(run_dir, 'models') if run_dir else config.model_dir
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(
        model_dir,
        f"{config.model_type}_shift_{shift_ms}ms_checkpoint.pt",
    )
    log_path = os.path.join(
        model_dir,
        f"metrics_{config.model_type}_shift_{shift_ms}ms.csv",
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

        is_best = val_loss < best_val_loss
        log_rows.append({
            'epoch': epoch,
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss:.6f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            'elapsed_s': f'{elapsed:.1f}',
        })

        if logger is not None:
            from .logging_utils import log_epoch
            log_epoch(
                logger, epoch, config.max_epochs,
                train_loss, val_loss, scheduler.get_last_lr()[0],
                elapsed, is_best=is_best,
            )
        else:
            best_tag = '  *** NEW BEST ***' if is_best else ''
            print(
                f"[shift={shift_ms}ms | epoch {epoch:03d}/{config.max_epochs}] "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.1e}  ({elapsed:.1f}s){best_tag}"
            )

        # Early stopping
        if is_best:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save({'state_dict': best_state, 'config': config, 'shift_ms': shift_ms},
                       ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                _log(f"Early stopping at epoch {epoch} (patience={config.patience})")
                break

    # Write CSV log
    if log_rows:
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

        # Save training-curves figure if run_dir provided
        if run_dir is not None:
            try:
                from . import figures
                fig_path = os.path.join(
                    run_dir, 'figures', 'training',
                    f"loss_curves_shift_{shift_ms}ms.png",
                )
                figures.save_training_curves(
                    log_path, fig_path,
                    title=f"{config.model_type.upper()} shift={shift_ms}ms",
                )
            except Exception as exc:
                _log(f"Warning: could not save training curves figure: {exc}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_all_shifts(
    file_paths: List[str],
    config: TrainingConfig,
    device: str = 'cpu',
    logger: Optional[logging.Logger] = None,
    run_dir: Optional[str] = None,
    val_records=None,
) -> Dict[int, PhasePredictor]:
    """Train and export a model for each shift value in config.shift_ms_list.

    Args:
        file_paths:  Paths to all EEG subject files.
        config:      TrainingConfig.
        device:      Torch device string.
        logger:      Optional logger (falls back to print if None).
        run_dir:     Optional run directory for figures / model storage.
        val_records: Optional list of ``(subject, run_num)`` or file paths
                     used to generate evaluation figures after each shift.

    Returns:
        Dictionary mapping shift_ms -> trained model.
    """
    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    models = {}
    for shift_ms in config.shift_ms_list:
        _log(f"\n{'='*60}")
        _log(f"Training {config.model_type.upper()} for shift={shift_ms}ms")
        _log(f"{'='*60}")
        model = train_one_model(
            file_paths, config, shift_ms,
            device=device, logger=logger, run_dir=run_dir,
        )
        models[shift_ms] = model

        # Export immediately after training
        model_dir = os.path.join(run_dir, 'models') if run_dir else config.model_dir
        os.makedirs(model_dir, exist_ok=True)
        output_base = os.path.join(model_dir, f"{config.model_type}_shift_{shift_ms}ms")
        if config.model_type == 'lstm':
            export_lstm_to_mat(model, config.lstm_config, output_base + '.mat')
        elif config.model_type == 'tcn':
            export_tcn_to_onnx(model, config.tcn_config, output_base + '.onnx')

        # Generate evaluation figures from val_records if provided
        if val_records is not None and run_dir is not None:
            _generate_eval_figures(
                model, val_records, config, shift_ms,
                device, run_dir, logger,
            )

    return models


def _generate_eval_figures(
    model: PhasePredictor,
    val_records,
    config: TrainingConfig,
    shift_ms: int,
    device: str,
    run_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate prediction sample and error histogram figures for val records."""
    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    try:
        import numpy as np
        from . import figures
        from .evaluation_utils import _run_model_on_features
        from .features import build_feature_matrix_offline
        from .labels import generate_labels
        from .config import LabelConfig
        from ..evaluation.metrics import (
            mace as compute_mace, plv as compute_plv, from_cos_sin,
        )

        label_cfg = LabelConfig(
            shift_ms=shift_ms,
            label_bp_lo=config.label_config.label_bp_lo,
            label_bp_hi=config.label_config.label_bp_hi,
            label_bp_order=config.label_config.label_bp_order,
        )

        all_true, all_pred = [], []

        # val_records may be file paths (str) or (subject, run_num) tuples
        for record in val_records[:3]:  # limit to 3 records for speed
            try:
                if isinstance(record, str):
                    import numpy as _np
                    eeg = _np.load(record)['eeg'].astype(np.float64).ravel()
                else:
                    from ..data.data_utils import load_eeg
                    eeg = load_eeg(record[0], record[1],
                                   channel=config.data_config.eeg_channel,
                                   target_fs=config.feature_config.fs)
            except Exception as exc:
                _log(f"  Warning: could not load val record {record}: {exc}")
                continue

            features = build_feature_matrix_offline(eeg, config.feature_config)
            labels = generate_labels(eeg, config.feature_config.fs, label_cfg)

            pred_cos_sin = _run_model_on_features(model, features, device)
            valid = ~np.isnan(labels[0])
            if not valid.any():
                continue
            true_phase = from_cos_sin(labels[0][valid], labels[1][valid])
            pred_phase = from_cos_sin(pred_cos_sin[0][valid], pred_cos_sin[1][valid])
            all_true.append(true_phase)
            all_pred.append(pred_phase)

        if not all_true:
            return

        true_all = np.concatenate(all_true)
        pred_all = np.concatenate(all_pred)
        mace_val = compute_mace(pred_all, true_all)
        plv_val = compute_plv(pred_all, true_all)

        eval_dir = os.path.join(run_dir, 'figures', 'evaluation')

        figures.save_prediction_sample(
            true_all, pred_all,
            fs=config.feature_config.fs,
            shift_ms=shift_ms,
            mace_val=mace_val,
            plv_val=plv_val,
            out_path=os.path.join(eval_dir, f"prediction_sample_shift_{shift_ms}ms.png"),
        )

        errors_deg = np.degrees(
            np.arctan2(np.sin(pred_all - true_all), np.cos(pred_all - true_all))
        )
        figures.save_phase_error_hist(
            errors_deg,
            mace_deg=float(np.degrees(mace_val)),
            plv_val=plv_val,
            out_path=os.path.join(eval_dir, f"phase_error_hist_shift_{shift_ms}ms.png"),
        )
        _log(f"  Eval figures saved (shift={shift_ms}ms, MACE={np.degrees(mace_val):.1f}°, PLV={plv_val:.3f})")

    except Exception as exc:
        if logger is not None:
            logger.warning("Could not generate eval figures for shift=%dms: %s", shift_ms, exc)
        else:
            print(f"Warning: eval figures failed for shift={shift_ms}ms: {exc}")
