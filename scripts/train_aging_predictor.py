#!/usr/bin/env python
"""
Main CLI entry point for training CLAMS phase predictors on the ORCD cluster.

Usage (local smoke-test):
    python scripts/train_aging_predictor.py --debug --output_dir output/smoke_test/

Usage (cluster):
    python scripts/train_aging_predictor.py \
        --age_group aging --model_type lstm \
        --shift_ms 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 \
        --output_dir output/
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure src/ is on the Python path when running as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


# ---------------------------------------------------------------------------
# Default shift list (all 22 values: 0, 5, 10, …, 105 ms)
# ---------------------------------------------------------------------------
_ALL_SHIFTS = list(range(0, 110, 5))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train CLAMS aging phase predictors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--age_group', default='aging',
                   choices=['aging', 'young', 'all'],
                   help="Subject age group to train on")
    p.add_argument('--subjects', nargs='+', default=None,
                   help="Override subject list (space-separated IDs)")
    p.add_argument('--model_type', default='lstm',
                   choices=['lstm', 'tcn'],
                   help="Model architecture")
    p.add_argument('--shift_ms', nargs='+', type=int, default=_ALL_SHIFTS,
                   help="Prediction shift values in ms")
    p.add_argument('--output_dir', default='output/',
                   help="Root output directory")
    p.add_argument('--config', default=None,
                   help="Optional YAML config to load (CLI args override)")
    p.add_argument('--data_dir', default=None,
                   help="Override AGING_DATA_DIR env var")
    p.add_argument('--channel', type=int, default=16,
                   help="EEG channel index (0-indexed; 16 = Fpz)")
    p.add_argument('--device', default='cpu',
                   choices=['cpu', 'cuda'],
                   help="Torch device")
    p.add_argument('--max_epochs', type=int, default=100,
                   help="Maximum training epochs per shift")
    p.add_argument('--batch_size', type=int, default=32,
                   help="Mini-batch size")
    p.add_argument('--debug', action='store_true',
                   help="Smoke-test mode: 3 subjects, 5 epochs")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Run-directory naming
# ---------------------------------------------------------------------------

def _make_run_dir(output_dir: str, model_type: str, age_group: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{ts}_{model_type}_{age_group}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Synthetic data helpers (for --debug when no real data available)
# ---------------------------------------------------------------------------

def _make_synthetic_npz(path: str, n_seconds: int = 120, fs: int = 200) -> None:
    """Write a tiny synthetic EEG NPZ file (0.75 Hz sine + noise)."""
    rng = np.random.default_rng(42)
    T = n_seconds * fs
    t = np.arange(T) / fs
    eeg = np.sin(2 * np.pi * 0.75 * t) + 0.3 * rng.standard_normal(T)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, eeg=eeg.astype(np.float32))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Build TrainingConfig (YAML base + CLI overrides)
    # ------------------------------------------------------------------
    from src.training.config import (
        DataConfig, FeatureConfig, LabelConfig, LSTMConfig, TCNConfig,
        TrainingConfig, config_from_yaml, config_to_yaml,
    )

    if args.config:
        config = config_from_yaml(args.config)
    else:
        config = TrainingConfig()

    # Apply CLI overrides
    data_dir = args.data_dir or os.environ.get('AGING_DATA_DIR', config.data_config.data_dir)
    data_cfg = DataConfig(
        data_dir=data_dir,
        eeg_channel=args.channel,
        source_fs=config.data_config.source_fs,
        age_group=args.age_group,
    )

    max_epochs = 5 if args.debug else args.max_epochs

    config = TrainingConfig(
        model_type=args.model_type,
        shift_ms_list=args.shift_ms,
        batch_size=args.batch_size,
        seq_len=config.seq_len,
        learning_rate=config.learning_rate,
        max_epochs=max_epochs,
        patience=config.patience,
        seed=config.seed,
        clip_grad_norm=config.clip_grad_norm,
        model_dir=config.model_dir,
        output_dir=args.output_dir,
        data_config=data_cfg,
        feature_config=config.feature_config,
        label_config=config.label_config,
        lstm_config=config.lstm_config,
        tcn_config=config.tcn_config,
    )

    # ------------------------------------------------------------------
    # 2. Create run directory + logger
    # ------------------------------------------------------------------
    run_dir = _make_run_dir(args.output_dir, args.model_type, args.age_group)

    from src.training.logging_utils import setup_logger, log_config
    logger = setup_logger(run_dir)
    logger.info("Run directory: %s", run_dir)
    logger.info("Debug mode: %s", args.debug)

    # Save config
    config_path = os.path.join(run_dir, 'config.yaml')
    config_to_yaml(config, config_path)
    logger.info("Config saved → %s", config_path)

    # ------------------------------------------------------------------
    # 3. Discover (subject, run) records
    # ------------------------------------------------------------------
    subject_runs: List[Tuple[str, int]] = []
    file_paths: List[str] = []
    use_cluster_data = False

    if args.debug:
        # Create 3 synthetic NPZ files for smoke-testing
        logger.info("DEBUG MODE: generating synthetic data")
        synthetic_dir = os.path.join(run_dir, '_synthetic')
        file_paths = []
        for i in range(3):
            path = os.path.join(synthetic_dir, f'synthetic_{i:02d}.npz')
            _make_synthetic_npz(path, n_seconds=120)
            file_paths.append(path)
        logger.info("Synthetic files: %s", file_paths)

    else:
        # Try cluster data
        try:
            from src.data.data_utils import list_eeg_records
            from src.data.subject_utils import get_subjects_by_age_group

            if args.subjects:
                subjects = args.subjects
            else:
                subjects = get_subjects_by_age_group(
                    args.age_group, usable_only=True, data_dir=data_dir,
                )

            if args.debug:
                subjects = subjects[:3]

            subject_runs = list_eeg_records(subjects=subjects, data_dir=data_dir)
            logger.info(
                "Found %d (subject, run) records for %d subjects",
                len(subject_runs), len(subjects),
            )

            if not subject_runs:
                logger.error("No EEG records found — check AGING_DATA_DIR.")
                sys.exit(1)

            use_cluster_data = True

        except Exception as exc:
            logger.warning("Cluster data not available (%s); aborting.", exc)
            sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Build DataLoaders + subject splits
    # ------------------------------------------------------------------
    from src.training.config import LabelConfig as _LC

    label_cfg_base = config.label_config

    if use_cluster_data:
        from src.training.dataset import make_cluster_dataloaders
        # Use shift_ms=0 just to determine splits (we retrain per shift)
        label_cfg_zero = _LC(
            shift_ms=0,
            label_bp_lo=label_cfg_base.label_bp_lo,
            label_bp_hi=label_cfg_base.label_bp_hi,
            label_bp_order=label_cfg_base.label_bp_order,
        )
        (train_loader, val_loader, test_loader,
         train_records, val_records, test_records) = make_cluster_dataloaders(
            subject_runs,
            config.feature_config,
            label_cfg_zero,
            config,
            data_dir=data_dir,
            channel=args.channel,
        )
        subjects_train = sorted({r[0] for r in train_records})
        subjects_val = sorted({r[0] for r in val_records})
        subjects_test = sorted({r[0] for r in test_records})
        # file_paths not used for cluster training
        file_paths_train = None
        val_records_for_figs = val_records

    else:
        from src.training.dataset import make_dataloaders
        label_cfg_zero = _LC(
            shift_ms=0,
            label_bp_lo=label_cfg_base.label_bp_lo,
            label_bp_hi=label_cfg_base.label_bp_hi,
            label_bp_order=label_cfg_base.label_bp_order,
        )
        train_loader, val_loader, test_loader = make_dataloaders(
            file_paths, config.feature_config, label_cfg_zero, config,
        )
        subjects_train = [f'synthetic_{i}' for i in range(len(file_paths))]
        subjects_val = subjects_train[-1:]
        subjects_test = subjects_train[-1:]
        file_paths_train = file_paths
        val_records_for_figs = file_paths

    # ------------------------------------------------------------------
    # 5. Log config + write subjects_{split}.txt
    # ------------------------------------------------------------------
    log_config(logger, config, subjects_train, subjects_val, subjects_test)

    for split_name, split_subjects in [
        ('train', subjects_train), ('val', subjects_val), ('test', subjects_test),
    ]:
        txt_path = os.path.join(run_dir, f'subjects_{split_name}.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(split_subjects) + '\n')

    # ------------------------------------------------------------------
    # 6. Feature sample figures (first 3 records / files)
    # ------------------------------------------------------------------
    from src.training import figures as fig_mod
    from src.training.features import build_feature_matrix_offline
    from src.training.labels import generate_labels

    sample_sources = (train_records if use_cluster_data else file_paths)[:3]
    data_fig_dir = os.path.join(run_dir, 'figures', 'data')

    for record in sample_sources:
        try:
            if isinstance(record, str):
                eeg = np.load(record)['eeg'].astype(np.float64).ravel()
                rec_label = Path(record).stem
            else:
                from src.data.data_utils import load_eeg
                eeg = load_eeg(record[0], record[1],
                               channel=args.channel,
                               target_fs=config.feature_config.fs,
                               data_dir=data_dir)
                rec_label = f"{record[0]}_run{record[1]}"

            feats = build_feature_matrix_offline(eeg, config.feature_config)
            lbls = generate_labels(eeg, config.feature_config.fs, config.label_config)
            fig_path = os.path.join(data_fig_dir, f"feature_sample_{rec_label}.png")
            fig_mod.save_feature_sample(
                eeg, feats, lbls,
                fs=config.feature_config.fs,
                shift_ms=config.label_config.shift_ms,
                title=f"Feature sample — {rec_label}",
                out_path=fig_path,
            )
            logger.info("Feature sample saved → %s", fig_path)
        except Exception as exc:
            logger.warning("Feature sample figure failed for %s: %s", record, exc)

    # ------------------------------------------------------------------
    # 7. Label distribution figure
    # ------------------------------------------------------------------
    try:
        all_cos_list, all_sin_list = [], []
        for record in sample_sources:
            if isinstance(record, str):
                eeg = np.load(record)['eeg'].astype(np.float64).ravel()
            else:
                from src.data.data_utils import load_eeg
                eeg = load_eeg(record[0], record[1],
                               channel=args.channel,
                               target_fs=config.feature_config.fs,
                               data_dir=data_dir)
            lbls = generate_labels(eeg, config.feature_config.fs, config.label_config)
            valid = ~np.isnan(lbls[0])
            all_cos_list.append(lbls[0][valid])
            all_sin_list.append(lbls[1][valid])

        if all_cos_list:
            label_dist_path = os.path.join(data_fig_dir, 'label_distribution.png')
            fig_mod.save_label_distribution(
                np.concatenate(all_cos_list),
                np.concatenate(all_sin_list),
                out_path=label_dist_path,
            )
            logger.info("Label distribution saved → %s", label_dist_path)
    except Exception as exc:
        logger.warning("Label distribution figure failed: %s", exc)

    # ------------------------------------------------------------------
    # 8. Train all shifts
    # ------------------------------------------------------------------
    from src.training.train import train_all_shifts

    if use_cluster_data:
        # For cluster training, we pass (subject, run) records and rebuild
        # dataloaders inside train_all_shifts via a patched call.
        # We build file_paths_train as None to signal cluster mode.
        # Instead, re-expose cluster records as the training source.
        # train_all_shifts currently expects file_paths; we pass train_records
        # through a thin wrapper.
        _train_source = _ClusterTrainSource(
            train_records=train_records,
            config=config,
            data_dir=data_dir,
            channel=args.channel,
        )
        _trained = _train_cluster_all_shifts(
            _train_source, config,
            device=args.device,
            logger=logger,
            run_dir=run_dir,
            val_records=val_records_for_figs,
        )
    else:
        _trained = train_all_shifts(
            file_paths,
            config,
            device=args.device,
            logger=logger,
            run_dir=run_dir,
            val_records=val_records_for_figs,
        )

    logger.info("Training complete.  Run dir: %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Cluster training shim
# ---------------------------------------------------------------------------

class _ClusterTrainSource:
    """Thin holder so cluster records can be passed to train_all_shifts."""
    def __init__(self, train_records, config, data_dir, channel):
        self.train_records = train_records
        self.config = config
        self.data_dir = data_dir
        self.channel = channel


def _train_cluster_all_shifts(
    source: _ClusterTrainSource,
    config,
    device: str,
    logger,
    run_dir: str,
    val_records,
):
    """Cluster-mode analogue of train_all_shifts using ClusterEEGDataset."""
    import copy
    from src.training.config import LabelConfig
    from src.training.dataset import ClusterEEGDataset, make_cluster_dataloaders
    from src.training.train import train_one_model, _generate_eval_figures
    from src.training.export import export_lstm_to_mat, export_tcn_to_onnx
    import torch

    trained = {}
    for shift_ms in config.shift_ms_list:
        logger.info("\n" + "=" * 60)
        logger.info("Training %s for shift=%dms", config.model_type.upper(), shift_ms)
        logger.info("=" * 60)

        label_cfg = LabelConfig(
            shift_ms=shift_ms,
            label_bp_lo=config.label_config.label_bp_lo,
            label_bp_hi=config.label_config.label_bp_hi,
            label_bp_order=config.label_config.label_bp_order,
        )

        # Build train/val datasets directly from cluster records
        from torch.utils.data import DataLoader
        train_ds = ClusterEEGDataset(
            source.train_records,
            config.feature_config, label_cfg,
            seq_len=config.seq_len,
            data_dir=source.data_dir,
            channel=source.channel,
        )
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size,
            shuffle=True, num_workers=0,
        )

        # Use a small val set from val_records for early stopping
        val_ds = ClusterEEGDataset(
            val_records[:min(len(val_records), 2)],
            config.feature_config, label_cfg,
            seq_len=config.seq_len,
            data_dir=source.data_dir,
            channel=source.channel,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size,
            shuffle=False, num_workers=0,
        )

        # Reuse train_one_model internals via a direct training loop
        # (We already have loaders, so call the inner loop directly)
        from src.training.models import build_model
        from src.training.loss import circular_mse_loss
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from src.training.logging_utils import log_epoch
        import torch.nn as nn

        dev = torch.device(device)
        model = build_model(config).to(dev)
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=1e-5)

        model_dir = os.path.join(run_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        ckpt_path = os.path.join(model_dir,
                                 f"{config.model_type}_shift_{shift_ms}ms_checkpoint.pt")
        log_path = os.path.join(model_dir,
                                f"metrics_{config.model_type}_shift_{shift_ms}ms.csv")

        import csv, time, copy as _copy
        best_val_loss = float('inf')
        best_state = None
        patience_ctr = 0
        log_rows = []

        for epoch in range(1, config.max_epochs + 1):
            t0 = time.time()
            model.train()
            tr_sum, tr_n = 0.0, 0
            for X, Y in train_loader:
                X, Y = X.to(dev), Y.to(dev)
                optimizer.zero_grad()
                pred = model(X)
                loss = circular_mse_loss(pred, Y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                optimizer.step()
                tr_sum += loss.item(); tr_n += 1
            scheduler.step()
            tr_loss = tr_sum / max(tr_n, 1)

            model.eval()
            vl_sum, vl_n = 0.0, 0
            with torch.no_grad():
                for X, Y in val_loader:
                    X, Y = X.to(dev), Y.to(dev)
                    pred = model(X)
                    loss = circular_mse_loss(pred, Y)
                    vl_sum += loss.item(); vl_n += 1
            vl_loss = vl_sum / max(vl_n, 1)
            elapsed = time.time() - t0

            is_best = vl_loss < best_val_loss
            log_rows.append({
                'epoch': epoch,
                'train_loss': f'{tr_loss:.6f}',
                'val_loss': f'{vl_loss:.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'elapsed_s': f'{elapsed:.1f}',
            })
            log_epoch(logger, epoch, config.max_epochs, tr_loss, vl_loss,
                      scheduler.get_last_lr()[0], elapsed, is_best=is_best)

            if is_best:
                best_val_loss = vl_loss
                best_state = _copy.deepcopy(model.state_dict())
                patience_ctr = 0
                torch.save({'state_dict': best_state, 'config': config,
                            'shift_ms': shift_ms}, ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= config.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if log_rows:
            with open(log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                writer.writeheader()
                writer.writerows(log_rows)
            try:
                from src.training import figures as fig_mod
                fig_path = os.path.join(run_dir, 'figures', 'training',
                                        f"loss_curves_shift_{shift_ms}ms.png")
                fig_mod.save_training_curves(log_path, fig_path,
                                              title=f"{config.model_type.upper()} shift={shift_ms}ms")
            except Exception as exc:
                logger.warning("Training curves figure failed: %s", exc)

        if best_state is not None:
            model.load_state_dict(best_state)

        trained[shift_ms] = model

        # Export
        output_base = os.path.join(model_dir, f"{config.model_type}_shift_{shift_ms}ms")
        if config.model_type == 'lstm':
            export_lstm_to_mat(model, config.lstm_config, output_base + '.mat')
        elif config.model_type == 'tcn':
            export_tcn_to_onnx(model, config.tcn_config, output_base + '.onnx')

        # Eval figures
        _generate_eval_figures(model, val_records, config, shift_ms,
                               device, run_dir, logger)

    return trained


if __name__ == '__main__':
    main()
