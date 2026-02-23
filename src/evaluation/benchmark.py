"""
Benchmarking utilities: run all models on test EEG files and report metrics.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from ..training.config import FeatureConfig, LabelConfig, LSTMConfig, TCNConfig
from ..training.features import CausalBandpassFilter, extract_features, build_feature_matrix_offline
from ..training.labels import generate_labels
from ..training.models.lstm import PhaseLSTM
from ..training.models.tcn import PhaseTCN
from ..training.export import _lstm_forward_numpy
from .metrics import mace, plv, up_phase_rate, from_cos_sin


def run_model_on_eeg(
    model,
    eeg: np.ndarray,
    feature_config: FeatureConfig,
    device: str = 'cpu',
) -> np.ndarray:
    """Simulate online inference with a trained PyTorch model.

    Uses a persistent CausalBandpassFilter state to match deployment behavior.
    Processes the full recording as a single chunk (no chunked streaming needed
    for offline benchmark, but filter state is stateful across the recording).

    Args:
        model: Trained PhasePredictor (PhaseLSTM or PhaseTCN).
        eeg: 1-D raw EEG array (T,).
        feature_config: FeatureConfig for feature extraction.
        device: Torch device string.

    Returns:
        Predicted phase array (T,) in radians.
    """
    model.eval()
    dev = torch.device(device)

    # Build features using single causal filter pass (training-consistent)
    features = build_feature_matrix_offline(eeg, feature_config)  # (C, T)
    C, T = features.shape

    x = torch.from_numpy(features[np.newaxis].astype(np.float32)).to(dev)  # (1, C, T)

    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]  # (2, T)

    cos_pred = pred[0]
    sin_pred = pred[1]
    return from_cos_sin(cos_pred, sin_pred)


def _load_lstm_from_mat(mat_path: str, lstm_cfg: LSTMConfig) -> PhaseLSTM:
    """Reconstruct a PhaseLSTM from an exported .mat file for benchmarking."""
    import scipy.io
    mat = scipy.io.loadmat(mat_path)

    model = PhaseLSTM(lstm_cfg)
    sd = model.state_dict()

    for k in range(lstm_cfg.num_layers):
        layer_num = k + 1
        W_ih = torch.from_numpy(mat[f'W_ih_{layer_num}'].astype(np.float32))
        W_hh = torch.from_numpy(mat[f'W_hh_{layer_num}'].astype(np.float32))
        # bias was saved as combined (bias_ih + bias_hh); split equally for reconstruction
        combined = mat[f'bias_{layer_num}'].ravel().astype(np.float32)
        sd[f'lstm.weight_ih_l{k}'] = W_ih
        sd[f'lstm.weight_hh_l{k}'] = W_hh
        sd[f'lstm.bias_ih_l{k}'] = torch.from_numpy(combined)
        sd[f'lstm.bias_hh_l{k}'] = torch.zeros_like(torch.from_numpy(combined))

    sd['fc.weight'] = torch.from_numpy(mat['W_fc'].astype(np.float32))
    sd['fc.bias'] = torch.from_numpy(mat['b_fc'].ravel().astype(np.float32))
    model.load_state_dict(sd)
    return model


def run_benchmark(
    test_files: List[str],
    model_dir: str,
    shift_ms: int,
    feature_config: FeatureConfig,
    label_config: LabelConfig,
    output_path: str,
    device: str = 'cpu',
) -> pd.DataFrame:
    """Run all available models on test subjects and compile a metric table.

    Models loaded from model_dir (if present):
        lstm_shift_{shift_ms}ms.mat
        tcn_shift_{shift_ms}ms.onnx

    Placeholder rows are added for MATLAB-only baselines (AR, PV, SSPE).

    Args:
        test_files: List of test subject EEG file paths.
        model_dir: Directory with exported model files.
        shift_ms: Shift value to benchmark.
        feature_config: FeatureConfig for feature extraction.
        label_config: LabelConfig for ground-truth generation.
        output_path: Where to write CSV + markdown results.
        device: Torch device string.

    Returns:
        DataFrame with columns: model, subject, mace_rad, plv, up_phase_rate.
    """
    from ..training.dataset import _load_eeg_from_file

    rows = []
    models_to_run: dict[str, object] = {}

    # Discover exported LSTM
    lstm_path = os.path.join(model_dir, f'lstm_shift_{shift_ms}ms.mat')
    if os.path.exists(lstm_path):
        lstm_cfg = LSTMConfig()
        models_to_run['lstm'] = _load_lstm_from_mat(lstm_path, lstm_cfg)

    # Discover exported TCN (onnxruntime inference)
    tcn_path = os.path.join(model_dir, f'tcn_shift_{shift_ms}ms.onnx')
    if os.path.exists(tcn_path):
        models_to_run['tcn'] = tcn_path  # string sentinel -> ONNX path

    for subj_path in test_files:
        subj_name = Path(subj_path).stem
        eeg = _load_eeg_from_file(subj_path)
        true_labels = generate_labels(eeg, feature_config.fs, label_config)  # (2, T)
        true_phase = from_cos_sin(true_labels[0], true_labels[1])

        # Valid mask (non-NaN labels)
        valid = ~np.isnan(true_labels[0])

        for model_name, model_obj in models_to_run.items():
            if isinstance(model_obj, str):
                # ONNX model
                pred_phase = _run_onnx_on_eeg(model_obj, eeg, feature_config)
            else:
                pred_phase = run_model_on_eeg(model_obj, eeg, feature_config, device)

            if valid.any():
                p = pred_phase[valid]
                t = true_phase[valid]
                rows.append({
                    'model': model_name,
                    'subject': subj_name,
                    'shift_ms': shift_ms,
                    'mace_rad': round(mace(p, t), 4),
                    'plv': round(plv(p, t), 4),
                    'up_phase_rate': round(up_phase_rate(p), 4),
                    'n_samples': int(valid.sum()),
                })

        # Placeholder rows for MATLAB-only baselines
        for baseline in ['ar', 'pv', 'sspe']:
            rows.append({
                'model': baseline,
                'subject': subj_name,
                'shift_ms': shift_ms,
                'mace_rad': float('nan'),
                'plv': float('nan'),
                'up_phase_rate': float('nan'),
                'n_samples': int(valid.sum()),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        # Save CSV
        csv_path = output_path if output_path.endswith('.csv') else output_path + '.csv'
        df.to_csv(csv_path, index=False)

        # Save markdown table
        md_path = csv_path.replace('.csv', '.md')
        numeric_cols = ['mace_rad', 'plv', 'up_phase_rate']
        summary = (
            df.groupby('model')[numeric_cols]
            .mean()
            .round(4)
            .reset_index()
        )
        with open(md_path, 'w') as f:
            f.write(f"# Benchmark results — shift={shift_ms}ms\n\n")
            f.write(summary.to_markdown(index=False))
            f.write('\n')
        print(f"Results saved to {csv_path} and {md_path}")

    return df


def _run_onnx_on_eeg(
    onnx_path: str,
    eeg: np.ndarray,
    feature_config: FeatureConfig,
) -> np.ndarray:
    """Run an ONNX model on a full EEG recording."""
    import onnxruntime as ort

    features = build_feature_matrix_offline(eeg, feature_config)  # (C, T)
    C, T = features.shape

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    expected_len = sess.get_inputs()[0].shape[2]   # fixed seq_len from export

    # Pad or chunk to fixed length
    if T <= expected_len:
        padded = np.zeros((1, C, expected_len), dtype=np.float32)
        padded[0, :, :T] = features
        out = sess.run(None, {input_name: padded})[0][0]   # (2, expected_len)
        cos_pred = out[0, :T]
        sin_pred = out[1, :T]
    else:
        # Process in non-overlapping chunks of expected_len
        cos_pred = np.zeros(T, dtype=np.float32)
        sin_pred = np.zeros(T, dtype=np.float32)
        for start in range(0, T, expected_len):
            end = min(start + expected_len, T)
            chunk = np.zeros((1, C, expected_len), dtype=np.float32)
            chunk[0, :, :end - start] = features[:, start:end]
            out = sess.run(None, {input_name: chunk})[0][0]  # (2, expected_len)
            cos_pred[start:end] = out[0, :end - start]
            sin_pred[start:end] = out[1, :end - start]

    return from_cos_sin(cos_pred, sin_pred)
