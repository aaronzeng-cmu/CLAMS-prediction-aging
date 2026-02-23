"""
PyTorch Dataset for EEG phase prediction.
Splits by subject (file), not by sample, to prevent data leakage.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import FeatureConfig, LabelConfig, TrainingConfig
from .features import build_feature_matrix_offline
from .labels import generate_labels


def _load_eeg_from_file(file_path: str) -> np.ndarray:
    """Load a single-channel EEG recording from NPZ or HDF5 (.mat v7.3).

    Expected keys:
      - NPZ: 'eeg' (T,) float array
      - HDF5 (.mat v7.3): dataset named 'eeg' or first numeric dataset

    Returns:
        1-D float64 array of shape (T,).
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == '.npz':
        data = np.load(file_path)
        return data['eeg'].astype(np.float64).ravel()

    elif ext in ('.mat', '.h5', '.hdf5'):
        import h5py
        with h5py.File(file_path, 'r') as f:
            if 'eeg' in f:
                arr = f['eeg'][()]
            else:
                # Fall back to first numeric dataset
                for key in f.keys():
                    try:
                        arr = f[key][()]
                        break
                    except Exception:
                        continue
                else:
                    raise KeyError(f"No 'eeg' dataset found in {file_path}")
        return np.array(arr, dtype=np.float64).ravel()

    else:
        raise ValueError(f"Unsupported file format: {ext}")


class SubjectEEGData:
    """Precomputed features + labels for one subject file."""

    def __init__(
        self,
        file_path: str,
        feature_cfg: FeatureConfig,
        label_cfg: LabelConfig,
    ) -> None:
        eeg = _load_eeg_from_file(file_path)
        self.features = build_feature_matrix_offline(eeg, feature_cfg)   # (C, T)
        self.labels = generate_labels(eeg, feature_cfg.fs, label_cfg)    # (2, T)


class EEGPhaseDataset(Dataset):
    """Sliding-window EEG phase dataset over a list of subject files.

    Args:
        file_paths: List of paths to EEG recordings.
        feature_cfg: Feature extraction config.
        label_cfg: Label generation config.
        seq_len: Sliding window length in samples.
        stride: Step size between windows (samples). Default=10 (50 ms at 200 Hz).
    """

    def __init__(
        self,
        file_paths: List[str],
        feature_cfg: FeatureConfig,
        label_cfg: LabelConfig,
        seq_len: int = 200,
        stride: int = 10,
    ) -> None:
        self.seq_len = seq_len
        self.stride = stride

        # Pre-load all subjects and collect valid windows
        self._windows: List[Tuple[np.ndarray, np.ndarray]] = []  # (X, Y) pairs

        for path in file_paths:
            data = SubjectEEGData(path, feature_cfg, label_cfg)
            C, T = data.features.shape
            for start in range(0, T - seq_len + 1, stride):
                end = start + seq_len
                X = data.features[:, start:end]    # (C, seq_len)
                Y = data.labels[:, start:end]      # (2, seq_len)
                # Skip windows containing any NaN label
                if not np.any(np.isnan(Y)):
                    self._windows.append((
                        X.astype(np.float32),
                        Y.astype(np.float32),
                    ))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X, Y = self._windows[idx]
        return torch.from_numpy(X), torch.from_numpy(Y)


def make_dataloaders(
    file_paths: List[str],
    feature_config: FeatureConfig,
    label_config: LabelConfig,
    training_config: TrainingConfig,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split file_paths at the subject level and build DataLoaders.

    Splitting is done over files (subjects), not samples, to prevent leakage.

    Args:
        file_paths: Full list of subject EEG files.
        feature_config: Feature config.
        label_config: Label config.
        training_config: Training config (batch_size, seq_len, seed).
        val_frac: Fraction of subjects for validation.
        test_frac: Fraction of subjects for test.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    rng = np.random.default_rng(training_config.seed)
    indices = rng.permutation(len(file_paths))

    n_test = max(1, round(len(file_paths) * test_frac))
    n_val = max(1, round(len(file_paths) * val_frac))

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    def _paths(idx):
        return [file_paths[i] for i in idx]

    def _make(paths, shuffle):
        ds = EEGPhaseDataset(
            paths, feature_config, label_config,
            seq_len=training_config.seq_len,
        )
        return DataLoader(
            ds,
            batch_size=training_config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

    train_loader = _make(_paths(train_idx), shuffle=True)
    val_loader = _make(_paths(val_idx), shuffle=False)
    test_loader = _make(_paths(test_idx), shuffle=False)

    return train_loader, val_loader, test_loader
