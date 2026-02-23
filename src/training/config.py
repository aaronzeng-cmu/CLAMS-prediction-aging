"""
Single source of truth for all ablation-controllable parameters.
No logic — only frozen dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass(frozen=True)
class FeatureConfig:
    fs: int = 200
    bp_lo: float = 0.4           # online feature bandpass lower edge (Hz)
    bp_hi: float = 1.2           # online feature bandpass upper edge (Hz)
    bp_order: int = 4
    feature_names: tuple = ('raw', 'diff', 'bandpass')
    # Ablation: bp_hi=1.0 + add 'so_envelope' for aging-adapted features


@dataclass(frozen=True)
class LabelConfig:
    shift_ms: int = 100
    label_bp_lo: float = 0.4     # SEPARATE from FeatureConfig — offline filtfilt for labels
    label_bp_hi: float = 1.2
    label_bp_order: int = 4


@dataclass(frozen=True)
class LSTMConfig:
    input_size: int = 3          # must equal len(feature_names)
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class TCNConfig:
    input_size: int = 3
    num_channels: tuple = (32, 32, 32, 32)
    kernel_size: int = 3
    output_size: int = 2
    dropout: float = 0.1
    # Receptive field = (kernel_size-1) * sum(2^i for i in range(n_layers)) + 1
    # Default: (3-1)*(1+2+4+8)+1 = 31 samples = 155 ms at 200 Hz


@dataclass
class TrainingConfig:
    model_type: str = 'lstm'         # 'lstm' | 'tcn'
    shift_ms_list: list = field(default_factory=lambda: [100])
    batch_size: int = 32
    seq_len: int = 200               # training window (1 s at 200 Hz)
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    seed: int = 42
    clip_grad_norm: float = 1.0
    model_dir: str = 'models/'
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    label_config: LabelConfig = field(default_factory=LabelConfig)
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)
    tcn_config: TCNConfig = field(default_factory=TCNConfig)

    def __post_init__(self):
        n = len(self.feature_config.feature_names)
        assert self.lstm_config.input_size == n, (
            f"LSTMConfig.input_size={self.lstm_config.input_size} != {n} features"
        )
        assert self.tcn_config.input_size == n, (
            f"TCNConfig.input_size={self.tcn_config.input_size} != {n} features"
        )


def config_from_yaml(path: str) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file. Nested configs are reconstructed."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    feature_cfg = FeatureConfig(**data.pop('feature_config', {}))
    label_cfg = LabelConfig(**data.pop('label_config', {}))
    lstm_cfg = LSTMConfig(**data.pop('lstm_config', {}))
    tcn_raw = data.pop('tcn_config', {})
    if 'num_channels' in tcn_raw and isinstance(tcn_raw['num_channels'], list):
        tcn_raw['num_channels'] = tuple(tcn_raw['num_channels'])
    tcn_cfg = TCNConfig(**tcn_raw)

    return TrainingConfig(
        feature_config=feature_cfg,
        label_config=label_cfg,
        lstm_config=lstm_cfg,
        tcn_config=tcn_cfg,
        **data,
    )
