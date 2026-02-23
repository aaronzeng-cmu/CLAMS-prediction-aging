"""Model registry for the CLAMS phase predictor."""
from .base import PhasePredictor
from .lstm import PhaseLSTM
from .tcn import PhaseTCN
from ..config import LSTMConfig, TCNConfig, TrainingConfig

__all__ = ['PhasePredictor', 'PhaseLSTM', 'PhaseTCN', 'build_model']


def build_model(config: TrainingConfig) -> PhasePredictor:
    """Instantiate the model specified by config.model_type.

    Args:
        config: TrainingConfig with model_type and model-specific sub-configs.

    Returns:
        Initialized PhasePredictor subclass.
    """
    if config.model_type == 'lstm':
        return PhaseLSTM(config.lstm_config)
    elif config.model_type == 'tcn':
        return PhaseTCN(config.tcn_config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type!r}. Choose 'lstm' or 'tcn'.")
