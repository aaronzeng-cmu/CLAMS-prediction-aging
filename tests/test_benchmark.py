"""Tests for evaluation metrics and benchmark smoke test (Verification plan #7)."""
import os
import tempfile

import numpy as np
import pytest

from src.evaluation.metrics import mace, plv, up_phase_rate, from_cos_sin


FS = 200
F_SO = 0.75
DURATION_S = 60


def make_sine_phase(freq=F_SO, fs=FS, duration_s=DURATION_S):
    t = np.arange(duration_s * fs) / fs
    return (2 * np.pi * freq * t) % (2 * np.pi) - np.pi


class TestMACE:
    def test_perfect_prediction(self):
        """Perfect prediction -> MACE = 0."""
        phase = make_sine_phase()
        assert mace(phase, phase) < 1e-10

    def test_opposite_phase(self):
        """Opposite phase -> MACE = pi."""
        phase = make_sine_phase()
        assert abs(mace(phase + np.pi, phase) - np.pi) < 1e-3

    def test_90_deg_error(self):
        """90° error -> MACE = pi/2."""
        phase = make_sine_phase()
        assert abs(mace(phase + np.pi/2, phase) - np.pi/2) < 1e-3

    def test_wrap_around(self):
        """MACE handles ±pi wrap correctly."""
        pred  = np.array([np.pi - 0.01, -(np.pi - 0.01)])
        true  = np.array([-(np.pi - 0.01), np.pi - 0.01])
        # Circular distance ~0.02 rad, not ~2*pi
        assert mace(pred, true) < 0.05

    def test_output_range(self):
        """MACE is always in [0, pi]."""
        pred = np.random.uniform(-np.pi, np.pi, 1000)
        true = np.random.uniform(-np.pi, np.pi, 1000)
        m = mace(pred, true)
        assert 0 <= m <= np.pi


class TestPLV:
    def test_perfect_locking(self):
        """Perfect locking -> PLV = 1."""
        phase = make_sine_phase()
        assert abs(plv(phase, phase) - 1.0) < 1e-6

    def test_random_phases_low_plv(self):
        """Random phase -> PLV ~ 0."""
        np.random.seed(42)
        T = 10000
        pred = np.random.uniform(-np.pi, np.pi, T)
        true = np.random.uniform(-np.pi, np.pi, T)
        assert plv(pred, true) < 0.05

    def test_output_range(self):
        pred = np.random.uniform(-np.pi, np.pi, 1000)
        true = np.random.uniform(-np.pi, np.pi, 1000)
        p = plv(pred, true)
        assert 0 <= p <= 1


class TestUpPhaseRate:
    def test_uniform_distribution(self):
        """Uniform phase -> up_phase_rate ~ 0.5 (half the circle)."""
        phase = np.linspace(-np.pi, np.pi, 10000, endpoint=False)
        rate = up_phase_rate(phase)
        assert abs(rate - 0.5) < 0.01

    def test_all_in_window(self):
        phase = np.zeros(100)   # all at phase=0, in [-pi/2, pi/2]
        assert up_phase_rate(phase) == 1.0

    def test_none_in_window(self):
        phase = np.full(100, np.pi)   # all at phase=pi, not in window
        assert up_phase_rate(phase) == 0.0

    def test_custom_window(self):
        phase = np.array([-0.1, 0.0, 0.1, np.pi])
        rate = up_phase_rate(phase, window=(-0.2, 0.2))
        assert abs(rate - 0.75) < 1e-6


class TestFromCosSin:
    def test_identity(self):
        """from_cos_sin(cos(phi), sin(phi)) == phi."""
        phase = np.linspace(-np.pi, np.pi, 100, endpoint=False)
        recovered = from_cos_sin(np.cos(phase), np.sin(phase))
        assert np.allclose(recovered, phase, atol=1e-6)

    def test_shape_preserved(self):
        cos_p = np.random.randn(50)
        sin_p = np.random.randn(50)
        out = from_cos_sin(cos_p, sin_p)
        assert out.shape == (50,)


class TestBenchmarkSmokeTest:
    """Smoke test: 60 s synthetic EEG, MACE < pi, PLV in [0,1] (plan #7)."""

    def test_metrics_valid_range(self):
        """Basic metric sanity on synthetic EEG (no model needed)."""
        np.random.seed(0)
        T = DURATION_S * FS
        true_phase = make_sine_phase()
        # Simulate an imperfect predictor: shifted phase + noise
        pred_phase = true_phase + np.random.randn(T) * 0.5

        m = mace(pred_phase, true_phase)
        p = plv(pred_phase, true_phase)
        u = up_phase_rate(pred_phase)

        assert m < np.pi, f"MACE {m:.3f} >= pi"
        assert 0 <= p <= 1, f"PLV {p:.3f} out of [0,1]"
        assert 0 <= u <= 1, f"up_phase_rate {u:.3f} out of [0,1]"

    def test_lstm_inference_smoke(self):
        """LSTM forward pass on 60 s synthetic EEG: MACE < pi, PLV in [0,1]."""
        import torch
        from src.training.config import LSTMConfig, FeatureConfig
        from src.training.models.lstm import PhaseLSTM
        from src.evaluation.benchmark import run_model_on_eeg

        cfg = LSTMConfig(input_size=3, hidden_size=16, num_layers=1)
        feat_cfg = FeatureConfig()
        model = PhaseLSTM(cfg)
        model.eval()

        np.random.seed(1)
        t = np.arange(DURATION_S * FS) / FS
        eeg = np.sin(2 * np.pi * F_SO * t) + 0.1 * np.random.randn(DURATION_S * FS)

        pred_phase = run_model_on_eeg(model, eeg, feat_cfg, device='cpu')
        true_phase = make_sine_phase(F_SO, FS, DURATION_S)

        m = mace(pred_phase, true_phase)
        p = plv(pred_phase, true_phase)

        assert m < np.pi, f"MACE {m:.3f} >= pi (plan #7)"
        assert 0 <= p <= 1, f"PLV {p:.3f} out of [0,1] (plan #7)"
