"""Tests for model export (Verification plan #6)."""
import os
import tempfile

import numpy as np
import pytest
import torch

from src.training.config import LSTMConfig, TCNConfig
from src.training.models.lstm import PhaseLSTM
from src.training.models.tcn import PhaseTCN
from src.training.export import (
    export_lstm_to_mat,
    export_tcn_to_onnx,
    verify_export,
)


@pytest.fixture
def lstm_model():
    cfg = LSTMConfig(input_size=3, hidden_size=16, num_layers=2)
    model = PhaseLSTM(cfg)
    model.eval()
    return model, cfg


@pytest.fixture
def tcn_model():
    cfg = TCNConfig(input_size=3, num_channels=(8, 8), kernel_size=3, dropout=0.0)
    model = PhaseTCN(cfg)
    model.eval()
    return model, cfg


class TestExportLSTMToMat:
    def test_mat_file_created(self, lstm_model, tmp_path):
        model, cfg = lstm_model
        out = str(tmp_path / "test_lstm.mat")
        export_lstm_to_mat(model, cfg, out)
        assert os.path.exists(out)

    def test_mat_keys_present(self, lstm_model, tmp_path):
        import scipy.io
        model, cfg = lstm_model
        out = str(tmp_path / "test_lstm.mat")
        export_lstm_to_mat(model, cfg, out)
        mat = scipy.io.loadmat(out)
        # Check layer 1
        assert 'W_ih_1' in mat
        assert 'W_hh_1' in mat
        assert 'bias_1' in mat
        # Check layer 2
        assert 'W_ih_2' in mat
        assert 'W_hh_2' in mat
        assert 'bias_2' in mat
        # Check FC
        assert 'W_fc' in mat
        assert 'b_fc' in mat
        # Metadata
        assert 'hidden_size' in mat
        assert 'num_layers' in mat

    def test_weight_shapes(self, lstm_model, tmp_path):
        import scipy.io
        model, cfg = lstm_model
        out = str(tmp_path / "test_lstm.mat")
        export_lstm_to_mat(model, cfg, out)
        mat = scipy.io.loadmat(out)
        H, C = cfg.hidden_size, cfg.input_size
        assert mat['W_ih_1'].shape == (4*H, C)
        assert mat['W_hh_1'].shape == (4*H, H)
        assert mat['bias_1'].shape == (4*H, 1)
        assert mat['W_ih_2'].shape == (4*H, H)
        assert mat['W_fc'].shape == (2, H)
        assert mat['b_fc'].shape == (2, 1)

    def test_bias_is_combined(self, lstm_model, tmp_path):
        """Saved bias must equal bias_ih + bias_hh."""
        import scipy.io
        model, cfg = lstm_model
        out = str(tmp_path / "test_lstm.mat")
        export_lstm_to_mat(model, cfg, out)
        mat = scipy.io.loadmat(out)

        sd = model.state_dict()
        expected = (sd['lstm.bias_ih_l0'] + sd['lstm.bias_hh_l0']).numpy()
        saved = mat['bias_1'].ravel()
        assert np.allclose(expected, saved, atol=1e-6)

    def test_numpy_roundtrip_matches_torch(self, lstm_model, tmp_path):
        """verify_export max diff < 1e-4 (plan #6)."""
        model, cfg = lstm_model
        out = str(tmp_path / "test_lstm.mat")
        export_lstm_to_mat(model, cfg, out)

        np.random.seed(0)
        test_input = np.random.randn(3, 50).astype(np.float32)
        max_diff = verify_export(out, model, test_input)
        assert max_diff < 1e-4, f"Round-trip max diff {max_diff:.2e} >= 1e-4"


class TestExportTCNToONNX:
    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('onnxruntime'),
        reason="onnxruntime not installed"
    )
    def test_onnx_file_created(self, tcn_model, tmp_path):
        model, cfg = tcn_model
        out = str(tmp_path / "test_tcn.onnx")
        export_tcn_to_onnx(model, cfg, out, seq_len=64)
        assert os.path.exists(out)

    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('onnxruntime'),
        reason="onnxruntime not installed"
    )
    def test_onnx_output_shape(self, tcn_model, tmp_path):
        import onnxruntime as ort
        model, cfg = tcn_model
        out = str(tmp_path / "test_tcn.onnx")
        export_tcn_to_onnx(model, cfg, out, seq_len=64)

        sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
        dummy = np.zeros((1, cfg.input_size, 64), dtype=np.float32)
        result = sess.run(None, {'eeg_features': dummy})
        assert result[0].shape == (1, 2, 64)

    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('onnxruntime'),
        reason="onnxruntime not installed"
    )
    def test_onnx_no_nan(self, tcn_model, tmp_path):
        import onnxruntime as ort
        model, cfg = tcn_model
        out = str(tmp_path / "test_tcn.onnx")
        export_tcn_to_onnx(model, cfg, out, seq_len=64)

        sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
        dummy = np.random.randn(1, cfg.input_size, 64).astype(np.float32)
        result = sess.run(None, {'eeg_features': dummy})
        assert not np.any(np.isnan(result[0]))
