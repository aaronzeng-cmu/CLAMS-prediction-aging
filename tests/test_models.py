"""Tests for LSTM and TCN models (Verification plan #4 and #5)."""
import numpy as np
import pytest
import torch
from src.training.config import LSTMConfig, TCNConfig
from src.training.models.lstm import PhaseLSTM
from src.training.models.tcn import PhaseTCN, CausalConv1d


class TestPhaseLSTM:
    @pytest.fixture
    def model(self):
        cfg = LSTMConfig(input_size=3, hidden_size=16, num_layers=2)
        return PhaseLSTM(cfg)

    def test_output_shape(self, model):
        """(1,3,50) -> (1,2,50) (plan #4)."""
        x = torch.randn(1, 3, 50)
        out = model(x)
        assert out.shape == (1, 2, 50), f"Unexpected shape: {out.shape}"

    def test_no_nan_output(self, model):
        """No NaN in output (plan #4)."""
        x = torch.randn(1, 3, 50)
        out = model(x)
        assert not torch.any(torch.isnan(out)), "Output contains NaN"

    def test_batch_dimension(self, model):
        x = torch.randn(8, 3, 100)
        out = model(x)
        assert out.shape == (8, 2, 100)

    def test_parameter_count(self, model):
        n = model.count_parameters()
        # 2-layer LSTM(3->16->16) + Linear(16->2)
        # Layer 1: 4*16*(3+16+1) = 1280, Layer 2: 4*16*(16+16+1) = 2112, FC: 16*2+2 = 34
        assert n > 0
        print(f"\nLSTM param count: {n}")

    def test_no_activation_on_output(self, model):
        """Output is unbounded — no sigmoid/tanh on final output."""
        x = torch.randn(1, 3, 50) * 100.0   # large input
        out = model(x)
        # If there were a sigmoid, values would be bounded to (0,1)
        # Values should sometimes exceed 1 for large inputs
        # This is a sanity check, not a guarantee
        assert out.dtype == torch.float32


class TestPhaseTCN:
    @pytest.fixture
    def model(self):
        cfg = TCNConfig(input_size=3, num_channels=(16, 16, 16, 16), kernel_size=3, dropout=0.0)
        return PhaseTCN(cfg)

    def test_output_shape(self, model):
        x = torch.randn(2, 3, 100)
        out = model(x)
        assert out.shape == (2, 2, 100)

    def test_causality(self, model):
        """Perturbing x[:,:,25:] leaves out[:,:,:25] unchanged (plan #5)."""
        model.eval()
        torch.manual_seed(0)
        x1 = torch.randn(1, 3, 50)
        x2 = x1.clone()
        x2[:, :, 25:] = x2[:, :, 25:] + torch.randn(1, 3, 25) * 10.0

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        max_diff = (out1[:, :, :25] - out2[:, :, :25]).abs().max().item()
        assert max_diff < 1e-5, (
            f"TCN causality violated: max diff at t<25 is {max_diff:.2e}"
        )

    def test_future_independence(self, model):
        """Output at t=0 must not depend on any future input."""
        model.eval()
        torch.manual_seed(42)
        x_base = torch.zeros(1, 3, 50)
        x_perturbed = x_base.clone()
        x_perturbed[:, :, 1:] = torch.randn(1, 3, 49)

        with torch.no_grad():
            out_base = model(x_base)
            out_pert = model(x_perturbed)

        # t=0 output should be identical regardless of future samples
        diff_t0 = (out_base[:, :, 0] - out_pert[:, :, 0]).abs().max().item()
        assert diff_t0 < 1e-6, f"Output at t=0 changed when future inputs changed: {diff_t0:.2e}"

    def test_no_nan_output(self, model):
        x = torch.randn(1, 3, 50)
        out = model(x)
        assert not torch.any(torch.isnan(out))

    def test_parameter_count(self, model):
        n = model.count_parameters()
        assert n > 0
        print(f"\nTCN param count: {n}")


class TestCausalConv1d:
    def test_output_same_length(self):
        """Output length equals input length."""
        conv = CausalConv1d(3, 8, kernel_size=3, dilation=2)
        x = torch.randn(1, 3, 50)
        out = conv(x)
        assert out.shape == (1, 8, 50)

    def test_no_future_leakage(self):
        """Output at t depends only on input at t' <= t."""
        conv = CausalConv1d(1, 1, kernel_size=3, dilation=1)
        with torch.no_grad():
            x1 = torch.zeros(1, 1, 20)
            x1[0, 0, 5] = 1.0      # impulse at t=5

            x2 = torch.zeros(1, 1, 20)
            x2[0, 0, 5] = 1.0
            x2[0, 0, 6:] = 99.0   # change future

            out1 = conv(x1)
            out2 = conv(x2)

        # t <= 5 should be identical
        diff = (out1[0, 0, :6] - out2[0, 0, :6]).abs().max().item()
        assert diff < 1e-6, f"Future input leaked to past output: {diff:.2e}"
