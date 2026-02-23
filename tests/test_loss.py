"""Tests for circular loss functions (Verification plan #3)."""
import numpy as np
import pytest
import torch
from src.training.loss import circular_mse_loss, unit_norm_regularizer


def make_cos_sin(phase: torch.Tensor) -> torch.Tensor:
    """Convert phase (B, T) -> (B, 2, T) as [cos, sin]."""
    return torch.stack([torch.cos(phase), torch.sin(phase)], dim=1)


class TestCircularMSELoss:
    def test_perfect_prediction_near_zero(self):
        """Perfect prediction -> loss < 1e-6 (plan #3)."""
        B, T = 4, 100
        phase = torch.rand(B, T) * 2 * np.pi - np.pi
        cs = make_cos_sin(phase)
        loss = circular_mse_loss(cs, cs)
        assert loss.item() < 1e-6, f"Perfect prediction loss: {loss.item():.2e}"

    def test_opposite_phase_near_two(self):
        """Opposite phase -> loss ≈ 2.0 (plan #3)."""
        B, T = 4, 100
        phase = torch.rand(B, T) * 2 * np.pi - np.pi
        pred = make_cos_sin(phase)
        target = make_cos_sin(phase + np.pi)   # exactly opposite
        loss = circular_mse_loss(pred, target)
        assert abs(loss.item() - 2.0) < 1e-4, f"Opposite phase loss: {loss.item():.4f}"

    def test_90_deg_error_near_one(self):
        """90° phase error -> loss ≈ 1.0 (geometry of unit circle)."""
        B, T = 4, 100
        phase = torch.rand(B, T) * 2 * np.pi - np.pi
        pred = make_cos_sin(phase)
        target = make_cos_sin(phase + np.pi / 2)
        loss = circular_mse_loss(pred, target)
        assert abs(loss.item() - 1.0) < 1e-4, f"90° loss: {loss.item():.4f}"

    def test_mask_zeros_nans(self):
        """Masked timesteps do not contribute to loss."""
        B, T = 2, 50
        phase = torch.zeros(B, T)
        pred = make_cos_sin(phase)
        target = make_cos_sin(phase + np.pi)   # opposite phase, loss=2 if unmasked

        # Mask only first half valid
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :25] = True

        loss_masked = circular_mse_loss(pred, target, mask=mask)
        assert abs(loss_masked.item() - 2.0) < 1e-3, (
            f"Masked loss should still be ~2: {loss_masked.item():.4f}"
        )

    def test_gradient_flows(self):
        """Loss must be differentiable w.r.t. predictions."""
        B, T = 2, 20
        pred = make_cos_sin(torch.rand(B, T)).requires_grad_(True)
        target = make_cos_sin(torch.rand(B, T))
        loss = circular_mse_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.any(torch.isnan(pred.grad))

    def test_no_atan2_in_computation(self):
        """Verify loss is ~MSE in (cos,sin) space, independent of atan2 discontinuity."""
        # Create a discontinuity boundary: phase alternating between +pi and -pi
        B, T = 1, 10
        phases = torch.tensor([np.pi - 0.01, -(np.pi - 0.01)] * (T // 2)).unsqueeze(0)
        pred   = make_cos_sin(phases)
        target = make_cos_sin(phases)
        loss = circular_mse_loss(pred, target)
        assert loss.item() < 1e-4, "Loss should be ~0 at ±pi boundary"

    def test_scalar_output(self):
        B, T = 4, 200
        pred = make_cos_sin(torch.rand(B, T))
        target = make_cos_sin(torch.rand(B, T))
        loss = circular_mse_loss(pred, target)
        assert loss.shape == torch.Size([]), "Loss must be a scalar"


class TestUnitNormRegularizer:
    def test_unit_vectors_zero_reg(self):
        """Unit-norm predictions -> regularizer ≈ 0."""
        B, T = 4, 100
        phase = torch.rand(B, T)
        cs = make_cos_sin(phase)   # already unit norm
        reg = unit_norm_regularizer(cs, weight=1.0)
        assert reg.item() < 1e-6

    def test_zero_vector_high_reg(self):
        """Zero predictions -> regularizer = weight * 1.0."""
        B, T = 4, 100
        pred = torch.zeros(B, 2, T)
        weight = 0.5
        reg = unit_norm_regularizer(pred, weight=weight)
        assert abs(reg.item() - weight) < 1e-4
