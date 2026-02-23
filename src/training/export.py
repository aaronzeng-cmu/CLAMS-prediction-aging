"""
Model export utilities.

MATLAB integration contract:
  - LSTM: exported as .mat containing raw weight matrices for dlnetwork reconstruction
  - TCN: exported as ONNX for importNetworkFromONNX (R2023b+)

Gate ordering note (critical):
  PyTorch weight_ih_l{k} layout: rows [0:H]=i, [H:2H]=f, [2H:3H]=g, [3H:4H]=o
  MATLAB lstmLayer InputWeights layout: same [i;f;g;o] order
  NO REORDERING NEEDED. Do not follow TensorFlow [i|f|o|g] guides.

Bias note:
  MATLAB lstmLayer uses a single bias vector per layer.
  Save combined = bias_ih_lk + bias_hh_lk, shape (4H, 1).
"""
from __future__ import annotations

import numpy as np
import scipy.io
import torch

from .config import LSTMConfig, TCNConfig
from .models.lstm import PhaseLSTM
from .models.tcn import PhaseTCN


def export_lstm_to_mat(
    model: PhaseLSTM,
    config: LSTMConfig,
    output_path: str,
) -> None:
    """Export LSTM weights to .mat for MATLAB dlnetwork reconstruction.

    Saved keys:
        W_ih_1, W_hh_1, bias_1       — layer 1 (shape: (4H,C), (4H,H), (4H,1))
        W_ih_2, W_hh_2, bias_2       — layer 2 if num_layers >= 2
        W_fc, b_fc                   — linear head ((2,H), (2,1))
        hidden_size, input_size, num_layers  — scalars as [[val]]

    Args:
        model: Trained PhaseLSTM.
        config: LSTMConfig used to build the model.
        output_path: Output .mat file path.
    """
    model.eval()
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    mat_data: dict = {}

    for k in range(config.num_layers):
        suffix = f'_l{k}'
        W_ih = sd[f'lstm.weight_ih{suffix}']    # (4H, C_in)
        W_hh = sd[f'lstm.weight_hh{suffix}']    # (4H, H)
        b_ih = sd[f'lstm.bias_ih{suffix}']      # (4H,)
        b_hh = sd[f'lstm.bias_hh{suffix}']      # (4H,)

        combined_bias = (b_ih + b_hh).reshape(-1, 1)   # (4H, 1)

        layer_num = k + 1
        mat_data[f'W_ih_{layer_num}'] = W_ih
        mat_data[f'W_hh_{layer_num}'] = W_hh
        mat_data[f'bias_{layer_num}'] = combined_bias

    # Linear head
    W_fc = sd['fc.weight']              # (2, H)
    b_fc = sd['fc.bias'].reshape(-1, 1) # (2, 1)
    mat_data['W_fc'] = W_fc
    mat_data['b_fc'] = b_fc

    # Scalar metadata — stored as [[val]] so MATLAB reads as scalar, not char
    mat_data['hidden_size'] = np.array([[config.hidden_size]], dtype=np.float64)
    mat_data['input_size'] = np.array([[config.input_size]], dtype=np.float64)
    mat_data['num_layers'] = np.array([[config.num_layers]], dtype=np.float64)

    scipy.io.savemat(output_path, mat_data)
    print(f"LSTM weights exported to: {output_path}")


def export_tcn_to_onnx(
    model: PhaseTCN,
    config: TCNConfig,
    output_path: str,
    seq_len: int = 512,
) -> None:
    """Export TCN to ONNX for MATLAB importNetworkFromONNX (R2023b+).

    Uses a static shape (1, C, seq_len) — MATLAB requires fixed input size.
    Verifies the export with onnxruntime before returning.

    Args:
        model: Trained PhaseTCN.
        config: TCNConfig used to build the model.
        output_path: Output .onnx file path.
        seq_len: Fixed sequence length for the static ONNX graph.
    """
    model.eval()
    dummy = torch.zeros(1, config.input_size, seq_len)

    # Use legacy export API to avoid onnxscript dependency (torch >= 2.5)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=['eeg_features'],
            output_names=['phase_pred'],
            dynamic_axes=None,      # static shape required for MATLAB
            opset_version=17,
            export_params=True,
        )

    # Verify with onnxruntime
    _verify_onnx(output_path, dummy.numpy(), expected_shape=(1, 2, seq_len))
    print(f"TCN exported to ONNX: {output_path}")


def _verify_onnx(
    onnx_path: str,
    dummy_input: np.ndarray,
    expected_shape: tuple,
) -> None:
    """Run ONNX model with onnxruntime and check output shape."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    outputs = sess.run(None, {'eeg_features': dummy_input.astype(np.float32)})
    out = outputs[0]
    assert out.shape == expected_shape, (
        f"ONNX output shape {out.shape} != expected {expected_shape}"
    )
    assert not np.any(np.isnan(out)), "ONNX output contains NaN"
    print(f"  ONNX verification passed: shape={out.shape}")


def _lstm_forward_numpy(
    weights: dict,
    x: np.ndarray,
    num_layers: int,
    hidden_size: int,
) -> np.ndarray:
    """Numpy reconstruction of LSTM forward pass for verify_export.

    Implements the standard LSTM equations matching PyTorch's nn.LSTM:
        i = σ(W_ih x + W_hh h + b)
        f = σ(...)
        g = tanh(...)
        o = σ(...)
        c = f*c + i*g
        h = o * tanh(c)
    """

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    T = x.shape[0]   # (T, C)
    outputs = x

    for k in range(num_layers):
        layer_num = k + 1
        W_ih = weights[f'W_ih_{layer_num}']    # (4H, C_in)
        W_hh = weights[f'W_hh_{layer_num}']    # (4H, H)
        bias  = weights[f'bias_{layer_num}'].ravel()   # (4H,)

        H = hidden_size
        h = np.zeros(H)
        c = np.zeros(H)
        out_seq = []

        for t in range(T):
            xt = outputs[t]   # (C_in,)
            gates = W_ih @ xt + W_hh @ h + bias   # (4H,)
            i_g = sigmoid(gates[0:H])
            f_g = sigmoid(gates[H:2*H])
            g_g = np.tanh(gates[2*H:3*H])
            o_g = sigmoid(gates[3*H:4*H])
            c = f_g * c + i_g * g_g
            h = o_g * np.tanh(c)
            out_seq.append(h.copy())

        outputs = np.stack(out_seq)   # (T, H)

    # Linear head
    W_fc = weights['W_fc']           # (2, H)
    b_fc = weights['b_fc'].ravel()   # (2,)
    result = outputs @ W_fc.T + b_fc  # (T, 2)
    return result.T   # (2, T)


def verify_export(
    mat_path: str,
    model: PhaseLSTM,
    test_input: np.ndarray,
) -> float:
    """Verify LSTM export by comparing numpy reconstruction to torch output.

    Args:
        mat_path: Path to the exported .mat file.
        model: PhaseLSTM to compare against.
        test_input: (C, T) or (1, C, T) float32 input array.

    Returns:
        Maximum absolute difference between torch and numpy outputs.

    Raises:
        AssertionError if max diff >= 1e-4 (gate ordering error likely).
    """
    if test_input.ndim == 2:
        test_input_3d = test_input[np.newaxis, :, :]   # (1, C, T)
    else:
        test_input_3d = test_input

    # Torch reference output
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(test_input_3d.astype(np.float32))
        torch_out = model(x_t).numpy()[0]   # (2, T)

    # Load saved weights
    mat = scipy.io.loadmat(mat_path)
    num_layers = int(mat['num_layers'].flat[0])
    hidden_size = int(mat['hidden_size'].flat[0])

    # Numpy reconstruction
    x_np = test_input_3d[0].T   # (T, C)
    numpy_out = _lstm_forward_numpy(mat, x_np, num_layers, hidden_size)  # (2, T)

    max_diff = float(np.max(np.abs(torch_out - numpy_out)))
    assert max_diff < 1e-4, (
        f"Export verification FAILED: max diff={max_diff:.2e} >= 1e-4. "
        "Check gate ordering (should be [i,f,g,o] not TF-style [i,f,o,g])."
    )
    print(f"verify_export passed: max diff = {max_diff:.2e}")
    return max_diff
