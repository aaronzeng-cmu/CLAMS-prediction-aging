"""Tests for label generation (Verification plan #2)."""
import numpy as np
import pytest
from src.training.config import LabelConfig
from src.training.labels import bandpass_filtfilt, compute_instantaneous_phase, generate_labels


FS = 200
T_SEC = 10
T = T_SEC * FS
F_SO = 0.75   # Hz — typical aging SO frequency


def make_sine(freq=F_SO, fs=FS, duration_s=T_SEC, phase_offset=0.0):
    t = np.arange(duration_s * fs) / fs
    return np.sin(2 * np.pi * freq * t + phase_offset)


class TestBandpassFiltfilt:
    def test_passband_sine_preserved(self):
        """0.75 Hz sine passes through 0.4–1.2 Hz filter with amplitude ~1.

        filtfilt is zero-phase so edge effects are the main source of deviation.
        Use a 30 s signal and skip the first 5 s to ensure steady-state.
        """
        sig = make_sine(F_SO, duration_s=30)
        filt = bandpass_filtfilt(sig, FS, 0.4, 1.2, order=4)
        skip = 5 * FS
        rms_in  = np.sqrt(np.mean(sig[skip:]**2))
        rms_out = np.sqrt(np.mean(filt[skip:]**2))
        assert abs(rms_in - rms_out) / rms_in < 0.05, (
            f"Passband attenuation too large: {rms_out/rms_in:.3f}"
        )

    def test_stopband_attenuated(self):
        """5 Hz signal (well above passband) is heavily attenuated."""
        sig = make_sine(5.0)
        filt = bandpass_filtfilt(sig, FS, 0.4, 1.2, order=4)
        rms_out = np.sqrt(np.mean(filt[2*FS:]**2))
        assert rms_out < 0.05, f"Stopband attenuation insufficient: rms={rms_out:.4f}"

    def test_output_shape(self):
        sig = make_sine()
        out = bandpass_filtfilt(sig, FS, 0.4, 1.2, order=4)
        assert out.shape == sig.shape


class TestInstantaneousPhase:
    def test_sine_phase_linear(self):
        """Phase of a clean cosine should match 2*pi*f*t (Hilbert convention).

        We use a cosine rather than a sine because Hilbert(cos(wt)) gives
        analytic signal with phase = wt exactly, avoiding the -pi/2 offset
        that Hilbert(sin(wt)) introduces.
        """
        f = F_SO
        t = np.arange(T) / FS
        expected_phase = 2 * np.pi * f * t
        sig = np.cos(expected_phase)
        phase = compute_instantaneous_phase(sig)
        # After transient (skip first 2 cycles), check wrapped difference
        skip = int(2 / f * FS)
        diff = np.abs(np.arctan2(
            np.sin(phase[skip:] - expected_phase[skip:]),
            np.cos(phase[skip:] - expected_phase[skip:]),
        ))
        assert np.mean(diff) < 0.2, f"Phase deviation: mean={np.mean(diff):.3f} rad"

    def test_output_range(self):
        phase = compute_instantaneous_phase(make_sine())
        assert np.all(phase >= -np.pi - 1e-9) and np.all(phase <= np.pi + 1e-9)


class TestGenerateLabels:
    def test_shape(self):
        """Labels have shape (2, T)."""
        sig = make_sine()
        cfg = LabelConfig(shift_ms=100)
        labels = generate_labels(sig, FS, cfg)
        assert labels.shape == (2, T)

    def test_nan_at_end(self):
        """Last shift_samples entries are NaN."""
        sig = make_sine()
        shift_ms = 100
        shift_samples = round(shift_ms * FS / 1000)
        cfg = LabelConfig(shift_ms=shift_ms)
        labels = generate_labels(sig, FS, cfg)
        assert np.all(np.isnan(labels[:, -shift_samples:])), (
            "Last shift_samples entries must be NaN"
        )

    def test_valid_entries_not_nan(self):
        sig = make_sine()
        cfg = LabelConfig(shift_ms=100)
        labels = generate_labels(sig, FS, cfg)
        shift_samples = round(100 * FS / 1000)
        assert not np.any(np.isnan(labels[:, :-shift_samples]))

    def test_unit_norm(self):
        """(cos, sin) pairs should lie on the unit circle."""
        sig = make_sine()
        cfg = LabelConfig(shift_ms=50)
        labels = generate_labels(sig, FS, cfg)
        shift_samples = round(50 * FS / 1000)
        valid = labels[:, :-shift_samples]
        norms = np.sqrt(valid[0]**2 + valid[1]**2)
        assert np.allclose(norms, 1.0, atol=1e-5), f"Max norm deviation: {np.max(np.abs(norms-1)):.2e}"

    def test_shift_accuracy(self):
        """cos/sin labels match cos/sin of phase shifted by shift_samples (plan #2)."""
        f = F_SO
        shift_ms = 100
        shift_samples = round(shift_ms * FS / 1000)

        from src.training.labels import bandpass_filtfilt, compute_instantaneous_phase
        sig = make_sine(f)
        cfg = LabelConfig(shift_ms=shift_ms)
        labels = generate_labels(sig, FS, cfg)

        filt = bandpass_filtfilt(sig, FS, cfg.label_bp_lo, cfg.label_bp_hi, cfg.label_bp_order)
        phase = compute_instantaneous_phase(filt)

        skip = 2 * FS   # skip filter transient
        T_valid = T - shift_samples
        err_cos = np.abs(labels[0, skip:T_valid] - np.cos(phase[shift_samples + skip:]))
        err_sin = np.abs(labels[1, skip:T_valid] - np.sin(phase[shift_samples + skip:]))

        assert np.mean(err_cos) < 0.02, f"cos label error: {np.mean(err_cos):.4f}"
        assert np.mean(err_sin) < 0.02, f"sin label error: {np.mean(err_sin):.4f}"
