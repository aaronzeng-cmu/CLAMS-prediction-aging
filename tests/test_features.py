"""Tests for causal feature extraction (Verification plan #1)."""
import numpy as np
import pytest
from src.training.config import FeatureConfig
from src.training.features import CausalBandpassFilter, extract_features, build_feature_matrix_offline


FS = 200
F_SO = 0.75


def make_sine(freq=F_SO, fs=FS, duration_s=10, phase_offset=0.0):
    t = np.arange(duration_s * fs) / fs
    return np.sin(2 * np.pi * freq * t + phase_offset).astype(np.float32)


class TestCausalBandpassFilter:
    def test_initialization(self):
        f = CausalBandpassFilter(FS, 0.4, 1.2, order=4)
        assert f._zi is not None
        assert f._zi.shape[1] == 2  # SOS form: 2 states per section

    def test_reset_zeros_state(self):
        f = CausalBandpassFilter(FS, 0.4, 1.2, order=4)
        f.apply(make_sine())  # process some data to change state
        f.reset()
        assert np.all(f._zi == 0)

    def test_stateful_across_chunks(self):
        """State is preserved across chunk calls — different from re-init."""
        sig = make_sine()
        mid = len(sig) // 2

        # Single pass
        f1 = CausalBandpassFilter(FS, 0.4, 1.2, order=4)
        out_full = f1.apply(sig)

        # Two chunks
        f2 = CausalBandpassFilter(FS, 0.4, 1.2, order=4)
        out1 = f2.apply(sig[:mid])
        out2 = f2.apply(sig[mid:])
        out_chunked = np.concatenate([out1, out2])

        assert np.allclose(out_full, out_chunked, atol=1e-5), (
            "Chunked and full-pass outputs must be identical"
        )

    def test_passband_preserves_so_sine(self):
        """0.75 Hz sine RMS is preserved by causal bandpass (plan #1 amplitude check).

        A causal filter has group delay, so sample-by-sample comparison to the
        input is not meaningful. The verification criterion is that the RMS
        amplitude is preserved within 5% after the transient settles.
        """
        sig = make_sine(F_SO)
        f = CausalBandpassFilter(FS, 0.4, 1.2, order=4)
        out = f.apply(sig)
        # Skip first 4 s to allow group delay transient to settle
        skip = 4 * FS
        rms_in  = np.sqrt(np.mean(sig[skip:] ** 2))
        rms_out = np.sqrt(np.mean(out[skip:] ** 2))
        rel_err = abs(rms_in - rms_out) / rms_in
        assert rel_err < 0.05, f"Causal bandpass RMS deviation: {rel_err:.3f} > 5%"

    def test_output_dtype(self):
        f = CausalBandpassFilter(FS, 0.4, 1.2, order=4)
        out = f.apply(make_sine())
        assert out.dtype == np.float32


class TestExtractFeatures:
    def test_shape_default_features(self):
        """Default 3 features -> shape (3, T)."""
        sig = make_sine(duration_s=5)
        chunk_with_prev = np.concatenate([[0.0], sig])
        f = CausalBandpassFilter(FS, 0.4, 1.2, order=4)
        out = extract_features(chunk_with_prev, f, ('raw', 'diff', 'bandpass'))
        assert out.shape == (3, len(sig))

    def test_raw_feature(self):
        """Raw feature equals chunk_with_prev[1:]."""
        sig = np.random.randn(100).astype(np.float32)
        prev = np.array([0.0], dtype=np.float32)
        cwp = np.concatenate([prev, sig])
        f = CausalBandpassFilter(FS, 0.4, 1.2, order=4)
        out = extract_features(cwp, f, ('raw',))
        assert np.allclose(out[0], sig), "Raw feature mismatch"

    def test_diff_feature(self):
        """Diff feature equals np.diff(chunk_with_prev)."""
        sig = np.arange(10, dtype=np.float32)
        prev = np.array([-1.0], dtype=np.float32)
        cwp = np.concatenate([prev, sig])
        f = CausalBandpassFilter(FS, 0.4, 1.2, order=4)
        out = extract_features(cwp, f, ('diff',))
        expected = np.diff(cwp.astype(np.float32))
        assert np.allclose(out[0], expected), "Diff feature mismatch"

    def test_optional_so_envelope(self):
        """so_envelope feature has correct shape."""
        sig = make_sine(duration_s=3)
        cwp = np.concatenate([[0.0], sig])
        f = CausalBandpassFilter(FS, 0.4, 1.2, order=4)
        out = extract_features(cwp, f, ('raw', 'diff', 'bandpass', 'so_envelope'))
        assert out.shape == (4, len(sig))
        # Envelope should be non-negative
        assert np.all(out[3] >= 0), "Envelope contains negative values"


class TestBuildFeatureMatrixOffline:
    def test_shape(self):
        sig = make_sine(duration_s=5)
        cfg = FeatureConfig()
        out = build_feature_matrix_offline(sig, cfg)
        assert out.shape == (len(cfg.feature_names), len(sig))

    def test_causality_equivalent_to_chunked(self):
        """Offline single-pass must match concatenated chunk processing."""
        sig = make_sine(duration_s=5)
        cfg = FeatureConfig()

        # Offline (single pass)
        out_offline = build_feature_matrix_offline(sig, cfg)

        # Chunked (simulate online)
        from src.training.features import CausalBandpassFilter, extract_features
        bp = CausalBandpassFilter(cfg.fs, cfg.bp_lo, cfg.bp_hi, cfg.bp_order)
        T = len(sig)
        chunk_size = 10
        chunks_out = []
        prev = 0.0
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            cwp = np.concatenate([[prev], sig[start:end]])
            feat = extract_features(cwp, bp, cfg.feature_names)
            chunks_out.append(feat)
            prev = sig[end - 1]
        out_chunked = np.concatenate(chunks_out, axis=1)

        assert np.allclose(out_offline, out_chunked, atol=1e-4), (
            f"Offline vs chunked max diff: {np.max(np.abs(out_offline - out_chunked)):.2e}"
        )
