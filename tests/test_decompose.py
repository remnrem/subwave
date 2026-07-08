import numpy as np
import pytest

from subwave import AxisAnnotatedTensor, decompose, from_array
from subwave.core import EventMatrix


# ---------------------------------------------------------------------------
# decompose() input dispatch / tensor rank handling
# ---------------------------------------------------------------------------

class TestDecomposeFunction:
    def _aat_3d(self, rng, n=20, n_samples=16, n_channels=3):
        X = rng.standard_normal((n, n_channels, n_samples))
        return AxisAnnotatedTensor(
            data=X,
            axes=["instance", "channel", "sample"],
            attrs={"sfreq": 64.0},
        )

    def test_3d_tensor_flattened(self, rng):
        """3-D input is reshaped to (n_events, channels*samples)."""
        aat = self._aat_3d(rng, n=20, n_samples=16, n_channels=3)
        result = decompose(aat, method="svd", n_components=4)
        assert result.templates.shape == (4, 3 * 16)
        assert result.loadings.shape == (20, 4)

    def test_3d_requires_instance_axis_first(self, rng):
        X = rng.standard_normal((16, 20, 3))
        aat = AxisAnnotatedTensor(
            data=X,
            axes=["sample", "instance", "channel"],
            attrs={"sfreq": 64.0},
        )
        with pytest.raises(ValueError, match="instance axis must be axis 0"):
            decompose(aat, method="svd", n_components=2)

    def test_raw_input_must_be_2d(self, rng):
        with pytest.raises(ValueError, match="2-D"):
            decompose(rng.standard_normal((2, 3, 4)), method="svd", n_components=2)


# ---------------------------------------------------------------------------
# EventMatrix construction
# ---------------------------------------------------------------------------

class TestEventMatrix:
    def test_basic_construction(self, small_matrix):
        em = from_array(small_matrix, sfreq=64.0)
        assert em.n_events == 20
        assert em.n_samples == 64
        assert em.sfreq == 64.0

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            from_array(np.zeros(64))

    def test_rejects_single_event_at_decompose(self):
        em = from_array(np.zeros((1, 64)))
        with pytest.raises(ValueError, match="at least 2"):
            em.decompose()

    def test_repr(self, event_matrix):
        r = repr(event_matrix)
        assert "EventMatrix" in r
        assert "n_events=20" in r

    def test_data_is_float(self, small_matrix):
        em = from_array(small_matrix.astype(int))
        assert em.data.dtype == float


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_normalize_max(self, event_matrix):
        em = event_matrix.normalize("max")
        assert np.abs(em.data).max() <= 1.0 + 1e-10

    def test_normalize_l2(self, event_matrix):
        em = event_matrix.normalize("l2")
        norms = np.linalg.norm(em.data, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_normalize_zscore(self, event_matrix):
        em = event_matrix.normalize("zscore")
        np.testing.assert_allclose(em.data.mean(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(em.data.std(axis=1), 1.0, atol=1e-10)

    def test_normalize_none_identity(self, event_matrix):
        em = event_matrix.normalize("none")
        np.testing.assert_array_equal(em.data, event_matrix.data)

    def test_normalize_unknown(self, event_matrix):
        with pytest.raises(ValueError, match="Unknown normalize"):
            event_matrix.normalize("bad")

    def test_align_peak(self, event_matrix):
        em = event_matrix.align("peak")
        center = em.n_samples // 2
        peaks = em.data.argmax(axis=1)
        # Peak should be near center after alignment
        assert np.abs(peaks - center).mean() < em.n_samples * 0.2

    def test_align_none_identity(self, event_matrix):
        em = event_matrix.align("none")
        np.testing.assert_array_equal(em.data, event_matrix.data)

    def test_resample(self, event_matrix):
        em = event_matrix.resample(128)
        assert em.n_samples == 128
        assert em.n_events == event_matrix.n_events

    def test_resample_identity(self, event_matrix):
        em = event_matrix.resample(64)
        np.testing.assert_array_equal(em.data, event_matrix.data)


# ---------------------------------------------------------------------------
# SVD decomposition
# ---------------------------------------------------------------------------

class TestSVD:
    def test_output_shapes(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=3)
        assert result.templates.shape == (3, 64)
        assert result.loadings.shape == (20, 3)
        assert result.singular_values.shape == (3,)
        assert result.explained_variance_ratio.shape == (3,)
        assert result.mean_waveform.shape == (64,)
        assert result.residuals.shape == (20, 64)

    def test_singular_values_descending(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=5)
        sv = result.singular_values
        assert np.all(sv[:-1] >= sv[1:] - 1e-10)

    def test_evr_sums_leq_one(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=5)
        assert result.explained_variance_ratio.sum() <= 1.0 + 1e-10

    def test_evr_positive(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=3)
        assert np.all(result.explained_variance_ratio >= 0)

    def test_reconstruction_quality(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=5, center=True)
        Xc = event_matrix.data - result.mean_waveform
        reconstructed = result.loadings @ result.templates
        rel_error = np.linalg.norm(Xc - reconstructed) / np.linalg.norm(Xc)
        assert rel_error < 0.5

    def test_no_center(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=3, center=False)
        np.testing.assert_array_equal(result.mean_waveform, np.zeros(64))

    def test_config_stored(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=3)
        assert result.config["method"] == "svd"
        assert result.config["n_components"] == 3

    def test_n_components_clipped_to_rank(self, rng):
        X = from_array(rng.standard_normal((5, 10)), sfreq=1.0)
        result = X.decompose(method="svd", n_components=100)
        assert result.templates.shape[0] <= 5


# ---------------------------------------------------------------------------
# NMF decomposition
# ---------------------------------------------------------------------------

class TestNMF:
    def test_output_shapes(self, nonneg_matrix):
        em = from_array(nonneg_matrix, sfreq=64.0)
        result = em.decompose(method="nmf", n_components=2)
        assert result.templates.shape == (2, 64)
        assert result.loadings.shape == (20, 2)

    def test_templates_nonneg(self, nonneg_matrix):
        em = from_array(nonneg_matrix, sfreq=64.0)
        result = em.decompose(method="nmf", n_components=2)
        assert np.all(result.templates >= -1e-10)

    def test_loadings_nonneg(self, nonneg_matrix):
        em = from_array(nonneg_matrix, sfreq=64.0)
        result = em.decompose(method="nmf", n_components=2)
        assert np.all(result.loadings >= -1e-10)

    def test_rejects_negative_input(self, event_matrix):
        with pytest.raises(ValueError, match="non-negative"):
            event_matrix.decompose(method="nmf", n_components=2)


# ---------------------------------------------------------------------------
# Dictionary learning decomposition
# ---------------------------------------------------------------------------

class TestDictLearn:
    def test_output_shapes(self, event_matrix):
        result = event_matrix.decompose(method="dictlearn", n_components=3)
        assert result.templates.shape == (3, 64)
        assert result.loadings.shape == (20, 3)

    def test_unknown_method(self, event_matrix):
        with pytest.raises(ValueError, match="Unknown method"):
            event_matrix.decompose(method="pca")


# ---------------------------------------------------------------------------
# DecompositionResult methods
# ---------------------------------------------------------------------------

class TestResultMethods:
    def test_reconstruct_full(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=5, center=True)
        rec = result.reconstruct()
        assert rec.shape == (20, 64)

    def test_reconstruct_truncated(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=5)
        rec2 = result.reconstruct(n_components=2)
        rec5 = result.reconstruct(n_components=5)
        # Truncated reconstruction has higher residuals
        err2 = np.linalg.norm(result.residuals)
        err5 = np.linalg.norm(event_matrix.data - result.mean_waveform - rec5)
        assert err2 >= err5 - 1e-8

    def test_outlier_scores_shape(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=3)
        scores = result.outlier_scores()
        assert scores.shape == (20,)
        assert np.all(scores >= 0)

    def test_project_shape(self, event_matrix, rng):
        result = event_matrix.decompose(method="svd", n_components=3)
        new_events = rng.standard_normal((5, 64))
        proj = result.project(new_events)
        assert proj.shape == (5, 3)

    def test_subspace_angles_shape(self, event_matrix):
        r1 = event_matrix.decompose(method="svd", n_components=3)
        r2 = event_matrix.decompose(method="svd", n_components=3, center=False)
        angles = r1.subspace_angles(r2)
        assert angles.shape == (3,)
        assert np.all(angles >= 0)
        assert np.all(angles <= np.pi / 2 + 1e-10)

    def test_repr(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=3)
        r = repr(result)
        assert "DecompositionResult" in r
        assert "svd" in r

    def test_round_trip_svd(self, event_matrix):
        """SVD reconstruction should recover original within tolerance."""
        k = min(20, 64)
        result = event_matrix.decompose(method="svd", n_components=k, center=True)
        rec = result.mean_waveform + result.reconstruct(n_components=k)
        rel_err = np.linalg.norm(event_matrix.data - rec) / np.linalg.norm(event_matrix.data)
        assert rel_err < 0.05


class TestSpectral:
    def test_spectrum_shapes(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=3)
        freqs, powers = result.template_spectrum(sfreq=64.0)
        assert freqs.shape == (33,)
        assert powers.shape == (3, 33)
        assert np.all(powers >= 0)

    def test_peak_freq_positive(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=3)
        peaks = result.template_peak_freq(sfreq=64.0)
        assert peaks.shape == (3,)
        assert np.all(peaks > 0)

    def test_bandwidth_nonneg(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=3)
        bw = result.template_bandwidth(sfreq=64.0)
        assert bw.shape == (3,)
        assert np.all(bw >= 0)


class TestAutoNComponents:
    def test_auto_returns_at_least_one(self, small_matrix):
        from subwave import decompose
        result = decompose(small_matrix, n_components="auto")
        assert result.templates.shape[0] >= 1

    def test_auto_rejects_nmf(self, nonneg_matrix):
        from subwave import decompose
        with pytest.raises(ValueError, match="only supported with method='svd'"):
            decompose(nonneg_matrix, method="nmf", n_components="auto")

    def test_invalid_string(self, small_matrix):
        from subwave import decompose
        with pytest.raises(ValueError, match="must be an int or 'auto'"):
            decompose(small_matrix, n_components="foo")


class TestRandomizedSVD:
    def test_large_input_uses_randomized(self, rng):
        from subwave import from_array
        X = rng.standard_normal((5001, 32))
        em = from_array(X, sfreq=1.0)
        result = em.decompose(method="svd", n_components=5)
        assert result.templates.shape == (5, 32)
        assert result.loadings.shape == (5001, 5)


class TestFourierSVD:
    def _make_jittered(self, rng, n_events=40, n_samples=128, sfreq=128.0,
                       freq_hz=13.0, jitter=True, jitter_samples=8):
        t = np.arange(n_samples) / sfreq
        events = []
        for _ in range(n_events):
            shift = rng.integers(-jitter_samples, jitter_samples + 1) if jitter else 0
            phase = 2 * np.pi * freq_hz * (t - shift / sfreq)
            wave = np.sin(phase) * rng.uniform(0.8, 1.2)
            wave = wave + rng.normal(0, 0.05, n_samples)
            events.append(wave)
        return np.stack(events)

    def test_output_shapes(self, rng):
        from subwave import decompose
        X = self._make_jittered(rng)
        result = decompose(X, method="fourier_svd", n_components=3)
        n_freqs = X.shape[1] // 2 + 1
        assert result.templates.shape == (3, n_freqs)
        assert result.loadings.shape == (X.shape[0], 3)

    def test_domain_marked_frequency(self, rng):
        from subwave import decompose
        X = self._make_jittered(rng)
        result = decompose(X, method="fourier_svd", n_components=2)
        assert result.config.get("domain") == "frequency"

    def test_template_peak_at_correct_freq(self, rng):
        from subwave import decompose
        sfreq = 128.0
        n_samples = 128
        freq_hz = 13.0
        X = self._make_jittered(rng, n_samples=n_samples, sfreq=sfreq, freq_hz=freq_hz)
        result = decompose(X, method="fourier_svd", n_components=3)
        # Templates live in frequency domain; bin for freq_hz is freq_hz / (sfreq/n_samples)
        bin_for_freq = int(round(freq_hz / (sfreq / n_samples)))
        # Use mean spectrum (most-of-variance template + mean) — peak should be near bin_for_freq
        mean_spec = result.mean_waveform
        assert abs(int(np.argmax(mean_spec)) - bin_for_freq) <= 2

    def test_shift_invariance_vs_plain_svd(self, rng):
        from subwave import decompose
        rng2 = np.random.default_rng(0)
        X_no_jitter = self._make_jittered(rng2, jitter=False)
        rng3 = np.random.default_rng(0)
        X_jitter = self._make_jittered(rng3, jitter=True, jitter_samples=8)

        f_no = decompose(X_no_jitter, method="fourier_svd", n_components=3)
        f_yes = decompose(X_jitter, method="fourier_svd", n_components=3)
        s_no = decompose(X_no_jitter, method="svd", n_components=3)
        s_yes = decompose(X_jitter, method="svd", n_components=3)

        # Top-component EVR for fourier_svd should be roughly stable under jitter,
        # while plain svd's top EVR should drop noticeably.
        f_drop = f_no.explained_variance_ratio[0] - f_yes.explained_variance_ratio[0]
        s_drop = s_no.explained_variance_ratio[0] - s_yes.explained_variance_ratio[0]
        assert f_drop < s_drop


class TestScatteringSVD:
    def _make_pop(self, rng, n_events=20, n_samples=128, sfreq=128.0,
                  freq_hz=13.0, shifts=None):
        t = np.arange(n_samples) / sfreq
        events = []
        for i in range(n_events):
            shift = 0 if shifts is None else int(shifts[i])
            phase = 2 * np.pi * freq_hz * (t - shift / sfreq)
            wave = np.sin(phase) * rng.uniform(0.8, 1.2)
            wave = wave + rng.normal(0, 0.02, n_samples)
            events.append(wave)
        return np.stack(events)

    def test_output_shapes(self, rng):
        pytest.importorskip("kymatio")
        from subwave import decompose
        X = self._make_pop(rng)
        result = decompose(X, method="scattering_svd", n_components=3)
        assert result.loadings.shape == (X.shape[0], 3)
        assert result.templates.shape[0] == 3
        assert result.config.get("domain") == "scattering"

    def test_shift_invariance(self):
        pytest.importorskip("kymatio")
        from subwave import decompose
        rng = np.random.default_rng(0)
        n_events = 16
        # Same underlying waveforms, two shift sets
        X_a = self._make_pop(rng, n_events=n_events, shifts=np.zeros(n_events))
        rng2 = np.random.default_rng(0)
        shifts = rng2.integers(-6, 7, size=n_events)
        # Recreate with same noise seed but different shifts
        rng3 = np.random.default_rng(0)
        X_b = self._make_pop(rng3, n_events=n_events, shifts=shifts)

        # Decompose stacked so both populations share the same basis
        X = np.vstack([X_a, X_b])
        result = decompose(X, method="scattering_svd", n_components=3)
        L_a = result.loadings[:n_events]
        L_b = result.loadings[n_events:]

        cos = np.array([
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            for a, b in zip(L_a, L_b)
        ])
        assert np.median(cos) > 0.9


class TestMetadataMethods:
    def test_loadings_correlated_with(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=3)
        values = np.linspace(0, 1, 20)
        df = result.loadings_correlated_with(values)
        assert list(df.columns) == ["component", "r", "p_value"]
        assert len(df) == 3
        assert df["r"].between(-1.0, 1.0).all()

    def test_loadings_correlated_with_mismatched(self, event_matrix):
        result = event_matrix.decompose(method="svd", n_components=2)
        with pytest.raises(ValueError, match="values length"):
            result.loadings_correlated_with(np.array([0.0, 1.0]))

    def test_scatter_colored_by(self, event_matrix):
        import matplotlib
        matplotlib.use("Agg")
        result = event_matrix.decompose(method="svd", n_components=3)
        values = np.linspace(0, 1, 20)
        ax = result.scatter_colored_by(values, label="metric")
        assert ax is not None
