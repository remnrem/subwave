import numpy as np
import pytest

from subwave import AxisAnnotatedTensor, from_array, from_npz
from subwave.core import EventMatrix


class TestFromArray:
    def test_returns_event_matrix(self, small_matrix):
        em = from_array(small_matrix, sfreq=256.0)
        assert isinstance(em, EventMatrix)
        assert em.sfreq == 256.0

    def test_default_sfreq(self, small_matrix):
        em = from_array(small_matrix)
        assert em.sfreq == 1.0

    def test_integer_input_converted(self, small_matrix):
        em = from_array((small_matrix * 100).astype(int))
        assert em.data.dtype == float

    def test_list_input(self):
        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        em = from_array(X)
        assert em.n_events == 2
        assert em.n_samples == 3


class TestDecomposeEventMatrix:
    def test_decompose_dispatches_to_event_matrix(self, event_matrix):
        """sw.decompose() accepts an EventMatrix and routes through its own
        decompose() method (core.decompose EventMatrix branch)."""
        import subwave as sw
        from subwave.result import DecompositionResult

        result = sw.decompose(event_matrix, method="svd", n_components=3)
        assert isinstance(result, DecompositionResult)
        assert result.templates.shape == (3, event_matrix.n_samples)
        assert result.loadings.shape == (event_matrix.n_events, 3)

    def test_matches_direct_event_matrix_decompose(self, event_matrix):
        import subwave as sw

        via_dispatch = sw.decompose(event_matrix, method="svd", n_components=2)
        direct = event_matrix.decompose(method="svd", n_components=2)
        np.testing.assert_allclose(via_dispatch.templates, direct.templates)


class TestFromNpz:
    def test_returns_aat(self, spindle_path):
        aat = from_npz(spindle_path)
        assert isinstance(aat, AxisAnnotatedTensor)

    def test_axes(self, spindle_path):
        aat = from_npz(spindle_path)
        assert aat.axes == ["instance", "sample"]

    def test_spindle_shape(self, spindle_path):
        aat = from_npz(spindle_path)
        assert aat.shape == (1972, 513)

    def test_ecg_shape(self, ecg_path):
        aat = from_npz(ecg_path)
        assert aat.shape == (42335, 513)

    def test_sfreq_inferred(self, spindle_path):
        aat = from_npz(spindle_path)
        assert abs(aat.attrs["sfreq"] - 256.0) < 1.0

    def test_time_axis_range(self, spindle_path):
        aat = from_npz(spindle_path)
        t = aat.axis_index["sample"]
        assert abs(t[0] - (-1.0)) < 0.01
        assert abs(t[-1] - 1.0) < 0.01

    def test_instance_ids(self, spindle_path):
        aat = from_npz(spindle_path)
        np.testing.assert_array_equal(aat.axis_index["instance"], np.arange(1972))

    def test_instance_meta(self, spindle_path):
        aat = from_npz(spindle_path)
        assert "instance_id" in aat.axis_meta["instance"].columns

    def test_sample_meta(self, spindle_path):
        aat = from_npz(spindle_path)
        assert "time" in aat.axis_meta["sample"].columns

    def test_attrs_populated(self, spindle_path):
        aat = from_npz(spindle_path)
        assert "channel_names" in aat.attrs
        assert "x_label" in aat.attrs
        assert "value_label" in aat.attrs

    def test_decomposable(self, spindle_path):
        """from_npz output feeds directly into decompose()."""
        import subwave as sw
        aat = from_npz(spindle_path)
        result = sw.decompose(aat, method="svd", n_components=5)
        assert result.templates.shape == (5, 513)
        assert result.loadings.shape == (1972, 5)

    def test_ecg_decomposable(self, ecg_path):
        import subwave as sw
        aat = from_npz(ecg_path)
        result = sw.decompose(aat, method="svd", n_components=3)
        assert result.templates.shape == (3, 513)
        assert result.loadings.shape == (42335, 3)


class TestFromYasa:
    def test_basic_extraction(self, rng):
        import pandas as pd
        from subwave import from_yasa

        sfreq = 256.0
        duration = 60.0
        raw = np.sin(2 * np.pi * 13 * np.arange(int(sfreq * duration)) / sfreq)
        peaks = [5.0, 10.0, 15.0, 20.0, 25.0]
        df = pd.DataFrame({"Peak": peaks})

        em = from_yasa(df, raw, sfreq=sfreq, window_sec=0.5)
        assert em.n_events == 5
        assert em.n_samples == 2 * int(0.5 * sfreq)

    def test_skips_out_of_bounds(self, rng):
        import pandas as pd
        from subwave import from_yasa

        sfreq = 256.0
        raw = np.zeros(int(sfreq * 10))
        df = pd.DataFrame({"Peak": [0.1, 5.0, 9.9]})
        em = from_yasa(df, raw, sfreq=sfreq, window_sec=1.0)
        assert em.n_events == 1  # only 5.0 fits

    def test_raises_when_no_valid_windows(self):
        import pandas as pd
        from subwave import from_yasa

        sfreq = 256.0
        raw = np.zeros(100)
        df = pd.DataFrame({"Peak": [0.0]})
        with pytest.raises(ValueError, match="No valid"):
            from_yasa(df, raw, sfreq=sfreq, window_sec=1.0)


class TestFromMne:
    def test_requires_mne(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "mne", None)
        from subwave import from_mne
        with pytest.raises(ImportError, match="mne"):
            from_mne(object())


class TestFromEdfBatch:
    def _make_edf(self, tmp_path, name, sfreq, duration, channel, freq_hz, rng):
        mne = pytest.importorskip("mne")
        n_samples = int(sfreq * duration)
        t = np.arange(n_samples) / sfreq
        data = (np.sin(2 * np.pi * freq_hz * t) + rng.normal(0, 0.05, n_samples))[None, :]
        info = mne.create_info(ch_names=[channel], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        path = tmp_path / name
        try:
            mne.export.export_raw(str(path), raw, fmt="edf", overwrite=True, verbose=False)
        except Exception as e:
            pytest.skip(f"EDF export not available: {e}")
        return path

    def test_basic_batch(self, tmp_path, rng):
        pytest.importorskip("mne")
        pytest.importorskip("yasa")
        channel = "C3"
        sfreq = 200.0
        p1 = self._make_edf(tmp_path, "subj1.edf", sfreq, 30.0, channel, 13.0, rng)
        p2 = self._make_edf(tmp_path, "subj2.edf", sfreq, 30.0, channel, 12.0, rng)

        from subwave import from_edf_batch
        try:
            em = from_edf_batch([p1, p2], channel=channel, window_sec=0.5)
        except ValueError as e:
            if "No events" in str(e):
                pytest.skip("YASA detected no events on synthetic signal")
            raise

        expected_samples = 2 * int(0.5 * sfreq)
        assert em.data.shape[1] == expected_samples
        assert em.meta is not None
        assert list(em.meta.columns) == ["subject", "file", "event_index", "peak_sec"]
        assert len(em.meta) == em.data.shape[0]
        assert set(em.meta["subject"].unique()).issubset({0, 1})

    def test_unknown_detector(self, tmp_path, rng):
        pytest.importorskip("mne")
        channel = "C3"
        sfreq = 200.0
        p = self._make_edf(tmp_path, "subj.edf", sfreq, 5.0, channel, 13.0, rng)

        from subwave import from_edf_batch
        with pytest.raises(ValueError, match="Unknown detector"):
            from_edf_batch([p], channel=channel, detector="bogus")
