import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest

from subwave import TensorResult


def _tensorly_or_skip():
    return pytest.importorskip("tensorly")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _make_result(rng, *, method="cp", n_events=20, n_samples=32, n_channels=4, rank=2):
    """Build a TensorResult directly (no tensorly needed) for unit tests of
    its presentation/helper methods."""
    core = None
    if method == "tucker":
        core = rng.standard_normal((rank, rank, rank))
    return TensorResult(
        method=method,
        event_factors=rng.standard_normal((n_events, rank)),
        temporal_factors=rng.standard_normal((n_samples, rank)),
        spatial_factors=rng.standard_normal((n_channels, rank)),
        core=core,
        reconstruction_error=0.123,
        config={"method": method, "rank": rank},
    )


class TestTensorResultMethods:
    def test_rank_property(self, rng):
        res = _make_result(rng, rank=3)
        assert res.rank == 3

    def test_component_waveform_returns_temporal(self, rng):
        res = _make_result(rng, n_samples=32)
        wf = res.component_waveform(0)
        np.testing.assert_array_equal(wf, res.temporal_factors[:, 0])

    def test_component_waveform_scaled_by_channel(self, rng):
        res = _make_result(rng)
        wf = res.component_waveform(1, channel=2)
        expected = res.temporal_factors[:, 1] * res.spatial_factors[2, 1]
        np.testing.assert_allclose(wf, expected)

    def test_repr_cp(self, rng):
        res = _make_result(rng, method="cp")
        text = repr(res)
        assert "method='cp'" in text
        assert "recon_error=0.1230" in text

    def test_repr_handles_missing_error(self, rng):
        res = _make_result(rng)
        res.reconstruction_error = None
        assert "recon_error=n/a" in repr(res)


class TestTensorResultPlot:
    def test_returns_axes_grid(self, rng):
        res = _make_result(rng, rank=2)
        axes = res.plot()
        assert axes.shape == (2, 3)

    def test_single_component(self, rng):
        """rank=1 path reshapes the 1-D axes array to 2-D."""
        res = _make_result(rng, rank=1)
        axes = res.plot()
        assert axes.shape == (1, 3)

    def test_rank_limit(self, rng):
        res = _make_result(rng, rank=4)
        axes = res.plot(rank=2)
        assert axes.shape == (2, 3)

    def test_custom_time_and_labels(self, rng):
        res = _make_result(rng, n_samples=32, n_channels=4, rank=2)
        t = np.linspace(-1, 1, 32)
        axes = res.plot(time=t, ch_labels=["a", "b", "c", "d"])
        assert axes[-1, 0].get_xlabel() == "Time (s)"

    def test_default_xlabel_is_samples(self, rng):
        res = _make_result(rng, rank=1)
        axes = res.plot()
        assert axes[-1, 0].get_xlabel() == "Samples"


class TestCPBasic:
    def test_shapes_and_recon_error(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((50, 64, 3))
        result = tensor_decompose(X, method="cp", rank=2, random_state=0)

        assert result.event_factors.shape == (50, 2)
        assert result.temporal_factors.shape == (64, 2)
        assert result.spatial_factors.shape == (3, 2)
        assert result.core is None
        assert result.reconstruction_error < 1.0


class TestTuckerBasic:
    def test_shapes_and_core(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((50, 64, 3))
        result = tensor_decompose(X, method="tucker", rank=2, random_state=0)

        assert result.event_factors.shape == (50, 2)
        assert result.temporal_factors.shape == (64, 2)
        assert result.spatial_factors.shape == (3, 2)
        assert result.core is not None
        assert result.core.shape == (2, 2, 2)


class TestCPSynthetic:
    def test_recovers_low_rank_signal(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        n_events, n_samples, n_channels = 40, 64, 3
        # Two rank-1 components
        e1 = rng.standard_normal(n_events)
        t1 = np.sin(np.linspace(0, 4 * np.pi, n_samples))
        c1 = np.array([1.0, 0.5, 0.2])

        e2 = rng.standard_normal(n_events)
        t2 = np.cos(np.linspace(0, 6 * np.pi, n_samples))
        c2 = np.array([0.2, 1.0, 0.5])

        X = (
            np.einsum("e,s,c->esc", e1, t1, c1)
            + np.einsum("e,s,c->esc", e2, t2, c2)
        )
        X = X + rng.normal(0, 0.01, X.shape)

        result = tensor_decompose(X, method="cp", rank=2, random_state=0)
        assert result.reconstruction_error < 0.1


class TestComponentWaveform:
    def test_returns_temporal_factor(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((20, 32, 4))
        result = tensor_decompose(X, method="cp", rank=2, random_state=0)
        wf = result.component_waveform(0)
        assert wf.shape == (32,)

    def test_scaled_by_channel(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((20, 32, 4))
        result = tensor_decompose(X, method="cp", rank=2, random_state=0)
        wf_raw = result.component_waveform(0)
        wf_ch = result.component_waveform(0, channel=1)
        np.testing.assert_allclose(wf_ch, wf_raw * result.spatial_factors[1, 0])


class TestMissingTensorly:
    def test_helpful_error_when_tensorly_absent(self, rng, monkeypatch):
        """tensor_decompose surfaces an install hint if tensorly is missing."""
        import sys

        monkeypatch.setitem(sys.modules, "tensorly", None)
        from subwave import tensor_decompose

        with pytest.raises(ImportError, match="tensorly"):
            tensor_decompose(rng.standard_normal((10, 16, 2)))


class TestInvalidInput:
    def test_2d_raises(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((20, 32))
        with pytest.raises(ValueError, match="3D tensor"):
            tensor_decompose(X)

    def test_unknown_method(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((10, 16, 2))
        with pytest.raises(ValueError, match="Unknown method"):
            tensor_decompose(X, method="bogus")
