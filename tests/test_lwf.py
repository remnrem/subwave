"""Tests for the Luna ``.lwf`` binary loader (``subwave.lwf``).

The reader (:mod:`subwave.lwf`) parses a bespoke little-endian binary format.
There is no public writer, so these tests ship a small writer
(:func:`write_lwf`) that mirrors the byte layout the reader expects. This lets
us round-trip well-formed files and also synthesise deliberately malformed
files to exercise the validation/error paths.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subwave import AxisAnnotatedTensor, from_lwf, lwf_summary

# ---------------------------------------------------------------------------
# Binary writing primitives (inverse of subwave.lwf._read_* helpers)
# ---------------------------------------------------------------------------

_MAGIC = b"LWF1"
_VERSION = 3


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _i32(v: int) -> bytes:
    return struct.pack("<i", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _f64(v: float) -> bytes:
    return struct.pack("<d", v)


# ---------------------------------------------------------------------------
# Wave + file builders
# ---------------------------------------------------------------------------


def make_wave(
    n_ch: int,
    ns: int,
    *,
    annot: str = "SP",
    instance: str = "1",
    annot_ch: str = "C3",
    meta: str = "m",
    annot_start: float = 10.0,
    annot_stop: float = 10.5,
    anchor: float = 10.25,
    wave_start: float = 9.75,
    wave_stop: float = 10.75,
    data=None,
    ns_per_ch=None,
):
    """Build a single wave/event description.

    ``data`` is a list of per-channel 1-D arrays. By default each channel gets a
    distinct ramp (``arange(ns) + 100*channel``) so we can assert the reader
    routes channel payloads into the right slot.
    """
    if data is None:
        data = [np.arange(ns, dtype=np.float32) + 100.0 * c for c in range(n_ch)]
    if ns_per_ch is None:
        ns_per_ch = [len(d) for d in data]
    return {
        "annot": annot,
        "instance": instance,
        "annot_ch": annot_ch,
        "meta": meta,
        "annot_start": annot_start,
        "annot_stop": annot_stop,
        "anchor": anchor,
        "wave_start": wave_start,
        "wave_stop": wave_stop,
        "data": data,
        "ns_per_ch": ns_per_ch,
    }


def write_lwf(
    path,
    *,
    id_: str = "subj1",
    tag: str = "SP",
    align: str = "peak",
    startdate: str = "01.01.20",
    starttime: str = "22.00.00",
    edf: str = "subj1.edf",
    outfile: str = "out.lwf",
    annots=("SP",),
    channels=(("C3", "uV", 256.0),),
    feature_names=(),
    waves=None,
    magic: bytes = _MAGIC,
    version: int = _VERSION,
    force_index_nblocks=None,
) -> Path:
    """Serialise a ``.lwf`` file matching the layout in ``subwave.lwf``.

    ``force_index_nblocks`` overrides the per-event block count written in the
    *index* section (used to trigger the block-count validation error).
    """
    if waves is None:
        waves = [make_wave(len(channels), 64)]

    n_ch = len(channels)
    n_features = len(feature_names)
    buf = bytearray()

    # --- header ---
    buf += magic
    buf += _i32(version)
    buf += _s(id_)
    buf += _s(edf)
    buf += _s(outfile)
    buf += _s(startdate)
    buf += _s(starttime)
    buf += _s(tag)
    buf += _s(align)
    buf += _i32(len(annots))
    for a in annots:
        buf += _s(a)
    buf += _i32(n_ch)
    for label, unit, sr in channels:
        buf += _s(label)
        buf += _s(unit)
        buf += _u64(0)  # sample_step_tp (discarded by reader)
        buf += _f64(sr)
    buf += _i32(n_features)
    for fn in feature_names:
        buf += _s(fn)
    buf += _i32(len(waves))

    # --- index section ---
    for w in waves:
        buf += _s(w["annot"])
        buf += _s(w["instance"])
        buf += _s(w["annot_ch"])
        buf += _f64(w["annot_start"])
        buf += _f64(w["annot_stop"])
        buf += _f64(w["anchor"])
        buf += _f64(w["wave_start"])
        buf += _f64(w["wave_stop"])
        buf += _u64(0)  # payload_offset (discarded; reader is sequential)
        nb = force_index_nblocks if force_index_nblocks is not None else n_ch
        buf += _i32(nb)
        for c in range(n_ch):
            buf += _i32(w["ns_per_ch"][c])
            buf += _f64(0.0)  # data_start_sec
            buf += _f64(0.0)  # data_stop_sec

    # --- payload section ---
    for w in waves:
        buf += _s(w["annot"])
        buf += _s(w["instance"])
        buf += _s(w["annot_ch"])
        buf += _s(w["meta"])
        buf += _f64(w["annot_start"])
        buf += _f64(w["annot_stop"])
        buf += _f64(w["anchor"])
        buf += _f64(w["wave_start"])
        buf += _f64(w["wave_stop"])
        buf += _i32(n_ch)
        for c in range(n_ch):
            arr = np.asarray(w["data"][c], dtype="<f4")
            buf += _i32(len(arr))
            buf += _f64(0.0)  # data_start_sec
            buf += _f64(0.0)  # data_stop_sec
            if n_features > 0:
                buf += _i32(0)  # feature_qc
                buf += b"\x00" * (n_features * 8)  # n_features float64
            buf += arr.tobytes()

    path = Path(path)
    path.write_bytes(bytes(buf))
    return path


# ---------------------------------------------------------------------------
# from_lwf — happy paths
# ---------------------------------------------------------------------------


class TestFromLwfBasic:
    def test_returns_axis_annotated_tensor(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf")
        aat = from_lwf(p)
        assert isinstance(aat, AxisAnnotatedTensor)

    def test_axes_order(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf")
        aat = from_lwf(p)
        assert aat.axes == ["instance", "channel", "sample"]

    def test_shape(self, tmp_path):
        waves = [make_wave(2, 32), make_wave(2, 32), make_wave(2, 32)]
        p = write_lwf(
            tmp_path / "a.lwf",
            channels=(("C3", "uV", 256.0), ("C4", "uV", 256.0)),
            waves=waves,
        )
        aat = from_lwf(p)
        assert aat.shape == (3, 2, 32)

    def test_accepts_string_path(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf")
        aat = from_lwf(str(p))
        assert aat.shape == (1, 1, 64)

    def test_payload_values_round_trip(self, tmp_path):
        """Per-channel ramps come back in the correct (event, channel) slot."""
        waves = [make_wave(2, 16)]
        p = write_lwf(
            tmp_path / "a.lwf",
            channels=(("C3", "uV", 256.0), ("C4", "uV", 256.0)),
            waves=waves,
        )
        aat = from_lwf(p)
        np.testing.assert_allclose(aat.data[0, 0], np.arange(16))
        np.testing.assert_allclose(aat.data[0, 1], np.arange(16) + 100.0)

    def test_data_dtype_is_float64(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf")
        aat = from_lwf(p)
        assert aat.data.dtype == np.float64


class TestFromLwfMetadata:
    def test_channel_meta(self, tmp_path):
        # mixed SR within a file is rejected, so use equal SRs for this check
        p = write_lwf(
            tmp_path / "b.lwf",
            channels=(("C3", "uV", 256.0), ("C4", "mV", 256.0)),
            waves=[make_wave(2, 16)],
        )
        aat = from_lwf(p)
        ch = aat.axis_meta["channel"]
        assert list(ch["label"]) == ["C3", "C4"]
        assert list(ch["unit"]) == ["uV", "mV"]
        assert list(ch["sr"]) == [256.0, 256.0]

    def test_instance_meta_columns(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf")
        aat = from_lwf(p)
        cols = list(aat.axis_meta["instance"].columns)
        assert cols == [
            "id",
            "tag",
            "file",
            "annot",
            "instance",
            "annot_ch",
            "meta",
            "anchor_sec",
            "annot_start_sec",
            "annot_stop_sec",
            "wave_start_sec",
            "wave_stop_sec",
        ]

    def test_instance_meta_values(self, tmp_path):
        waves = [make_wave(1, 8, instance="42", meta="hello", annot="SS")]
        p = write_lwf(tmp_path / "a.lwf", id_="S7", tag="T", waves=waves)
        aat = from_lwf(p)
        row = aat.axis_meta["instance"].iloc[0]
        assert row["id"] == "S7"
        assert row["tag"] == "T"
        assert row["instance"] == "42"
        assert row["meta"] == "hello"
        assert row["annot"] == "SS"
        assert row["file"] == str(p)

    def test_sample_meta(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf", waves=[make_wave(1, 10)])
        aat = from_lwf(p)
        sm = aat.axis_meta["sample"]
        assert list(sm.columns) == ["sample_index", "time"]
        np.testing.assert_array_equal(sm["sample_index"], np.arange(10))

    def test_attrs(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf", align="trough")
        aat = from_lwf(p)
        assert aat.attrs["sfreq"] == 256.0
        assert aat.attrs["align"] == "trough"
        assert aat.attrs["source_files"] == [str(p)]


class TestFromLwfTimeAxis:
    def test_offset_and_spacing(self, tmp_path):
        # offset = wave_start - anchor = 9.75 - 10.25 = -0.5
        p = write_lwf(
            tmp_path / "a.lwf",
            channels=(("C3", "uV", 256.0),),
            waves=[make_wave(1, 4, anchor=10.25, wave_start=9.75)],
        )
        aat = from_lwf(p)
        t = aat.axis_index["sample"]
        assert t[0] == pytest.approx(-0.5)
        assert (t[1] - t[0]) == pytest.approx(1.0 / 256.0)

    def test_time_axis_matches_sample_meta(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf", waves=[make_wave(1, 12)])
        aat = from_lwf(p)
        np.testing.assert_allclose(
            aat.axis_index["sample"], aat.axis_meta["sample"]["time"]
        )


class TestFromLwfFeatures:
    def test_features_are_skipped(self, tmp_path):
        """Per-channel feature blocks must be skipped without corrupting data."""
        waves = [make_wave(1, 16)]
        p = write_lwf(
            tmp_path / "a.lwf",
            feature_names=("PEAK", "FRQ", "DUR"),
            waves=waves,
        )
        aat = from_lwf(p)
        assert aat.shape == (1, 1, 16)
        np.testing.assert_allclose(aat.data[0, 0], np.arange(16))

    def test_summary_reports_feature_count(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf", feature_names=("PEAK", "FRQ"))
        df = lwf_summary(p)
        assert df.loc[0, "n_features"] == 2


# ---------------------------------------------------------------------------
# from_lwf — multiple files
# ---------------------------------------------------------------------------


class TestFromLwfMultipleFiles:
    def test_concatenates_events(self, tmp_path):
        p1 = write_lwf(tmp_path / "a.lwf", id_="A", waves=[make_wave(1, 16)])
        p2 = write_lwf(
            tmp_path / "b.lwf",
            id_="B",
            waves=[make_wave(1, 16), make_wave(1, 16)],
        )
        aat = from_lwf([p1, p2])
        assert aat.shape == (3, 1, 16)
        ids = list(aat.axis_meta["instance"]["id"])
        assert ids == ["A", "B", "B"]

    def test_source_files_tracks_all(self, tmp_path):
        p1 = write_lwf(tmp_path / "a.lwf", waves=[make_wave(1, 8)])
        p2 = write_lwf(tmp_path / "b.lwf", waves=[make_wave(1, 8)])
        aat = from_lwf([p1, p2])
        assert aat.attrs["source_files"] == [str(p1), str(p2)]


class TestFromLwfDirectory:
    def test_loads_directory(self, tmp_path):
        write_lwf(tmp_path / "a.lwf", waves=[make_wave(1, 8)])
        write_lwf(tmp_path / "b.lwf", waves=[make_wave(1, 8)])
        aat = from_lwf(tmp_path)
        assert aat.shape == (2, 1, 8)

    def test_non_recursive_ignores_subdirs(self, tmp_path):
        sub = tmp_path / "nested"
        sub.mkdir()
        write_lwf(sub / "deep.lwf", waves=[make_wave(1, 8)])
        # No .lwf at the top level -> nothing found
        with pytest.raises(ValueError, match="no .lwf files found"):
            from_lwf(tmp_path)

    def test_recursive_finds_nested(self, tmp_path):
        sub = tmp_path / "nested"
        sub.mkdir()
        write_lwf(sub / "deep.lwf", waves=[make_wave(1, 8)])
        aat = from_lwf(tmp_path, recur=True)
        assert aat.shape == (1, 1, 8)


# ---------------------------------------------------------------------------
# from_lwf — error / validation paths
# ---------------------------------------------------------------------------


class TestFromLwfErrors:
    def test_no_files_found(self, tmp_path):
        with pytest.raises(ValueError, match="no .lwf files found"):
            from_lwf(tmp_path)

    def test_bad_magic(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf", magic=b"XXXX")
        with pytest.raises(ValueError, match="bad magic"):
            from_lwf(p)

    def test_bad_version(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf", version=99)
        with pytest.raises(ValueError, match="unsupported .lwf version"):
            from_lwf(p)

    def test_channel_label_mismatch_across_files(self, tmp_path):
        p1 = write_lwf(
            tmp_path / "a.lwf",
            channels=(("C3", "uV", 256.0),),
            waves=[make_wave(1, 8)],
        )
        p2 = write_lwf(
            tmp_path / "b.lwf",
            channels=(("C4", "uV", 256.0),),
            waves=[make_wave(1, 8)],
        )
        with pytest.raises(ValueError, match="channel mismatch"):
            from_lwf([p1, p2])

    def test_mixed_sample_rates_within_file(self, tmp_path):
        p = write_lwf(
            tmp_path / "a.lwf",
            channels=(("C3", "uV", 256.0), ("C4", "uV", 128.0)),
            waves=[make_wave(2, 8)],
        )
        with pytest.raises(ValueError, match="mixed sample rates"):
            from_lwf(p)

    def test_sample_count_mismatch_across_files(self, tmp_path):
        p1 = write_lwf(tmp_path / "a.lwf", waves=[make_wave(1, 16)])
        p2 = write_lwf(tmp_path / "b.lwf", waves=[make_wave(1, 32)])
        with pytest.raises(ValueError, match="sample count mismatch"):
            from_lwf([p1, p2])

    def test_block_count_mismatch(self, tmp_path):
        p = write_lwf(
            tmp_path / "a.lwf",
            channels=(("C3", "uV", 256.0),),
            waves=[make_wave(1, 8)],
            force_index_nblocks=2,
        )
        with pytest.raises(ValueError, match="block count"):
            from_lwf(p)

    def test_channels_different_sample_counts_within_event(self, tmp_path):
        ragged = make_wave(
            2,
            0,
            data=[np.arange(10, dtype=np.float32), np.arange(20, dtype=np.float32)],
        )
        p = write_lwf(
            tmp_path / "a.lwf",
            channels=(("C3", "uV", 256.0), ("C4", "uV", 256.0)),
            waves=[ragged],
        )
        with pytest.raises(ValueError, match="different sample counts"):
            from_lwf(p)

    def test_non_uniform_waveform_length(self, tmp_path):
        p = write_lwf(
            tmp_path / "a.lwf",
            channels=(("C3", "uV", 256.0),),
            waves=[make_wave(1, 16), make_wave(1, 32)],
        )
        with pytest.raises(ValueError, match="non-uniform waveform length"):
            from_lwf(p)


# ---------------------------------------------------------------------------
# lwf_summary
# ---------------------------------------------------------------------------


class TestLwfSummary:
    def test_columns(self, tmp_path):
        p = write_lwf(tmp_path / "a.lwf")
        df = lwf_summary(p)
        assert list(df.columns) == [
            "file",
            "id",
            "tag",
            "startdate",
            "starttime",
            "align",
            "n_waves",
            "n_channels",
            "channels",
            "srs",
            "annots",
            "n_features",
        ]

    def test_single_file_values(self, tmp_path):
        p = write_lwf(
            tmp_path / "a.lwf",
            id_="S1",
            tag="SP",
            align="peak",
            startdate="02.02.21",
            starttime="23.30.00",
            annots=("SP", "SO"),
            channels=(("C3", "uV", 256.0), ("C4", "uV", 256.0)),
            waves=[make_wave(2, 8), make_wave(2, 8)],
        )
        df = lwf_summary(p)
        row = df.iloc[0]
        assert row["id"] == "S1"
        assert row["tag"] == "SP"
        assert row["align"] == "peak"
        assert row["startdate"] == "02.02.21"
        assert row["starttime"] == "23.30.00"
        assert row["n_waves"] == 2
        assert row["n_channels"] == 2
        assert row["channels"] == "C3,C4"
        assert row["srs"] == "256.0,256.0"
        assert row["annots"] == "SP,SO"
        assert row["n_features"] == 0

    def test_does_not_load_signal(self, tmp_path):
        """Summary works on a header-only file (no payload section)."""
        # A file with n_waves declared but zero waves still has a valid header.
        p = write_lwf(tmp_path / "a.lwf", waves=[make_wave(1, 4)])
        df = lwf_summary(p)
        assert len(df) == 1

    def test_multiple_files(self, tmp_path):
        p1 = write_lwf(tmp_path / "a.lwf", id_="A", waves=[make_wave(1, 4)])
        p2 = write_lwf(tmp_path / "b.lwf", id_="B", waves=[make_wave(1, 4)])
        df = lwf_summary([p1, p2])
        assert len(df) == 2
        assert set(df["id"]) == {"A", "B"}

    def test_directory_recursive(self, tmp_path):
        sub = tmp_path / "nested"
        sub.mkdir()
        write_lwf(tmp_path / "top.lwf", waves=[make_wave(1, 4)])
        write_lwf(sub / "deep.lwf", waves=[make_wave(1, 4)])
        df = lwf_summary(tmp_path, recur=True)
        assert len(df) == 2

    def test_no_files_found(self, tmp_path):
        with pytest.raises(ValueError, match="no .lwf files found"):
            lwf_summary(tmp_path)
