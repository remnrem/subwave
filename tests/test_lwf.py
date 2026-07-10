"""Tests for the new LWF v3 format (int16 samples, phys_min/phys_max in header,
payload contains only meta + n_blocks + samples)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subwave.lwf import (
    _read_header,
    from_lwf,
    lwf_summary,
    lwf_event_counts,
    LwfSummary,
)

# ---------------------------------------------------------------------------
# Synthetic LWF writer (mirrors the C++ write_* functions)
# ---------------------------------------------------------------------------

_MAGIC   = b"LWF1"
_VERSION = 3


def _wu32(v: int) -> bytes:
    return struct.pack('<I', v)

def _wi32(v: int) -> bytes:
    return struct.pack('<i', v)

def _wu64(v: int) -> bytes:
    return struct.pack('<Q', v)

def _wf64(v: float) -> bytes:
    return struct.pack('<d', v)

def _wstr(s: str) -> bytes:
    b = s.encode('utf-8')
    return _wu32(len(b)) + b

def _encode_int16(values: np.ndarray, phys_min: float, phys_max: float) -> bytes:
    gain   = 65535.0 / (phys_max - phys_min) if phys_max != phys_min else 1.0
    offset = -32768.0 - gain * phys_min
    dv = np.clip(np.round(gain * values + offset), -32768, 32767).astype(np.int16)
    return dv.tobytes()


def make_lwf(
    path: Path,
    *,
    id_: str = "TEST01",
    tag: str = "test",
    align: str = "mid",
    annots: list[str] | None = None,
    channels: list[dict] | None = None,
    waves: list[dict] | None = None,
    annot_ch_match: bool = False,
    feature_names: tuple = (),
    force_index_nblocks: int | None = None,
) -> None:
    """Write a minimal synthetic .lwf file in the current format.

    Each entry in *waves* is a dict with keys:
      annot, instance, annot_ch, annot_start_sec, annot_stop_sec,
      anchor_sec, wave_start_sec, wave_stop_sec, meta,
      blocks: list of {'label': str, 'values': np.ndarray}
    """
    annots   = annots   or ["SO_neg_pk"]
    channels = channels or [{'label': 'CZ', 'unit': 'uV', 'sr': 128.0,
                              'phys_min': -200.0, 'phys_max': 200.0}]
    waves    = waves    or []

    n_ch = len(channels)

    # ---- header bytes ----
    hdr = (
        _MAGIC
        + _wi32(_VERSION)
        + _wstr(id_)
        + _wstr("test.edf")
        + _wstr(str(path))
        + _wstr("01.01.24")
        + _wstr("00.00.00")
        + _wstr(tag)
        + _wstr(align)
    )
    hdr += _wi32(len(annots))
    for a in annots:
        hdr += _wstr(a)

    hdr += _wi32(n_ch)
    sample_step_tp = int(round(1e9 / channels[0]['sr']))
    for ch in channels:
        hdr += (_wstr(ch['label']) + _wstr(ch['unit'])
                + _wu64(sample_step_tp) + _wf64(ch['sr'])
                + _wf64(ch['phys_min']) + _wf64(ch['phys_max']))

    n_features = len(feature_names)
    hdr += _wi32(n_features)
    for fn in feature_names:
        hdr += _wstr(fn)
    hdr += _wi32(len(waves))

    # ---- index bytes ----
    ch_map = {c['label']: c for c in channels}

    index_bytes = b""
    for w in waves:
        blocks = w['blocks']
        n_blocks = len(blocks)
        index_bytes += (
            _wstr(w['annot'])
            + _wstr(w.get('instance', '.'))
            + _wstr(w.get('annot_ch', '.'))
            + _wf64(w.get('annot_start_sec', 0.0))
            + _wf64(w.get('annot_stop_sec',  0.0))
            + _wf64(w.get('anchor_sec',       0.0))
            + _wf64(w.get('wave_start_sec',  -1.5))
            + _wf64(w.get('wave_stop_sec',    1.5))
            + _wu64(0)         # payload_offset placeholder
            + _wi32(force_index_nblocks if force_index_nblocks is not None else n_blocks)
        )
        for blk in blocks:
            ns = len(blk['values'])
            index_bytes += _wi32(ns) + _wf64(-1.5) + _wf64(1.5)

    # ---- payload bytes ----
    payload_bytes = b""
    for w in waves:
        blocks = w['blocks']
        payload_bytes += _wstr(w.get('meta', ''))
        payload_bytes += _wi32(len(blocks))
        for blk in blocks:
            label = blk['label']
            ch    = ch_map[label]
            if n_features:
                payload_bytes += _wi32(0) + b"\x00" * (n_features * 8)  # feature_qc + values
            payload_bytes += _encode_int16(
                np.asarray(blk['values'], dtype=np.float64),
                ch['phys_min'], ch['phys_max'],
            )

    # patch payload_offsets in index_bytes (sequential — just compute them)
    header_size  = len(hdr)
    index_size   = len(index_bytes)
    payload_base = header_size + index_size

    # re-build index with real offsets
    index_bytes = b""
    running_offset = payload_base
    for w in waves:
        blocks = w['blocks']
        n_blocks = len(blocks)
        index_bytes += (
            _wstr(w['annot'])
            + _wstr(w.get('instance', '.'))
            + _wstr(w.get('annot_ch', '.'))
            + _wf64(w.get('annot_start_sec', 0.0))
            + _wf64(w.get('annot_stop_sec',  0.0))
            + _wf64(w.get('anchor_sec',       0.0))
            + _wf64(w.get('wave_start_sec',  -1.5))
            + _wf64(w.get('wave_stop_sec',    1.5))
            + _wu64(running_offset)
            + _wi32(force_index_nblocks if force_index_nblocks is not None else n_blocks)
        )
        payload_sz = 4 + len(w.get('meta', '').encode()) + 4  # meta str + n_blocks
        for blk in blocks:
            if n_features:
                payload_sz += 4 + n_features * 8
            payload_sz += len(blk['values']) * 2
        for blk in blocks:
            ns = len(blk['values'])
            index_bytes += _wi32(ns) + _wf64(-1.5) + _wf64(1.5)
        running_offset += payload_sz

    path.write_bytes(hdr + index_bytes + payload_bytes)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR      = 128.0
N_SAMP  = 384          # 3 s at 128 Hz
W_LEFT  = N_SAMP // 2  # anchor at midpoint

def _sine_wave(freq: float = 1.0, amp: float = 50.0) -> np.ndarray:
    t = np.linspace(-1.5, 1.5, N_SAMP, endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)


@pytest.fixture
def two_channel_lwf(tmp_path) -> Path:
    """Standard mode: 4 waves, 2 channels each."""
    p = tmp_path / "standard.lwf"
    channels = [
        {'label': 'CZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
        {'label': 'FZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
    ]
    rng = np.random.default_rng(42)
    waves = []
    for i in range(4):
        waves.append({
            'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': '.',
            'anchor_sec': 100.0 + i * 10,
            'wave_start_sec': 100.0 + i * 10 - 1.5,
            'wave_stop_sec':  100.0 + i * 10 + 1.5,
            'meta': '',
            'blocks': [
                {'label': 'CZ', 'values': _sine_wave(1.0) + rng.normal(0, 5, N_SAMP)},
                {'label': 'FZ', 'values': _sine_wave(2.0) + rng.normal(0, 5, N_SAMP)},
            ],
        })
    make_lwf(p, channels=channels, waves=waves)
    return p


@pytest.fixture
def annot_ch_match_lwf(tmp_path) -> Path:
    """annot-ch-match mode: 6 waves, 1 block per wave (alternating CZ/FZ)."""
    p = tmp_path / "annot_ch_match.lwf"
    channels = [
        {'label': 'CZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
        {'label': 'FZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
    ]
    rng = np.random.default_rng(7)
    waves = []
    for i in range(6):
        ch = 'CZ' if i % 2 == 0 else 'FZ'
        waves.append({
            'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': ch,
            'anchor_sec': 100.0 + i * 10,
            'wave_start_sec': 100.0 + i * 10 - 1.5,
            'wave_stop_sec':  100.0 + i * 10 + 1.5,
            'meta': f'wave_{i}',
            'blocks': [
                {'label': ch, 'values': _sine_wave() + rng.normal(0, 5, N_SAMP)},
            ],
        })
    make_lwf(p, channels=channels, waves=waves, annot_ch_match=True)
    return p


@pytest.fixture
def imbalanced_acm_lwf(tmp_path) -> Path:
    """annot-ch-match mode with an imbalanced channel mix: CZ x10, FZ x4."""
    p = tmp_path / "imbalanced_acm.lwf"
    channels = [
        {'label': 'CZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
        {'label': 'FZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
    ]
    rng = np.random.default_rng(11)
    waves = []
    labels = ['CZ'] * 10 + ['FZ'] * 4
    for i, ch in enumerate(labels):
        waves.append({
            'annot': 'SO_neg_pk', 'instance': str(i), 'annot_ch': ch,
            'anchor_sec': 100.0 + i * 10,
            'wave_start_sec': 100.0 + i * 10 - 1.5,
            'wave_stop_sec':  100.0 + i * 10 + 1.5,
            'meta': f'wave_{i}',
            'blocks': [{'label': ch, 'values': _sine_wave() + rng.normal(0, 5, N_SAMP)}],
        })
    make_lwf(p, channels=channels, waves=waves, annot_ch_match=True)
    return p


@pytest.fixture
def three_channel_lwf(tmp_path) -> Path:
    """Standard mode: 8 waves, 3 channels (CZ, FZ, OZ), unique instance ids."""
    p = tmp_path / "three_channel.lwf"
    channels = [
        {'label': 'CZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
        {'label': 'FZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
        {'label': 'OZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
    ]
    rng = np.random.default_rng(3)
    waves = []
    for i in range(8):
        waves.append({
            'annot': 'SO_neg_pk', 'instance': str(i), 'annot_ch': '.',
            'anchor_sec': 100.0 + i * 10,
            'wave_start_sec': 100.0 + i * 10 - 1.5,
            'wave_stop_sec':  100.0 + i * 10 + 1.5,
            'meta': '',
            'blocks': [
                {'label': 'CZ', 'values': _sine_wave(1.0) + rng.normal(0, 5, N_SAMP)},
                {'label': 'FZ', 'values': _sine_wave(2.0) + rng.normal(0, 5, N_SAMP)},
                {'label': 'OZ', 'values': _sine_wave(3.0) + rng.normal(0, 5, N_SAMP)},
            ],
        })
    make_lwf(p, channels=channels, waves=waves)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReadHeader:
    def test_reads_phys_min_max(self, two_channel_lwf):
        with open(two_channel_lwf, 'rb') as f:
            h = _read_header(f)
        cz = next(c for c in h['channels'] if c['label'] == 'CZ')
        assert cz['phys_min'] == pytest.approx(-200.0)
        assert cz['phys_max'] == pytest.approx(200.0)

    def test_channel_count(self, two_channel_lwf):
        with open(two_channel_lwf, 'rb') as f:
            h = _read_header(f)
        assert len(h['channels']) == 2
        assert [c['label'] for c in h['channels']] == ['CZ', 'FZ']

    def test_bad_magic_raises(self, tmp_path):
        p = tmp_path / "bad.lwf"
        p.write_bytes(b"NOPE" + b"\x00" * 100)
        with open(p, 'rb') as f:
            with pytest.raises(ValueError, match="bad magic"):
                _read_header(f)

    def test_bad_version_raises(self, tmp_path):
        p = tmp_path / "badver.lwf"
        p.write_bytes(b"LWF1" + struct.pack('<i', 99) + b"\x00" * 100)
        with open(p, 'rb') as f:
            with pytest.raises(ValueError, match="version"):
                _read_header(f)


class TestLwfSummary:
    def test_returns_dataframe(self, two_channel_lwf):
        df = lwf_summary(two_channel_lwf)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_n_waves(self, two_channel_lwf):
        df = lwf_summary(two_channel_lwf)
        assert df['n_waves'].iloc[0] == 4

    def test_n_channels(self, two_channel_lwf):
        df = lwf_summary(two_channel_lwf)
        assert df['n_channels'].iloc[0] == 2

    def test_channel_labels(self, two_channel_lwf):
        df = lwf_summary(two_channel_lwf)
        assert df['channels'].iloc[0] == 'CZ,FZ'

    def test_directory_scan(self, tmp_path, two_channel_lwf, annot_ch_match_lwf):
        df = lwf_summary(tmp_path)
        assert len(df) == 2


class TestFromLwfStandard:
    def test_shape(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        assert aat.shape == (4, 2, N_SAMP)

    def test_axes(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        assert aat.axes == ['instance', 'channel', 'sample']

    def test_channel_meta(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        labels = list(aat.axis_meta['channel']['label'])
        assert labels == ['CZ', 'FZ']

    def test_instance_meta_columns(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        for col in ('annot', 'annot_ch', 'anchor_sec', 'wave_start_sec'):
            assert col in aat.axis_meta['instance'].columns

    def test_time_axis_range(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        t = aat.axis_index['sample']
        assert t[0]  == pytest.approx(-1.5, abs=0.02)
        assert t[-1] == pytest.approx( 1.5, abs=0.02)

    def test_not_annot_ch_match(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        assert aat.attrs['annot_ch_match'] is False

    def test_values_finite(self, two_channel_lwf):
        aat = from_lwf(two_channel_lwf)
        assert np.all(np.isfinite(aat.data))

    def test_int16_decode_precision(self, tmp_path):
        """Round-trip encode→int16→decode should stay within 1 LSB."""
        phys_min, phys_max = -100.0, 100.0
        rng = np.random.default_rng(0)
        original = rng.uniform(phys_min, phys_max, N_SAMP)

        channels = [{'label': 'CZ', 'unit': 'uV', 'sr': SR,
                     'phys_min': phys_min, 'phys_max': phys_max}]
        waves = [{'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': '.',
                  'anchor_sec': 0.0, 'wave_start_sec': -1.5, 'wave_stop_sec': 1.5,
                  'meta': '', 'blocks': [{'label': 'CZ', 'values': original}]}]
        p = tmp_path / "precision.lwf"
        make_lwf(p, channels=channels, waves=waves)

        aat = from_lwf(p)
        recovered = aat.data[0, 0, :]
        lsb = (phys_max - phys_min) / 65535.0
        assert np.max(np.abs(recovered - original)) <= lsb + 1e-9

    def test_multi_file_concat(self, tmp_path):
        """Two files with same channels are concatenated along instance axis."""
        channels = [{'label': 'CZ', 'unit': 'uV', 'sr': SR,
                     'phys_min': -200.0, 'phys_max': 200.0}]
        rng = np.random.default_rng(1)
        for name, n in [("a.lwf", 3), ("b.lwf", 5)]:
            waves = [{'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': '.',
                      'anchor_sec': float(i), 'wave_start_sec': float(i) - 1.5,
                      'wave_stop_sec': float(i) + 1.5, 'meta': '',
                      'blocks': [{'label': 'CZ',
                                  'values': rng.normal(0, 50, N_SAMP)}]}
                     for i in range(n)]
            make_lwf(tmp_path / name, channels=channels, waves=waves)

        aat = from_lwf(tmp_path)
        assert aat.shape[0] == 8      # 3 + 5
        assert aat.shape[1] == 1
        assert aat.shape[2] == N_SAMP

    def test_channel_mismatch_raises(self, tmp_path):
        ch_a = [{'label': 'CZ', 'unit': 'uV', 'sr': SR,
                 'phys_min': -200.0, 'phys_max': 200.0}]
        ch_b = [{'label': 'FZ', 'unit': 'uV', 'sr': SR,
                 'phys_min': -200.0, 'phys_max': 200.0}]
        rng = np.random.default_rng(2)
        for name, ch in [("x.lwf", ch_a), ("y.lwf", ch_b)]:
            waves = [{'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': '.',
                      'anchor_sec': 0.0, 'wave_start_sec': -1.5, 'wave_stop_sec': 1.5,
                      'meta': '', 'blocks': [{'label': ch[0]['label'],
                                              'values': rng.normal(0, 50, N_SAMP)}]}]
            make_lwf(tmp_path / name, channels=ch, waves=waves)

        with pytest.raises(ValueError, match="channel mismatch"):
            from_lwf(tmp_path)


class TestFromLwfAnnotChMatch:
    def test_shape_one_block_per_wave(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf)
        assert aat.shape == (6, 1, N_SAMP)

    def test_annot_ch_match_flag(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf)
        assert aat.attrs['annot_ch_match'] is True

    def test_annot_ch_in_instance_meta(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf)
        ch_vals = list(aat.axis_meta['instance']['annot_ch'])
        assert ch_vals == ['CZ', 'FZ', 'CZ', 'FZ', 'CZ', 'FZ']

    def test_meta_preserved(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf)
        metas = list(aat.axis_meta['instance']['meta'])
        assert metas == [f'wave_{i}' for i in range(6)]

    def test_values_finite(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf)
        assert np.all(np.isfinite(aat.data))

    def test_decode_uses_correct_channel_phys(self, tmp_path):
        """Each channel uses its own phys_min/phys_max for decoding."""
        channels = [
            {'label': 'CZ', 'unit': 'uV', 'sr': SR,
             'phys_min': -100.0, 'phys_max': 100.0},
            {'label': 'FZ', 'unit': 'uV', 'sr': SR,
             'phys_min': -500.0, 'phys_max': 500.0},
        ]
        cz_vals = np.full(N_SAMP, 80.0)   # within CZ range
        fz_vals = np.full(N_SAMP, -400.0)  # outside CZ range, within FZ range

        waves = [
            {'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': 'CZ',
             'anchor_sec': 0.0, 'wave_start_sec': -1.5, 'wave_stop_sec': 1.5,
             'meta': '', 'blocks': [{'label': 'CZ', 'values': cz_vals}]},
            {'annot': 'SO_neg_pk', 'instance': '.', 'annot_ch': 'FZ',
             'anchor_sec': 10.0, 'wave_start_sec': 8.5, 'wave_stop_sec': 11.5,
             'meta': '', 'blocks': [{'label': 'FZ', 'values': fz_vals}]},
        ]
        p = tmp_path / "phys_check.lwf"
        make_lwf(p, channels=channels, waves=waves, annot_ch_match=True)

        aat = from_lwf(p)
        lsb_cz = 200.0 / 65535.0
        lsb_fz = 1000.0 / 65535.0
        assert np.max(np.abs(aat.data[0, 0, :] - 80.0))   <= lsb_cz + 1e-9
        assert np.max(np.abs(aat.data[1, 0, :] - (-400.0))) <= lsb_fz + 1e-9


class TestFromLwfChannels:
    def test_standard_single_channel_narrows_axis(self, three_channel_lwf):
        aat = from_lwf(three_channel_lwf, channels='FZ')
        assert aat.shape == (8, 1, N_SAMP)
        assert list(aat.axis_meta['channel']['label']) == ['FZ']

    def test_standard_single_channel_str_or_list_equivalent(self, three_channel_lwf):
        a = from_lwf(three_channel_lwf, channels='FZ')
        b = from_lwf(three_channel_lwf, channels=['FZ'])
        assert np.array_equal(a.data, b.data)

    def test_standard_channel_subset_and_order(self, three_channel_lwf):
        aat = from_lwf(three_channel_lwf, channels=['CZ', 'OZ'])
        assert aat.shape == (8, 2, N_SAMP)
        # order follows the file's channel order, not the argument order
        assert list(aat.axis_meta['channel']['label']) == ['CZ', 'OZ']

    def test_standard_selected_block_matches_full_load(self, three_channel_lwf):
        full = from_lwf(three_channel_lwf)             # (8, 3, N)
        sel  = from_lwf(three_channel_lwf, channels='FZ')
        # FZ is column index 1 in the full load
        assert np.array_equal(sel.data[:, 0, :], full.data[:, 1, :])

    def test_annot_ch_match_filters_events(self, annot_ch_match_lwf):
        aat = from_lwf(annot_ch_match_lwf, channels='CZ')
        assert aat.shape == (3, 1, N_SAMP)   # 3 of the 6 events are CZ
        assert set(aat.axis_meta['instance']['annot_ch']) == {'CZ'}

    def test_annot_ch_match_multiple_channels(self, imbalanced_acm_lwf):
        aat = from_lwf(imbalanced_acm_lwf, channels=['CZ', 'FZ'])
        assert aat.shape[0] == 14   # all events retained
        aat_cz = from_lwf(imbalanced_acm_lwf, channels='CZ')
        assert aat_cz.shape[0] == 10

    def test_unknown_channel_raises(self, three_channel_lwf):
        with pytest.raises(ValueError, match="not present"):
            from_lwf(three_channel_lwf, channels='ZZ')


class TestFromLwfSample:
    def test_sample_n_limits_count(self, three_channel_lwf):
        aat = from_lwf(three_channel_lwf, sample_n=3, seed=0)
        assert aat.shape == (3, 3, N_SAMP)

    def test_sample_n_reproducible_with_seed(self, three_channel_lwf):
        a = from_lwf(three_channel_lwf, sample_n=3, seed=42)
        b = from_lwf(three_channel_lwf, sample_n=3, seed=42)
        assert np.array_equal(a.data, b.data)
        assert list(a.axis_meta['instance']['instance']) == \
               list(b.axis_meta['instance']['instance'])

    def test_sample_n_larger_than_events_loads_all(self, three_channel_lwf):
        aat = from_lwf(three_channel_lwf, sample_n=999, seed=1)
        assert aat.shape[0] == 8

    def test_sample_n_preserves_file_order(self, three_channel_lwf):
        # instances kept must be a sorted subset (draw is re-sorted per file)
        aat = from_lwf(three_channel_lwf, sample_n=4, seed=7)
        insts = [int(x) for x in aat.axis_meta['instance']['instance']]
        assert insts == sorted(insts)

    def test_sample_n_is_per_file(self, tmp_path):
        channels = [{'label': 'CZ', 'unit': 'uV', 'sr': SR,
                     'phys_min': -200.0, 'phys_max': 200.0}]
        rng = np.random.default_rng(1)
        for name, n in [("a.lwf", 6), ("b.lwf", 6)]:
            waves = [{'annot': 'SO_neg_pk', 'instance': str(i), 'annot_ch': '.',
                      'anchor_sec': float(i), 'wave_start_sec': float(i) - 1.5,
                      'wave_stop_sec': float(i) + 1.5, 'meta': '',
                      'blocks': [{'label': 'CZ', 'values': rng.normal(0, 50, N_SAMP)}]}
                     for i in range(n)]
            make_lwf(tmp_path / name, id_=name, channels=channels, waves=waves)

        aat = from_lwf(tmp_path, sample_n=2, seed=5)
        assert aat.shape[0] == 4   # 2 per file x 2 files
        assert aat.axis_meta['instance']['id'].value_counts().to_dict() == \
               {"a.lwf": 2, "b.lwf": 2}

    def test_sample_by_channel_balances(self, imbalanced_acm_lwf):
        # CZ x10, FZ x4 ; N=5 -> min(10,5)=5 CZ, min(4,5)=4 FZ
        aat = from_lwf(imbalanced_acm_lwf, sample_n=5, sample_by='channel', seed=1)
        counts = aat.axis_meta['instance']['annot_ch'].value_counts().to_dict()
        assert counts == {'CZ': 5, 'FZ': 4}

    def test_sample_by_record_follows_prevalence(self, imbalanced_acm_lwf):
        aat = from_lwf(imbalanced_acm_lwf, sample_n=5, sample_by='record', seed=1)
        assert aat.shape[0] == 5   # 5 total, mix not guaranteed balanced

    def test_sample_by_channel_noop_in_standard(self, three_channel_lwf):
        a = from_lwf(three_channel_lwf, sample_n=4, seed=2, sample_by='record')
        b = from_lwf(three_channel_lwf, sample_n=4, seed=2, sample_by='channel')
        assert np.array_equal(a.data, b.data)

    def test_channels_and_sample_combine(self, imbalanced_acm_lwf):
        aat = from_lwf(imbalanced_acm_lwf, channels='CZ', sample_n=3, seed=1)
        assert aat.shape[0] == 3
        assert set(aat.axis_meta['instance']['annot_ch']) == {'CZ'}

    def test_invalid_sample_by_raises(self, three_channel_lwf):
        with pytest.raises(ValueError, match="sample_by"):
            from_lwf(three_channel_lwf, sample_n=2, sample_by='bogus')


class TestLwfSummaryEstimate:
    def test_n_samples_column(self, three_channel_lwf):
        df = lwf_summary(three_channel_lwf)
        assert df['n_samples'].iloc[0] == N_SAMP

    def test_annot_ch_match_flag(self, three_channel_lwf, annot_ch_match_lwf):
        assert lwf_summary(three_channel_lwf)['annot_ch_match'].iloc[0] == False  # noqa: E712
        assert lwf_summary(annot_ch_match_lwf)['annot_ch_match'].iloc[0] == True   # noqa: E712

    def test_est_bytes_matches_full_load_standard(self, three_channel_lwf):
        df = lwf_summary(three_channel_lwf)
        aat = from_lwf(three_channel_lwf)
        assert int(df['est_bytes'].iloc[0]) == aat.data.nbytes

    def test_est_bytes_matches_full_load_annot_ch_match(self, imbalanced_acm_lwf):
        df = lwf_summary(imbalanced_acm_lwf)
        aat = from_lwf(imbalanced_acm_lwf)
        assert int(df['est_bytes'].iloc[0]) == aat.data.nbytes

    def test_est_mb_is_bytes_over_mib(self, three_channel_lwf):
        df = lwf_summary(three_channel_lwf)
        assert df['est_mb'].iloc[0] == pytest.approx(
            df['est_bytes'].iloc[0] / 1024**2, abs=0.01)


class TestLwfEventCounts:
    def test_long_form_columns(self, imbalanced_acm_lwf):
        df = lwf_event_counts(imbalanced_acm_lwf)
        assert list(df.columns) == ['file', 'id', 'annot_ch', 'n_events']

    def test_per_channel_counts(self, imbalanced_acm_lwf):
        df = lwf_event_counts(imbalanced_acm_lwf)
        counts = dict(zip(df['annot_ch'], df['n_events']))
        assert counts == {'CZ': 10, 'FZ': 4}

    def test_total_matches_n_waves(self, imbalanced_acm_lwf):
        df = lwf_event_counts(imbalanced_acm_lwf)
        assert df['n_events'].sum() == lwf_summary(imbalanced_acm_lwf)['n_waves'].iloc[0]

    def test_predicts_sample_by_channel_result(self, imbalanced_acm_lwf):
        counts = lwf_event_counts(imbalanced_acm_lwf).set_index('annot_ch')['n_events']
        n = 5
        predicted = int(sum(min(v, n) for v in counts))
        actual = from_lwf(imbalanced_acm_lwf, sample_n=n, sample_by='channel',
                          seed=1).shape[0]
        assert predicted == actual

    def test_multi_file_rows(self, tmp_path, three_channel_lwf, imbalanced_acm_lwf):
        df = lwf_event_counts(tmp_path)
        # three_channel has annot_ch '.', imbalanced_acm has CZ/FZ
        assert set(df['annot_ch']) == {'.', 'CZ', 'FZ'}
        assert len(df) == 3


class TestLwfSummaryOverview:
    def _two_subjects(self, tmp_path):
        channels = [
            {'label': 'CZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
            {'label': 'FZ', 'unit': 'uV', 'sr': SR, 'phys_min': -200.0, 'phys_max': 200.0},
        ]
        rng = np.random.default_rng(0)
        for sub, n in [('subA', 4), ('subB', 6)]:
            waves = [{'annot': 'SO_neg_pk', 'instance': str(i), 'annot_ch': '.',
                      'anchor_sec': float(i), 'wave_start_sec': float(i) - 1.5,
                      'wave_stop_sec': float(i) + 1.5, 'meta': '',
                      'blocks': [{'label': 'CZ', 'values': rng.normal(0, 50, N_SAMP)},
                                 {'label': 'FZ', 'values': rng.normal(0, 50, N_SAMP)}]}
                     for i in range(n)]
            make_lwf(tmp_path / f"{sub}.lwf", id_=sub, channels=channels, waves=waves)

    def test_returns_lwf_summary_subclass(self, three_channel_lwf):
        df = lwf_summary(three_channel_lwf)
        assert isinstance(df, LwfSummary)
        assert isinstance(df, pd.DataFrame)

    def test_summary_is_series(self, three_channel_lwf):
        s = lwf_summary(three_channel_lwf).summary()
        assert isinstance(s, pd.Series)

    def test_summary_aggregates(self, tmp_path):
        self._two_subjects(tmp_path)
        s = lwf_summary(tmp_path).summary()
        assert s['files'] == 2
        assert s['subjects'] == 2
        assert s['events'] == 10            # 4 + 6
        assert s['channels'] == 2
        assert s['channel_labels'] == 'CZ,FZ'
        assert s['samples_per_event'] == N_SAMP
        assert s['annot_ch_match'] is False

    def test_summary_est_bytes_matches_full_load(self, tmp_path):
        self._two_subjects(tmp_path)
        df = lwf_summary(tmp_path)
        total = sum(from_lwf(f).data.nbytes for f in df['file'])
        assert int(df.summary()['est_bytes']) == total

    def test_summary_after_slice(self, tmp_path):
        self._two_subjects(tmp_path)
        df = lwf_summary(tmp_path)
        s = df[df['id'] == 'subA'].summary()
        assert s['files'] == 1
        assert s['events'] == 4

    def test_summary_mixed_annot_ch_match(self, tmp_path, three_channel_lwf,
                                          annot_ch_match_lwf):
        s = lwf_summary(tmp_path).summary()
        assert s['annot_ch_match'] == 'mixed'


# ---------------------------------------------------------------------------
# Cases ported from the upstream suite, rewritten for the int16 make_lwf writer.
# (Dropped: block-count-mismatch and different-sample-counts-within-event —
#  those target the old float32 reader and have no int16 equivalent.)
# ---------------------------------------------------------------------------

_PMIN, _PMAX = -200.0, 200.0
_LSB = (_PMAX - _PMIN) / 65535.0


def _chs(*specs):
    """specs: (label, unit, sr) -> list of channel dicts (phys -200..200)."""
    return [{'label': l, 'unit': u, 'sr': s, 'phys_min': _PMIN, 'phys_max': _PMAX}
            for (l, u, s) in specs]


def _wave(labels, ns, *, instance='1', annot='SP', annot_ch='.', meta='m',
          anchor=10.25, wave_start=9.75, wave_stop=10.75,
          annot_start=10.0, annot_stop=10.5, values=None):
    if values is None:
        values = [np.arange(n, dtype=float) + 100.0 * i
                  for i, n in enumerate([ns] * len(labels))]
    return {'annot': annot, 'instance': instance, 'annot_ch': annot_ch, 'meta': meta,
            'anchor_sec': anchor, 'wave_start_sec': wave_start, 'wave_stop_sec': wave_stop,
            'annot_start_sec': annot_start, 'annot_stop_sec': annot_stop,
            'blocks': [{'label': lab, 'values': v} for lab, v in zip(labels, values)]}


class TestFromLwfBasicPorted:
    def test_returns_axis_annotated_tensor(self, tmp_path):
        from subwave import AxisAnnotatedTensor
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 16)])
        assert isinstance(from_lwf(p), AxisAnnotatedTensor)

    def test_axes_order(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 16)])
        assert from_lwf(p).axes == ['instance', 'channel', 'sample']

    def test_accepts_string_path(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 16)])
        assert from_lwf(str(p)).shape == (1, 1, 16)

    def test_payload_round_trip_channel_routing(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0), ('C4', 'uV', 256.0)),
                 waves=[_wave(['C3', 'C4'], 16)])
        aat = from_lwf(p)
        np.testing.assert_allclose(aat.data[0, 0], np.arange(16), atol=_LSB)
        np.testing.assert_allclose(aat.data[0, 1], np.arange(16) + 100.0, atol=_LSB)

    def test_data_dtype_is_float64(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 16)])
        assert from_lwf(p).data.dtype == np.float64


class TestFromLwfMetadataPorted:
    def test_channel_meta(self, tmp_path):
        p = tmp_path / "b.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0), ('C4', 'mV', 256.0)),
                 waves=[_wave(['C3', 'C4'], 16)])
        ch = from_lwf(p).axis_meta['channel']
        assert list(ch['label']) == ['C3', 'C4']
        assert list(ch['unit']) == ['uV', 'mV']
        assert list(ch['sr']) == [256.0, 256.0]

    def test_instance_meta_columns(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 16)])
        cols = list(from_lwf(p).axis_meta['instance'].columns)
        assert cols == ['id', 'tag', 'file', 'annot', 'instance', 'annot_ch', 'meta',
                        'anchor_sec', 'annot_start_sec', 'annot_stop_sec',
                        'wave_start_sec', 'wave_stop_sec']

    def test_instance_meta_values(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, id_='S7', tag='T', channels=_chs(('C3', 'uV', 256.0)),
                 waves=[_wave(['C3'], 8, instance='42', meta='hello', annot='SS')])
        row = from_lwf(p).axis_meta['instance'].iloc[0]
        assert row['id'] == 'S7' and row['tag'] == 'T'
        assert row['instance'] == '42' and row['meta'] == 'hello'
        assert row['annot'] == 'SS' and row['file'] == str(p)

    def test_sample_meta(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 10)])
        sm = from_lwf(p).axis_meta['sample']
        assert list(sm.columns) == ['sample_index', 'time']
        np.testing.assert_array_equal(sm['sample_index'], np.arange(10))

    def test_attrs(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, align='trough', channels=_chs(('C3', 'uV', 256.0)),
                 waves=[_wave(['C3'], 8)])
        aat = from_lwf(p)
        assert aat.attrs['sfreq'] == 256.0
        assert aat.attrs['align'] == 'trough'
        assert aat.attrs['source_files'] == [str(p)]


class TestFromLwfTimeAxisPorted:
    def test_offset_and_spacing(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)),
                 waves=[_wave(['C3'], 4, anchor=10.25, wave_start=9.75)])
        t = from_lwf(p).axis_index['sample']
        assert t[0] == pytest.approx(-0.5)
        assert (t[1] - t[0]) == pytest.approx(1.0 / 256.0)

    def test_time_axis_matches_sample_meta(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 12)])
        aat = from_lwf(p)
        np.testing.assert_allclose(aat.axis_index['sample'],
                                   aat.axis_meta['sample']['time'])


class TestFromLwfFeaturesPorted:
    def test_features_are_skipped(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)),
                 feature_names=('PEAK', 'FRQ', 'DUR'), waves=[_wave(['C3'], 16)])
        aat = from_lwf(p)
        assert aat.shape == (1, 1, 16)
        np.testing.assert_allclose(aat.data[0, 0], np.arange(16), atol=_LSB)

    def test_summary_reports_feature_count(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)),
                 feature_names=('PEAK', 'FRQ'), waves=[_wave(['C3'], 8)])
        assert lwf_summary(p).loc[0, 'n_features'] == 2


class TestFromLwfMultipleFilesPorted:
    def test_concatenates_events(self, tmp_path):
        pa = tmp_path / "a.lwf"; pb = tmp_path / "b.lwf"
        make_lwf(pa, id_='A', channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 16)])
        make_lwf(pb, id_='B', channels=_chs(('C3', 'uV', 256.0)),
                 waves=[_wave(['C3'], 16), _wave(['C3'], 16)])
        aat = from_lwf([pa, pb])
        assert aat.shape == (3, 1, 16)
        assert list(aat.axis_meta['instance']['id']) == ['A', 'B', 'B']

    def test_source_files_tracks_all(self, tmp_path):
        pa = tmp_path / "a.lwf"; pb = tmp_path / "b.lwf"
        make_lwf(pa, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 8)])
        make_lwf(pb, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 8)])
        assert from_lwf([pa, pb]).attrs['source_files'] == [str(pa), str(pb)]


class TestFromLwfDirectoryPorted:
    def test_loads_directory(self, tmp_path):
        make_lwf(tmp_path / "a.lwf", channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 8)])
        make_lwf(tmp_path / "b.lwf", channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 8)])
        assert from_lwf(tmp_path).shape == (2, 1, 8)

    def test_non_recursive_ignores_subdirs(self, tmp_path):
        sub = tmp_path / "nested"; sub.mkdir()
        make_lwf(sub / "deep.lwf", channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 8)])
        with pytest.raises(ValueError, match="no .lwf files found"):
            from_lwf(tmp_path)

    def test_recursive_finds_nested(self, tmp_path):
        sub = tmp_path / "nested"; sub.mkdir()
        make_lwf(sub / "deep.lwf", channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 8)])
        assert from_lwf(tmp_path, recur=True).shape == (1, 1, 8)


class TestFromLwfErrorsPorted:
    def test_no_files_found(self, tmp_path):
        with pytest.raises(ValueError, match="no .lwf files found"):
            from_lwf(tmp_path)

    def test_channel_label_mismatch_across_files(self, tmp_path):
        pa = tmp_path / "a.lwf"; pb = tmp_path / "b.lwf"
        make_lwf(pa, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 8)])
        make_lwf(pb, channels=_chs(('C4', 'uV', 256.0)), waves=[_wave(['C4'], 8)])
        with pytest.raises(ValueError, match="channel mismatch"):
            from_lwf([pa, pb])

    def test_mixed_sample_rates_within_file(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0), ('C4', 'uV', 128.0)),
                 waves=[_wave(['C3', 'C4'], 8)])
        with pytest.raises(ValueError, match="mixed sample rates"):
            from_lwf(p)

    def test_sample_count_mismatch_across_files(self, tmp_path):
        pa = tmp_path / "a.lwf"; pb = tmp_path / "b.lwf"
        make_lwf(pa, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 16)])
        make_lwf(pb, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 32)])
        with pytest.raises(ValueError, match="sample count mismatch"):
            from_lwf([pa, pb])

    def test_non_uniform_waveform_length(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)),
                 waves=[_wave(['C3'], 16), _wave(['C3'], 32)])
        with pytest.raises(ValueError, match="non-uniform waveform length"):
            from_lwf(p)


class TestLwfSummaryColumnsPorted:
    def test_columns(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, channels=_chs(('C3', 'uV', 256.0)), waves=[_wave(['C3'], 8)])
        assert list(lwf_summary(p).columns) == [
            'file', 'id', 'tag', 'startdate', 'starttime', 'align',
            'n_waves', 'n_channels', 'channels', 'srs', 'annots', 'n_features',
            'n_samples', 'annot_ch_match', 'est_bytes', 'est_mb']

    def test_single_file_values(self, tmp_path):
        p = tmp_path / "a.lwf"
        make_lwf(p, id_='S1', tag='SP', align='peak',
                 channels=_chs(('C3', 'uV', 256.0), ('C4', 'uV', 256.0)),
                 annots=['SP', 'SO'], waves=[_wave(['C3', 'C4'], 8), _wave(['C3', 'C4'], 8)])
        row = lwf_summary(p).iloc[0]
        assert row['id'] == 'S1' and row['tag'] == 'SP' and row['align'] == 'peak'
        assert row['n_waves'] == 2 and row['n_channels'] == 2
        assert row['channels'] == 'C3,C4' and row['srs'] == '256.0,256.0'
        assert row['annots'] == 'SP,SO' and row['n_features'] == 0
