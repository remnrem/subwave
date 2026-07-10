from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from .tensor import AxisAnnotatedTensor

_LWF_MAGIC = b"LWF1"
_LWF_VERSION = 3

# ---------------------------------------------------------------------------
# Binary primitives
# ---------------------------------------------------------------------------

def _read_u32(f) -> int:
    return struct.unpack('<I', f.read(4))[0]

def _read_i32(f) -> int:
    return struct.unpack('<i', f.read(4))[0]

def _read_u64(f) -> int:
    return struct.unpack('<Q', f.read(8))[0]

def _read_f64(f) -> float:
    return struct.unpack('<d', f.read(8))[0]

def _read_string(f) -> str:
    n = _read_u32(f)
    return f.read(n).decode('utf-8') if n else ''

def _decode_int16(raw: bytes, phys_min: float, phys_max: float) -> np.ndarray:
    digital = np.frombuffer(raw, dtype='<i2').astype(np.float64)
    return (digital + 32768.0) * (phys_max - phys_min) / 65535.0 + phys_min

# ---------------------------------------------------------------------------
# Header reader  (cursor lands at start of index section on return)
# ---------------------------------------------------------------------------

def _read_header(f) -> dict:
    magic = f.read(4)
    if magic != _LWF_MAGIC:
        raise ValueError(f"not a valid .lwf file (bad magic: {magic!r})")
    version = _read_i32(f)
    if version != _LWF_VERSION:
        raise ValueError(f"unsupported .lwf version {version} (expected {_LWF_VERSION})")

    id_   = _read_string(f)
    _edf  = _read_string(f)
    _read_string(f)          # stored output filename — discard
    startdate = _read_string(f)
    starttime = _read_string(f)
    tag   = _read_string(f)
    align = _read_string(f)

    n_annots = _read_i32(f)
    annots = [_read_string(f) for _ in range(n_annots)]

    n_ch = _read_i32(f)
    channels = []
    for _ in range(n_ch):
        label    = _read_string(f)
        unit     = _read_string(f)
        _read_u64(f)          # sample_step_tp — not needed in Python
        sr       = _read_f64(f)
        phys_min = _read_f64(f)
        phys_max = _read_f64(f)
        channels.append({'label': label, 'unit': unit, 'sr': sr,
                         'phys_min': phys_min, 'phys_max': phys_max})

    n_features = _read_i32(f)
    feature_names = [_read_string(f) for _ in range(n_features)]

    n_waves = _read_i32(f)

    return {
        'id':            id_,
        'startdate':     startdate,
        'starttime':     starttime,
        'tag':           tag,
        'align':         align,
        'annots':        annots,
        'channels':      channels,
        'feature_names': feature_names,
        'n_waves':       n_waves,
    }

# ---------------------------------------------------------------------------
# Index entry reader  (one event's index record; leaves cursor at the next one)
# ---------------------------------------------------------------------------

def _read_index_entry(f) -> dict:
    """Read a single index record without touching the payload section.

    Returns the per-event fields plus ``n_blocks``, ``payload_offset`` (absolute
    file offset of the event's payload) and ``ns`` (samples in the first block).
    """
    annot    = _read_string(f)
    instance = _read_string(f)
    annot_ch = _read_string(f)
    annot_start = _read_f64(f)
    annot_stop  = _read_f64(f)
    anchor      = _read_f64(f)
    wave_start  = _read_f64(f)
    wave_stop   = _read_f64(f)
    payload_offset = _read_u64(f)   # absolute file offset of this wave's payload

    n_blocks = _read_i32(f)
    ns_first: int | None = None
    for b in range(n_blocks):
        ns = _read_i32(f)
        _read_f64(f)          # data_start_sec
        _read_f64(f)          # data_stop_sec
        if b == 0:
            ns_first = ns

    return {
        'annot':           annot,
        'instance':        instance,
        'annot_ch':        annot_ch,
        'annot_start_sec': annot_start,
        'annot_stop_sec':  annot_stop,
        'anchor_sec':      anchor,
        'wave_start_sec':  wave_start,
        'wave_stop_sec':   wave_stop,
        'n_blocks':        n_blocks,
        'payload_offset':  payload_offset,
        'ns':              ns_first if ns_first is not None else 0,
    }

# ---------------------------------------------------------------------------
# Events reader  (index section then payload section, single sequential pass)
# ---------------------------------------------------------------------------

def _read_events(
    f,
    header: dict,
    *,
    channels: set[str] | None = None,
    sample_n: int | None = None,
    sample_by: str = 'record',
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, pd.DataFrame, bool, list[str] | None]:
    """Read the events of a single file.

    ``channels`` restricts what is loaded.  In annot-ch-match files (one block
    per event) it keeps only events whose detecting channel (``annot_ch``) is in
    the set.  In standard files (all channels per event) it selects the matching
    channel *blocks* instead, narrowing the channel axis.

    ``sample_n`` randomly keeps at most that many events from this file, drawn
    (without replacement) *after* any channel-based event filtering, using
    ``rng``.  ``sample_by`` controls the draw in annot-ch-match files:
    ``'record'`` keeps N events total (channel mix follows prevalence);
    ``'channel'`` keeps up to N events for *each* detecting channel (balanced).
    In standard files every event already carries all channels, so ``sample_by``
    has no effect there.
    """
    n_waves    = header['n_waves']
    n_ch       = len(header['channels'])
    n_features = len(header['feature_names'])

    # build label -> (phys_min, phys_max) for int16 decoding
    ch_phys = {c['label']: (c['phys_min'], c['phys_max']) for c in header['channels']}

    # --- pass 1: index section ---
    index_rows: list[dict] = []
    expected_n: int | None = None
    annot_ch_match = False  # detected if any event has fewer blocks than channels

    for w in range(n_waves):
        e = _read_index_entry(f)
        if e['n_blocks'] < n_ch:
            annot_ch_match = True

        ns = e['ns']
        if expected_n is None:
            expected_n = ns
        elif ns != expected_n:
            raise ValueError(
                f"non-uniform waveform length at event {w}: "
                f"got {ns} samples, expected {expected_n}"
            )

        index_rows.append(e)

    if expected_n is None:
        expected_n = 0

    def _draw(pool: list[int]) -> list[int]:
        if len(pool) <= sample_n:
            return list(pool)
        picks = rng.choice(len(pool), size=sample_n, replace=False)
        return [pool[i] for i in picks]

    # --- select which events (waves) to keep ---
    keep = list(range(n_waves))
    if channels is not None and annot_ch_match:
        keep = [w for w in keep if index_rows[w]['annot_ch'] in channels]
    if sample_n is not None:
        if sample_by == 'channel' and annot_ch_match:
            groups: dict[str, list[int]] = {}
            for w in keep:
                groups.setdefault(index_rows[w]['annot_ch'], []).append(w)
            kept: list[int] = []
            for ws in groups.values():
                kept.extend(_draw(ws))
            keep = sorted(kept)
        else:
            keep = sorted(_draw(keep))

    # --- select which channel blocks to keep (standard files only) ---
    if annot_ch_match:
        out_n_ch = 1
        sel_labels: list[str] | None = None
        block_to_col: dict[int, int] = {}
    else:
        if channels is not None:
            sel_blocks = [i for i, c in enumerate(header['channels'])
                          if c['label'] in channels]
        else:
            sel_blocks = list(range(n_ch))
        out_n_ch = len(sel_blocks)
        sel_labels = [header['channels'][i]['label'] for i in sel_blocks]
        block_to_col = {b: col for col, b in enumerate(sel_blocks)}

    data = np.full((len(keep), out_n_ch, expected_n), np.nan, dtype=np.float64)
    meta_col: list[str] = []

    # --- pass 2: payload section (seek to each kept wave) ---
    for out_i, w in enumerate(keep):
        f.seek(index_rows[w]['payload_offset'])
        meta_col.append(_read_string(f))  # meta — only in payload
        n_blocks = _read_i32(f)

        for b in range(n_blocks):
            if n_features > 0:
                f.read(4 + n_features * 8)   # feature_qc (int32) + feature values (float64)

            raw = f.read(expected_n * 2)     # int16 = 2 bytes; uniform ns (validated)

            if annot_ch_match:
                label = index_rows[w]['annot_ch']
                phys_min, phys_max = ch_phys.get(label, (0.0, 1.0))
                data[out_i, 0, :] = _decode_int16(raw, phys_min, phys_max)
            elif b in block_to_col:
                label = header['channels'][b]['label']
                phys_min, phys_max = ch_phys.get(label, (0.0, 1.0))
                data[out_i, block_to_col[b], :] = _decode_int16(raw, phys_min, phys_max)

    index_cols = ['annot', 'instance', 'annot_ch', 'annot_start_sec',
                  'annot_stop_sec', 'anchor_sec', 'wave_start_sec', 'wave_stop_sec']
    kept_rows = [index_rows[w] for w in keep]
    if kept_rows:
        event_meta = pd.DataFrame(kept_rows)[index_cols]
    else:
        event_meta = pd.DataFrame(columns=index_cols)
    event_meta['meta'] = meta_col
    return data, event_meta, annot_ch_match, sel_labels

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_paths(
    paths: Union[str, Path, Iterable],
    recur: bool,
) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    resolved: list[Path] = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            pattern = '**/*.lwf' if recur else '*.lwf'
            resolved.extend(sorted(p.glob(pattern)))
        else:
            resolved.append(p)

    if not resolved:
        raise ValueError("no .lwf files found in the given paths")
    return resolved

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LwfSummary(pd.DataFrame):
    """A per-file .lwf summary table (see :func:`lwf_summary`).

    Behaves as an ordinary :class:`pandas.DataFrame`, with an extra
    :meth:`summary` method that collapses the rows into a whole-sample overview.
    """

    @property
    def _constructor(self):
        return LwfSummary

    def summary(self) -> pd.Series:
        """Collapse the per-file rows into one whole-sample overview.

        Returns a :class:`pandas.Series` with the total number of files,
        subjects (unique ``id``), events, unique channels, per-event sample
        length, and the estimated full-load memory footprint.
        """
        chans = sorted({c for row in self['channels'] for c in row.split(',') if c})
        n_samp = sorted(set(int(x) for x in self['n_samples']))
        acm = sorted(set(bool(x) for x in self['annot_ch_match']))
        total_bytes = int(self['est_bytes'].sum())
        return pd.Series({
            'files':             len(self),
            'subjects':          int(self['id'].nunique()),
            'events':            int(self['n_waves'].sum()),
            'channels':          len(chans),
            'channel_labels':    ','.join(chans),
            'samples_per_event': n_samp[0] if len(n_samp) == 1 else 'varies',
            'annot_ch_match':    acm[0] if len(acm) == 1 else 'mixed',
            'est_bytes':         total_bytes,
            'est_mb':            round(total_bytes / 1024**2, 2),
            'est_gb':            round(total_bytes / 1024**3, 2),
        })


def lwf_summary(
    paths: Union[str, Path, Iterable],
    *,
    recur: bool = False,
) -> LwfSummary:
    """Summarise one or more .lwf files without loading signal data.

    Parameters
    ----------
    paths:
        A .lwf file path, a directory containing .lwf files, or a list of
        either.
    recur:
        Recurse into subdirectories when *paths* contains a directory.

    Returns
    -------
    LwfSummary (a pandas.DataFrame subclass) with one row per file and columns:
    ``file``, ``id``, ``tag``, ``startdate``, ``starttime``, ``align``,
    ``n_waves``, ``n_channels``, ``channels``, ``srs``, ``annots``,
    ``n_features``, ``n_samples``, ``annot_ch_match``, ``est_bytes``,
    ``est_mb``.

    ``n_samples`` is the per-event waveform length and ``est_bytes`` /
    ``est_mb`` estimate the in-memory (float64) size of loading the whole file
    with :func:`from_lwf` (``n_waves * out_channels * n_samples * 8``).  These
    come from a cheap peek at the first index record, not a full load.

    Call ``.summary()`` on the result for a whole-sample overview (total files,
    subjects, events, unique channels, total memory).
    """
    files = _resolve_paths(paths, recur)
    rows = []
    for p in files:
        with open(p, 'rb') as f:
            h = _read_header(f)
            if h['n_waves'] > 0:
                e0 = _read_index_entry(f)
                n_samples = e0['ns']
                annot_ch_match = e0['n_blocks'] < len(h['channels'])
            else:
                n_samples = 0
                annot_ch_match = False
        out_n_ch = 1 if annot_ch_match else len(h['channels'])
        est_bytes = h['n_waves'] * out_n_ch * n_samples * 8
        rows.append({
            'file':           str(p),
            'id':             h['id'],
            'tag':            h['tag'],
            'startdate':      h['startdate'],
            'starttime':      h['starttime'],
            'align':          h['align'],
            'n_waves':        h['n_waves'],
            'n_channels':     len(h['channels']),
            'channels':       ','.join(c['label'] for c in h['channels']),
            'srs':            ','.join(str(c['sr']) for c in h['channels']),
            'annots':         ','.join(h['annots']),
            'n_features':     len(h['feature_names']),
            'n_samples':      n_samples,
            'annot_ch_match': annot_ch_match,
            'est_bytes':      est_bytes,
            'est_mb':         round(est_bytes / 1024**2, 2),
        })
    return LwfSummary(rows)


def lwf_event_counts(
    paths: Union[str, Path, Iterable],
    *,
    recur: bool = False,
) -> pd.DataFrame:
    """Count events per detecting channel without loading signal data.

    Scans the (small) index section of each file — no waveform payloads are
    decoded — and reports how many events are associated with each ``annot_ch``.
    Useful for planning a load, e.g. deciding ``sample_n`` / ``sample_by`` in
    :func:`from_lwf`.

    Parameters
    ----------
    paths:
        A .lwf file path, a directory containing .lwf files, or a list of
        either.
    recur:
        Recurse into subdirectories when *paths* contains a directory.

    Returns
    -------
    pandas.DataFrame in long form with one row per (file, channel):
    ``file``, ``id``, ``annot_ch``, ``n_events``.
    """
    files = _resolve_paths(paths, recur)
    rows = []
    for p in files:
        with open(p, 'rb') as f:
            h = _read_header(f)
            counts: dict[str, int] = {}
            for _ in range(h['n_waves']):
                e = _read_index_entry(f)
                counts[e['annot_ch']] = counts.get(e['annot_ch'], 0) + 1
        for ch, n in sorted(counts.items()):
            rows.append({
                'file':     str(p),
                'id':       h['id'],
                'annot_ch': ch,
                'n_events': n,
            })
    return pd.DataFrame(rows)


def from_lwf(
    paths: Union[str, Path, Iterable],
    *,
    recur: bool = False,
    channels: Union[str, Iterable[str], None] = None,
    sample_n: int | None = None,
    sample_by: str = 'record',
    seed: int | None = None,
) -> AxisAnnotatedTensor:
    """Load one or more .lwf files into an AxisAnnotatedTensor.

    All files must share the same channel labels and per-channel sample rates.
    All waveform events must have the same number of samples (i.e. Luna's
    ``require=full`` mode, or any mode that produced uniform windows).

    When the file was written with ``annot-ch-match=T``, each event contains
    one channel block (the detecting channel).  The tensor shape is then
    ``(n_waves, 1, n_samples)`` and ``axis_meta['instance']['annot_ch']``
    identifies the channel for each event.

    Parameters
    ----------
    paths:
        A .lwf file path, a directory containing .lwf files, or a list of
        either.
    recur:
        Recurse into subdirectories when *paths* contains a directory.
    channels:
        Restrict which channels are loaded (a label or an iterable of labels).
        In annot-ch-match files this keeps only events whose detecting channel
        (``annot_ch``) is one of ``channels``.  In standard files it selects
        the matching channel blocks, narrowing the channel axis.  ``None``
        (default) loads everything.
    sample_n:
        If given, randomly keep at most this many events, drawn without
        replacement after any ``channels`` filtering.  Groups with fewer events
        are loaded in full.
    sample_by:
        Grouping for ``sample_n`` in annot-ch-match files: ``'record'``
        (default) keeps N events per file (channel mix follows prevalence);
        ``'channel'`` keeps up to N events per detecting channel *per file*
        (balanced across channels).  Ignored for standard files, where every
        event already carries all channels.
    seed:
        Seed for the ``sample_n`` random draw, for reproducibility.

    Returns
    -------
    AxisAnnotatedTensor with axes ``['instance', 'channel', 'sample']``.
    """
    files = _resolve_paths(paths, recur)

    if channels is not None:
        channels = {channels} if isinstance(channels, str) else set(channels)
    if sample_by not in ('record', 'channel'):
        raise ValueError(f"sample_by must be 'record' or 'channel', got {sample_by!r}")
    rng = np.random.default_rng(seed) if sample_n is not None else None

    # --- pass 1: read all headers and validate consistency ---
    headers: list[dict] = []
    for p in files:
        with open(p, 'rb') as f:
            h = _read_header(f)
        h['_path'] = p
        headers.append(h)

    ref_ch = headers[0]['channels']

    ref_sig = [(c['label'], c['sr']) for c in ref_ch]
    for h in headers[1:]:
        sig = [(c['label'], c['sr']) for c in h['channels']]
        if sig != ref_sig:
            raise ValueError(
                f"channel mismatch:\n"
                f"  {headers[0]['_path']}: {ref_sig}\n"
                f"  {h['_path']}: {sig}"
            )

    if channels is not None:
        avail = {c['label'] for c in ref_ch}
        missing = channels - avail
        if missing:
            raise ValueError(
                f"requested channel(s) not present: {sorted(missing)}; "
                f"available: {sorted(avail)}"
            )

    # when selecting a channel subset, only those channels' sample rates matter
    sel_ch = [c for c in ref_ch if channels is None or c['label'] in channels]
    all_srs = [c['sr'] for c in sel_ch]
    if len(set(all_srs)) > 1:
        raise ValueError(
            f"channels have mixed sample rates {all_srs}; "
            f"select a single channel before loading"
        )
    sr = all_srs[0]

    ref_align = headers[0]['align']

    # --- pass 2: read events from each file ---
    all_data:  list[np.ndarray] = []
    all_meta:  list[pd.DataFrame] = []
    expected_n: int | None = None
    detected_annot_ch_match: bool | None = None
    sel_labels: list[str] | None = None

    for h in headers:
        with open(h['_path'], 'rb') as f:
            _read_header(f)
            data, event_meta, annot_ch_match, file_sel_labels = _read_events(
                f, h, channels=channels, sample_n=sample_n,
                sample_by=sample_by, rng=rng,
            )
        if file_sel_labels is not None:
            sel_labels = file_sel_labels

        if detected_annot_ch_match is None:
            detected_annot_ch_match = annot_ch_match
        elif annot_ch_match != detected_annot_ch_match:
            raise ValueError(
                f"mixed annot-ch-match modes across files: {h['_path']}"
            )

        ns = data.shape[2]
        if expected_n is None:
            expected_n = ns
        elif ns != expected_n:
            raise ValueError(
                f"sample count mismatch across files: "
                f"expected {expected_n}, got {ns} in {h['_path']}"
            )

        event_meta.insert(0, 'file', str(h['_path']))
        event_meta.insert(0, 'tag',  h['tag'])
        event_meta.insert(0, 'id',   h['id'])

        all_data.append(data)
        all_meta.append(event_meta)

    if expected_n is None:
        expected_n = 0
    if detected_annot_ch_match is None:
        detected_annot_ch_match = False

    data       = np.concatenate(all_data, axis=0)
    event_meta = pd.concat(all_meta, ignore_index=True)

    n_events, n_ch_out, n_samples = data.shape

    if n_events > 0:
        offset = (
            event_meta['wave_start_sec'].iloc[0]
            - event_meta['anchor_sec'].iloc[0]
        )
    else:
        offset = 0.0
    time_axis = offset + np.arange(n_samples) / sr

    # channel axis metadata
    if detected_annot_ch_match:
        ch_meta = pd.DataFrame([{'label': '(annot_ch)', 'unit': '', 'sr': sr}])
    else:
        lbl_map = {c['label']: c for c in ref_ch}
        labels = sel_labels if sel_labels is not None else [c['label'] for c in ref_ch]
        ch_meta = pd.DataFrame({
            'label': labels,
            'unit':  [lbl_map[l]['unit'] for l in labels],
            'sr':    [lbl_map[l]['sr']   for l in labels],
        })

    col_order = [
        'id', 'tag', 'file',
        'annot', 'instance', 'annot_ch', 'meta',
        'anchor_sec', 'annot_start_sec', 'annot_stop_sec',
        'wave_start_sec', 'wave_stop_sec',
    ]
    event_meta = event_meta[col_order].reset_index(drop=True)

    return AxisAnnotatedTensor(
        data=data,
        axes=['instance', 'channel', 'sample'],
        axis_index={
            'instance': np.arange(n_events),
            'channel':  np.arange(n_ch_out),
            'sample':   time_axis,
        },
        axis_meta={
            'instance': event_meta,
            'channel':  ch_meta,
            'sample': pd.DataFrame({
                'sample_index': np.arange(n_samples),
                'time':         time_axis,
            }),
        },
        attrs={
            'sfreq':             sr,
            'source_files':      [str(h['_path']) for h in headers],
            'align':             ref_align,
            'annot_ch_match':    detected_annot_ch_match,
        },
    )
