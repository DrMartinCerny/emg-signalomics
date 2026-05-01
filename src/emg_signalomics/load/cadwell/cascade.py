# emg_signalomics/load/cadwell/cascade.py
"""
Cadwell Cascade / IOMax CSV -> MNE Raw loader (single file).

This loader expects a typical Cadwell "Raw EMG Raw Data.csv" style export:
- semicolon-separated CSV
- decimal comma in numeric fields
- first row: column headers (sample index + channel names)
- next ~7 rows: per-channel metadata (Time, Gain, filters, Period (s), Units, ...)
- then: waveform samples, where the first column is a sample index/counter

Public API
----------
load_cascade(path, *, channels_to_keep=None, max_samples=None) -> mne.io.Raw
    Load a Cadwell Cascade CSV and return an MNE Raw object — nothing more,
    nothing less.  The Cadwell metadata block is consumed only to set the
    standard MNE fields (channel names, sampling rate, measurement date,
    channel data in Volts) and to surface sample-index discontinuities as
    ``BAD_gap`` annotations.

head_cascade(path) -> dict
    Read only the header + per-channel metadata block (the first 8 rows of
    the file) and return ``{"channel_names": ..., "sfreq": ..., "meas_date":
    ..., "highpass": ..., "lowpass": ..., "notch": ..., "units": ...,
    "format": "cadwell_cascade_csv", "source_file": ...}``.  Useful for
    deciding whether to load the full file.

Sanitized example (NOT real data; channel names and values are illustrative only).
You can copy-paste this into a .csv and it should parse:
Name;EMG_CH01;EMG_CH02;EMG_CH03
Time;20250101120000;20250101120000;20250101120000
Gain (µV/Div);1000;1000;1000
Highcut (Hz);2000;2000;2000
Lowcut (Hz);30;30;30
Notch;On;On;On
Period (s);0,00080000;0,00080000;0,00080000
Units;µV;µV;µV
0;0,152588;-3,051758;0,305176
1;0,762940;-3,814697;0,152588
2;0,610352;-3,967285;0,000000
3;0,305176;-3,509521;-0,152588
4;0,152588;-2,899170;-0,305176
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import mne


# -----------------------------
# Public API
# -----------------------------

def load_cascade(
    path: Union[str, Path],
    *,
    channels_to_keep: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
) -> mne.io.Raw:
    """
    Load a Cadwell Cascade / IOMax CSV export into an MNE Raw object.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the CSV file.
    channels_to_keep : sequence of str, optional
        If given, restricts loading to these channel names (must match the
        header exactly).  Defaults to all channels other than the timing
        column.
    max_samples : int, optional
        If given, only the first ``max_samples`` waveform rows are loaded.

    Returns
    -------
    mne.io.Raw
        Preloaded Raw object with EMG channels in Volts, sampling rate and
        measurement date populated from the Cadwell header, and ``BAD_gap``
        annotations covering any sample-index discontinuities.
    """
    path = Path(path)

    cols = _read_header_columns_fast(path)
    if not cols:
        raise ValueError(f"No columns found in {path}")

    timing_col = "Name" if "Name" in cols else cols[0]

    if channels_to_keep is None:
        channels = [c for c in cols if c != timing_col]
    else:
        req = [_norm_col(c) for c in channels_to_keep]
        missing = [c for c in req if c not in cols]
        if missing:
            raise ValueError(
                "Some requested channels are not present in the file.\n"
                f"Missing: {missing}\nAvailable: {cols}"
            )
        channels = req

    if len(channels) == 0:
        raise ValueError(f"No EMG channels selected. timing_col={timing_col!r}, cols={cols}")

    usecols = [timing_col] + list(channels)

    # ---- metadata block (7 rows after header) ----
    # Parsed only to derive the standard MNE fields below.  Not stored on the
    # Raw object — callers who need the raw Cadwell header should use
    # ``head_cascade`` or re-read the CSV themselves.
    channel_meta = _read_channel_meta_block(path, cols, list(channels))

    fs = _infer_fs_from_channel_meta(channel_meta, list(channels))
    start_ts_naive = _infer_start_timestamp(channel_meta, list(channels))
    units = _infer_units(channel_meta, list(channels))

    # ---- waveform block ----
    skiprows = list(range(1, 8))  # skip metadata lines

    dtype_map: Dict[str, Any] = {ch: np.float32 for ch in channels}
    dtype_map[timing_col] = np.float64

    df = pd.read_csv(
        path,
        sep=";",
        header=0,
        usecols=usecols,
        skiprows=skiprows,
        nrows=max_samples,
        decimal=",",
        encoding="utf-8",
        dtype=dtype_map,
        memory_map=True,
        low_memory=False,
    )

    if timing_col not in df.columns:
        raise ValueError(
            f"Timing column {timing_col!r} not found after loading. "
            f"Loaded columns: {list(df.columns)}"
        )

    time_col = df[timing_col].to_numpy(dtype=np.float64, copy=False)
    df = df.drop(columns=[timing_col])

    # (n_channels, n_times)
    data = np.vstack(
        [df[ch].to_numpy(dtype=np.float32, copy=False) for ch in channels]
    ).astype(np.float32, copy=False)

    # Convert units -> Volts (MNE convention)
    data_v = _to_volts(data, units)

    info = mne.create_info(
        ch_names=list(channels),
        sfreq=float(fs),
        ch_types=["emg"] * len(channels),
    )

    raw = mne.io.RawArray(data_v, info, verbose="ERROR")

    # Attach meas_date (MNE requires UTC-aware datetime)
    if start_ts_naive is not None:
        raw.set_meas_date(start_ts_naive.replace(tzinfo=timezone.utc))

    # Surface sample-index discontinuities as BAD_gap annotations
    gaps = _detect_gaps(time_col)
    if gaps:
        raw.set_annotations(
            raw.annotations + _gaps_to_bad_annotations(gaps, sfreq=float(fs))
        )

    return raw


def head_cascade(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read only the header + per-channel metadata block of a Cadwell CSV.

    This touches only the first 8 rows of the file — the column header and
    the seven metadata rows — and never opens the waveform block, so it is
    cheap even for multi-gigabyte exports.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the CSV file.

    Returns
    -------
    dict
        ``{
            "format":        "cadwell_cascade_csv",
            "source_file":   str(path),
            "channel_names": [...],
            "n_channels":    int,
            "sfreq":         float (Hz, derived from "Period (s)"),
            "meas_date":     UTC-aware datetime | None,
            "units":         str,        # the original "Units" field
            "highpass":      float | None,  # from "Lowcut (Hz)"
            "lowpass":       float | None,  # from "Highcut (Hz)"
            "notch":         str | None,    # the raw "Notch" field, e.g. "On"/"Off"
        }``
    """
    path = Path(path)

    cols = _read_header_columns_fast(path)
    if not cols:
        raise ValueError(f"No columns found in {path}")
    timing_col = "Name" if "Name" in cols else cols[0]
    channels = [c for c in cols if c != timing_col]
    if not channels:
        raise ValueError(
            f"No EMG channels found in {path} (timing_col={timing_col!r})."
        )

    channel_meta = _read_channel_meta_block(path, cols, channels)

    fs = _infer_fs_from_channel_meta(channel_meta, channels)
    start_ts_naive = _infer_start_timestamp(channel_meta, channels)
    units = _infer_units(channel_meta, channels)
    meas_date = (
        start_ts_naive.replace(tzinfo=timezone.utc)
        if start_ts_naive is not None
        else None
    )

    first = channel_meta[channels[0]]
    lff = _parse_float_maybe(first.get("Lowcut (Hz)"))
    hff = _parse_float_maybe(first.get("Highcut (Hz)"))
    notch_raw = first.get("Notch")
    notch = str(notch_raw).strip() if notch_raw is not None else None

    return {
        "format": "cadwell_cascade_csv",
        "source_file": str(path),
        "channel_names": list(channels),
        "n_channels": len(channels),
        "sfreq": float(fs),
        "meas_date": meas_date,
        "units": units,
        "highpass": lff,
        "lowpass": hff,
        "notch": notch,
    }


# -----------------------------
# Helpers
# -----------------------------

def _norm_col(s: str) -> str:
    return str(s).replace("﻿", "").strip()


def _read_header_columns_fast(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        line = f.readline().strip("\n\r")
    return [_norm_col(x) for x in line.split(";")]


def _read_channel_meta_block(
    path: Path, cols: List[str], channels: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Read the 7-row Cadwell metadata block (Time, Gain, Highcut, Lowcut, Notch,
    Period (s), Units) and return ``{channel_name: {key: value}}``.
    """
    meta_block = pd.read_csv(
        path,
        sep=";",
        header=None,
        nrows=7,
        skiprows=1,
        encoding="utf-8",
    )

    col_index = {name: idx for idx, name in enumerate(cols)}
    channel_meta: Dict[str, Dict[str, Any]] = {ch: {} for ch in channels}
    for _, row in meta_block.iterrows():
        key = str(row.iloc[0]).strip()
        if not key or key.lower() == "nan":
            continue
        for ch in channels:
            idx = col_index[ch]
            channel_meta[ch][key] = row.iloc[idx] if idx < len(row) else None
    return channel_meta


def _parse_float_maybe(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_timestamp_yyyymmddhhmmss(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _infer_fs_from_channel_meta(
    channel_meta: Dict[str, Dict[str, Any]], channels: List[str]
) -> float:
    for ch in channels:
        period = _parse_float_maybe(channel_meta[ch].get("Period (s)", None))
        if period and period > 0:
            return float(1.0 / period)
    raise ValueError("Could not infer sampling rate from 'Period (s)' in metadata block.")


def _infer_start_timestamp(
    channel_meta: Dict[str, Dict[str, Any]], channels: List[str]
) -> Optional[datetime]:
    for ch in channels:
        ts = _parse_timestamp_yyyymmddhhmmss(channel_meta[ch].get("Time", None))
        if ts is not None:
            return ts
    return None


def _infer_units(
    channel_meta: Dict[str, Dict[str, Any]], channels: List[str]
) -> str:
    units_set = set()
    for ch in channels:
        u = channel_meta[ch].get("Units", None)
        if u is not None and str(u).strip():
            units_set.add(str(u).strip())
    if len(units_set) == 1:
        return units_set.pop()
    if len(units_set) > 1:
        return "mixed"
    return "unknown"


def _to_volts(data: np.ndarray, units: str) -> np.ndarray:
    """
    Convert *data* to Volts (MNE convention).  Unknown / mixed units are left
    unscaled; a UserWarning is emitted so the caller is not silently surprised.
    """
    u = (units or "").strip().lower()

    if u in {"uv", "µv", "microv", "microvolt", "microvolts"}:
        return data.astype(np.float64, copy=False) * 1e-6

    if u in {"mv", "millivolt", "millivolts"}:
        return data.astype(np.float64, copy=False) * 1e-3

    if u in {"v", "volt", "volts"}:
        return data.astype(np.float64, copy=False)

    import warnings
    warnings.warn(
        f"Cadwell metadata reported units={units!r}; data left unscaled. "
        "MNE expects Volts — set units in the source file or convert manually.",
        UserWarning,
        stacklevel=3,
    )
    return data.astype(np.float64, copy=False)


def _detect_gaps(time_col: Optional[np.ndarray]) -> List[Dict[str, int]]:
    """
    Walk the sample-index column and return a list of
    ``{"after_sample", "before_sample", "gap_samples"}`` dicts for each
    missing run.  Empty list if the column is monotonic, too short, or
    contains NaN/inf.
    """
    if time_col is None or len(time_col) < 2:
        return []
    if np.any(~np.isfinite(time_col)):
        return []

    t_int = np.rint(time_col.astype(np.float64)).astype(np.int64)
    bad = np.where(np.diff(t_int) != 1)[0]
    if len(bad) == 0:
        return []

    gaps: List[Dict[str, int]] = []
    for i in bad[:5000]:
        after_sample = int(t_int[i])
        before_sample = int(t_int[i + 1])
        gap_samples = before_sample - after_sample - 1
        if gap_samples > 0:
            gaps.append(
                {
                    "after_sample": after_sample,
                    "before_sample": before_sample,
                    "gap_samples": gap_samples,
                }
            )
    return gaps


def _gaps_to_bad_annotations(
    gaps: List[Dict[str, int]], sfreq: float
) -> mne.Annotations:
    """
    Build BAD_gap annotations for missing sample intervals.

    Annotated region corresponds to missing samples
    ``(after_sample + 1) ... (before_sample - 1)``.
    """
    onsets: List[float] = []
    durations: List[float] = []
    desc: List[str] = []

    for g in gaps:
        after_sample = g["after_sample"]
        gap_samples = g["gap_samples"]

        onset_s = (after_sample + 1) / float(sfreq)
        duration_s = gap_samples / float(sfreq)

        if duration_s <= 0:
            continue

        onsets.append(onset_s)
        durations.append(duration_s)
        desc.append("BAD_gap")

    return mne.Annotations(onset=onsets, duration=durations, description=desc)
