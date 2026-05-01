# emg_signalomics/load/nim/eclipse.py
"""
Medtronic NIM Eclipse "Raw EMG" CSV -> MNE Raw loader (single file).

The Eclipse export is much less tidy than the Cadwell one.  After four lines
of free-form text header (Test name / Test start / Test end / Duration) and a
blank line, the file has a column header

    Mod No, Mod Name, Tr No, Tr Name, Date Time, LFF, HFF, Notch, 0, 1, ..., 9999

and then one record per "trace" (= one channel at one wall-clock second).
Each trace is serialised as a single fully-quoted CSV cell whose value is
itself a CSV record of 8 metadata fields followed by ~10 000 sample values.
The Eclipse writer additionally hard-breaks any line longer than 32 KiB at a
byte boundary, so the *quoted* cell may end on one physical line and the
remaining samples come back unquoted on the next physical line(s).  Both the
column header and individual traces can therefore wrap across 2-3 physical
lines.  csv.reader sees one single-field row followed by zero or more
multi-field rows per trace; this loader re-glues them.

Channels are recovered from the ``Tr Name`` field (one entry per channel),
and the sampling rate is inferred from the per-epoch sample count and the
spacing between consecutive ``Date Time`` timestamps (the test file has 1 s
epochs of 10 000 samples, giving 10 kHz).  Non-contiguous epochs are
surfaced as ``BAD_gap`` annotations on the returned Raw.

Public API
----------
load_eclipse(path, *, channels_to_keep=None, max_samples=None) -> mne.io.Raw
    Load a NIM Eclipse Raw EMG CSV and return an MNE Raw object.

Units
-----
The Eclipse file does not carry an explicit unit field for the Raw EMG
modality.  Sample values are assumed to be microvolts, which matches the
typical Eclipse default; the returned Raw is in Volts as MNE expects.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import csv
import sys
import warnings

import numpy as np
import mne


_META_COLS = (
    "Mod No",
    "Mod Name",
    "Tr No",
    "Tr Name",
    "Date Time",
    "LFF",
    "HFF",
    "Notch",
)


# -----------------------------
# Public API
# -----------------------------

def load_eclipse(
    path: Union[str, Path],
    *,
    channels_to_keep: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
) -> mne.io.Raw:
    """
    Load a Medtronic NIM Eclipse Raw EMG CSV export into an MNE Raw object.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the CSV file.
    channels_to_keep : sequence of str, optional
        If given, restricts loading to these channel names (matched against
        ``Tr Name`` after stripping whitespace).  Defaults to all channels
        present in the file, in the order they first appear.
    max_samples : int, optional
        If given, the returned Raw is truncated to the first ``max_samples``
        samples in the time dimension.

    Returns
    -------
    mne.io.Raw
        Preloaded Raw object with EMG channels in Volts, sampling rate and
        measurement date populated from the Eclipse header, ``info["highpass"]``
        / ``info["lowpass"]`` set from the ``LFF`` / ``HFF`` fields, and
        ``BAD_gap`` annotations covering any gap between consecutive epochs.
    """
    path = Path(path)
    rows = _read_csv_rows(path)

    # ---- locate column header ---------------------------------------------
    hdr_start = next(
        (i for i, r in enumerate(rows) if r and r[0].strip().startswith("Mod No")),
        None,
    )
    if hdr_start is None:
        raise ValueError(
            f"Could not find 'Mod No' column header in {path}; this does not "
            f"look like a NIM Eclipse Raw EMG CSV export."
        )

    # The column header may wrap across multiple physical lines; concatenate
    # consecutive non-blank rows.
    j = hdr_start
    column_header: List[str] = []
    while j < len(rows) and rows[j]:
        column_header.extend(s.strip() for s in rows[j])
        j += 1
    n_cols = len(column_header)
    if n_cols < len(_META_COLS) + 1:
        raise ValueError(
            f"Column header in {path} has only {n_cols} fields, expected at "
            f"least {len(_META_COLS) + 1}."
        )
    samples_per_epoch_decl = n_cols - len(_META_COLS)

    # Skip blank separator rows
    while j < len(rows) and not rows[j]:
        j += 1

    # ---- walk per-trace blocks --------------------------------------------
    traces = _parse_trace_blocks(rows, start_idx=j)
    if not traces:
        raise ValueError(f"No trace records found after the header in {path}.")

    # ---- channel selection -------------------------------------------------
    available: List[str] = []
    for t in traces:
        nm = t["Tr Name"]
        if nm not in available:
            available.append(nm)

    if channels_to_keep is None:
        channels = available
    else:
        req = [str(c).strip() for c in channels_to_keep]
        missing = [c for c in req if c not in available]
        if missing:
            raise ValueError(
                "Some requested channels are not present in the file.\n"
                f"Missing: {missing}\nAvailable: {available}"
            )
        channels = req

    # ---- group by channel, sort by Date Time ------------------------------
    by_channel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in traces:
        if t["Tr Name"] in channels:
            by_channel[t["Tr Name"]].append(t)
    for ch in channels:
        by_channel[ch].sort(key=lambda t: t["Date Time"])

    n_epochs_per_channel = {ch: len(by_channel[ch]) for ch in channels}
    if len(set(n_epochs_per_channel.values())) > 1:
        warnings.warn(
            f"Inconsistent epoch counts across channels: {n_epochs_per_channel}. "
            "Using the minimum so all channels have equal length.",
            UserWarning,
        )
    n_epochs = min(n_epochs_per_channel.values())
    if n_epochs == 0:
        raise ValueError("No epochs survived per-channel grouping.")

    # ---- align per-epoch sample counts ------------------------------------
    # The column header advertises N samples but individual traces sometimes
    # carry one or two extra; truncate all to the declared length so every
    # channel/epoch has identical shape.
    samples_per_epoch = samples_per_epoch_decl

    # ---- infer sampling rate ----------------------------------------------
    epoch_starts = [t["Date Time"] for t in by_channel[channels[0]][:n_epochs]]
    fs, typical_epoch_dur_s = _infer_sampling_rate(
        epoch_starts, samples_per_epoch
    )

    # ---- build the continuous timeline (with gap padding) -----------------
    # Each epoch contributes `samples_per_epoch` samples.  If the wall-clock
    # gap between two consecutive epoch starts exceeds the typical epoch
    # duration, we insert NaN-filled padding so the output timeline matches
    # real elapsed time and emit a BAD_gap annotation covering the padding.
    pieces_per_channel: Dict[str, List[np.ndarray]] = {ch: [] for ch in channels}
    gap_onsets_s: List[float] = []
    gap_durations_s: List[float] = []

    cumulative_samples = 0
    for ei in range(n_epochs):
        if ei > 0:
            delta_s = (epoch_starts[ei] - epoch_starts[ei - 1]).total_seconds()
            extra_s = delta_s - typical_epoch_dur_s
            if extra_s > 0.5 / fs:
                pad_n = int(round(extra_s * fs))
                onset_s = cumulative_samples / fs
                duration_s = pad_n / fs
                gap_onsets_s.append(onset_s)
                gap_durations_s.append(duration_s)
                for ch in channels:
                    pieces_per_channel[ch].append(
                        np.full(pad_n, np.nan, dtype=np.float64)
                    )
                cumulative_samples += pad_n

        for ch in channels:
            samples = by_channel[ch][ei]["samples"]
            if len(samples) >= samples_per_epoch:
                seg = samples[:samples_per_epoch]
            else:
                seg = np.concatenate(
                    [samples, np.full(samples_per_epoch - len(samples), np.nan)]
                )
            pieces_per_channel[ch].append(seg)
        cumulative_samples += samples_per_epoch

    data_uv = np.vstack(
        [np.concatenate(pieces_per_channel[ch]) for ch in channels]
    )

    # µV -> V (Eclipse Raw EMG is conventionally in microvolts)
    data_v = data_uv.astype(np.float64, copy=False) * 1e-6

    if max_samples is not None:
        data_v = data_v[:, :max_samples]

    # ---- build the MNE Raw -------------------------------------------------
    info = mne.create_info(
        ch_names=list(channels),
        sfreq=float(fs),
        ch_types=["emg"] * len(channels),
    )
    raw = mne.io.RawArray(data_v, info, verbose="ERROR")

    # meas_date — first epoch's start time, forced to UTC as MNE requires
    raw.set_meas_date(epoch_starts[0].replace(tzinfo=timezone.utc))

    # Filter cutoffs from the LFF/HFF fields of the first trace
    lff = _parse_float_maybe(traces[0]["LFF"])
    hff = _parse_float_maybe(traces[0]["HFF"])
    if lff is not None and hff is not None and 0 <= lff < hff:
        with raw.info._unlock():
            raw.info["highpass"] = float(lff)
            raw.info["lowpass"] = float(hff)

    # BAD_gap annotations for any padded gaps between epochs
    if max_samples is not None:
        max_t = max_samples / fs
        keep = [(o, d) for o, d in zip(gap_onsets_s, gap_durations_s) if o < max_t]
        gap_onsets_s = [o for o, d in keep]
        gap_durations_s = [min(d, max_t - o) for o, d in keep]

    if gap_onsets_s:
        raw.set_annotations(
            raw.annotations
            + mne.Annotations(
                onset=gap_onsets_s,
                duration=gap_durations_s,
                description=["BAD_gap"] * len(gap_onsets_s),
            )
        )

    return raw


# -----------------------------
# Helpers
# -----------------------------

def _read_csv_rows(path: Path) -> List[List[str]]:
    """Read the whole file with csv.reader, allowing very long fields."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.reader(f))


def _parse_trace_blocks(
    rows: List[List[str]], *, start_idx: int
) -> List[Dict[str, Any]]:
    """
    Walk *rows* from *start_idx* and yield one dict per trace.

    A trace block is a single-field row (the fully-quoted CSV cell carrying
    metadata + first chunk of samples) followed by zero or more multi-field
    rows that contain additional samples (the writer's 32 KiB hard-break).
    """
    out: List[Dict[str, Any]] = []
    i = start_idx
    while i < len(rows):
        r = rows[i]
        if not r:
            i += 1
            continue
        if len(r) != 1:
            # Stray multi-field row not preceded by a quoted record — skip.
            i += 1
            continue

        inner = next(csv.reader([r[0]]))
        i += 1
        while i < len(rows) and len(rows[i]) > 1:
            cont = rows[i]
            # The Eclipse writer hard-breaks every 32 KiB.  When the break
            # falls right after a comma the continuation row's csv.reader
            # output starts with an empty field; drop it so we don't inject
            # spurious NaN samples into the timeline.
            if cont and cont[0] == "":
                cont = cont[1:]
            inner.extend(cont)
            i += 1

        if len(inner) < len(_META_COLS) + 1:
            continue  # malformed trace; skip

        meta = {col: str(inner[k]).strip() for k, col in enumerate(_META_COLS)}
        meta["Tr Name"] = meta["Tr Name"].strip()
        meta["Date Time"] = _parse_eclipse_datetime(meta["Date Time"])
        if meta["Date Time"] is None:
            continue  # cannot place the trace on a timeline

        sample_strs = inner[len(_META_COLS) :]
        meta["samples"] = _parse_samples(sample_strs)
        out.append(meta)
    return out


def _parse_samples(sample_strs: List[str]) -> np.ndarray:
    """
    Convert the trailing sample strings into a float64 array.

    Empty/blank tokens are skipped (rather than emitted as NaN) so that the
    occasional empty field produced by the Eclipse 32 KiB hard-wrap doesn't
    pollute the per-trace sample count.  Tokens that fail to parse (e.g.,
    half a number from a mid-value wrap) are kept as NaN — there is no way
    to recover the original value, but the loader downstream still aligns
    everything to the declared samples-per-epoch.
    """
    out: List[float] = []
    for s in sample_strs:
        s = s.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except ValueError:
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float64)


def _parse_eclipse_datetime(s: str) -> Optional[datetime]:
    """
    Parse an Eclipse "Date Time" field.  The field looks like:
        "2/20/2025 10:20:14"
    sometimes with surrounding whitespace and stray quotes from the CSV
    quoting layer.
    """
    if s is None:
        return None
    s = s.strip().strip('"').strip()
    if not s:
        return None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _parse_float_maybe(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _infer_sampling_rate(
    epoch_starts: List[datetime], samples_per_epoch: int
) -> Tuple[float, float]:
    """
    Return ``(fs_hz, typical_epoch_duration_s)`` inferred from the file.

    Strategy:
      * If at least two epochs are present, the typical epoch duration is the
        smallest positive gap between consecutive ``Date Time`` values
        (longer gaps are dropouts/pauses).
      * If only one epoch is present, fall back to assuming a 1 s epoch (the
        Eclipse default for Raw EMG export) and emit a warning.
    """
    if len(epoch_starts) < 2:
        warnings.warn(
            "Single-epoch NIM Eclipse file; assuming a 1 s epoch and "
            f"sfreq = {samples_per_epoch} Hz.",
            UserWarning,
        )
        return float(samples_per_epoch), 1.0

    deltas = np.array(
        [
            (epoch_starts[i + 1] - epoch_starts[i]).total_seconds()
            for i in range(len(epoch_starts) - 1)
        ],
        dtype=np.float64,
    )
    positive = deltas[deltas > 0]
    if len(positive) == 0:
        warnings.warn(
            "Could not derive epoch duration from Date Time stamps; "
            f"assuming a 1 s epoch and sfreq = {samples_per_epoch} Hz.",
            UserWarning,
        )
        return float(samples_per_epoch), 1.0

    typical = float(np.min(positive))
    return float(samples_per_epoch) / typical, typical
