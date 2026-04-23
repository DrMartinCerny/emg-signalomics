"""
atrain.py — A-train detection in MNE Raw objects.

Background
----------
An "A-train" (afterdischarge train) is a sustained burst of high-frequency
oscillatory activity, here defined by spectral periodicity in the LOWCUT-HIGHCUT
band (default 100-200 Hz).  Detection is based on a sliding window autocorrelation:
a window of signal is periodic at frequency f if its autocorrelation function shows
a clear peak at lag 1/f.  We summarise each window by its *peak* autocorrelation
across all lags that correspond to the target frequency band, giving a single
"how periodic is this window" score between -1 and 1.

Detection pipeline (five stages)
---------------------------------
  1. Silence gate     — windows whose RMS amplitude falls below a threshold are
                        excluded before any AC logic runs.

  2. Core windows     — windows where peak AC ≥ ac_core_threshold seed candidate
                        A-trains.

  3. Grow (flood-fill)— each core is expanded left and right into adjacent windows
                        satisfying peak AC ≥ ac_adjacent_threshold AND not silenced.

  4. Gap merge        — two A-train candidates separated by a gap shorter than
                        ac_merge_gap_ms are fused into one.

  5. Duration filter  — A-train candidates shorter than ac_min_dur_ms are discarded.

Public API
----------
detect_atrains(raw, *, picks, lowcut, highcut,
                 ac_window_ms, ac_step_ms,
                 ac_core_threshold, ac_adjacent_threshold,
                 ac_merge_gap_ms, ac_min_dur_ms,
                 rms_silence_threshold_uv)
    -> mne.Annotations

Runs detection on each picked channel independently and returns an
mne.Annotations object containing one annotation per detected A-train
segment, with description "ATRAIN_{channel_name}".  Because MNE annotations
are global (not per-channel), the channel identity is encoded in the
description string.  The annotations can be appended to the Raw object with:

    raw.set_annotations(raw.annotations + detect_atrains(raw, ...))

and later filtered by channel with:

    [a for a in raw.annotations if a["description"] == "ATRAIN_Fp1"]
"""

import numpy as np
import mne


# ---------------------------------------------------------------------------
# Internal helper: fast vectorised rolling RMS via stride tricks
# ---------------------------------------------------------------------------

def _rolling_rms(x, win_samp, step_samp):
    starts  = np.arange(0, len(x) - win_samp + 1, step_samp)
    shape   = (len(starts), win_samp)
    strides = (x.strides[0] * step_samp, x.strides[0])
    frames  = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    rms     = np.sqrt(np.mean(frames ** 2, axis=1))
    return rms, starts


# ---------------------------------------------------------------------------
# Internal helper: find active sample ranges from a per-window above-threshold mask
# ---------------------------------------------------------------------------

def _active_chunks(above_mask, win_starts, win_samp, pad_samp, signal_len):
    if not np.any(above_mask):
        return []

    active_starts = win_starts[above_mask]
    active_ends   = active_starts + win_samp

    active_starts = np.maximum(0,           active_starts - pad_samp)
    active_ends   = np.minimum(signal_len,  active_ends   + pad_samp)

    order  = np.argsort(active_starts)
    starts = active_starts[order]
    ends   = active_ends[order]

    merged = []
    cs, ce = int(starts[0]), int(ends[0])
    for s, e in zip(starts[1:], ends[1:]):
        s, e = int(s), int(e)
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


# ---------------------------------------------------------------------------
# Internal helper: sliding peak autocorrelation (no matrix retained)
# ---------------------------------------------------------------------------

def _sliding_peak_ac(x, fs, win_ms, step_ms, lag_lo_hz, lag_hi_hz):
    """
    Compute per-window peak autocorrelation over the target lag range.

    Returns
    -------
    peak_ac        : (n_windows,) float
    win_centers_ms : (n_windows,) float  — relative to x[0]
    win_starts     : (n_windows,) int
    win_samp       : int
    """
    win_samp  = int(win_ms  / 1000.0 * fs)
    step_samp = max(1, int(step_ms / 1000.0 * fs))

    lag_lo_samp = max(1, int(fs / lag_lo_hz))
    lag_hi_samp = int(fs / lag_hi_hz) + 1
    lags        = np.arange(lag_lo_samp, lag_hi_samp + 1)

    win_samp = max(win_samp, lag_hi_samp * 3)
    win_samp = min(win_samp, len(x) // 2)

    starts = np.arange(0, len(x) - win_samp, step_samp)
    if len(starts) == 0:
        return np.array([np.nan]), np.array([len(x) / 2 / fs * 1000.0]), np.array([0]), win_samp

    peak_ac = np.full(len(starts), np.nan)
    for j, s in enumerate(starts):
        win  = x[s : s + win_samp]
        win  = win - win.mean()
        norm = np.dot(win, win)
        if norm == 0:
            continue
        ac_vals = np.array([
            np.dot(win[:-lag], win[lag:]) / norm
            for lag in lags if lag < win_samp
        ])
        if len(ac_vals):
            peak_ac[j] = np.nanmax(ac_vals)

    win_centers_ms = (starts + win_samp // 2) / fs * 1000.0
    return peak_ac, win_centers_ms, starts, win_samp


# ---------------------------------------------------------------------------
# Internal helper: gap merge on a boolean window-timeline mask
# ---------------------------------------------------------------------------

def _merge_atrain_gaps(win_mask, max_gap_steps):
    merged    = win_mask.copy()
    in_gap    = False
    gap_start = 0

    for i in range(len(merged)):
        if merged[i]:
            if in_gap:
                gap_len = i - gap_start
                if gap_len <= max_gap_steps:
                    merged[gap_start:i] = True
                in_gap = False
        else:
            if not in_gap:
                in_gap    = True
                gap_start = i

    return merged


# ---------------------------------------------------------------------------
# Internal helper: discard short A-trains on the sample timeline
# ---------------------------------------------------------------------------

def _discard_short_atrains(atrain_mask, times_s, min_dur_s):
    out    = atrain_mask.copy()
    edges  = np.diff(atrain_mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)

    if atrain_mask[0]:
        starts = [0] + starts
    if atrain_mask[-1]:
        ends = ends + [len(atrain_mask)]

    for s, e in zip(starts, ends):
        if (times_s[e - 1] - times_s[s]) < min_dur_s:
            out[s:e] = False

    return out


# ---------------------------------------------------------------------------
# Internal: single-channel detection (numpy in, bool mask out)
# ---------------------------------------------------------------------------

def _detect_atrains_single_channel(
    signal_uv, fs, *,
    lowcut, highcut, ac_window_ms, ac_step_ms,
    ac_core_threshold, ac_adjacent_threshold,
    ac_merge_gap_ms, ac_min_dur_ms, rms_silence_threshold_uv,
):
    """Run the full five-stage A-train pipeline on a single 1-D signal array."""
    signal_uv = np.asarray(signal_uv, dtype=float)
    T         = len(signal_uv)
    times_s   = np.arange(T) / fs

    step_samp = max(1, int(ac_step_ms / 1000.0 * fs))
    step_s    = ac_step_ms / 1000.0

    lag_hi_samp  = int(fs / lowcut) + 1
    win_samp_req = int(ac_window_ms / 1000.0 * fs)
    win_samp_est = max(win_samp_req, lag_hi_samp * 3)
    pad_samp     = win_samp_est // 2 + 1

    rms_wins_full, win_starts_full = _rolling_rms(signal_uv, win_samp_est, step_samp)
    above_silence_wins_full        = rms_wins_full >= rms_silence_threshold_uv

    chunks = _active_chunks(
        above_silence_wins_full, win_starts_full, win_samp_est, pad_samp, T
    )

    atrain_raw = np.zeros(T, dtype=bool)

    for (chunk_start, chunk_end) in chunks:
        chunk = signal_uv[chunk_start:chunk_end]

        peak_ac_wins, win_centers_ms_local, win_starts_local, win_samp = \
            _sliding_peak_ac(
                chunk, fs,
                win_ms    = ac_window_ms,
                step_ms   = ac_step_ms,
                lag_lo_hz = highcut,
                lag_hi_hz = lowcut,
            )

        n_wins_chunk        = len(peak_ac_wins)
        win_centers_s_local = win_centers_ms_local / 1000.0

        rms_wins_chunk = np.array([
            np.sqrt(np.mean(chunk[s : s + win_samp] ** 2))
            for s in win_starts_local
        ])
        above_silence_chunk = rms_wins_chunk >= rms_silence_threshold_uv

        core_wins_chunk   = (peak_ac_wins >= ac_core_threshold) & above_silence_chunk
        adj_eligible      = (peak_ac_wins >= ac_adjacent_threshold) & above_silence_chunk
        atrain_wins_chunk = core_wins_chunk.copy()

        while True:
            expanded       = atrain_wins_chunk.copy()
            expanded[1:]  |= atrain_wins_chunk[:-1]
            expanded[:-1] |= atrain_wins_chunk[1:]
            expanded      &= adj_eligible
            if np.array_equal(expanded, atrain_wins_chunk):
                break
            atrain_wins_chunk = expanded

        merge_gap_steps   = int(np.ceil(ac_merge_gap_ms / ac_step_ms))
        atrain_wins_chunk = _merge_atrain_gaps(atrain_wins_chunk, merge_gap_steps)

        for wi in range(n_wins_chunk):
            if not atrain_wins_chunk[wi]:
                continue
            tc  = chunk_start / fs + win_centers_s_local[wi]
            idx = np.where(
                (times_s >= tc - step_s / 2) & (times_s < tc + step_s / 2)
            )[0]
            atrain_raw[idx] = True

    return _discard_short_atrains(atrain_raw, times_s, ac_min_dur_ms / 1000.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_atrains(
    raw: "mne.io.BaseRaw",
    *,
    picks=None,
    lowcut:                   float = 100.0,
    highcut:                  float = 200.0,
    ac_window_ms:             float = 40.0,
    ac_step_ms:               float = 0.5,
    ac_core_threshold:        float = 0.70,
    ac_adjacent_threshold:    float = 0.50,
    ac_merge_gap_ms:          float = 10.0,
    ac_min_dur_ms:            float = 30.0,
    rms_silence_threshold_uv: float = 2.0,
) -> "mne.Annotations":
    """
    Detect A-trains in every picked channel of *raw* and return annotations.

    Each detected A-train segment becomes one mne.Annotation with description
    ``"ATRAIN_{channel_name}"``.  Annotations are global (MNE has no native
    per-channel annotation support), so the channel is encoded in the name.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        A preloaded MNE Raw object.  The data are read in µV (MNE default
        unit conversion is applied via ``raw.get_data(units="uV")``).
    picks : str | list | slice | None
        Channels to analyse — passed directly to ``mne.pick_types`` /
        ``raw.pick`` logic via ``mne.io.pick.pick_info``.  None → all channels.
    lowcut, highcut : float
        Target oscillation band edges (Hz).
    ac_window_ms : float
        Sliding autocorrelation window duration (ms).
    ac_step_ms : float
        Step between successive windows (ms).
    ac_core_threshold : float in [0, 1]
        Peak-AC threshold to seed an A-train candidate.
    ac_adjacent_threshold : float in [0, 1]
        Peak-AC threshold to extend an existing candidate.
    ac_merge_gap_ms : float
        Maximum gap (ms) between candidates to bridge.
    ac_min_dur_ms : float
        Minimum surviving A-train duration (ms).
    rms_silence_threshold_uv : float
        Windows below this RMS (µV) are excluded from detection.

    Returns
    -------
    mne.Annotations
        One annotation per detected A-train segment; onset and duration are in
        seconds relative to ``raw.first_time``.  Append to existing annotations
        with ``raw.set_annotations(raw.annotations + detect_atrains(raw, ...))``.
    """
    fs = raw.info["sfreq"]
    t0 = raw.first_time  # recording start offset (s) — needed for onset alignment

    # mne.pick_channels / mne.pick_types accept many formats; the canonical
    # portable way to resolve the 'picks' argument is _picks_to_idx.
    picked_idx = mne.io.pick._picks_to_idx(raw.info, picks, none="all", exclude=())

    onsets     = []
    durations  = []
    descriptions = []

    for ch_idx in picked_idx:
        ch_name   = raw.ch_names[ch_idx]
        # get_data returns (1, T) in Volts; convert to µV
        signal_uv = raw.get_data(picks=[ch_idx], units="uV")[0]

        atrain_mask = _detect_atrains_single_channel(
            signal_uv, fs,
            lowcut=lowcut, highcut=highcut,
            ac_window_ms=ac_window_ms, ac_step_ms=ac_step_ms,
            ac_core_threshold=ac_core_threshold,
            ac_adjacent_threshold=ac_adjacent_threshold,
            ac_merge_gap_ms=ac_merge_gap_ms, ac_min_dur_ms=ac_min_dur_ms,
            rms_silence_threshold_uv=rms_silence_threshold_uv,
        )

        # Convert boolean mask → (onset, duration) pairs
        times_s = np.arange(len(signal_uv)) / fs
        edges   = np.diff(atrain_mask.astype(int))
        starts  = list(np.where(edges == 1)[0] + 1)
        ends    = list(np.where(edges == -1)[0] + 1)

        if atrain_mask[0]:
            starts = [0] + starts
        if atrain_mask[-1]:
            ends = ends + [len(atrain_mask)]

        for s, e in zip(starts, ends):
            onsets.append(t0 + times_s[s])
            durations.append(times_s[e - 1] - times_s[s])
            descriptions.append(f"ATRAIN_{ch_name}")

    if not onsets:
        return mne.Annotations([], [], [])

    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=raw.info.get("meas_date"),
    )