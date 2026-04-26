"""
atrain.py — A-train detection in MNE Raw objects.

Background
----------
An "A-train" (afterdischarge train) is a sustained burst of high-frequency
oscillatory activity, defined by spectral periodicity in the LOWCUT-HIGHCUT
band (default 100-200 Hz).  Detection is based on a sliding-window
autocorrelation: a segment of signal is periodic at frequency f if its
normalised autocorrelation function shows a clear peak at lag 1/f.  Each
window is summarised by its *peak* autocorrelation across all lags that
correspond to the target frequency band, giving a single "how periodic is
this window" score in [-1, 1].

Detection pipeline
------------------
  1. Silence gate       — a fast global RMS pass on the native-rate signal
                          identifies non-silent regions (chunks).  Only these
                          chunks are subsequently upsampled and analysed.

  2. Upsample           — each active chunk is resampled to ac_target_fs so
                          that the lag grid is fine enough for accurate
                          frequency estimates (important when the native rate
                          is low relative to the target band).

  3. Core windows       — windows where peak AC ≥ ac_core_threshold AND the
                          native-rate RMS is above rms_silence_threshold_uv
                          seed candidate A-trains.  Contiguous core runs
                          shorter than ac_core_min_dur_ms are discarded before
                          seeding to suppress isolated noise spikes.

  4. Grow (flood-fill)  — each surviving core is expanded left and right into
                          neighbouring windows where peak AC ≥
                          ac_adjacent_threshold.  The silence gate does NOT
                          restrict expansion — only AC matters here, so brief
                          amplitude dips within a train are not artificially
                          split.

  5. Gap merge          — candidates separated by a gap ≤ ac_merge_gap_ms are
                          fused.  Applied twice: at the window level (within a
                          chunk) and at the sample level (across chunk
                          boundaries).

  6. Duration filter    — candidates shorter than ac_min_dur_ms are discarded.

Public API
----------
detect_atrains_single_channel(signal_uv, fs, *, ...)
    Accepts a 1-D numpy array in µV and returns a dict with four sample-level
    arrays: atrain_mask, rms_envelope, peak_ac, peak_lag_hz.  Useful for
    inspection and diagnostic plotting without an MNE object.

detect_atrains(raw, *, picks, ...)
    Wraps detect_atrains_single_channel for every picked channel of an MNE
    Raw object and returns mne.Annotations.  Each annotation has the
    description "ATRAIN_{channel_name}" so the source channel is recoverable.
    Append to the recording with:

        raw.set_annotations(raw.annotations + detect_atrains(raw, ...))

    Filter by channel with:

        [a for a in raw.annotations if a["description"] == "ATRAIN_Fp1"]
"""

import numpy as np
import mne
from scipy.signal import resample_poly
from math import gcd


# ---------------------------------------------------------------------------
# Internal helper: fast vectorised rolling RMS via stride tricks
# ---------------------------------------------------------------------------

def _rolling_rms(x, win_samp, step_samp):
    """
    Compute RMS for every sliding window of length win_samp with stride
    step_samp over the 1-D array x.  Uses as_strided for zero-copy framing.

    Returns
    -------
    rms    : (n_windows,) float
    starts : (n_windows,) int   — index of the first sample of each window
    """
    starts  = np.arange(0, len(x) - win_samp + 1, step_samp)
    shape   = (len(starts), win_samp)
    strides = (x.strides[0] * step_samp, x.strides[0])
    frames  = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    rms     = np.sqrt(np.mean(frames ** 2, axis=1))
    return rms, starts


# ---------------------------------------------------------------------------
# Internal helper: find contiguous active sample ranges from a per-window
# above-threshold mask, with symmetric padding for AC window context
# ---------------------------------------------------------------------------

def _active_chunks(above_mask, win_starts, win_samp, pad_samp, signal_len):
    """
    Convert a per-window boolean mask into a list of (start, end) sample
    index pairs for non-silent regions.  Adjacent/overlapping regions are
    merged.  Each boundary is padded by pad_samp to ensure windows centred
    near chunk edges have enough context.

    Returns
    -------
    list of (int, int) — (chunk_start, chunk_end) in sample indices
    """
    if not np.any(above_mask):
        return []

    active_starts = win_starts[above_mask]
    active_ends   = active_starts + win_samp

    # Pad outward and clamp to signal boundaries
    active_starts = np.maximum(0,           active_starts - pad_samp)
    active_ends   = np.minimum(signal_len,  active_ends   + pad_samp)

    # Sort and merge overlapping intervals
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
# Internal helper: sliding-window peak autocorrelation
# ---------------------------------------------------------------------------

def _sliding_peak_ac(x, fs, win_ms, step_ms, lag_lo_hz, lag_hi_hz):
    """
    For each sliding window of signal x (already at rate fs), compute:
      - the peak normalised autocorrelation across all lags corresponding
        to frequencies in [lag_hi_hz, lag_lo_hz]  (higher freq = shorter lag)
      - the frequency (Hz) of that dominant lag

    Window size is forced to be at least 3x the longest lag to ensure
    reliable AC estimation; capped at half the signal length.

    Parameters
    ----------
    x          : 1-D array — signal at rate fs (upsample before calling if needed)
    fs         : float — sampling rate of x (Hz)
    win_ms     : float — requested window duration (ms); may be enlarged
    step_ms    : float — step between successive windows (ms)
    lag_lo_hz  : float — lower bound of target band (Hz) → longest lag
    lag_hi_hz  : float — upper bound of target band (Hz) → shortest lag

    Returns
    -------
    peak_ac        : (n_windows,) float — peak AC in [-1, 1]; NaN for flat windows
    peak_lag_hz    : (n_windows,) float — dominant frequency (Hz); NaN for flat
    win_centers_ms : (n_windows,) float — window centre times in ms from x[0]
    win_starts     : (n_windows,) int   — window start sample indices in x
    win_samp       : int                — actual window size used (samples in x)
    """
    win_samp  = int(win_ms  / 1000.0 * fs)
    step_samp = max(1, int(step_ms / 1000.0 * fs))

    # Convert frequency bounds to lag bounds (frequency and lag are inverse)
    lag_lo_samp = max(1, int(fs / lag_lo_hz))   # shortest lag (highest freq)
    lag_hi_samp = int(fs / lag_hi_hz) + 1        # longest  lag (lowest  freq)
    lags        = np.arange(lag_lo_samp, lag_hi_samp + 1)

    # Enforce minimum window length and cap at half signal length
    win_samp = max(win_samp, lag_hi_samp * 3)
    win_samp = min(win_samp, len(x) // 2)

    starts = np.arange(0, len(x) - win_samp, step_samp)
    if len(starts) == 0:
        # Signal too short — return a single NaN placeholder
        return (np.array([np.nan]), np.array([np.nan]),
                np.array([len(x) / 2 / fs * 1000.0]), np.array([0]), win_samp)

    peak_ac     = np.full(len(starts), np.nan)
    peak_lag_hz = np.full(len(starts), np.nan)

    for j, s in enumerate(starts):
        win  = x[s : s + win_samp]
        win  = win - win.mean()          # remove DC offset before correlating
        norm = np.dot(win, win)
        if norm == 0:
            continue                     # flat window — leave NaN

        # Only include lags that fit within this window
        valid_lags = [lag for lag in lags if lag < win_samp]
        ac_vals = np.array([
            np.dot(win[:-lag], win[lag:]) / norm
            for lag in valid_lags
        ])

        if len(ac_vals):
            best           = np.nanargmax(ac_vals)
            peak_ac[j]     = ac_vals[best]
            peak_lag_hz[j] = fs / valid_lags[best]   # lag in samples → Hz

    win_centers_ms = (starts + win_samp // 2) / fs * 1000.0
    return peak_ac, peak_lag_hz, win_centers_ms, starts, win_samp


# ---------------------------------------------------------------------------
# Internal helper: bridge short gaps in a boolean window-level mask
# ---------------------------------------------------------------------------

def _merge_atrain_gaps(win_mask, max_gap_steps):
    """
    Fill gaps of ≤ max_gap_steps consecutive False values between True runs.
    Operates on the window timeline (not the sample timeline).

    Returns a new boolean array of the same length as win_mask.
    """
    merged    = win_mask.copy()
    in_gap    = False
    gap_start = 0

    for i in range(len(merged)):
        if merged[i]:
            if in_gap:
                if (i - gap_start) <= max_gap_steps:
                    merged[gap_start:i] = True
                in_gap = False
        else:
            if not in_gap:
                in_gap    = True
                gap_start = i

    return merged


# ---------------------------------------------------------------------------
# Internal helper: remove short A-train segments from a sample-level mask
# ---------------------------------------------------------------------------

def _discard_short_atrains(atrain_mask, times_s, min_dur_s):
    """
    Zero out any True run in atrain_mask whose duration is < min_dur_s seconds.
    times_s must be the per-sample time array aligned to atrain_mask.

    Returns a new boolean array of the same length.
    """
    out    = atrain_mask.copy()
    edges  = np.diff(atrain_mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)

    # Handle mask that starts or ends mid-run
    if atrain_mask[0]:
        starts = [0] + starts
    if atrain_mask[-1]:
        ends = ends + [len(atrain_mask)]

    for s, e in zip(starts, ends):
        if (times_s[e - 1] - times_s[s]) < min_dur_s:
            out[s:e] = False

    return out


# ---------------------------------------------------------------------------
# Internal helper: interpolate window-level scalars onto the sample timeline
# ---------------------------------------------------------------------------

def _interp_window_to_samples(win_centers_s, win_values, times_s):
    """
    Linearly interpolate per-window scalar values onto a dense sample timeline.
    NaN entries in win_values are excluded before interpolation.
    Samples outside the range of finite window centres are filled with the
    value of the nearest finite window (nearest-edge extrapolation).

    Returns (T,) float — all NaN if no finite windows exist.
    """
    finite = np.isfinite(win_values)
    if not np.any(finite):
        return np.full(len(times_s), np.nan)
    xp = win_centers_s[finite]
    fp = win_values[finite]
    return np.interp(times_s, xp, fp, left=fp[0], right=fp[-1])


# ---------------------------------------------------------------------------
# Public: single-channel detection
# ---------------------------------------------------------------------------

def detect_atrains_single_channel(
    signal_uv,
    fs,
    *,
    lowcut:                   float = 100.0,
    highcut:                  float = 200.0,
    ac_window_ms:             float = 40.0,
    ac_step_ms:               float = 0.5,
    ac_core_threshold:        float = 0.70,
    ac_adjacent_threshold:    float = 0.50,
    ac_core_min_dur_ms:       float = 10.0,
    ac_merge_gap_ms:          float = 10.0,
    ac_min_dur_ms:            float = 30.0,
    rms_silence_threshold_uv: float = 2.0,
    rms_envelope_window_ms:   float = 100.0,
    ac_target_fs:             float = 5000.0,
):
    """
    Run the full A-train detection pipeline on a single 1-D signal array.

    Parameters
    ----------
    signal_uv : array-like, shape (T,)
        Raw signal in microvolts.
    fs : float
        Native sampling frequency of signal_uv (Hz).
    lowcut : float, default 100.0
        Lower edge of the target oscillation band (Hz).
    highcut : float, default 200.0
        Upper edge of the target oscillation band (Hz).
    ac_window_ms : float, default 40.0
        Sliding autocorrelation window duration (ms).  The actual window used
        may be longer if the band geometry requires it (≥ 3x the longest lag).
    ac_step_ms : float, default 0.5
        Step between successive AC windows (ms).
    ac_core_threshold : float, default 0.70
        Peak-AC value a window must reach to seed a new A-train candidate.
        The RMS silence gate also applies at this stage.
    ac_adjacent_threshold : float, default 0.50
        Peak-AC value required for a window to be absorbed into an existing
        candidate during flood-fill.  The silence gate does NOT apply here
        so brief amplitude dips within a train are not artificially split.
    ac_core_min_dur_ms : float, default 10.0
        Minimum duration (ms) of a contiguous core run before it is allowed
        to seed a flood-fill.  At ac_step_ms=0.5 this means ≥ 20 consecutive
        windows.  Suppresses isolated noise spikes.
    ac_merge_gap_ms : float, default 10.0
        Maximum gap (ms) between two A-train candidates that will be bridged.
        Applied at both the window level (within a chunk) and the sample level
        (across chunk boundaries).
    ac_min_dur_ms : float, default 30.0
        Minimum duration (ms) of a surviving A-train after all merging.
        Shorter detections are discarded.
    rms_silence_threshold_uv : float, default 2.0
        RMS amplitude (µV) below which a window is considered silent.  Silent
        windows cannot seed core detections and define chunk boundaries used
        to skip quiet signal regions entirely.
    rms_envelope_window_ms : float, default 100.0
        Window duration (ms) for the smooth RMS envelope returned for display.
        Longer than the silence-gate window so the envelope tracks sustained
        energy rather than individual cycle amplitude.
    ac_target_fs : float, default 5000.0
        Target sampling rate (Hz) for autocorrelation.  Active chunks are
        upsampled to this rate before AC to increase lag resolution.  If the
        native fs is already ≥ ac_target_fs, no upsampling is performed.

    Returns
    -------
    dict with the following keys (all arrays shape (T,), aligned to signal_uv):

    "atrain_mask" : np.ndarray, dtype bool
        Sample-level binary A-train label.
    "rms_envelope" : np.ndarray, dtype float
        Smooth RMS amplitude envelope (µV), window = rms_envelope_window_ms.
    "peak_ac" : np.ndarray, dtype float
        Peak autocorrelation in [-1, 1] per sample, interpolated from the AC
        window grid.  NaN in silent regions where no AC was computed.
    "peak_lag_hz" : np.ndarray, dtype float
        Dominant oscillation frequency (Hz) per sample, derived from the lag
        that produced the peak AC.  NaN in silent regions.
    """
    signal_uv = np.asarray(signal_uv, dtype=float)
    T         = len(signal_uv)
    times_s   = np.arange(T) / fs

    step_samp = max(1, int(ac_step_ms / 1000.0 * fs))
    step_s    = ac_step_ms / 1000.0

    # ---- rational upsampling factors for AC ---------------------------------
    # resample_poly requires integer up/down ratios; reduce by GCD to keep
    # the ratio minimal and avoid unnecessary computation.
    if fs < ac_target_fs:
        _g    = gcd(int(ac_target_fs), int(fs))
        _up   = int(ac_target_fs) // _g
        _dn   = int(fs)           // _g
        fs_ac = fs * _up / _dn    # actual rate achieved after rational resample
    else:
        _up, _dn = 1, 1
        fs_ac    = fs              # native rate is already sufficient

    # ---- silence-gate window sizing -----------------------------------------
    # The AC window must span at least 3 full cycles of the lowest target
    # frequency so autocorrelation at the longest lag is reliable.
    lag_hi_samp  = int(fs / lowcut) + 1       # longest lag at native rate
    win_samp_req = int(ac_window_ms / 1000.0 * fs)
    win_samp_est = max(win_samp_req, lag_hi_samp * 3)
    pad_samp     = win_samp_est // 2 + 1      # symmetric padding around chunks

    # ---- global RMS pass: silence gate and chunk identification -------------
    # Runs on the full native-rate signal.  Cheap stride-trick RMS; only
    # windows above rms_silence_threshold_uv define active chunks for AC.
    rms_wins_full, win_starts_full = _rolling_rms(signal_uv, win_samp_est, step_samp)
    above_silence_wins_full        = rms_wins_full >= rms_silence_threshold_uv

    # ---- separate longer-window RMS for smooth diagnostic envelope ----------
    # Uses rms_envelope_window_ms (default 100 ms) so the displayed envelope
    # reflects sustained amplitude, not individual oscillation cycles.
    env_win_samp  = max(win_samp_est, int(rms_envelope_window_ms / 1000.0 * fs))
    env_wins, env_starts = _rolling_rms(signal_uv, env_win_samp, step_samp)
    env_centers_s = (env_starts + env_win_samp / 2) / fs
    rms_envelope  = _interp_window_to_samples(env_centers_s, env_wins, times_s)

    # Identify contiguous active chunks; silent gaps are skipped entirely
    chunks = _active_chunks(
        above_silence_wins_full, win_starts_full, win_samp_est, pad_samp, T
    )

    # Output diagnostic maps — NaN where no AC was computed (silent regions)
    peak_ac_map  = np.full(T, np.nan)
    peak_lag_map = np.full(T, np.nan)
    atrain_raw   = np.zeros(T, dtype=bool)

    for (chunk_start, chunk_end) in chunks:
        chunk = signal_uv[chunk_start:chunk_end]

        # ---- upsample chunk for AC ------------------------------------------
        # Higher sample rate → more lag steps in the target band → smoother
        # frequency estimates and better AC resolution.
        if _up > 1:
            chunk_ac = resample_poly(chunk, _up, _dn)
        else:
            chunk_ac = chunk

        # ---- sliding peak AC on upsampled chunk -----------------------------
        peak_ac_wins, peak_lag_hz_wins, win_centers_ms_local, win_starts_local, win_samp = \
            _sliding_peak_ac(
                chunk_ac, fs_ac,
                win_ms    = ac_window_ms,
                step_ms   = ac_step_ms,
                lag_lo_hz = highcut,   # higher freq bound → shorter lag bound
                lag_hi_hz = lowcut,    # lower  freq bound → longer  lag bound
            )

        n_wins_chunk        = len(peak_ac_wins)
        win_centers_s_local = win_centers_ms_local / 1000.0  # local seconds

        # ---- fill diagnostic maps -------------------------------------------
        # Interpolate AC and lag values onto the native-rate sample timeline.
        # Only samples within the window-centre span are filled; the dead-zone
        # half-window margins at each chunk edge remain NaN.
        finite = np.isfinite(peak_ac_wins)
        if np.any(finite):
            xp = win_centers_s_local[finite]
            chunk_times_local = times_s[chunk_start:chunk_end] - times_s[chunk_start]
            in_span = (chunk_times_local >= xp[0]) & (chunk_times_local <= xp[-1])
            if np.any(in_span):
                peak_ac_map[chunk_start:chunk_end][in_span] = np.interp(
                    chunk_times_local[in_span], xp, peak_ac_wins[finite]
                )
                peak_lag_map[chunk_start:chunk_end][in_span] = np.interp(
                    chunk_times_local[in_span], xp, peak_lag_hz_wins[finite]
                )

        # ---- per-chunk silence gate (native rate) ---------------------------
        # Re-compute RMS on the native-rate chunk aligned to the same window
        # geometry as the AC windows so the silence mask maps correctly onto
        # peak_ac_wins.  This is used only for core seeding, not flood-fill.
        win_samp_native = int(ac_window_ms / 1000.0 * fs)
        win_samp_native = max(win_samp_native, int(fs / lowcut) * 3)
        win_samp_native = min(win_samp_native, len(chunk) // 2)
        step_samp_local = max(1, int(ac_step_ms / 1000.0 * fs))
        starts_native   = np.arange(0, len(chunk) - win_samp_native, step_samp_local)

        rms_wins_chunk = (
            np.array([np.sqrt(np.mean(chunk[s : s + win_samp_native] ** 2))
                      for s in starts_native])
            if len(starts_native)
            else np.array([np.sqrt(np.mean(chunk ** 2))])
        )
        above_silence_chunk = rms_wins_chunk >= rms_silence_threshold_uv

        # Interpolate native-rate silence decisions onto the AC window grid
        if len(starts_native) > 1:
            rms_centers_local = (starts_native + win_samp_native / 2) / fs
            above_silence_ac  = np.interp(
                win_centers_s_local,
                rms_centers_local,
                above_silence_chunk.astype(float),
                left=above_silence_chunk[0],
                right=above_silence_chunk[-1],
            ) >= 0.5
        else:
            above_silence_ac = np.full(n_wins_chunk, bool(above_silence_chunk[0]))

        # ---- core and adjacent window masks ---------------------------------
        # Core requires high AC AND sufficient RMS — real signal must be present
        # to seed a detection.
        # Adjacent requires only high AC — the silence gate is intentionally
        # excluded so the flood-fill can cross brief amplitude dips within a
        # sustained oscillatory burst without artificially splitting it.
        core_wins_chunk = (peak_ac_wins >= ac_core_threshold) & above_silence_ac
        adj_eligible    = (peak_ac_wins >= ac_adjacent_threshold)

        # ---- core minimum duration filter -----------------------------------
        # Remove contiguous core runs shorter than ac_core_min_dur_ms to
        # suppress isolated noise spikes from seeding detections.
        core_min_steps = max(1, int(np.ceil(ac_core_min_dur_ms / ac_step_ms)))
        if core_min_steps > 1:
            filtered_core = core_wins_chunk.copy()
            c_edges  = np.diff(core_wins_chunk.astype(int))
            c_starts = list(np.where(c_edges == 1)[0] + 1)
            c_ends   = list(np.where(c_edges == -1)[0] + 1)
            if core_wins_chunk[0]:
                c_starts = [0] + c_starts
            if core_wins_chunk[-1]:
                c_ends = c_ends + [len(core_wins_chunk)]
            for cs, ce in zip(c_starts, c_ends):
                if (ce - cs) < core_min_steps:
                    filtered_core[cs:ce] = False
            core_wins_chunk = filtered_core

        # ---- flood-fill expansion -------------------------------------------
        # Grow outward from surviving core windows one step per iteration.
        # Only adj_eligible windows are absorbed.  Already-confirmed windows
        # are never re-evaluated so they cannot be erased by a later silence
        # check — this was a subtle prior bug.
        atrain_wins_chunk = core_wins_chunk.copy()
        while True:
            new_candidates       = np.zeros(n_wins_chunk, dtype=bool)
            new_candidates[1:]  |= atrain_wins_chunk[:-1]   # right neighbours
            new_candidates[:-1] |= atrain_wins_chunk[1:]    # left  neighbours
            new_candidates      &= adj_eligible
            new_candidates      &= ~atrain_wins_chunk        # only new windows
            if not np.any(new_candidates):
                break
            atrain_wins_chunk |= new_candidates

        # ---- window-level gap merge (within this chunk) ---------------------
        merge_gap_steps   = int(np.ceil(ac_merge_gap_ms / ac_step_ms))
        atrain_wins_chunk = _merge_atrain_gaps(atrain_wins_chunk, merge_gap_steps)

        # ---- project window mask → sample mask ------------------------------
        # Each detected window claims all native-rate samples within ±½ step
        # of its centre time.
        for wi in range(n_wins_chunk):
            if not atrain_wins_chunk[wi]:
                continue
            tc  = chunk_start / fs + win_centers_s_local[wi]
            idx = np.where(
                (times_s >= tc - step_s / 2) & (times_s < tc + step_s / 2)
            )[0]
            atrain_raw[idx] = True

    # ---- sample-level gap merge (across chunk boundaries) -------------------
    # The window-level merge cannot bridge gaps that span two separate active
    # chunks split by a brief silence dropout.  This pass operates on the
    # final sample mask and closes those cross-chunk gaps.
    merge_gap_samp = int(ac_merge_gap_ms / 1000.0 * fs)
    s_edges    = np.diff(atrain_raw.astype(int))
    seg_starts = list(np.where(s_edges == 1)[0] + 1)
    seg_ends   = list(np.where(s_edges == -1)[0] + 1)
    if atrain_raw[0]:
        seg_starts = [0] + seg_starts
    if atrain_raw[-1]:
        seg_ends = seg_ends + [len(atrain_raw)]
    for i in range(len(seg_starts) - 1):
        if (seg_starts[i + 1] - seg_ends[i]) <= merge_gap_samp:
            atrain_raw[seg_ends[i]:seg_starts[i + 1]] = True

    # ---- final duration filter ----------------------------------------------
    atrain_mask = _discard_short_atrains(atrain_raw, times_s, ac_min_dur_ms / 1000.0)

    return {
        "atrain_mask":  atrain_mask,
        "rms_envelope": rms_envelope,
        "peak_ac":      peak_ac_map,
        "peak_lag_hz":  peak_lag_map,
    }


# ---------------------------------------------------------------------------
# Public API: MNE wrapper
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
    ac_core_min_dur_ms:       float = 10.0,
    ac_merge_gap_ms:          float = 10.0,
    ac_min_dur_ms:            float = 30.0,
    rms_silence_threshold_uv: float = 2.0,
    rms_envelope_window_ms:   float = 100.0,
    ac_target_fs:             float = 5000.0,
) -> "mne.Annotations":
    """
    Detect A-trains in every picked channel of *raw* and return annotations.

    Each detected segment becomes one mne.Annotation with description
    ``"ATRAIN_{channel_name}"``.  MNE annotations are global (not per-channel),
    so the channel identity is encoded in the description string.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        A preloaded MNE Raw object.  Data are read in µV via
        ``raw.get_data(units="uV")``.
    picks : str | list | slice | None
        Channels to analyse, passed to MNE's picks resolver.  None → all channels.
    lowcut : float, default 100.0
        Lower edge of the target oscillation band (Hz).
    highcut : float, default 200.0
        Upper edge of the target oscillation band (Hz).
    ac_window_ms : float, default 40.0
        Sliding autocorrelation window duration (ms).
    ac_step_ms : float, default 0.5
        Step between successive AC windows (ms).
    ac_core_threshold : float, default 0.70
        Peak-AC threshold to seed an A-train candidate.
    ac_adjacent_threshold : float, default 0.50
        Peak-AC threshold to extend an existing candidate during flood-fill.
    ac_core_min_dur_ms : float, default 10.0
        Minimum duration (ms) of a contiguous core run before it seeds a detection.
    ac_merge_gap_ms : float, default 10.0
        Maximum gap (ms) between candidates that will be bridged into one detection.
    ac_min_dur_ms : float, default 30.0
        Minimum duration (ms) of a surviving A-train detection.
    rms_silence_threshold_uv : float, default 2.0
        RMS amplitude (µV) below which windows are silent and cannot seed cores.
    rms_envelope_window_ms : float, default 100.0
        Window duration (ms) for the smooth RMS envelope (diagnostic output only).
    ac_target_fs : float, default 5000.0
        Target sampling rate (Hz) for AC computation.  No-op if native fs ≥ target.

    Returns
    -------
    mne.Annotations
        One annotation per detected A-train segment.  Onset and duration are in
        seconds relative to ``raw.first_time``.  Append to existing annotations
        with ``raw.set_annotations(raw.annotations + detect_atrains(raw, ...))``.
    """
    fs = raw.info["sfreq"]
    t0 = raw.first_time

    picked_idx = mne.io.pick._picks_to_idx(raw.info, picks, none="all", exclude=())

    onsets       = []
    durations    = []
    descriptions = []

    for ch_idx in picked_idx:
        ch_name   = raw.ch_names[ch_idx]
        signal_uv = raw.get_data(picks=[ch_idx], units="uV")[0]

        result = detect_atrains_single_channel(
            signal_uv, fs,
            lowcut                   = lowcut,
            highcut                  = highcut,
            ac_window_ms             = ac_window_ms,
            ac_step_ms               = ac_step_ms,
            ac_core_threshold        = ac_core_threshold,
            ac_adjacent_threshold    = ac_adjacent_threshold,
            ac_core_min_dur_ms       = ac_core_min_dur_ms,
            ac_merge_gap_ms          = ac_merge_gap_ms,
            ac_min_dur_ms            = ac_min_dur_ms,
            rms_silence_threshold_uv = rms_silence_threshold_uv,
            rms_envelope_window_ms   = rms_envelope_window_ms,
            ac_target_fs             = ac_target_fs,
        )
        atrain_mask = result["atrain_mask"]

        # Convert boolean sample mask to (onset, duration) annotation pairs
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