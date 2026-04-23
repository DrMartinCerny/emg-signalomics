"""
atrain.py — A-train detection in single-channel LFP/EEG signals.

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
                        excluded before any AC logic runs.  This prevents the
                        autocorrelation of flat/noise-floor signal from producing
                        spurious periodicity scores.

  2. Core windows     — windows where peak AC ≥ ac_core_threshold seed candidate
                        A-trains.  A high threshold here keeps false-positive seeds
                        rare; it is intentionally strict.

  3. Grow (flood-fill)— each core is iteratively expanded left and right into
                        adjacent windows that satisfy peak AC ≥ ac_adjacent_threshold
                        AND are not silenced.  This softer threshold captures the
                        rising and falling edges of a burst that may not meet the
                        strict core criterion.  Expansion halts when no further
                        eligible window is reachable.

  4. Gap merge        — two A-train candidates separated by a gap shorter than
                        ac_merge_gap_ms are fused into one.  This tolerates brief
                        within-burst interruptions (e.g. a single noisy window).
                        Note: gaps that span silenced regions are NOT merged, because
                        silenced windows are not eligible and cannot become True.

  5. Duration filter  — A-train candidates shorter than ac_min_dur_ms are discarded.
                        Very brief detections are most likely artefactual.

Performance optimisation
------------------------
Because silence typically constitutes ~95 % of the signal, autocorrelation (the
dominant cost) is computed only on *active chunks*: contiguous regions whose
rolling RMS exceeds rms_silence_threshold_uv.  Each active region is padded on
both sides by enough samples to guarantee that AC windows centred near the chunk
boundary have complete data.  Overlapping padded chunks are merged before
processing.  The full (T, n_lags) AC matrix is allocated up-front and filled
only in the processed regions; all other entries remain NaN.

Public API
----------
detect_atrains(signal_uv, fs, *, lowcut, highcut,
               ac_window_ms, ac_step_ms,
               ac_core_threshold, ac_adjacent_threshold,
               ac_merge_gap_ms, ac_min_dur_ms,
               rms_silence_threshold_uv)
    -> dict:
         "atrains"   : np.ndarray bool   (T,)        — final A-train mask
         "silence"   : np.ndarray bool   (T,)        — silence mask
         "ac_matrix" : np.ndarray float  (T, n_lags) — autocorrelation map
         "peak_ac"   : np.ndarray float  (T,)        — per-sample peak AC score
         "rms"       : np.ndarray float  (T,)        — per-sample rolling RMS
         "ac_lag_ms" : np.ndarray float  (n_lags,)   — lag axis in ms

All output arrays are aligned to the input sample timeline: index i corresponds
to time i/fs seconds.  Regions not covered by any AC window (the first and last
~win_samp/2 samples at each end) are filled with NaN for continuous outputs and
False for boolean masks.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helper: fast vectorised rolling RMS via stride tricks
# ---------------------------------------------------------------------------

def _rolling_rms(x, win_samp, step_samp):
    """
    Compute rolling RMS of *x* using a strided view — no Python loop.

    Returns
    -------
    rms    : (n_windows,) float
    starts : (n_windows,) int   — start index of each window in x
    """
    starts  = np.arange(0, len(x) - win_samp + 1, step_samp)
    # Build a (n_windows, win_samp) view without copying data
    shape   = (len(starts), win_samp)
    strides = (x.strides[0] * step_samp, x.strides[0])
    frames  = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    rms     = np.sqrt(np.mean(frames ** 2, axis=1))
    return rms, starts


# ---------------------------------------------------------------------------
# Internal helper: find active sample ranges from a per-window above-threshold mask
# ---------------------------------------------------------------------------

def _active_chunks(above_mask, win_starts, win_samp, pad_samp, signal_len):
    """
    Convert a per-window above-silence mask into merged, padded sample intervals.

    Each window that is above threshold contributes the half-open interval
    [win_start, win_start + win_samp).  These intervals are unioned, then padded
    on both sides by *pad_samp* samples, clipped to [0, signal_len), and finally
    adjacent/overlapping intervals are merged.

    Parameters
    ----------
    above_mask  : (n_windows,) bool
    win_starts  : (n_windows,) int
    win_samp    : int
    pad_samp    : int   — padding added to both sides of each active region
    signal_len  : int   — total signal length T

    Returns
    -------
    chunks : list of (start, end) int tuples — half-open sample intervals,
             sorted and non-overlapping.  Empty list if no active windows.
    """
    if not np.any(above_mask):
        return []

    # Build sorted list of active windows
    active_starts = win_starts[above_mask]
    active_ends   = active_starts + win_samp

    # Pad
    active_starts = np.maximum(0,           active_starts - pad_samp)
    active_ends   = np.minimum(signal_len,  active_ends   + pad_samp)

    # Sort by start, then merge overlapping / adjacent intervals
    order  = np.argsort(active_starts)
    starts = active_starts[order]
    ends   = active_ends[order]

    merged = []
    cs, ce = int(starts[0]), int(ends[0])
    for s, e in zip(starts[1:], ends[1:]):
        s, e = int(s), int(e)
        if s <= ce:         # overlap or adjacent → extend
            ce = max(ce, e)
        else:               # gap → commit current chunk, start new one
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


# ---------------------------------------------------------------------------
# Internal helper: sliding autocorrelation map
# ---------------------------------------------------------------------------

def _sliding_autocorr_map(x, fs, win_ms, step_ms, lag_lo_hz, lag_hi_hz):
    """
    Compute a time-lag autocorrelation map of signal *x*.

    For each overlapping window of length *win_ms*, the normalised
    autocorrelation is computed at every integer-sample lag that corresponds to
    oscillation frequencies between *lag_hi_hz* and *lag_lo_hz*
    (lag = fs / freq, so a higher frequency → shorter lag).

    The normalised autocorrelation at lag k is:
        AC(k) = [ Σ_t  x[t] · x[t+k] ]  /  [ Σ_t  x[t]² ]
    where x is mean-subtracted within each window.  This gives AC ∈ [-1, 1]
    with AC(0) = 1 by definition.

    The window length is automatically clamped:
      - minimum: 3× the longest lag (= 3 full cycles at the lowest target freq),
                 ensuring at least 3 oscillation cycles are visible per window.
      - maximum: half the signal length, ensuring at least 2 windows exist.

    Parameters
    ----------
    x          : (T,) float — input signal (may be a chunk, not the full signal)
    fs         : float      — sampling rate (Hz)
    win_ms     : float      — requested window length (ms); may be extended by clamping
    step_ms    : float      — step between successive window centres (ms)
    lag_lo_hz  : float      — upper frequency bound → shortest lag  (pass HIGHCUT)
    lag_hi_hz  : float      — lower frequency bound → longest  lag  (pass LOWCUT)

    Returns
    -------
    ac_matrix      : (n_lags, n_windows) float — autocorrelation values
    lag_ms         : (n_lags,) float           — lag axis in ms
    win_centers_ms : (n_windows,) float        — window centre times in ms
                                                  relative to x[0]
    win_starts     : (n_windows,) int          — window start indices into x
    win_samp       : int                       — final (clamped) window length in samples
    """
    win_samp  = int(win_ms  / 1000.0 * fs)
    step_samp = max(1, int(step_ms / 1000.0 * fs))

    # Convert frequency bounds to sample lags.
    lag_lo_samp = max(1, int(fs / lag_lo_hz))
    lag_hi_samp = int(fs / lag_hi_hz) + 1
    lags   = np.arange(lag_lo_samp, lag_hi_samp + 1)
    lag_ms = lags / fs * 1000.0

    # Clamp window length
    win_samp = max(win_samp, lag_hi_samp * 3)
    win_samp = min(win_samp, len(x) // 2)

    starts = np.arange(0, len(x) - win_samp, step_samp)
    if len(starts) == 0:
        dummy = np.full((len(lags), 1), np.nan)
        return dummy, lag_ms, np.array([len(x) / 2 / fs * 1000.0]), np.array([0]), win_samp

    ac_matrix = np.full((len(lags), len(starts)), np.nan)
    for j, s in enumerate(starts):
        win  = x[s : s + win_samp]
        win  = win - win.mean()
        norm = np.dot(win, win)
        if norm == 0:
            continue
        for li, lag in enumerate(lags):
            if lag >= win_samp:
                continue
            ac_matrix[li, j] = np.dot(win[:-lag], win[lag:]) / norm

    win_centers_ms = (starts + win_samp // 2) / fs * 1000.0
    return ac_matrix, lag_ms, win_centers_ms, starts, win_samp


# ---------------------------------------------------------------------------
# Internal helper: gap merge on a boolean window-timeline mask
# ---------------------------------------------------------------------------

def _merge_atrain_gaps(win_mask, max_gap_steps):
    """
    Bridge gaps between A-train candidate windows when the gap is short enough.

    Scans *win_mask* for runs of False ("gaps") sandwiched between runs of
    True ("A-train candidates").  Any gap whose length in windows is ≤
    *max_gap_steps* is filled with True, fusing the flanking candidates into
    a single longer A-train.  Gaps that are too long are left unchanged.

    Parameters
    ----------
    win_mask      : (n_windows,) bool — window-level A-train candidate mask
                                        before gap merging
    max_gap_steps : int               — maximum gap length to bridge (in windows)

    Returns
    -------
    merged : (n_windows,) bool — mask after gap bridging
    """
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
    """
    Remove A-train runs from *atrain_mask* that are shorter than *min_dur_s*.

    Operates on the *sample-level* mask so that duration is measured in real
    time rather than in windows.  Contiguous True runs are identified via edge
    detection; those that span less than *min_dur_s* seconds are zeroed out.

    Parameters
    ----------
    atrain_mask : (T,) bool  — sample-level A-train mask before duration filter
    times_s     : (T,) float — sample timestamps in seconds (times_s[i] = i/fs)
    min_dur_s   : float      — minimum A-train duration in seconds

    Returns
    -------
    out : (T,) bool — mask with short A-trains removed
    """
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
# Public API
# ---------------------------------------------------------------------------

def detect_atrains(
    signal_uv: np.ndarray,
    fs: float,
    *,
    lowcut:                   float = 100.0,
    highcut:                  float = 200.0,
    ac_window_ms:             float = 40.0,
    ac_step_ms:               float = 0.5,
    ac_core_threshold:        float = 0.70,
    ac_adjacent_threshold:    float = 0.50,
    ac_merge_gap_ms:          float = 10.0,
    ac_min_dur_ms:            float = 30.0,
    rms_silence_threshold_uv: float = 2.0,
) -> dict:
    """
    Detect A-trains (sustained HFO bursts) in a 1-D bandpassed signal.

    All intermediate quantities (AC map, RMS, silence mask) are returned
    alongside the final binary A-train mask so that the caller can render
    diagnostic visualisations without recomputing anything.

    Parameters
    ----------
    signal_uv : (T,) array
        Input signal in µV.  Should already be bandpassed to the target
        frequency range (lowcut-highcut Hz) before calling this function.
    fs : float
        Sampling rate in Hz.
    lowcut : float
        Lower edge of the target oscillation band (Hz).
    highcut : float
        Upper edge of the target oscillation band (Hz).
    ac_window_ms : float
        Duration of each sliding autocorrelation window (ms).
    ac_step_ms : float
        Step between successive window centres (ms).
    ac_core_threshold : float  in [0, 1]
        Peak-AC value required to *seed* an A-train candidate.
    ac_adjacent_threshold : float  in [0, 1]
        Peak-AC value required to *extend* an existing A-train candidate.
    ac_merge_gap_ms : float
        Maximum gap duration (ms) between candidates to bridge.
    ac_min_dur_ms : float
        Minimum duration (ms) for a surviving A-train.
    rms_silence_threshold_uv : float
        Windows below this RMS (µV) are excluded from all detection stages.

    Returns
    -------
    dict with keys:

    "atrains" : (T,) bool
        Final A-train mask.

    "silence" : (T,) bool
        Silence mask (True = silent).

    "ac_matrix" : (T, n_lags) float
        Full autocorrelation map on the sample timeline.  NaN where skipped.

    "peak_ac" : (T,) float
        Maximum autocorrelation across target lags per sample.  NaN where skipped.

    "rms" : (T,) float
        Rolling RMS per sample.  NaN where skipped.

    "ac_lag_ms" : (n_lags,) float
        Lag axis in milliseconds.
    """
    signal_uv = np.asarray(signal_uv, dtype=float)
    T         = len(signal_uv)
    times_s   = np.arange(T) / fs

    step_samp     = max(1, int(ac_step_ms / 1000.0 * fs))
    step_s        = ac_step_ms / 1000.0

    # ------------------------------------------------------------------
    # Pre-compute lag bounds so we know the padding size before any AC work.
    #
    # lag_hi_samp is the longest lag (= fs / lowcut), which sets both:
    #   • the lag range evaluated by _sliding_autocorr_map
    #   • the minimum window length (win_samp ≥ lag_hi_samp * 3)
    # We use the same clamped window estimate for padding so that windows
    # centred near a chunk boundary always have full data available.
    # ------------------------------------------------------------------
    lag_lo_samp   = max(1, int(fs / highcut))
    lag_hi_samp   = int(fs / lowcut) + 1
    lags          = np.arange(lag_lo_samp, lag_hi_samp + 1)
    n_lags        = len(lags)
    ac_lag_ms     = lags / fs * 1000.0

    # Conservative estimate of the clamped window length (mirrors logic in
    # _sliding_autocorr_map) — used only to size the padding.
    win_samp_req  = int(ac_window_ms / 1000.0 * fs)
    win_samp_est  = max(win_samp_req, lag_hi_samp * 3)
    # pad = half a window so AC windows near the chunk edge are fully inside
    pad_samp      = win_samp_est // 2 + 1

    # ------------------------------------------------------------------
    # Stage 1: Fast vectorised rolling RMS over the whole signal.
    # This is cheap (no Python loop) and determines which regions need AC.
    # ------------------------------------------------------------------
    rms_wins_full, win_starts_full = _rolling_rms(signal_uv, win_samp_est, step_samp)
    n_wins_full                    = len(rms_wins_full)

    above_silence_wins_full = rms_wins_full >= rms_silence_threshold_uv

    # ------------------------------------------------------------------
    # Stage 2: Find active chunks — contiguous groups of above-threshold
    # windows, padded and merged.
    # ------------------------------------------------------------------
    chunks = _active_chunks(
        above_silence_wins_full, win_starts_full, win_samp_est,
        pad_samp, T
    )

    # ------------------------------------------------------------------
    # Stage 3: Compute silence mask from the full-signal RMS pass.
    # This mirrors the original logic exactly:
    #   silence_samples = ~_wins_to_samples_bool(above_silence_wins_full)
    # Every sample whose covering window was below threshold → True (silent).
    # Samples at the signal edges not covered by any window → False.
    # ------------------------------------------------------------------
    def _wins_to_samples_bool(win_mask, win_starts, win_samp, step_s_inner):
        out = np.zeros(T, dtype=bool)
        for wi, v in enumerate(win_mask):
            if not v:
                continue
            tc  = (win_starts[wi] + win_samp // 2) / fs
            idx = np.where(
                (times_s >= tc - step_s_inner / 2) &
                (times_s <  tc + step_s_inner / 2)
            )[0]
            out[idx] = True
        return out

    silence_samples = ~_wins_to_samples_bool(
        above_silence_wins_full, win_starts_full, win_samp_est, step_s
    )

    # ------------------------------------------------------------------
    # Allocate remaining full-signal output arrays (NaN / False by default).
    # AC, peak_ac, and rms are only filled inside processed chunks.
    # ------------------------------------------------------------------
    ac_matrix_samples = np.full((T, n_lags), np.nan, dtype=float)
    peak_ac_samples   = np.full(T, np.nan,   dtype=float)
    rms_samples       = np.full(T, np.nan,   dtype=float)
    atrain_raw        = np.zeros(T,          dtype=bool)

    # ------------------------------------------------------------------
    # Stage 4: Process each active chunk independently.
    # ------------------------------------------------------------------
    for (chunk_start, chunk_end) in chunks:
        chunk     = signal_uv[chunk_start:chunk_end]
        chunk_len = len(chunk)

        # --- AC map on this chunk ---
        ac_mat_wins, _, win_centers_ms_local, win_starts_local, win_samp = \
            _sliding_autocorr_map(
                chunk, fs,
                win_ms    = ac_window_ms,
                step_ms   = ac_step_ms,
                lag_lo_hz = highcut,
                lag_hi_hz = lowcut,
            )

        n_wins_chunk  = ac_mat_wins.shape[1]
        win_centers_s_local = win_centers_ms_local / 1000.0
        # Convert chunk-local sample indices to full-signal sample indices
        win_starts_global   = win_starts_local + chunk_start

        # --- Peak AC and RMS per window ---
        peak_ac_wins_chunk = np.nanmax(ac_mat_wins, axis=0)   # (n_wins_chunk,)
        rms_wins_chunk     = np.array([
            np.sqrt(np.mean(chunk[s : s + win_samp] ** 2))
            for s in win_starts_local
        ])

        above_silence_chunk = rms_wins_chunk >= rms_silence_threshold_uv

        # --- Silence and A-train detection on this chunk's windows ---
        core_wins_chunk = (peak_ac_wins_chunk >= ac_core_threshold) & above_silence_chunk
        adj_eligible    = (peak_ac_wins_chunk >= ac_adjacent_threshold) & above_silence_chunk

        atrain_wins_chunk = core_wins_chunk.copy()
        while True:
            expanded        = atrain_wins_chunk.copy()
            expanded[1:]   |= atrain_wins_chunk[:-1]
            expanded[:-1]  |= atrain_wins_chunk[1:]
            expanded       &= adj_eligible
            if np.array_equal(expanded, atrain_wins_chunk):
                break
            atrain_wins_chunk = expanded

        merge_gap_steps   = int(np.ceil(ac_merge_gap_ms / ac_step_ms))
        atrain_wins_chunk = _merge_atrain_gaps(atrain_wins_chunk, merge_gap_steps)

        # --- Project window-level results back to full-signal sample timeline ---
        # Only write into the portion of the full arrays covered by this chunk.
        chunk_times_s = times_s[chunk_start:chunk_end]

        for wi in range(n_wins_chunk):
            tc  = win_starts_s = chunk_start / fs + win_centers_s_local[wi]
            # Samples in the full signal that this window covers
            idx = np.where(
                (times_s >= tc - step_s / 2) & (times_s < tc + step_s / 2)
            )[0]
            if len(idx) == 0:
                continue

            ac_matrix_samples[idx, :] = ac_mat_wins[:, wi]
            peak_ac_samples[idx]      = peak_ac_wins_chunk[wi]
            rms_samples[idx]          = rms_wins_chunk[wi]

            if atrain_wins_chunk[wi]:
                atrain_raw[idx] = True

    # ------------------------------------------------------------------
    # Stage 5: Duration filter on the full-signal sample mask.
    # ------------------------------------------------------------------
    atrain_samples = _discard_short_atrains(
        atrain_raw, times_s, ac_min_dur_ms / 1000.0
    )

    return {
        "atrains":   atrain_samples,    # (T,) bool
        "silence":   silence_samples,   # (T,) bool
        "ac_matrix": ac_matrix_samples, # (T, n_lags) — NaN where skipped
        "peak_ac":   peak_ac_samples,   # (T,) float  — NaN where skipped
        "rms":       rms_samples,       # (T,) float  — NaN where skipped
        "ac_lag_ms": ac_lag_ms,         # (n_lags,)
    }