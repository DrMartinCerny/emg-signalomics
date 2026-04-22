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
    x          : (T,) float — input signal
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
    # lag_lo_hz (= HIGHCUT) → shorter lag (fewer samples)
    # lag_hi_hz (= LOWCUT)  → longer  lag (more  samples)
    lag_lo_samp = max(1, int(fs / lag_lo_hz))
    lag_hi_samp = int(fs / lag_hi_hz) + 1
    lags   = np.arange(lag_lo_samp, lag_hi_samp + 1)
    lag_ms = lags / fs * 1000.0

    # Clamp window length
    win_samp = max(win_samp, lag_hi_samp * 3)   # at least 3 cycles at lowest freq
    win_samp = min(win_samp, len(x) // 2)        # at most half the signal

    starts = np.arange(0, len(x) - win_samp, step_samp)
    if len(starts) == 0:
        # Signal too short for even one window — return a single NaN column
        dummy = np.full((len(lags), 1), np.nan)
        return dummy, lag_ms, np.array([len(x) / 2 / fs * 1000.0]), np.array([0]), win_samp

    ac_matrix = np.full((len(lags), len(starts)), np.nan)
    for j, s in enumerate(starts):
        win  = x[s : s + win_samp]
        win  = win - win.mean()           # remove DC offset within window
        norm = np.dot(win, win)           # Σ x² — zero norm means flat window
        if norm == 0:
            continue                      # leave this column as NaN
        for li, lag in enumerate(lags):
            if lag >= win_samp:
                continue
            # Dot product of win with its lagged copy, normalised by win energy
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
                    merged[gap_start:i] = True   # bridge this gap
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

    # Handle A-trains that begin or end at the array boundary
    if atrain_mask[0]:
        starts = [0] + starts
    if atrain_mask[-1]:
        ends = ends + [len(atrain_mask)]

    for s, e in zip(starts, ends):
        if (times_s[e - 1] - times_s[s]) < min_dur_s:
            out[s:e] = False   # too short — discard

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
        Defines the longest autocorrelation lag evaluated (lag = fs / lowcut).
    highcut : float
        Upper edge of the target oscillation band (Hz).
        Defines the shortest autocorrelation lag evaluated (lag = fs / highcut).
    ac_window_ms : float
        Duration of each sliding autocorrelation window (ms).
        Longer windows give more stable AC estimates but reduce temporal
        resolution.  The actual window used may be longer if 3-cycle clamping
        forces an extension.
    ac_step_ms : float
        Step between successive window centres (ms).  Controls temporal
        resolution of the A-train boundary estimates.  Smaller = finer, slower.
    ac_core_threshold : float  in [0, 1]
        Peak-AC value required to *seed* an A-train candidate.  Should be
        set high (e.g. 0.70) so that only clearly periodic windows act as
        seeds, keeping false-positive A-trains rare.
    ac_adjacent_threshold : float  in [0, 1]
        Peak-AC value required to *extend* an existing A-train candidate into
        a neighbouring window.  Must be ≤ ac_core_threshold.  The lower this
        is, the more the detected A-train extends into its flanks.
    ac_merge_gap_ms : float
        Maximum gap duration (ms) between two A-train candidates that will be
        bridged into a single A-train.
    ac_min_dur_ms : float
        Minimum duration (ms) an A-train must have to survive the final filter.
    rms_silence_threshold_uv : float
        Windows whose RMS amplitude falls below this value (µV) are excluded
        from all detection stages, including gap bridging.

    Returns
    -------
    dict with keys:

    "atrains" : (T,) bool
        Final A-train mask.  True at every sample that belongs to a confirmed
        A-train after all five pipeline stages.

    "silence" : (T,) bool
        Silence mask.  True at every sample whose AC window had
        RMS < rms_silence_threshold_uv.  Samples not covered by any window
        (signal edges) are marked False (not silent).

    "ac_matrix" : (T, n_lags) float
        Full autocorrelation map on the sample timeline.  Entry [i, j] is the
        autocorrelation at lag ac_lag_ms[j] ms for the window covering sample i.
        NaN at samples not covered by any window.

    "peak_ac" : (T,) float
        Maximum autocorrelation across all target lags for the window covering
        each sample.  This scalar is what the detection thresholds operate on.
        NaN at uncovered samples.

    "rms" : (T,) float
        Rolling RMS computed over the same windows as the AC map.
        NaN at uncovered samples.

    "ac_lag_ms" : (n_lags,) float
        Lag axis in milliseconds, corresponding to the columns of "ac_matrix".
        Spans from fs/highcut ms (shortest lag, highest freq) to fs/lowcut ms
        (longest lag, lowest freq).
    """
    signal_uv = np.asarray(signal_uv, dtype=float)
    T         = len(signal_uv)
    times_s   = np.arange(T) / fs   # sample timestamps in seconds

    # ------------------------------------------------------------------
    # Stage 1: Sliding autocorrelation
    # Produces ac_mat_wins (n_lags, n_windows) and scalar diagnostics per window.
    # ------------------------------------------------------------------
    ac_mat_wins, ac_lag_ms, win_centers_ms, win_starts, win_samp = \
        _sliding_autocorr_map(
            signal_uv, fs,
            win_ms    = ac_window_ms,
            step_ms   = ac_step_ms,
            lag_lo_hz = highcut,   # higher frequency → shorter lag
            lag_hi_hz = lowcut,    # lower  frequency → longer  lag
        )
    n_lags, n_wins = ac_mat_wins.shape
    win_centers_s  = win_centers_ms / 1000.0
    step_s         = ac_step_ms / 1000.0

    # Peak AC: the single most-correlated lag in each window.
    # This is the primary detection feature — one number per window.
    peak_ac_wins = np.nanmax(ac_mat_wins, axis=0)   # (n_windows,)

    # ------------------------------------------------------------------
    # Stage 2: Rolling RMS
    # Computed over the same windows as the AC map for consistent temporal
    # resolution.  Used by the silence gate and returned as a diagnostic.
    # ------------------------------------------------------------------
    rms_wins = np.array([
        np.sqrt(np.mean(signal_uv[s : s + win_samp] ** 2))
        for s in win_starts
    ])   # (n_windows,)

    # ------------------------------------------------------------------
    # Stage 3: Silence gate
    # Windows below the RMS threshold are excluded from all A-train logic.
    # above_silence_wins[i] = True  ↔  window i is active (not silent).
    # ------------------------------------------------------------------
    above_silence_wins = rms_wins >= rms_silence_threshold_uv   # (n_windows,) bool

    # ------------------------------------------------------------------
    # Stage 4a: Core A-train windows (seeds)
    # Must be both non-silent and strongly periodic.
    # ------------------------------------------------------------------
    core_wins = (peak_ac_wins >= ac_core_threshold) & above_silence_wins

    # ------------------------------------------------------------------
    # Stage 4b: Grow A-train cores (flood-fill)
    # A window is "adjacent eligible" if it is non-silent and meets the
    # weaker adjacent threshold.  The flood-fill propagates the current
    # A-train set into eligible neighbours until convergence.
    # ------------------------------------------------------------------
    adj_eligible_wins = (peak_ac_wins >= ac_adjacent_threshold) & above_silence_wins

    atrain_wins = core_wins.copy()
    while True:
        expanded        = atrain_wins.copy()
        expanded[1:]   |= atrain_wins[:-1]   # try to grow rightward (later in time)
        expanded[:-1]  |= atrain_wins[1:]    # try to grow leftward  (earlier in time)
        expanded       &= adj_eligible_wins   # only keep eligible positions
        if np.array_equal(expanded, atrain_wins):
            break   # stable — no further growth possible
        atrain_wins = expanded

    # ------------------------------------------------------------------
    # Stage 4c: Merge short gaps between A-train candidates
    # Convert the ms gap threshold to window steps, then bridge short gaps.
    # ------------------------------------------------------------------
    merge_gap_steps    = int(np.ceil(ac_merge_gap_ms / ac_step_ms))
    atrain_wins_merged = _merge_atrain_gaps(atrain_wins, merge_gap_steps)

    # ------------------------------------------------------------------
    # Stage 5: Project window-level results back to the sample timeline.
    #
    # Each window covers a half-open interval centred on win_centers_s[wi]:
    #   [center - step/2,  center + step/2)
    # Samples at the signal edges that fall outside all windows retain their
    # fill values (NaN for float, False for bool).
    # ------------------------------------------------------------------

    def _wins_to_samples_float(win_vals, fill=np.nan):
        """Project (n_windows,) float values onto the (T,) sample timeline."""
        out = np.full(T, fill, dtype=float)
        for wi, v in enumerate(win_vals):
            tc  = win_centers_s[wi]
            idx = np.where(
                (times_s >= tc - step_s / 2) & (times_s < tc + step_s / 2)
            )[0]
            out[idx] = v
        return out

    def _wins_to_samples_bool(win_mask):
        """Project (n_windows,) bool mask onto the (T,) sample timeline."""
        out = np.zeros(T, dtype=bool)
        for wi, v in enumerate(win_mask):
            if not v:
                continue
            tc  = win_centers_s[wi]
            idx = np.where(
                (times_s >= tc - step_s / 2) & (times_s < tc + step_s / 2)
            )[0]
            out[idx] = True
        return out

    # Continuous diagnostics — returned as-is for display
    peak_ac_samples = _wins_to_samples_float(peak_ac_wins, fill=np.nan)
    rms_samples     = _wins_to_samples_float(rms_wins,     fill=np.nan)

    # Silence: invert above_silence so True = "this sample is silent"
    silence_samples = ~_wins_to_samples_bool(above_silence_wins)

    # A-train mask before duration filter
    atrain_samples_raw = _wins_to_samples_bool(atrain_wins_merged)

    # Duration filter: discard A-trains shorter than ac_min_dur_ms.
    # Applied on the sample-level mask for sample-accurate timing.
    atrain_samples = _discard_short_atrains(
        atrain_samples_raw, times_s, ac_min_dur_ms / 1000.0
    )

    # ------------------------------------------------------------------
    # Build sample-aligned AC matrix: (T, n_lags)
    # Each row receives the AC vector of the window covering that sample.
    # Orientation: rows = time (samples), columns = lags (ascending ms).
    # ------------------------------------------------------------------
    ac_matrix_samples = np.full((T, n_lags), np.nan, dtype=float)
    for wi in range(n_wins):
        tc  = win_centers_s[wi]
        idx = np.where(
            (times_s >= tc - step_s / 2) & (times_s < tc + step_s / 2)
        )[0]
        ac_matrix_samples[idx, :] = ac_mat_wins[:, wi]   # broadcast column to rows

    return {
        "atrains":   atrain_samples,      # (T,) bool   — final A-train mask
        "silence":   silence_samples,     # (T,) bool   — silence mask
        "ac_matrix": ac_matrix_samples,   # (T, n_lags) — autocorrelation map
        "peak_ac":   peak_ac_samples,     # (T,) float  — peak AC per sample
        "rms":       rms_samples,         # (T,) float  — RMS per sample
        "ac_lag_ms": ac_lag_ms,           # (n_lags,)   — lag axis in ms
    }
