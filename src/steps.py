"""Stage 3 -- footstrike detection and cadence from vertical acceleration.

Input is the frame-resolved vertical acceleration produced by stage 2.
Output is step timestamps and a cadence time series in steps per minute.

What this module detects, precisely
-----------------------------------
A **step-rate phase marker**, not a true initial contact. The detection
band deliberately stops below the second harmonic (see `LOWPASS_STEP_FREQ_
MULTIPLE`), so what is found is the crest of the near-fundamental vertical
oscillation. That crest sits at a fixed, unknown offset from real
footstrike. This is the right trade for *cadence*, where only the rate and
the phase consistency matter, and the wrong trade for anything needing true
contact timing. At 50 Hz on a pocket-worn phone, true initial contact is
not recoverable anyway: the impact transient that marks it lives above the
usable band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from . import dsp

# --- steady-state segmentation -------------------------------------------

# The MotionSense protocol has the participant start the app, pocket the
# phone, run, then stop and retrieve it. Both ends of every trial therefore
# contain handling that is not running, and it must be cut before anything
# is measured.
#
# 1.0 s window: spans ~3 steps at running cadence -- long enough to average
# over a stride, short enough to localise the start/stop transition to
# about a second.
STEADY_WINDOW_S = 1.0
# A window counts as running if its acceleration RMS is at least this
# fraction of the trial median. Measured on this dataset, running windows
# stay within roughly +/-25% of the median while pocketing/retrieval
# windows fall to 0.3x or below, so 0.5 sits in the empty gap between the
# two populations rather than inside either.
STEADY_RMS_FRACTION = 0.5
# ...and, whatever the median, at least this absolute RMS in g. The
# median-relative rule assumes running is the majority of the record. When
# it is not -- ten minutes standing at the start line, five minutes
# running -- the median IS the standing noise, the threshold is half of
# that, every window passes, and the band-pass then turns sensor noise
# into ~1700 fabricated "steps" at exactly the step frequency, which the
# two cadence estimators agree on because they share the band. Measured on
# MotionSense (1 s windows of |userAcceleration|): standing and sitting
# never exceed 0.13 g even at the per-trial median (95th-percentile window
# 0.11 g), while every locomotion class sits above it -- stairs 0.23 g,
# walking 0.41 g, jogging 0.67 g at the 1st percentile of windows. 0.2 g
# is above the highest non-locomotion window and below the lowest
# locomotion window. It is a floor against records with no locomotion in
# them, not a walking/running separator: walking clears it easily.
STEADY_RMS_FLOOR_G = 0.2
# Trials shorter than this after trimming are not worth reporting a cadence
# for: 5 s is ~14 steps, the minimum for a stable interval statistic.
MIN_STEADY_SECONDS = 5.0
# A dip below the RMS threshold shorter than this is bridged, not treated
# as a break between bouts. Measured on MotionSense, 14 of 48 continuous
# jog trials contained a single sub-threshold second -- a turn, a soft
# step -- and a one-window break rule cut each of them to roughly half its
# length without a word. 3 s is longer than any single-step anomaly
# (~0.35 s) or a turn (1-2 s) and far shorter than any real stop, which
# lasts tens of seconds.
MIN_BREAK_SECONDS = 3.0

# --- detection band ------------------------------------------------------

# Both cutoffs are multiples of the *estimated step frequency*, not fixed Hz.
# That keeps the filter identical in relative terms at any cadence and any
# sample rate -- the requirement that nothing hardcode 50 Hz.
#
# High-pass at 0.7 x f_step: sits between the stride subharmonic (0.5
# f_step) and the fundamental. A 4th-order Butterworth puts the subharmonic
# down ~12 dB while costing the fundamental 0.25 dB. That subharmonic is
# the step-to-step asymmetry term; leaving it in makes alternate peaks
# systematically shorter and invites missed steps. It also kills any DC
# offset left by imperfect gravity removal.
HIGHPASS_STEP_FREQ_MULTIPLE = 0.7
# Low-pass at 1.5 x f_step: keeps the fundamental, excludes the second
# harmonic. Measured on this dataset the second harmonic carries 11.6% of
# fundamental power at the median and up to 67%, and passing it puts a
# genuine *second* crest inside every step cycle -- ensemble-averaging the
# step cycle shows 2.0 peaks per cycle at a 3 x f_step cutoff versus 1.0 at
# 1.5 x. Sweeping the cutoff against an independent spectral cadence
# estimate, 1.5 x gave 0.58% median error and was the only setting
# insensitive to the peak-spacing constraint (x1.0 with the rule disabled,
# vs x2.4 at 2.0 x and x30 at 3.0 x), i.e. it rejects the harmonic on
# filter shape alone rather than leaning on the minimum-distance rule to
# hide it.
LOWPASS_STEP_FREQ_MULTIPLE = 1.5
# Never let the low cutoff approach Nyquist. 0.4 x fs leaves the Butterworth
# transition band comfortably inside the spectrum; matters at low rates or
# very high cadences, never at 50 Hz / 3 Hz.
MAX_CUTOFF_NYQUIST_FRACTION = 0.4
FILTER_ORDER = 4

# --- peak picking --------------------------------------------------------

# Minimum spacing between footstrikes, as a fraction of the estimated step
# period. 0.6 permits an instantaneous cadence up to 1/0.6 = 1.67x the
# trial mean -- far more variation than a runner shows -- while blocking
# double-hits on a single crest.
MIN_PEAK_DISTANCE_STEP_PERIODS = 0.6
# Peak prominence as a multiple of the robust (MAD-based) sigma of the
# filtered trace. 0.5 sigma accepts the weaker step of an asymmetric pair
# -- essential here, where the two steps of a stride differ substantially
# -- while rejecting ripple. Raising it to 1.5 multiplied the disagreement
# with the spectral cadence estimate by 5 by starting to drop alternate steps.
PEAK_PROMINENCE_SIGMA = 0.5
# Below this ratio of robust sigma to the segment's own peak amplitude,
# the trace carries no variation and there is nothing to detect. A constant
# input makes robust sigma exactly 0, which would otherwise set the
# prominence threshold to 0 -- maximally permissive at the precise moment
# the signal is empty -- and `find_peaks` would return every local maximum
# in filtfilt's numerical noise (~47 "steps" from `np.ones(1000)`).
# 1e-9 sits many orders above double-precision filter noise (~1e-16 of the
# input scale) and many orders below any real signal, where the ratio is
# order 0.1-0.5.
DEGENERATE_SIGMA_FRACTION = 1e-9

# --- cadence validation --------------------------------------------------

# The band the spec asks trials to be judged against.
EXPECTED_CADENCE_SPM = (150.0, 190.0)
# Peak-counted rate vs the independent spectral rate. Beyond 10% the two
# disagree by more than estimator noise and the detector is at fault.
DETECTOR_SPECTRAL_TOLERANCE = 0.10
# Within 15% of exactly half or double the spectral rate = a classic
# harmonic (halving/doubling) error rather than generic miscounting. The
# doubling branch is reachable in principle only: the 1.5 x f_step low-pass
# leaves no second crest to count, and the 0.6-period minimum spacing caps
# the ratio at 1.67 anyway. A doubled cadence in practice comes from the
# spectral seed landing on a harmonic, which `estimate_step_frequency`
# reports as `harmonic_ambiguous` and `diagnose_cadence` checks first.
HARMONIC_ERROR_TOLERANCE = 0.15
# Below this stride regularity the trace is not steady running, so no
# cadence claim -- of any kind -- is defensible.
MIN_STRIDE_REGULARITY = 0.30
# A stride interval is "regular" when it lies within this band of the
# median stride. Stride-to-stride variability in steady running is 2-4%
# (coefficient of variation), so +/-30% is far outside gait variability
# and well inside the failure modes it exists to exclude: a missed step
# makes a stride of 1.5x, a double detection one of 0.5x, and a pause
# inside a bridged bout one of 2x or more. Strides -- pairs of consecutive
# step intervals -- rather than steps, because the detector's crest shifts
# by a different amount on the two steps of a stride (the left/right
# waveform asymmetry at a pocket), so step intervals alternate short/long
# by up to 24% while their sums do not.
REGULAR_STRIDE_BAND = (0.7, 1.3)
# Above this spread of the smoothed cadence series inside one bout -- the
# 90th minus the 10th percentile, as a fraction of the median -- the bout
# holds more than one gait and no single cadence describes it. Steady
# running varies a few percent over a run (5.6% median, 11.6% max on the
# 48 MotionSense trials), a hard/easy interval session about 12%, and a
# synthetic walk merged into a running bout is a 30-50% jump. 0.2 sits
# between.
#
# Known limit, measured on real data: a real MotionSense walk spliced
# ahead of the same subject's jog is caught on 4 of 8 subjects. On the
# other 4 the detector -- band-passed around the RUNNING fundamental --
# counts walking's stride harmonics at close to the running rate, so the
# spread reads 0.11-0.19: the cadence comes out within a few percent of
# the run's, but the walking seconds are counted as running steps. Two
# detector-independent rules were tried and fail on the same data for the
# same reason: walking cadence is typically two thirds of running cadence,
# so walking's 2nd and 3rd stride harmonics land on 1.5x and 1.0x the
# running fundamental, which defeats any spectral-peak or band-power rule;
# and within-bout amplitude bimodality (window RMS p10/p90) overlaps
# between steady runs (min 0.47) and splices (0.31-0.57). At the trunk,
# walking is roughly 0.3x running RMS and is excluded by the running-state
# rule instead; at a pocket it is ~0.5x, right at that rule's threshold.
MIXED_CADENCE_SPREAD = 0.20
# Cadence above which the sample rate, not the runner, is the prime
# suspect. Distance runners top out near 200 spm; only sprinting exceeds
# 210. The spectral rate check is blind to an overstated rate until the
# true fundamental leaves the search band (about 1.6x at running cadence),
# and 1.25x at 168 spm lands exactly here, so this is where an asserted
# rate stops being distinguishable from a fast runner. Below the ceiling
# no signal-only check can tell them apart, which is why logger sessions
# measure their rate from timestamps instead of asserting it.
PLAUSIBLE_CADENCE_MAX_SPM = 210.0
# The spectral peak is called ambiguous when the band at half its
# frequency holds at least this fraction of its power, because a genuine
# fundamental whose second harmonic was picked instead shows its own line
# at half the picked frequency with comparable power. Measured on
# MotionSense the stride subharmonic of a real step-periodic signal holds
# 0.03 of the fundamental's power at the median and 0.11 at most, so 0.5
# never fires on a correctly picked peak, while a second harmonic only
# slightly stronger than its fundamental gives a ratio near 1. A second
# harmonic more than twice as strong as its fundamental is NOT caught;
# no such signal exists in the data seen so far.
SUBHARMONIC_AMBIGUITY_RATIO = 0.5
# Sample-rate plausibility. If more spectral power sits OUTSIDE the
# plausible human step band than at the strongest peak inside it, the real
# gait fundamental is not where `fs_hz` says it is, and the rate is the
# prime suspect. Measured on this dataset: across all 48 trials at the
# correct rate the out-of-band / in-band peak-power ratio never exceeds
# 0.44 (the second harmonic is always weaker than the fundamental), while
# claiming 100 Hz for 50 Hz data gives 15.1 and claiming 200 Hz gives 1002.
# 1.0 sits in that empty gap and is also the natural physical boundary:
# more power outside the human band than inside it.
SAMPLE_RATE_OUT_OF_BAND_RATIO = 1.0


def steady_state_segment(
    accel_magnitude: np.ndarray,
    fs_hz: float,
    window_s: float = STEADY_WINDOW_S,
    rms_fraction: float = STEADY_RMS_FRACTION,
    rms_floor_g: float = STEADY_RMS_FLOOR_G,
) -> dict:
    """Longest contiguous stretch of sustained motion, in samples.

    Uses acceleration *magnitude*, which is rotation-invariant (and
    polarity-invariant), so this runs before and independently of stage 2.

    Every return path carries the same keys. `segments` lists every bout
    found, `no_motion` is True when no window cleared the threshold -- in
    which case the whole record is returned so a caller can still inspect
    it, but nothing in it should be summarised.
    """
    x = np.asarray(accel_magnitude, float)
    n = len(x)
    w = int(max(1, round(window_s * fs_hz)))
    k = n // w
    if k < 3:
        # Too short to segment; take it whole and say so.
        return {
            "start": 0,
            "stop": n,
            "n_windows": k,
            "segmented": False,
            "trimmed_start_s": 0.0,
            "trimmed_end_s": 0.0,
            "kept_fraction": 1.0,
            "segments": [(0, n)],
            "n_segments": 1,
            "discarded_steady_s": 0.0,
            "threshold_g": float("nan"),
            "no_motion": False,
        }
    rms = np.sqrt((x[: k * w].reshape(k, w) ** 2).mean(axis=1))
    threshold = max(rms_fraction * float(np.median(rms)), rms_floor_g)
    active = rms >= threshold

    # Bridge inactive runs shorter than MIN_BREAK_SECONDS: they are dips
    # inside a bout, not breaks between bouts. Leading and trailing
    # inactive runs are never bridged -- those are the handling transients
    # this function exists to remove.
    min_break_windows = int(np.ceil(MIN_BREAK_SECONDS / window_s))
    i = 0
    while i < k:
        if not active[i]:
            j = i
            while j < k and not active[j]:
                j += 1
            if i > 0 and j < k and (j - i) < min_break_windows:
                active[i:j] = True
            i = j
        else:
            i += 1

    # Every contiguous run of active windows, as (start_sample, stop_sample).
    # The longest is returned as the segment to analyse -- unchanged
    # behaviour -- but the others are no longer thrown away in silence.
    segments: list[tuple[int, int]] = []
    i = 0
    while i < k:
        if active[i]:
            j = i
            while j < k and active[j]:
                j += 1
            segments.append((i * w, min(n, j * w)))
            i = j
        else:
            i += 1
    if not segments:
        return {
            "start": 0,
            "stop": n,
            "n_windows": k,
            "segmented": False,
            "trimmed_start_s": 0.0,
            "trimmed_end_s": 0.0,
            "kept_fraction": 1.0,
            "window_rms": rms,
            "segments": [],
            "n_segments": 0,
            "discarded_steady_s": 0.0,
            "threshold_g": float(threshold),
            "no_motion": True,
        }
    start, stop = max(segments, key=lambda s: s[1] - s[0])
    # `start`/`stop` is the longest bout, which the pipeline uses for the
    # frame and the exploratory stage. Steady motion in the *other* bouts is
    # reported here and analysed for cadence by `pipeline.run_stages`, so a
    # traffic-light stop no longer drops half the run from the summary.
    discarded = sum(b - a for a, b in segments) - (stop - start)
    return {
        "start": int(start),
        "stop": int(stop),
        "n_windows": int(k),
        "segmented": True,
        "trimmed_start_s": float(start / fs_hz),
        "trimmed_end_s": float((n - stop) / fs_hz),
        "kept_fraction": float((stop - start) / n),
        "window_rms": rms,
        "segments": [(int(a), int(b)) for a, b in segments],
        "n_segments": int(len(segments)),
        "discarded_steady_s": float(discarded / fs_hz),
        "threshold_g": float(threshold),
        "no_motion": False,
    }


def estimate_step_frequency(a_vert: np.ndarray, fs_hz: float) -> dict:
    """Detector-free estimate of step rate from the spectrum.

    Used both to seed the detector's band and to audit its count. Those
    two roles are not independent, and the audit's limits follow from
    that: because the detection band is built around this peak, a peak
    that lands on the wrong spectral line takes the detector with it, and
    the two then agree while both are wrong. The audit still catches what
    it was built for -- the detector missing alternate steps or splitting
    crests within the band it was given -- but a wrong seed has to be
    caught here, at the source, which `harmonic_ambiguous` does: it fires
    when the line at half the picked frequency holds a comparable share of
    the power (`SUBHARMONIC_AMBIGUITY_RATIO`), i.e. when the picked peak
    may be a second harmonic. That can only happen at cadences below 135
    spm, where the second harmonic is still inside the search band.
    """
    pk = dsp.spectral_peak(a_vert, fs_hz)
    f_peak = pk["f_peak_hz"]
    f, p = pk["freqs"], pk["psd"]
    peak_power = dsp.band_power(f, p, f_peak)
    sub_power = dsp.band_power(f, p, f_peak / 2.0)
    sub_ratio = float(sub_power / peak_power) if peak_power > 0 else np.nan
    band = dsp.STEP_FREQ_SEARCH_BAND_HZ
    ambiguous = bool(f_peak / 2.0 >= band[0] and sub_ratio >= SUBHARMONIC_AMBIGUITY_RATIO)
    return {
        "f_step_hz": f_peak,
        "cadence_spm": f_peak * 60.0,
        "spectral_bin_hz": pk["bin_hz"],
        "band_power_fraction": pk["band_power_fraction"],
        "subharmonic_power_ratio": sub_ratio,
        "harmonic_ambiguous": ambiguous,
        "alternative_f_step_hz": float(f_peak / 2.0) if ambiguous else np.nan,
    }


def check_sample_rate(
    a_vert: np.ndarray,
    fs_hz: float,
    band: tuple[float, float] = dsp.STEP_FREQ_SEARCH_BAND_HZ,
) -> dict:
    """Is `fs_hz` consistent with the data, or is the rate probably wrong?

    `fs_hz` is an input assertion: nothing in a bare array of samples states
    the rate it was captured at, and getting it wrong rescales every result
    linearly. The failure is quiet and dangerous, because the step-frequency
    search band clamps the estimate back into a plausible-looking range --
    50 Hz data presented as 200 Hz yields ~206 spm rather than an obviously
    absurd number, and the two cadence estimators then *agree with each
    other* because they share the same wrong rate.

    The test compares the strongest spectral peak inside the plausible human
    step band against the strongest peak outside it. Under a correct rate
    the fundamental dominates and the ratio is well below 1; under an
    overstated rate the real fundamental is pushed above the band and the
    ratio explodes.

    Known blind spots, stated rather than papered over:

    * **Understating the rate by about 2x is undetectable here**, because
      it moves a 2.7 Hz fundamental to 1.36 Hz, which is still inside the
      human band and is genuinely indistinguishable from a slow walker.
    * **Overstating it by less than about 1.6x is undetectable here too**:
      the fundamental only leaves the 4.5 Hz top of the band at 2.8 x 1.6
      Hz. Measured, 45 of 48 MotionSense trials pass this check when
      claimed at 75 Hz (1.5x). What catches that case is the cadence
      ceiling (`PLAUSIBLE_CADENCE_MAX_SPM`) in `diagnose_cadence`, and
      only once the apparent cadence passes 210 spm, i.e. above ~1.25x.
      Between 1x and 1.25x nothing signal-only can tell an overstated
      rate from a fast runner, which is why logger sessions measure the
      rate from their timestamps instead of asserting it.
    * A second harmonic *stronger* than the fundamental at a cadence above
      135 spm sits outside the band and would be reported as a wrong rate.
      On the pocket data the second harmonic never exceeds 0.67 of the
      fundamental; at the lower back this ratio is unmeasured.
    """
    pk = dsp.spectral_peak(a_vert, fs_hz, band)
    f, p = pk["freqs"], pk["psd"]
    in_band = (f >= band[0]) & (f <= band[1])
    out_band = (~in_band) & (f > 0)

    in_peak = float(p[in_band].max()) if in_band.any() else 0.0
    if out_band.any():
        j = int(np.argmax(p[out_band]))
        out_peak = float(p[out_band][j])
        out_hz = float(f[out_band][j])
    else:
        out_peak, out_hz = 0.0, float("nan")

    ratio = out_peak / in_peak if in_peak > 0 else float("inf")
    in_hz = float(pk["f_peak_hz"])
    return {
        "in_band_peak_hz": in_hz,
        "in_band_peak_power": in_peak,
        "out_of_band_peak_hz": out_hz,
        "out_of_band_peak_power": out_peak,
        "out_of_band_ratio": float(ratio),
        # Where the strongest out-of-band line sits relative to the in-band
        # peak. An integer or half-integer multiple is what a gait harmonic
        # gives -- and also what a 2x-overstated rate gives (the true
        # fundamental at exactly 2.0x the stride subharmonic), so this is
        # reported to make the diagnosis legible, not used to excuse it.
        "out_of_band_multiple": float(out_hz / in_hz) if in_hz > 0 and np.isfinite(out_hz) else float("nan"),
        "sample_rate_plausible": bool(ratio <= SAMPLE_RATE_OUT_OF_BAND_RATIO),
        "fs_hz": float(fs_hz),
    }


def bandpass_for_steps(
    a_vert: np.ndarray, fs_hz: float, f_step_hz: float
) -> tuple[np.ndarray, tuple[float, float]]:
    """Apply the step-detection band, with cutoffs tied to `f_step_hz`."""
    low = HIGHPASS_STEP_FREQ_MULTIPLE * f_step_hz
    high = LOWPASS_STEP_FREQ_MULTIPLE * f_step_hz
    nyq_cap = MAX_CUTOFF_NYQUIST_FRACTION * fs_hz
    if high >= nyq_cap:
        high = nyq_cap
    # Clamping `high` down to the Nyquist cap can push the step fundamental
    # itself out of the passband while still leaving low < high, which would
    # yield a filter that removes the very component being detected and
    # report no error. Require the fundamental to survive, not just the band
    # to be non-empty. This binds when fs < 2.5 * f_step -- never at 50 Hz
    # and running cadence, but it is a real cliff and it fails loudly.
    if low >= high or high <= f_step_hz:
        raise ValueError(
            f"cannot build a step band at f_step={f_step_hz:.2f} Hz with "
            f"fs={fs_hz:g} Hz: the usable band [{low:.2f}, {high:.2f}] Hz does "
            f"not contain the step fundamental. The sample rate is too low "
            f"for this cadence (need fs > {f_step_hz / MAX_CUTOFF_NYQUIST_FRACTION:.1f} Hz)."
        )
    return dsp.bandpass(a_vert, fs_hz, low, high, order=FILTER_ORDER), (low, high)


def detect_steps(
    a_vert: np.ndarray,
    fs_hz: float,
    f_step_hz: float | None = None,
    t0_s: float = 0.0,
    sample_times_s: np.ndarray | None = None,
) -> dict:
    """Detect step-rate peaks in frame-resolved vertical acceleration.

    `t0_s` offsets the returned timestamps so they refer to the original
    trial clock even when a trimmed segment was passed in. When the record
    carries real per-sample times (`sample_times_s`, same length as
    `a_vert`, on whatever origin the caller wants), step times are read
    from them instead of from `index / fs_hz + t0_s`: the filter still runs
    on the uniform grid, but a dropped sample then no longer shifts every
    later step time by one period.
    """
    a_vert = np.asarray(a_vert, float)
    if sample_times_s is not None:
        sample_times_s = np.asarray(sample_times_s, float)
        if len(sample_times_s) != len(a_vert):
            raise ValueError(
                f"sample_times_s has {len(sample_times_s)} entries for {len(a_vert)} samples"
            )
    if f_step_hz is None:
        f_step_hz = estimate_step_frequency(a_vert, fs_hz)["f_step_hz"]

    filtered, (low, high) = bandpass_for_steps(a_vert, fs_hz, f_step_hz)
    sigma = dsp.robust_sigma(filtered)
    distance = max(1, int(round(MIN_PEAK_DISTANCE_STEP_PERIODS / f_step_hz * fs_hz)))
    prominence = PEAK_PROMINENCE_SIGMA * sigma

    # A degenerate trace is "no signal", never "accept every peak".
    #
    # The reference scale is the *input* amplitude, not the filtered trace's.
    # Comparing the filtered trace against itself is circular: on a constant
    # input the band-pass output and its robust sigma are both ~1e-16, so
    # their ratio is order 1 and nothing looks wrong. Measured against the
    # input, a constant gives sigma/scale ~ 1e-16 while any real signal gives
    # order 0.1-0.5.
    #
    # Written as `not (sigma > threshold)` so a NaN sigma also lands here:
    # `nan > x` is False.
    scale = float(np.max(np.abs(a_vert))) if a_vert.size else 0.0
    degenerate = not (sigma > DEGENERATE_SIGMA_FRACTION * scale)
    if degenerate:
        idx = np.empty(0, dtype=int)
        props: dict = {}
    else:
        idx, props = signal.find_peaks(filtered, distance=distance, prominence=prominence)
    times = sample_times_s[idx] if sample_times_s is not None else idx / fs_hz + t0_s
    intervals = np.diff(times)

    return {
        "step_indices": idx,
        "step_times_s": times,
        "step_intervals_s": intervals,
        "filtered": filtered,
        "band_hz": (float(low), float(high)),
        "f_step_hz_used": float(f_step_hz),
        "prominence_threshold": float(prominence),
        "robust_sigma": float(sigma),
        "min_distance_samples": int(distance),
        "peak_prominences": props.get("prominences", np.array([])),
        "n_steps": int(len(idx)),
        # True when the trace carried no variation to detect steps in, as
        # distinct from a live trace in which no peak cleared the threshold.
        "degenerate_signal": bool(degenerate),
        "step_times_from_hardware_clock": sample_times_s is not None,
    }


def cadence_series(
    step_times_s: np.ndarray, smooth_steps: int = 6
) -> pd.DataFrame:
    """Instantaneous and smoothed cadence, in steps per minute.

    Instantaneous cadence is 60 / (step interval), timestamped at the
    midpoint of the interval it describes. The smoothed column is a rolling
    *median* over `smooth_steps` intervals -- median, not mean, so that one
    missed step (which doubles a single interval) cannot drag the whole
    window with it. It is not immune: the two central order statistics
    move by one rank at a missed or doubled step, so the trace still
    shows a bump of a few percent for `smooth_steps` intervals around it.

    `smooth_steps` must be **even**, and defaults to 6 (three strides).
    Step intervals alternate short-long with the runner's two legs, a
    period-2 sequence. An odd-length median window sits on one side of that
    alternation and returns a short or a long interval alternately, so the
    "smoothed" trace sawtooths at exactly the rate it is supposed to
    suppress. An even window averages the two central order statistics,
    which cancels a period-2 alternation exactly while keeping the median's
    resistance to a single bad interval.

    The smoothed column is NaN for the first and last `smooth_steps // 2`
    intervals. A partial window there is odd-length or one-sided, and it
    sawtoothed by the full alternation amplitude at both edges of every
    segment; an honest gap is better than a fabricated edge.
    """
    if smooth_steps % 2:
        raise ValueError(
            f"smooth_steps must be even to cancel left/right step-time "
            f"alternation, got {smooth_steps}"
        )
    t = np.asarray(step_times_s, float)
    if len(t) < 2:
        return pd.DataFrame(columns=["t_s", "cadence_spm", "cadence_spm_smooth"])
    intervals = np.diff(t)
    mid = t[:-1] + intervals / 2.0
    inst = 60.0 / intervals
    df = pd.DataFrame({"t_s": mid, "cadence_spm": inst})
    df["cadence_spm_smooth"] = (
        df["cadence_spm"].rolling(smooth_steps, center=True, min_periods=smooth_steps).median()
    )
    return df


def cadence_spread(series: pd.DataFrame) -> float:
    """Spread of the smoothed cadence inside a bout: (p90 - p10) / median.

    The statistic `MIXED_CADENCE_SPREAD` is judged against. NaN with fewer
    than ten smoothed values, which is too few to call anything a regime.
    """
    if "cadence_spm_smooth" not in series:
        return float("nan")
    sm = series["cadence_spm_smooth"].to_numpy(float)
    sm = sm[np.isfinite(sm)]
    if sm.size < 10:
        return float("nan")
    med = float(np.median(sm))
    if med <= 0:
        return float("nan")
    return float((np.percentile(sm, 90) - np.percentile(sm, 10)) / med)


def _empty_summary(n_steps: int, span_s: float, min_span_s: float) -> dict:
    return {
        "n_steps": int(n_steps),
        "n_strides": 0,
        "n_regular_strides": 0,
        "cadence_spm": np.nan,
        "cadence_spm_median": np.nan,
        "cadence_spm_span": np.nan,
        "cadence_cv": np.nan,
        "irregular_stride_fraction": np.nan,
        "alternating_interval_asymmetry_abs_pct": np.nan,
        "span_s": float(span_s),
        "span_too_short": bool(span_s < min_span_s),
    }


def _stride_statistics(strides: np.ndarray) -> dict:
    """Cadence statistics from a pool of stride intervals (seconds).

    The band test is against the pooled median, so a pool drawn from
    several bouts of the same run shares one reference.
    """
    strides = np.asarray(strides, float)
    med = float(np.median(strides))
    lo, hi = REGULAR_STRIDE_BAND
    regular = (strides >= lo * med) & (strides <= hi * med)
    reg = strides[regular]
    n_reg = int(reg.size)
    mean_stride = float(reg.mean()) if n_reg else np.nan
    return {
        "n_strides": int(strides.size),
        "n_regular_strides": n_reg,
        # Steps per minute from the mean REGULAR stride: two steps per
        # stride, so 120 / stride. A missed step, a double detection or a
        # pause removes its strides from both numerator and denominator,
        # which leaves the estimate unbiased by them -- the property the
        # old whole-span rate had for missed steps but not for pauses.
        "cadence_spm": float(120.0 / mean_stride) if n_reg else np.nan,
        "cadence_spm_median": float(120.0 / med) if med > 0 else np.nan,
        # Stride-interval coefficient of variation over regular strides.
        # This is the runner's cadence variability. The step-interval CV
        # is NOT: it is dominated by the detector's crest shift between
        # the two steps of a stride (median 0.11, max 0.41 on MotionSense,
        # where the true value is a few percent), so it is not reported as
        # a cadence statistic at all.
        "cadence_cv": float(reg.std() / mean_stride) if n_reg >= 2 else np.nan,
        "irregular_stride_fraction": float(1.0 - n_reg / strides.size) if strides.size else np.nan,
    }


def cadence_summary(
    step_times_s: np.ndarray, min_span_s: float = MIN_STEADY_SECONDS
) -> dict:
    """Whole-bout cadence statistics from detected step times.

    Everything here is built on **stride** intervals -- the time from one
    detected step to the one after next -- rather than step intervals. The
    detector marks the two steps of a stride at different phases of their
    crests (see `alternating_interval_asymmetry_abs_pct`), so consecutive
    step intervals alternate short/long by up to a quarter on real data. A
    median or an irregularity count taken over step intervals then flips
    with the PARITY of how many intervals there are: a perfectly regular
    0.30/0.42 s alternation reported a median cadence of 166.7 or 200.0
    spm and an irregular fraction of 0.00 or 0.49 depending on whether the
    last interval was a short or a long one. Stride intervals are the sum
    of a short and a long, and have neither problem.

    The short-record guard lives here, not in the pipeline, because this is
    the one function every entry point goes through to obtain a cadence.
    Enforcing it further up left `scripts/run_invariance_checks.py` and
    `tests/test_invariance.py` free to report a confident 162.79 spm from a
    3 s record. Below `min_span_s` the cadence is NaN, the same way it
    already is below three steps: not a number, so nothing downstream can
    quietly treat it as one.
    """
    t = np.asarray(step_times_s, float)
    span_now = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    if len(t) < 3 or span_now < min_span_s:
        return _empty_summary(len(t), span_now, min_span_s)
    intervals = np.diff(t)
    strides = t[2:] - t[:-2]  # overlapping: every step interval is in two strides
    out = _empty_summary(len(t), span_now, min_span_s)
    out.update(_stride_statistics(strides))
    # The whole-span rate a step counter would report, kept as a
    # diagnostic. It is biased low by any pause inside the bout (a bridged
    # 2.5 s standstill in an 80 s bout reads 163.6 for a true 168).
    out["cadence_spm_span"] = float(60.0 * (len(t) - 1) / span_now)

    # Alternating-interval asymmetry: the difference between the two
    # interleaved sets of step intervals, as a percentage of the mean
    # interval. Reported as an ABSOLUTE value.
    #
    # The sign is meaningless and is therefore discarded. Which interleaved
    # set is "even" depends entirely on which step happens to land at index
    # 0, which is set by where detection started -- so a signed value would
    # flip on a one-step change in the trim boundary while describing the
    # identical gait.
    #
    # READ THIS AS A DETECTOR DIAGNOSTIC, NOT AS THE RUNNER'S STEP-TIME
    # SYMMETRY. When the two steps of a stride produce differently shaped
    # vertical waveforms -- which they do at a pocket, strongly -- the crest
    # of the band-passed fundamental shifts by a different amount on
    # alternate steps, so the detected timestamps alternate short-long even
    # if the runner's true step times do not. On this dataset this quantity
    # correlates at r = -0.61 with the amplitude-domain step symmetry index,
    # which is what that artifact looks like. Values here reach 24% (median
    # 8%), far beyond the 1-3% real runners show.
    even, odd = intervals[0::2], intervals[1::2]
    n = min(len(even), len(odd))
    if n >= 2:
        out["alternating_interval_asymmetry_abs_pct"] = float(
            100.0 * abs(even[:n].mean() - odd[:n].mean()) / np.mean(intervals)
        )
    out["span_too_short"] = False
    return out


def cadence_summary_pooled(
    step_times_per_bout: list, min_span_s: float = MIN_STEADY_SECONDS
) -> dict:
    """Cadence statistics pooled over several bouts of one run.

    Strides are formed inside each bout and never across a bout boundary,
    then pooled, so a stop between bouts contributes no interval. Bouts
    whose own step span is shorter than `min_span_s` contribute nothing,
    the same rule `cadence_summary` applies to a single bout.
    """
    pools = []
    n_steps = 0
    span_total = 0.0
    n_bouts = 0
    for times in step_times_per_bout:
        t = np.asarray(times, float)
        span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        if len(t) < 3 or span < min_span_s:
            continue
        pools.append(t[2:] - t[:-2])
        n_steps += int(len(t))
        span_total += span
        n_bouts += 1
    if not pools:
        out = _empty_summary(n_steps, span_total, min_span_s)
        out["n_bouts_pooled"] = 0
        return out
    out = _empty_summary(n_steps, span_total, min_span_s)
    out.update(_stride_statistics(np.concatenate(pools)))
    out["span_too_short"] = False
    out["n_bouts_pooled"] = int(n_bouts)
    return out


def diagnose_cadence(
    detected_spm: float,
    spectral_spm: float,
    stride_regularity: float,
    irregular_stride_fraction: float,
    expected: tuple[float, float] = EXPECTED_CADENCE_SPM,
    sample_rate_plausible: bool = True,
    out_of_band_peak_hz: float | None = None,
    cadence_spread: float = np.nan,
    harmonic_ambiguous: bool = False,
    subharmonic_power_ratio: float = np.nan,
    rate_is_measured: bool = False,
    out_of_band_multiple: float = np.nan,
) -> dict:
    """Flag out-of-band cadence and attribute the cause.

    Four outcomes: "none", "sample_rate", "algorithm", "trial".

    The algorithm-vs-trial attribution compares two estimators: peak
    counting and the spectral peak. If they agree, the detector counted
    what the spectrum shows and an out-of-band number is a fact about the
    trial. If they disagree, the detector is at fault -- and the ratio
    between them says how. They are not independent of each other (the
    detector's band is built around the spectral peak; see
    `estimate_step_frequency`), so a wrong spectral peak is caught by
    `harmonic_ambiguous`, checked here before the ratio is read.

    Two preconditions are checked before any of that:

    * `sample_rate_plausible`: the two estimators share the sample rate.
      Given a wrong `fs_hz` they agree with each other while both being
      wrong together, and their agreement was previously read as evidence
      about the runner. Agreement under a shared wrong assumption is not
      evidence. The cadence ceiling `PLAUSIBLE_CADENCE_MAX_SPM` extends
      the same reasoning to the overstated rates the spectral check cannot
      see.
    * `cadence_spread`: a bout whose smoothed cadence spans more than
      `MIXED_CADENCE_SPREAD` holds more than one gait -- typically a walk
      merged into a run. The detector counts both correctly, the spectral
      peak reports the dominant one, and the pooled number is a mix of the
      two. That used to be attributed to the detector; it is a fact about
      the record, and no single cadence describes it.

    `rate_is_measured` changes what a failed spectral rate check and a
    cadence above the ceiling mean. On a logger session the rate comes from
    hardware timestamps, so neither can be a rate error: the first is an
    unusually strong out-of-band line (a harmonic stronger than the
    fundamental, or a non-gait signal) and is reported as a suspect, not a
    blocker; the second is an implausible cadence for distance running --
    sprinting, or the detector counting a harmonic -- and is refused as
    such. Both keep the cause "trial".
    """
    in_band = bool(expected[0] <= detected_spm <= expected[1])
    ratio = detected_spm / spectral_spm if spectral_spm > 0 else np.nan
    agree = bool(abs(ratio - 1.0) <= DETECTOR_SPECTRAL_TOLERANCE)
    spread_high = bool(np.isfinite(cadence_spread) and cadence_spread > MIXED_CADENCE_SPREAD)
    # Set only by the branch that reaches it, so the flag and the message
    # never disagree: a spread above the rule on a record already refused
    # for a wrong rate or no periodicity is that failure, not a second one.
    mixed = False

    harmonic_suspect = False
    implausible = False
    mult = (
        f" at {out_of_band_multiple:.2f}x the in-band peak" if np.isfinite(out_of_band_multiple) else ""
    )
    if not sample_rate_plausible and rate_is_measured:
        cause = "trial"
        harmonic_suspect = True
        detail = (
            f"strongest spectral line sits outside the plausible step band "
            f"({out_of_band_peak_hz:.2f} Hz{mult}). The rate is measured from timestamps, "
            f"so this is not a rate error: a harmonic stronger than the fundamental, or a "
            f"non-gait signal. Cadence reported; read it with that in mind."
        )
    elif not sample_rate_plausible:
        cause = "sample_rate"
        where = (
            f" (strongest power sits at {out_of_band_peak_hz:.2f} Hz{mult}, outside the "
            f"plausible step band)" if out_of_band_peak_hz is not None else ""
        )
        detail = (
            f"the sample rate is probably wrong{where}. Both cadence estimators "
            f"share fs_hz, so their agreement says nothing about the runner here. "
            f"Check the rate before reading this cadence."
        )
    elif not np.isfinite(detected_spm):
        cause = "trial"
        detail = (
            "no defensible cadence: fewer than three steps, or a step span "
            f"shorter than {MIN_STEADY_SECONDS:g}s"
        )
    elif stride_regularity < MIN_STRIDE_REGULARITY:
        cause = "trial"
        detail = (
            f"stride regularity {stride_regularity:.2f} < {MIN_STRIDE_REGULARITY}: "
            f"this segment is not steady running, so no cadence is defensible"
        )
    elif harmonic_ambiguous:
        cause = "algorithm"
        detail = (
            f"step frequency ambiguous: the spectral line at half the picked "
            f"frequency ({spectral_spm / 2:.0f} spm) holds {subharmonic_power_ratio:.2f} "
            f"of its power, so the picked peak may be a second harmonic and the "
            f"detector, seeded by it, may be counting a harmonic"
        )
    elif spread_high:
        cause = "trial"
        mixed = True
        detail = (
            f"mixed cadences within the bout: the smoothed cadence spans "
            f"{100 * cadence_spread:.0f}% of its median (p10 to p90); walking merged "
            f"with running? No single cadence describes this bout"
        )
    elif detected_spm > PLAUSIBLE_CADENCE_MAX_SPM and rate_is_measured:
        cause = "trial"
        implausible = True
        detail = (
            f"{detected_spm:.0f} spm is above the {PLAUSIBLE_CADENCE_MAX_SPM:.0f} spm "
            f"ceiling for distance running, with a rate measured from timestamps: "
            f"sprinting, the detector counting a harmonic, or a timestamp column that "
            f"is not in seconds"
        )
    elif detected_spm > PLAUSIBLE_CADENCE_MAX_SPM:
        cause = "sample_rate"
        detail = (
            f"{detected_spm:.0f} spm is above the {PLAUSIBLE_CADENCE_MAX_SPM:.0f} spm "
            f"ceiling for distance running; an overstated sample rate, which the "
            f"spectral check cannot see below ~1.6x, is the likelier cause"
        )
    elif abs(ratio - 0.5) <= HARMONIC_ERROR_TOLERANCE:
        cause = "algorithm"
        detail = (
            "detector counted about half the spectral rate: alternate steps missed "
            "(step-to-step asymmetry exceeding the prominence threshold)"
        )
    elif abs(ratio - 2.0) <= HARMONIC_ERROR_TOLERANCE:
        cause = "algorithm"
        detail = (
            "detector counted about twice the spectral rate: harmonic leaking "
            "through the low-pass, producing two crests per step"
        )
    elif not agree:
        cause = "algorithm"
        detail = (
            f"peak-counted rate disagrees with the spectral rate by "
            f"{100 * abs(ratio - 1):.0f}% (irregular strides: "
            f"{100 * irregular_stride_fraction:.0f}%)"
        )
    elif not in_band:
        cause = "trial"
        detail = (
            f"detector agrees with the spectral estimate "
            f"({spectral_spm:.0f} spm) to {100 * abs(ratio - 1):.0f}%; the runner's "
            f"cadence really is outside {expected[0]:.0f}-{expected[1]:.0f} spm"
        )
    else:
        cause = "none"
        detail = "in band and consistent with the spectral estimate"

    return {
        "cadence_in_expected_band": in_band,
        "detector_spectral_ratio": float(ratio),
        "detector_agrees_with_spectrum": agree,
        "sample_rate_plausible": bool(sample_rate_plausible),
        "cadence_spread": float(cadence_spread),
        "mixed_gait": mixed,
        "harmonic_ambiguous": bool(harmonic_ambiguous),
        "harmonic_suspect": harmonic_suspect,
        "implausible_cadence": implausible,
        "rate_is_measured": bool(rate_is_measured),
        # "none" | "sample_rate" | "algorithm" | "trial"
        "failure_attributed_to": cause,
        "diagnosis": detail,
        "flagged": bool(cause != "none"),
    }
