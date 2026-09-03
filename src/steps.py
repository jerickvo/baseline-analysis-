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
# harmonic. Measured on this dataset the second harmonic carries up to 21%
# of fundamental power, and passing it puts a genuine *second* crest inside
# every step cycle -- ensemble-averaging the step cycle shows 2.0 peaks per
# cycle at a 3 x f_step cutoff versus 1.0 at 1.5 x. Sweeping the cutoff
# against an independent spectral cadence estimate, 1.5 x gave 1.1% median
# error and was the only setting insensitive to the peak-spacing
# constraint, i.e. it rejects the harmonic on filter shape alone rather
# than leaning on the minimum-distance rule to hide it.
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
# harmonic (halving/doubling) error rather than generic miscounting.
HARMONIC_ERROR_TOLERANCE = 0.15
# Below this stride regularity the trace is not steady running, so no
# cadence claim -- of any kind -- is defensible.
MIN_STRIDE_REGULARITY = 0.30
# Sample-rate plausibility. If more spectral power sits OUTSIDE the
# plausible human step band than at the strongest peak inside it, the real
# gait fundamental is not where `fs_hz` says it is, and the rate is the
# prime suspect. Measured on this dataset: across all 48 trials at the
# correct rate the out-of-band / in-band peak-power ratio never exceeds
# 0.435 (the second harmonic is always weaker than the fundamental), while
# claiming 100 Hz for 50 Hz data gives 15.1 and claiming 200 Hz gives 1002.
# 1.0 sits in that empty gap and is also the natural physical boundary:
# more power outside the human band than inside it.
SAMPLE_RATE_OUT_OF_BAND_RATIO = 1.0


def steady_state_segment(
    accel_magnitude: np.ndarray,
    fs_hz: float,
    window_s: float = STEADY_WINDOW_S,
    rms_fraction: float = STEADY_RMS_FRACTION,
) -> dict:
    """Longest contiguous stretch of sustained motion, in samples.

    Uses acceleration *magnitude*, which is rotation-invariant, so this runs
    before and independently of stage 2.
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
        }
    rms = np.sqrt((x[: k * w].reshape(k, w) ** 2).mean(axis=1))
    active = rms >= rms_fraction * np.median(rms)

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
            "segments": [],
            "n_segments": 0,
            "discarded_steady_s": 0.0,
        }
    start, stop = max(segments, key=lambda s: s[1] - s[0])
    # Steady motion in the *other* segments. The product targets 20-60 min
    # runs; a single traffic-light stop splits one into two bouts, and
    # keeping only the longer bout can drop half the run. That loss is now
    # measured and reported so a caller can refuse to summarise a run from
    # a fraction of it. Handling multiple bouts is a pipeline design
    # decision (per-bout vs pooled step lists) and is deliberately not made
    # here.
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
    }


def estimate_step_frequency(a_vert: np.ndarray, fs_hz: float) -> dict:
    """Independent, detector-free estimate of step rate from the spectrum.

    Kept separate from peak detection precisely so it can be used to *audit*
    peak detection: two estimators that share no machinery disagreeing is
    evidence, whereas a detector agreeing with itself is not.
    """
    pk = dsp.spectral_peak(a_vert, fs_hz)
    return {
        "f_step_hz": pk["f_peak_hz"],
        "cadence_spm": pk["f_peak_hz"] * 60.0,
        "spectral_bin_hz": pk["bin_hz"],
        "band_power_fraction": pk["band_power_fraction"],
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

    Known blind spot, stated rather than papered over: **understating the
    rate by about 2x is undetectable here**, because it moves a 2.7 Hz
    fundamental to 1.36 Hz, which is still inside the human band and is
    genuinely indistinguishable from a slow walker. This check catches
    overstatement and gross understatement only.
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
    return {
        "in_band_peak_hz": float(pk["f_peak_hz"]),
        "in_band_peak_power": in_peak,
        "out_of_band_peak_hz": out_hz,
        "out_of_band_peak_power": out_peak,
        "out_of_band_ratio": float(ratio),
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
) -> dict:
    """Detect step-rate peaks in frame-resolved vertical acceleration.

    `t0_s` offsets the returned timestamps so they refer to the original
    trial clock even when a trimmed segment was passed in.
    """
    a_vert = np.asarray(a_vert, float)
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
    times = idx / fs_hz + t0_s
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
    }


def cadence_series(
    step_times_s: np.ndarray, smooth_steps: int = 6
) -> pd.DataFrame:
    """Instantaneous and smoothed cadence, in steps per minute.

    Instantaneous cadence is 60 / (step interval), timestamped at the
    midpoint of the interval it describes. The smoothed column is a rolling
    *median* over `smooth_steps` intervals -- median, not mean, so that one
    missed step (which doubles a single interval) shifts the estimate by at
    most one sample's worth instead of dragging the whole window.

    `smooth_steps` must be **even**, and defaults to 6 (three strides).
    Step intervals alternate short-long with the runner's two legs, a
    period-2 sequence. An odd-length median window sits on one side of that
    alternation and returns a short or a long interval alternately, so the
    "smoothed" trace sawtooths at exactly the rate it is supposed to
    suppress. An even window averages the two central order statistics,
    which cancels a period-2 alternation exactly while keeping the median's
    resistance to a single bad interval.
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
        df["cadence_spm"].rolling(smooth_steps, center=True, min_periods=1).median()
    )
    return df


def cadence_summary(
    step_times_s: np.ndarray, min_span_s: float = MIN_STEADY_SECONDS
) -> dict:
    """Whole-trial cadence statistics from detected step times.

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
        return {
            "n_steps": int(len(t)),
            "cadence_spm": np.nan,
            "cadence_spm_median": np.nan,
            "cadence_cv": np.nan,
            "irregular_step_fraction": np.nan,
            "alternating_interval_asymmetry_abs_pct": np.nan,
            "span_s": span_now,
            "span_too_short": bool(span_now < min_span_s),
        }
    intervals = np.diff(t)
    span = span_now
    # Rate over the whole span rather than the mean of per-interval
    # cadences: robust to a few outlier intervals, and the quantity a
    # step counter would report.
    cadence = 60.0 * (len(t) - 1) / span
    med = float(np.median(intervals))
    # An interval more than 25% from the local median is very likely a
    # missed step (~2x) or a double detection (~0.5x); 25% is well outside
    # normal stride-to-stride variability (a few percent) and well inside
    # those two failure modes.
    irregular = float(np.mean(np.abs(intervals - med) > 0.25 * med))

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
    # correlates at r = -0.91 with the amplitude-domain step symmetry index,
    # which is what that artifact looks like. Values here reach 32%, far
    # beyond the 1-3% real runners show.
    even, odd = intervals[0::2], intervals[1::2]
    n = min(len(even), len(odd))
    if n >= 2:
        asym = float(
            100.0 * abs(even[:n].mean() - odd[:n].mean()) / np.mean(intervals)
        )
    else:
        asym = np.nan

    return {
        "n_steps": int(len(t)),
        "cadence_spm": float(cadence),
        "cadence_spm_median": float(60.0 / med),
        "cadence_cv": float(np.std(intervals) / np.mean(intervals)),
        "irregular_step_fraction": irregular,
        "alternating_interval_asymmetry_abs_pct": asym,
        "span_s": span,
        "span_too_short": False,
    }


def diagnose_cadence(
    detected_spm: float,
    spectral_spm: float,
    stride_regularity: float,
    irregular_step_fraction: float,
    expected: tuple[float, float] = EXPECTED_CADENCE_SPM,
    sample_rate_plausible: bool = True,
    out_of_band_peak_hz: float | None = None,
) -> dict:
    """Flag out-of-band cadence and attribute the cause.

    Four outcomes: "none", "sample_rate", "algorithm", "trial".

    The algorithm-vs-trial attribution rests on having two estimators that
    share no machinery: peak counting and the spectral peak. If they agree,
    the detector is doing its job and an out-of-band number is a fact about
    the trial. If they disagree, the detector is at fault -- and the ratio
    between them says how.

    That reasoning has one precondition, which is why `sample_rate_plausible`
    is checked *first*: the two estimators share the sample rate. Given a
    wrong `fs_hz` they agree with each other while both being wrong
    together, and their agreement was previously read as evidence about the
    runner -- reporting a rate error as "the runner's cadence really is
    outside 150-190 spm". Agreement under a shared wrong assumption is not
    evidence. Pass the verdict from `check_sample_rate` to keep that
    distinction.
    """
    in_band = bool(expected[0] <= detected_spm <= expected[1])
    ratio = detected_spm / spectral_spm if spectral_spm > 0 else np.nan
    agree = bool(abs(ratio - 1.0) <= DETECTOR_SPECTRAL_TOLERANCE)

    if not sample_rate_plausible:
        cause = "sample_rate"
        where = (
            f" (strongest power sits at {out_of_band_peak_hz:.2f} Hz, outside the "
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
            f"{100 * abs(ratio - 1):.0f}% (irregular intervals: "
            f"{100 * irregular_step_fraction:.0f}% of steps)"
        )
    elif not in_band:
        cause = "trial"
        detail = (
            f"detector agrees with the independent spectral estimate "
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
        # "none" | "sample_rate" | "algorithm" | "trial"
        "failure_attributed_to": cause,
        "diagnosis": detail,
        "flagged": bool(not in_band or not agree or not sample_rate_plausible),
    }
