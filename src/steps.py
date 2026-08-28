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
# -- while rejecting ripple. Raising it to 1.5 tripled the disagreement
# with the spectral cadence estimate by starting to drop alternate steps.
PEAK_PROMINENCE_SIGMA = 0.5

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

    best_start = best_len = 0
    i = 0
    while i < k:
        if active[i]:
            j = i
            while j < k and active[j]:
                j += 1
            if j - i > best_len:
                best_start, best_len = i, j - i
            i = j
        else:
            i += 1
    if best_len == 0:
        return {
            "start": 0,
            "stop": n,
            "n_windows": k,
            "segmented": False,
            "trimmed_start_s": 0.0,
            "trimmed_end_s": 0.0,
            "kept_fraction": 1.0,
        }
    start = best_start * w
    stop = min(n, (best_start + best_len) * w)
    return {
        "start": int(start),
        "stop": int(stop),
        "n_windows": int(k),
        "segmented": True,
        "trimmed_start_s": float(start / fs_hz),
        "trimmed_end_s": float((n - stop) / fs_hz),
        "kept_fraction": float((stop - start) / n),
        "window_rms": rms,
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


def cadence_summary(step_times_s: np.ndarray) -> dict:
    """Whole-trial cadence statistics from detected step times."""
    t = np.asarray(step_times_s, float)
    if len(t) < 3:
        return {
            "n_steps": int(len(t)),
            "cadence_spm": np.nan,
            "cadence_spm_median": np.nan,
            "cadence_cv": np.nan,
            "irregular_step_fraction": np.nan,
            "alternating_interval_asymmetry_pct": np.nan,
            "span_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        }
    intervals = np.diff(t)
    span = float(t[-1] - t[0])
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

    # Alternating-interval asymmetry: the signed difference between the two
    # interleaved sets of step intervals, as a percentage of the mean
    # interval.
    #
    # READ THIS AS A DETECTOR DIAGNOSTIC, NOT AS THE RUNNER'S STEP-TIME
    # SYMMETRY. When the two steps of a stride produce differently shaped
    # vertical waveforms -- which they do at a pocket, strongly -- the crest
    # of the band-passed fundamental shifts by a different amount on
    # alternate steps, so the detected timestamps alternate short-long even
    # if the runner's true step times do not. On this dataset the magnitude
    # of this quantity correlates at r = -0.91 with the amplitude-domain
    # step symmetry index, which is what that artifact looks like. Values
    # here reach +/-30%, far beyond the 1-3% real runners show.
    even, odd = intervals[0::2], intervals[1::2]
    n = min(len(even), len(odd))
    if n >= 2:
        asym = float(
            100.0 * (even[:n].mean() - odd[:n].mean()) / np.mean(intervals)
        )
    else:
        asym = np.nan

    return {
        "n_steps": int(len(t)),
        "cadence_spm": float(cadence),
        "cadence_spm_median": float(60.0 / med),
        "cadence_cv": float(np.std(intervals) / np.mean(intervals)),
        "irregular_step_fraction": irregular,
        "alternating_interval_asymmetry_pct": asym,
        "span_s": span,
    }


def diagnose_cadence(
    detected_spm: float,
    spectral_spm: float,
    stride_regularity: float,
    irregular_step_fraction: float,
    expected: tuple[float, float] = EXPECTED_CADENCE_SPM,
) -> dict:
    """Flag out-of-band cadence and attribute it to the algorithm or the trial.

    The attribution rests on having two estimators that share no machinery:
    peak counting and the spectral peak. If they agree, the detector is
    doing its job and an out-of-band number is a fact about the trial. If
    they disagree, the detector is at fault -- and the ratio between them
    says how.
    """
    in_band = bool(expected[0] <= detected_spm <= expected[1])
    ratio = detected_spm / spectral_spm if spectral_spm > 0 else np.nan
    agree = bool(abs(ratio - 1.0) <= DETECTOR_SPECTRAL_TOLERANCE)

    if stride_regularity < MIN_STRIDE_REGULARITY:
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
        "failure_attributed_to": cause,  # "none" | "algorithm" | "trial"
        "diagnosis": detail,
        "flagged": bool(not in_band or not agree),
    }
