"""Failure-mode probes: what happens on degenerate, short, or lying input.

Companion to `test_invariance.py`, which checks the properties the pipeline
claims to have. This file checks what it does when its assumptions are
violated -- the cases where silently returning a plausible number is worse
than raising.

Tests marked `xfail(strict=True)` document **confirmed, unfixed bugs**. They
assert the behaviour the code *should* have; strict mode means the suite
goes red the moment one is fixed, so the marker must be removed with the
fix. Everything unmarked is behaviour that is already correct and is pinned
here so it stays that way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal as _sig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import dsp, lateral, loader, orientation, pipeline, steps  # noqa: E402

FS_HZ = 50.0
F_STEP_HZ = 2.75
DURATION_S = 40.0


def _t(fs_hz: float = FS_HZ, seconds: float = DURATION_S) -> np.ndarray:
    return np.arange(0, seconds, 1.0 / fs_hz)


# --- constant / degenerate input -----------------------------------------


def test_robust_sigma_of_constant_is_zero():
    """Documented precondition for the prominence bug below."""
    assert dsp.robust_sigma(np.ones(1000)) == 0.0


@pytest.mark.xfail(
    strict=True,
    reason="BUG: robust_sigma of a constant signal is 0, so the peak "
    "prominence threshold becomes 0 and find_peaks accepts every local "
    "maximum in filtfilt's numerical noise. A constant input yields ~47 "
    "'steps' with no error or warning.",
)
def test_constant_signal_detects_no_steps():
    n_steps = steps.detect_steps(np.ones(1000), FS_HZ, F_STEP_HZ)["n_steps"]
    assert n_steps == 0, f"constant signal produced {n_steps} spurious steps"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _principal_axis_2d returns np.inf when the secondary "
    "eigenvalue is <= 0, and inf >= FORWARD_CONDITIONING_MIN_RATIO, so a "
    "rank-deficient (information-free) input is reported as perfectly "
    "well-conditioned. The semantics are inverted for exactly the input "
    "the check exists to reject.",
)
def test_degenerate_horizontal_accel_is_not_well_conditioned():
    est = orientation.forward_axis(
        np.ones((1000, 3)), np.array([0.0, 0.0, 1.0]), FS_HZ, method="pca"
    )
    assert not est["well_conditioned"], (
        f"degenerate input reported well_conditioned with ratio "
        f"{est['eigenvalue_ratio']}"
    )


def test_constant_signal_fails_the_periodicity_check():
    """This one the code gets right: a constant trace is not periodic."""
    per = orientation.verify_vertical_periodicity(np.ones(1000), FS_HZ)
    assert per["periodicity_ok"] is False


def test_zero_gravity_raises_rather_than_returning_nonsense():
    """Correct behaviour, pinned: an all-zero gravity column is refused."""
    with pytest.raises(ValueError, match="zero-length gravity"):
        orientation.unit_gravity(np.zeros((100, 3)))
    with pytest.raises(ValueError, match="zero-length gravity"):
        orientation.vertical_axis(np.zeros((100, 3)))


# --- NaN input ------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG: unit_gravity guards only against a zero norm. NaN norms "
    "are not equal to 0, so an all-NaN gravity column passes the guard and "
    "returns NaN, propagating silently through the whole frame. No stage "
    "checks for non-finite input; only the loader counts NaNs, and nothing "
    "acts on that count.",
)
def test_nan_gravity_is_rejected():
    with pytest.raises((ValueError, FloatingPointError)):
        orientation.unit_gravity(np.full((100, 3), np.nan))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: spectral_peak on an all-NaN signal returns the lowest bin "
    "of the search band (1.25 Hz = 75 spm) as a confident estimate instead "
    "of refusing. argmax over an all-NaN array returns index 0.",
)
def test_nan_signal_spectral_peak_is_rejected():
    with pytest.raises((ValueError, FloatingPointError)):
        dsp.spectral_peak(np.full(1000, np.nan), FS_HZ)


def test_nan_signal_detects_no_steps():
    """Correct behaviour, pinned: find_peaks finds nothing in all-NaN."""
    assert steps.detect_steps(np.full(1000, np.nan), FS_HZ, F_STEP_HZ)["n_steps"] == 0


def test_all_zero_signal_detects_no_steps():
    assert steps.detect_steps(np.zeros(1000), FS_HZ, F_STEP_HZ)["n_steps"] == 0


# --- known-frequency ground truth ----------------------------------------


@pytest.mark.parametrize("fs_hz", [50.0, 100.0, 200.0])
@pytest.mark.parametrize("f0_hz", [2.5, 2.75, 3.1])
def test_pure_sine_returns_the_right_cadence(fs_hz, f0_hz):
    """A pure sine is the one case where the true cadence is known exactly."""
    x = np.sin(2 * np.pi * f0_hz * _t(fs_hz))
    det = steps.detect_steps(x, fs_hz)
    cadence = steps.cadence_summary(det["step_times_s"])["cadence_spm"]
    expected = 60.0 * f0_hz
    # 0.5% covers the half-sample quantisation of the first and last peak
    # over a 40 s record; anything larger would be a real counting error.
    assert abs(cadence - expected) / expected < 0.005, (
        f"fs={fs_hz} f0={f0_hz}: got {cadence:.3f} spm, expected {expected:.3f}"
    )


@pytest.mark.parametrize("fs_hz", [50.0, 100.0, 200.0])
def test_spectral_estimator_recovers_a_known_frequency(fs_hz):
    x = np.sin(2 * np.pi * F_STEP_HZ * _t(fs_hz))
    got = steps.estimate_step_frequency(x, fs_hz)["f_step_hz"]
    assert abs(got - F_STEP_HZ) < 0.01, f"fs={fs_hz}: got {got:.4f} Hz"


# --- harmonic rejection ---------------------------------------------------


def _double_crest_signal(fs_hz: float = FS_HZ) -> np.ndarray:
    """Fundamental plus an in-phase 2nd harmonic: two crests per step cycle."""
    t = _t(fs_hz)
    return np.sin(2 * np.pi * F_STEP_HZ * t) - 0.9 * np.sin(2 * np.pi * 2 * F_STEP_HZ * t)


def test_lowpass_rejects_the_second_harmonic_on_filter_shape_alone():
    """The specific claim REPORT.md makes about the 1.5 x f_step cutoff.

    Peak spacing is disabled (distance=1) so the minimum-distance rule
    cannot mask a double detection. Only the filter can do the work.
    """
    x = _double_crest_signal()
    true_steps = int(DURATION_S * F_STEP_HZ)  # 110

    def n_peaks(sig_):
        pk, _ = _sig.find_peaks(sig_, prominence=0.5 * dsp.robust_sigma(sig_))
        return len(pk)

    unfiltered = n_peaks(x)
    shipped = n_peaks(steps.bandpass_for_steps(x, FS_HZ, F_STEP_HZ)[0])
    wide = n_peaks(dsp.bandpass(x, FS_HZ, 0.7 * F_STEP_HZ, 3.0 * F_STEP_HZ))

    assert unfiltered > 1.8 * true_steps, "test signal does not double-detect"
    assert wide > 1.8 * true_steps, "3 x f_step cutoff should pass the harmonic"
    assert abs(shipped - true_steps) <= 2, (
        f"1.5 x f_step cutoff left {shipped} peaks, expected ~{true_steps}"
    )


def test_harmonic_signal_gives_the_fundamental_cadence():
    det = steps.detect_steps(_double_crest_signal(), FS_HZ, F_STEP_HZ)
    cadence = steps.cadence_summary(det["step_times_s"])["cadence_spm"]
    assert abs(cadence - 60 * F_STEP_HZ) / (60 * F_STEP_HZ) < 0.005


# --- short records --------------------------------------------------------


def test_min_steady_seconds_is_enforced_only_by_the_pipeline():
    """Documents where the short-record guard lives, and where it does not.

    `pipeline.run_trial` sets `steady_too_short` and overrides the cadence
    diagnosis below MIN_STEADY_SECONDS. The stage-level functions have no
    such guard, so a caller assembling stages directly -- as
    `scripts/run_invariance_checks.py` and `tests/test_invariance.py` both
    do -- gets a confident cadence from a 3 s record with no flag.
    """
    tr = loader.load_trial("jog", 9, 1)
    n = int(3.0 * tr.fs_hz)
    a, g = tr.user_accel[:n], tr.gravity[:n]

    a_vert = orientation.vertical_component(a, g)
    f0 = steps.estimate_step_frequency(a_vert, tr.fs_hz)["f_step_hz"]
    det = steps.detect_steps(a_vert, tr.fs_hz, f0)
    summary = steps.cadence_summary(det["step_times_s"])
    diag = steps.diagnose_cadence(
        summary["cadence_spm"], 60 * f0, 0.64, summary["irregular_step_fraction"]
    )

    assert 3.0 < steps.MIN_STEADY_SECONDS
    # The stage path reports an unflagged, in-band cadence from 3 seconds.
    assert np.isfinite(summary["cadence_spm"])
    assert diag["flagged"] is False
    assert diag["failure_attributed_to"] == "none"


def test_cadence_summary_degrades_cleanly_below_three_steps():
    for times in ([], [1.0], [1.0, 1.4]):
        out = steps.cadence_summary(np.asarray(times, float))
        assert np.isnan(out["cadence_spm"])
        assert out["n_steps"] == len(times)


def test_cadence_series_is_empty_below_two_steps():
    df = steps.cadence_series(np.asarray([1.0]))
    assert len(df) == 0
    assert list(df.columns) == ["t_s", "cadence_spm", "cadence_spm_smooth"]


# --- lying about the sample rate ------------------------------------------


def test_wrong_sample_rate_scales_the_answer_undetectably():
    """Passing the wrong fs is an input assertion no code here can check.

    Pinned because the *size* of the error matters: the step-frequency
    search band (1.2-4.5 Hz) clamps the estimate into a plausible-looking
    range, so a 4x rate error yields ~206 spm rather than an obviously
    absurd number.
    """
    tr = loader.load_trial("jog", 9, 1)
    a_vert = orientation.vertical_component(tr.user_accel, tr.gravity)

    truth = steps.cadence_summary(
        steps.detect_steps(a_vert, 50.0, steps.estimate_step_frequency(a_vert, 50.0)["f_step_hz"])[
            "step_times_s"
        ]
    )["cadence_spm"]
    lied = steps.cadence_summary(
        steps.detect_steps(a_vert, 200.0, steps.estimate_step_frequency(a_vert, 200.0)["f_step_hz"])[
            "step_times_s"
        ]
    )["cadence_spm"]

    assert 160 < truth < 165
    assert lied > 190, "a 4x rate lie should at least leave the expected band"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: diagnose_cadence attributes a sample-rate error to the "
    "trial. Both estimators are computed under the same wrong fs, so they "
    "agree with each other, and the out-of-band result is reported as "
    "'the runner's cadence really is outside 150-190 spm'. Agreement "
    "between two estimators sharing a wrong fs is not evidence about the "
    "runner.",
)
def test_rate_error_is_not_blamed_on_the_trial():
    tr = loader.load_trial("jog", 9, 1)
    a_vert = orientation.vertical_component(tr.user_accel, tr.gravity)
    f0 = steps.estimate_step_frequency(a_vert, 200.0)["f_step_hz"]
    det = steps.detect_steps(a_vert, 200.0, f0)
    summary = steps.cadence_summary(det["step_times_s"])
    diag = steps.diagnose_cadence(summary["cadence_spm"], 60 * f0, 0.85, summary["irregular_step_fraction"])
    assert diag["failure_attributed_to"] != "trial"


# --- the Nyquist guard ----------------------------------------------------


def test_nyquist_guard_raises_when_the_fundamental_leaves_the_band():
    """Correct behaviour, pinned: fs < 2.5 x f_step must raise."""
    x = np.random.default_rng(0).normal(size=2000)
    with pytest.raises(ValueError, match="sample rate is too low"):
        steps.bandpass_for_steps(x, fs_hz=50.0, f_step_hz=20.0)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: for 0.2667*fs <= f_step < 0.4*fs the low-pass is silently "
    "clamped to 0.4*fs instead of 1.5*f_step. The band still contains the "
    "fundamental so nothing raises, but the filter is no longer the "
    "documented multiple of f_step -- contradicting 'both cutoffs are "
    "multiples of the estimated step frequency, never fixed Hz'. Does not "
    "bite at running cadence on this dataset (needs f_step >= 13.3 Hz at "
    "50 Hz), but it is a silent clamp, not a loud failure.",
)
def test_nyquist_clamp_is_never_silent():
    x = np.random.default_rng(0).normal(size=2000)
    fs_hz, f_step_hz = 50.0, 14.0  # inside the clamp window
    _, (low, high) = steps.bandpass_for_steps(x, fs_hz, f_step_hz)
    assert high == pytest.approx(steps.LOWPASS_STEP_FREQ_MULTIPLE * f_step_hz), (
        f"low-pass silently clamped from {steps.LOWPASS_STEP_FREQ_MULTIPLE * f_step_hz} "
        f"to {high} Hz"
    )


# --- autocorrelation lag indexing -----------------------------------------


@pytest.mark.parametrize("fs_hz,f0_hz", [(50.0, 2.75), (100.0, 3.0), (200.0, 2.5)])
def test_autocorrelation_lag_indices_have_no_off_by_one(fs_hz, f0_hz):
    """A pure sine must autocorrelate near +1 at whole periods, -1 at halves.

    The magnitude at a whole period is not exactly 1: it is deflated by the
    biased normaliser (see `test_autocorrelation_is_unbiased_as_documented`)
    and by lag quantisation when `period * fs` is not an integer. Both are
    bounded and neither is an indexing error, so this test checks the two
    things that *would* reveal an off-by-one: that the chosen index is the
    nearest sample to the target lag, and that whole and half periods come
    back with opposite sign and near-unit magnitude.
    """
    period = 1.0 / f0_hz
    x = np.sin(2 * np.pi * f0_hz * _t(fs_hz, 60.0))
    lags, ac = dsp.autocorrelation(x, fs_hz, 3.5 * period)
    for multiple in (1.0, 2.0, 3.0):
        i = int(np.argmin(np.abs(lags - multiple * period)))
        assert i == round(multiple * period * fs_hz), (
            f"{multiple}T: index {i}, expected {round(multiple * period * fs_hz)}"
        )
        assert abs(lags[i] - multiple * period) <= 0.5 / fs_hz + 1e-12
        assert ac[i] > 0.95, f"{multiple}T: ac={ac[i]:.4f}"
    for multiple in (0.5, 1.5, 2.5):
        i = int(np.argmin(np.abs(lags - multiple * period)))
        assert ac[i] < -0.95, f"{multiple}T: ac={ac[i]:.4f} (should be ~-1)"


def test_stride_lag_dominates_for_an_asymmetric_gait_signal():
    """The stride-vs-step logic, on a signal built to be stride-periodic."""
    t = _t(FS_HZ, 60.0)
    asym = np.sin(2 * np.pi * F_STEP_HZ * t) * (
        1 + 0.6 * np.sign(np.sin(2 * np.pi * (F_STEP_HZ / 2) * t))
    )
    per = orientation.verify_vertical_periodicity(asym, FS_HZ)
    assert per["stride_regularity"] > per["step_regularity"]
    assert per["periodicity_ok"] is True
    assert 0.0 < per["step_symmetry_index"] < 1.0


@pytest.mark.xfail(
    strict=True,
    reason="BUG: autocorrelation's docstring says 'unbiased-ish' but it is "
    "the plain biased estimator -- ac(k) is scaled by (n-k)/n with no "
    "correction. Because the stride lag is twice the step lag, stride "
    "regularity is deflated about twice as much as step regularity, which "
    "systematically inflates step_symmetry_index (their ratio).",
)
def test_autocorrelation_is_unbiased_as_documented():
    fs_hz, f0_hz, n = 200.0, 2.5, 12000
    x = np.sin(2 * np.pi * f0_hz * np.arange(n) / fs_hz)
    lags, ac = dsp.autocorrelation(x, fs_hz, 3.0 / f0_hz)
    i = int(np.argmin(np.abs(lags - 1.0 / f0_hz)))
    # Exact lag alignment here, so any shortfall is the missing 1/(n-k).
    assert ac[i] == pytest.approx(1.0, abs=1e-6), f"ac(1T) = {ac[i]:.6f}"


# --- stage 4 nulls --------------------------------------------------------


def test_phase_sweep_null_is_max_over_phase_not_mean():
    """The claim REPORT.md rests its 40/48 result on."""
    rng = np.random.default_rng(0)
    omega = np.sin(2 * np.pi * (F_STEP_HZ / 2) * _t(FS_HZ, 60.0)) + 0.3 * rng.normal(
        size=len(_t(FS_HZ, 60.0))
    )
    idx = np.arange(20, len(omega) - 40, 18)
    fixed = lateral.null_models(omega, idx, FS_HZ, F_STEP_HZ, n_surrogates=40, seed=0)
    swept = lateral.phase_sweep_null(omega, idx, FS_HZ, F_STEP_HZ, n_surrogates=40, seed=0)
    assert swept["max_mean"] > fixed["surrogate_mean"], (
        "sweep-and-maximise null must exceed the fixed-phase null, or the "
        "selection over offsets is uncorrected"
    )
    assert swept["max_p95"] >= swept["max_mean"]


def test_alternation_rate_edge_cases():
    assert np.isnan(lateral.alternation_rate(np.array([])))
    assert np.isnan(lateral.alternation_rate(np.array([1.0])))
    assert lateral.alternation_rate(np.array([1.0, -1.0, 1.0, -1.0])) == 1.0
    assert lateral.alternation_rate(np.array([1.0, 1.0, 1.0])) == 0.0
    # zeros cannot support a flip claim
    assert lateral.alternation_rate(np.array([0.0, 0.0, 0.0])) == 0.0


def test_surrogate_preserves_the_power_spectrum():
    rng = np.random.default_rng(3)
    x = rng.normal(size=2048)
    s = lateral.phase_randomised_surrogate(x, rng)
    assert np.allclose(np.abs(np.fft.rfft(x))[1:-1], np.abs(np.fft.rfft(s))[1:-1], rtol=1e-8)


# --- guards that already work ---------------------------------------------


def test_cadence_smoothing_rejects_odd_windows():
    with pytest.raises(ValueError, match="even"):
        steps.cadence_series(np.arange(10) * 0.36, smooth_steps=5)


def test_even_window_cancels_period_two_alternation():
    """The stated reason the window must be even."""
    short, long_ = 0.30, 0.42
    times = np.cumsum([0.0] + [short, long_] * 20)
    df = steps.cadence_series(times, smooth_steps=6)
    interior = df["cadence_spm_smooth"].iloc[5:-5]
    assert interior.std() < 1.0, (
        f"even window still sawtooths: std {interior.std():.3f} spm"
    )


def test_build_frame_rejects_an_unknown_mode():
    tr = loader.load_trial("jog", 9, 1)
    with pytest.raises(ValueError, match="mode must be"):
        orientation.build_frame(tr.user_accel, tr.gravity, tr.fs_hz, mode="nonsense")


def test_lateral_record_always_disclaims_ground_truth():
    result = pipeline.run_trial("jog", 16, 5)
    assert result["lateral"]["ground_truth_available"] is False
