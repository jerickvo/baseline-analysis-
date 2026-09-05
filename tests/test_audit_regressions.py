"""Regression tests from the audit: each pins a defect that was found, verified
and fixed, on a synthetic signal whose true answer is known.

Grouped by pipeline stage. Every test here ran red against the code as it
was before the fix; the docstring says what the code used to do.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import dsp, lateral, loader, orientation, pipeline, steps  # noqa: E402

FS_HZ = 100.0
F_STEP_HZ = 2.8


def _t(seconds: float = 60.0, fs_hz: float = FS_HZ) -> np.ndarray:
    return np.arange(0, seconds, 1.0 / fs_hz)


def _body_gait(seconds: float = 60.0, fs_hz: float = FS_HZ, f: float = F_STEP_HZ, ml_amp: float = 0.15):
    """Body-frame gait: fore-aft braking at the step rate in anti-phase with the
    vertical peak, mediolateral sway at the stride rate. Columns [fwd, ml, up]."""
    t = _t(seconds, fs_hz)
    fwd = -0.3 * np.cos(2 * np.pi * f * t)
    ml = ml_amp * np.sin(2 * np.pi * (f / 2) * t)
    up = 0.8 * np.cos(2 * np.pi * f * t)
    return np.column_stack([fwd, ml, up])


def _pitched_sensor(body: np.ndarray, pitch_deg: float, f: float = F_STEP_HZ, fs_hz: float = FS_HZ):
    """Rotate body-frame vectors into a sensor that pitches +/-pitch_deg about the
    body ML axis at the stride rate (thigh-like). Returns (accel, gravity)."""
    t = np.arange(len(body)) / fs_hz
    th = np.radians(pitch_deg) * np.sin(2 * np.pi * (f / 2) * t)
    fwd, ml, up = body.T
    acc = np.column_stack([fwd * np.cos(th) + up * np.sin(th), ml, -fwd * np.sin(th) + up * np.cos(th)])
    grav = np.column_stack([-np.sin(th), np.zeros_like(t), -np.cos(th)])
    return acc, grav


# --- stage 2: conditioning semantics ---------------------------------------


def test_single_axis_motion_is_perfectly_conditioned():
    """A signal with fore-aft on exactly one horizontal axis and nothing on the
    other is the best-determined case there is. It used to return NaN
    ('undefined') and fail the conditioning test."""
    body = _body_gait(ml_amp=0.0)
    est = orientation.forward_axis(body, np.array([0.0, 0.0, 1.0]), FS_HZ)
    assert est["well_conditioned"], est["eigenvalue_ratio"]
    assert np.isinf(est["eigenvalue_ratio"])


def test_no_power_at_all_is_still_not_conditioned():
    """The case the previous fix was for: a constant input has no axis."""
    est = orientation.forward_axis(np.ones((1000, 3)), np.array([0.0, 0.0, 1.0]), FS_HZ, method="pca")
    assert not est["well_conditioned"]
    assert np.isnan(est["eigenvalue_ratio"])


# --- stage 2: verdict gates -------------------------------------------------


@pytest.mark.parametrize("pitch_deg", [0.0, 10.0, 20.0, 30.0])
def test_ml_check_survives_stride_rate_pitch(pitch_deg):
    """A sensor pitching at the stride rate leaked vertical acceleration into the
    mean-plane stride band and failed the ML check (48 deg at 20 deg pitch)
    while the tracking resolution was exact to 1e-8 g. The check now runs on
    the resolved channels and holds at 90 deg."""
    acc, grav = _pitched_sensor(_body_gait(), pitch_deg)
    frame = orientation.build_frame(acc, grav, FS_HZ)
    v = orientation.verify_frame(acc, grav, frame, FS_HZ)
    assert v["mediolateral_check"]["step_stride_axis_angle_deg"] > 85.0
    assert v["verdict"] == "ok", v["reasons"]


@pytest.mark.parametrize("seed", range(6))
def test_ml_check_requires_a_determined_stride_axis(seed):
    """With NO stride-rate power anywhere (ML channel is noise), the stride axis
    points somewhere random and used to land >= 60 deg from the step axis
    about a third of the time -- a chance pass, which is how this dataset's
    only two 'ok' verdicts were produced. An undetermined stride axis is now
    its own non-passing state."""
    rng = np.random.default_rng(seed)
    body = _body_gait(ml_amp=0.0)
    # Isotropic noise on BOTH horizontal channels: noise on one channel only
    # would itself be a (determined) axis. Fore-aft keeps its step-rate
    # sinusoid, so the forward axis stays well-conditioned; the stride band
    # sees only noise, with no direction to prefer.
    body[:, :2] += 0.05 * rng.normal(size=(len(body), 2))
    frame = orientation.build_frame(body, np.tile([0.0, 0.0, -1.0], (len(body), 1)), FS_HZ)
    v = orientation.verify_frame(body, np.tile([0.0, 0.0, -1.0], (len(body), 1)), frame, FS_HZ)
    assert v["mediolateral_check"]["ml_check_state"] == "stride_axis_undetermined"
    assert v["verdict"] != "ok"


def test_drifting_forward_axis_cannot_be_ok():
    """The forward axis yawing 90 deg over the trial was measured (drift p95)
    but never consulted by the verdict."""
    body = _body_gait()
    t = _t()
    yaw = np.radians(90.0) * t / t[-1]
    fwd, ml, up = body.T
    acc = np.column_stack([fwd * np.cos(yaw) - ml * np.sin(yaw), fwd * np.sin(yaw) + ml * np.cos(yaw), up])
    grav = np.tile([0.0, 0.0, -1.0], (len(t), 1))
    frame = orientation.build_frame(acc, grav, FS_HZ)
    v = orientation.verify_frame(acc, grav, frame, FS_HZ)
    assert v["stability"]["forward_drift_p95_deg"] > orientation.FORWARD_DRIFT_MAX_P95_DEG
    assert v["verdict"] != "ok"
    assert any("wanders" in r and "forward" in r for r in v["reasons"]), v["reasons"]


def test_trial_too_short_to_measure_stability_cannot_be_ok():
    """Zero stability windows (< 8 s) is 'unmeasured', which is not 'stable'."""
    body = _body_gait(seconds=6.0)
    grav = np.tile([0.0, 0.0, -1.0], (len(body), 1))
    frame = orientation.build_frame(body, grav, FS_HZ)
    v = orientation.verify_frame(body, grav, frame, FS_HZ)
    assert v["stability"]["n_stability_windows"] == 0
    assert v["verdict"] != "ok"
    assert any("too short" in r for r in v["reasons"]), v["reasons"]


def test_forward_axis_band_matches_the_verifier_on_real_data():
    """jog_16/sub_1 built its forward axis at the 1.5 x f harmonic (4.16 Hz)
    because build_frame took f_step from the static projection while the
    verifier used the tracking vertical (2.78 Hz)."""
    res = pipeline.run_trial("jog", 16, 1)
    used = res["frame"].diagnostics["f_step_hz_used"]
    ver = res["verify"]["periodicity"]["f_step_hz"]
    assert abs(used / ver - 1.0) < orientation.HARMONIC_REL_BW, (used, ver)
    assert res["frame_f_step_mismatch"] is False


def test_resolve_stays_orthonormal_when_up_is_parallel_to_forward():
    """The degenerate fallback substituted the un-projected static forward,
    giving a non-orthonormal triad and NaN at exact parallelism."""
    frame = orientation.AnatomicalFrame(
        up=np.array([0.0, 0.0, 1.0]), forward=np.array([1.0, 0.0, 0.0]),
        mediolateral=np.array([0.0, 1.0, 0.0]), rotation=np.eye(3), mode="tracking",
        up_series=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [np.cos(5e-7), 0.0, np.sin(5e-7)]]),
    )
    v = np.tile([1.0, 2.0, 3.0], (3, 1))
    r = orientation.resolve(v, frame)
    assert np.isfinite(r).all()
    assert np.allclose(np.linalg.norm(r, axis=1), np.linalg.norm(v, axis=1), atol=1e-9)


@pytest.mark.parametrize(
    "axis,angle_deg,name",
    [((1, 0, 0), 180.0, "upside-down (roll 180)"), ((0, 0, 1), 180.0, "yawed 180"), ((0, 1, 0), 90.0, "on its side")],
)
def test_deliberate_placements_recover_the_same_vertical(axis, angle_deg, name):
    """Named placements, not just random ones: the resolved vertical channel
    and the step count must be identical to the un-rotated sensor."""
    body = _body_gait()
    grav0 = np.tile([0.0, 0.0, -1.0], (len(body), 1))
    R = dsp.rotation_matrix_from_axis_angle(np.array(axis, float), np.radians(angle_deg))
    acc, grav = body @ R.T, grav0 @ R.T
    a0 = orientation.vertical_component(body, grav0)
    a1 = orientation.vertical_component(acc, grav)
    assert np.allclose(a0, a1, atol=1e-12), name
    n0 = steps.detect_steps(a0, FS_HZ, F_STEP_HZ)["n_steps"]
    n1 = steps.detect_steps(a1, FS_HZ, F_STEP_HZ)["n_steps"]
    assert n0 == n1 and n0 > 100, name


# --- stage 2: regularity / symmetry ----------------------------------------


@pytest.mark.parametrize("asym", [0.2, 0.4, 0.6])
def test_symmetry_index_is_not_inflated_by_the_band_edge(asym):
    """The regularity band started at 0.5 x f_step -- the corner sat on the
    stride subharmonic, attenuating it 6 dB after filtfilt and biasing the
    symmetry index upward (0.72 reported vs 0.47 true at asym 0.6)."""
    fs = 50.0
    t = _t(60.0, fs)
    f = 2.75
    x = np.sin(2 * np.pi * f * t) * (1 + asym * np.sign(np.sin(2 * np.pi * (f / 2) * t)))
    per = orientation.verify_vertical_periodicity(x, fs)
    # The cited definition: autocorrelation of the signal itself.
    lags, ac = dsp.autocorrelation(x, fs, 3.0 / f)
    ref = ac[int(np.argmin(np.abs(lags - 1 / f)))] / ac[int(np.argmin(np.abs(lags - 2 / f)))]
    assert abs(per["step_symmetry_index"] - ref) < 0.05, (per["step_symmetry_index"], ref)


@pytest.mark.parametrize("f0", [2.5, 2.75, 3.1])
def test_symmetry_index_never_exceeds_one_and_is_rate_agnostic(f0):
    """Reading the autocorrelation at the nearest integer lag pushed the index
    above 1 on two real trials and made it depend on the sample rate. It is
    now read at the exact fractional lag."""
    vals = []
    for fs in (50.0, 200.0):
        t = _t(60.0, fs)
        x = np.sin(2 * np.pi * f0 * t) * (1 + 0.5 * np.sign(np.sin(2 * np.pi * (f0 / 2) * t)))
        per = orientation.verify_vertical_periodicity(x, fs)
        assert per["step_symmetry_index"] <= 1.0 + 1e-6
        vals.append(per["step_symmetry_index"])
    assert abs(vals[0] - vals[1]) < 0.02, vals


# --- stage 3: cadence attribution -------------------------------------------


def test_stride_subharmonic_lock_on_is_ambiguous_not_the_runners_fault():
    """A runner whose stride-rate power exceeds step-rate power: the spectral
    peak sits at f/2, the detector is band-passed around it, both halve
    together, and the halving was attributed to the TRIAL as 'cadence really
    is outside 150-190'."""
    t = _t()
    f = 170.0 / 60.0
    x = 0.6 * np.sin(2 * np.pi * f * t) + 1.0 * np.sin(2 * np.pi * (f / 2) * t)
    x += 0.05 * np.random.default_rng(0).normal(size=len(t))
    per = orientation.verify_vertical_periodicity(x, FS_HZ)
    det = steps.detect_steps(x, FS_HZ, f_step_hz=per["f_step_hz"])
    cs = steps.cadence_summary(det["step_times_s"])
    sp = steps.estimate_step_frequency(x, FS_HZ)
    assert cs["cadence_spm"] < 100  # both estimators halved
    dg = steps.diagnose_cadence(
        cs["cadence_spm"], sp["cadence_spm"], per["stride_regularity"], cs["irregular_step_fraction"],
        second_harmonic_ratio=per["second_harmonic_ratio"],
    )
    assert dg["failure_attributed_to"] == "ambiguous"
    assert dg["flagged"]


def _run_walk_run():
    def gait(sec, f, amp, seed):
        tt = _t(sec)
        return amp * np.cos(2 * np.pi * f * tt) + 0.2 * amp * np.cos(2 * np.pi * f / 2 * tt) \
            + 0.05 * np.random.default_rng(seed).normal(size=len(tt))
    return np.r_[gait(120, F_STEP_HZ, 0.8, 0), gait(60, 1.9, 0.35, 1), gait(120, F_STEP_HZ, 0.8, 2)]


def test_walk_break_is_a_detection_gap_attributed_to_the_trial():
    """A 60 s walk inside a 5 min run dragged the whole-span cadence from 168
    to 135 spm, and the resulting detector/spectral disagreement was blamed
    on the ALGORITHM."""
    x = _run_walk_run()
    sp = steps.estimate_step_frequency(x, FS_HZ)
    det = steps.detect_steps(x, FS_HZ, f_step_hz=sp["f_step_hz"])
    cs = steps.cadence_summary(det["step_times_s"])
    assert cs["detection_gap"] is True
    assert cs["largest_interval_step_periods"] > 50
    dg = steps.diagnose_cadence(
        cs["cadence_spm"], sp["cadence_spm"], 0.8, cs["irregular_step_fraction"],
        detection_gap=cs["detection_gap"], largest_interval_step_periods=cs["largest_interval_step_periods"],
    )
    assert dg["failure_attributed_to"] == "trial"
    assert "not continuous running" in dg["diagnosis"]


def test_two_consecutive_missed_steps_are_not_a_detection_gap():
    """~3 step periods is a detector miss (irregular_step_fraction's job), not a
    hole in the running. The largest such interval on the real data is 3.4x."""
    times = np.cumsum([0.0] + [0.36] * 40 + [0.36 * 3.4] + [0.36] * 40)
    cs = steps.cadence_summary(times)
    assert cs["detection_gap"] is False
    assert cs["irregular_step_fraction"] > 0


def test_diagnosis_without_a_spectral_estimate_does_not_print_nan():
    """spectral_spm of 0 or NaN produced cause='algorithm' and the message
    'disagrees with the spectral rate by nan%'."""
    for spectral in (0.0, np.nan):
        dg = steps.diagnose_cadence(165.0, spectral, 0.8, 0.02)
        assert dg["failure_attributed_to"] == "trial"
        assert "nan" not in dg["diagnosis"]


def test_flagged_is_exactly_cause_is_not_none():
    """A low stride regularity used to give cause='trial' with flagged=False."""
    cases = [
        dict(detected_spm=165.0, spectral_spm=166.0, stride_regularity=0.2, irregular_step_fraction=0.0),
        dict(detected_spm=165.0, spectral_spm=166.0, stride_regularity=0.8, irregular_step_fraction=0.0),
        dict(detected_spm=140.0, spectral_spm=141.0, stride_regularity=0.8, irregular_step_fraction=0.0),
        dict(detected_spm=np.nan, spectral_spm=166.0, stride_regularity=0.8, irregular_step_fraction=np.nan),
    ]
    for kw in cases:
        dg = steps.diagnose_cadence(**kw)
        assert dg["flagged"] == (dg["failure_attributed_to"] != "none"), (kw, dg)


# --- stage 3a: steady state ----------------------------------------------


def test_steady_state_reports_every_active_segment():
    """Only the longest active segment is analysed; on a run with a stop in the
    middle the rest vanished with nothing but `trimmed_end_s` to show for it."""
    def mag(sec, seed):
        return np.abs(_body_gait(sec)[:, 2]) + 0.6
    x = np.r_[mag(120, 0), 0.02 * np.abs(np.random.default_rng(3).normal(size=int(20 * FS_HZ))), mag(120, 2)]
    seg = steps.steady_state_segment(x, FS_HZ)
    assert seg["n_active_segments"] == 2
    assert abs(seg["kept_fraction_of_active"] - 0.5) < 0.05
    assert abs(seg["active_seconds"] - 240.0) < 3.0


def test_steady_state_refuses_non_finite_input():
    """A NaN poisoned the median so no window was 'active' and the whole record
    was silently kept as steady state."""
    x = np.r_[np.ones(500), np.nan, np.ones(500)]
    with pytest.raises(ValueError, match="non-finite"):
        steps.steady_state_segment(x, 50.0)


def test_pipeline_refuses_a_file_with_a_nan_sample(monkeypatch):
    """The integrity record said clean=False and nothing read it."""
    real = loader.load_trial

    def corrupt(*args, **kwargs):
        tr = real(*args, **kwargs)
        tr.df.iloc[100, tr.df.columns.get_loc("userAcceleration.x")] = np.nan
        tr.integrity = dict(tr.integrity, n_nan=1, clean=False, problems="non-finite samples")
        return tr

    monkeypatch.setattr(loader, "load_trial", corrupt)
    with pytest.raises(ValueError, match="unusable file"):
        pipeline.run_trial("jog", 9, 1)


# --- dsp ------------------------------------------------------------------


@pytest.mark.parametrize("n_seconds", [15.3, 23.9, 30.0, 47.0, 95.0, 110.7])
def test_welch_segments_tile_the_record(n_seconds):
    """With a fixed 16 s segment scipy.signal.welch silently dropped up to 7 s
    (30%) of a trial. The segment length now stretches so that k segments at
    50% overlap cover the whole record."""
    fs = 50.0
    n = int(round(n_seconds * fs))
    nps = dsp._nperseg(n, fs)
    half = nps // 2
    k = (n - half) // (nps - half)
    covered = half + k * (nps - half)
    assert n - covered <= max(2, nps // 20), (n, nps, covered)


# --- scripts ---------------------------------------------------------------


def test_verdict_summary_survives_a_trial_that_raised():
    """One trial raising produced a row of NaNs that crashed print_verdicts
    after the CSVs were written, so a single bad file killed the summary."""
    spec = importlib.util.spec_from_file_location("run_all", REPO / "scripts" / "run_all.py")
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)
    good = pd.read_csv(REPO / "reports" / "pipeline_results.csv")
    bad = pd.DataFrame([{"activity": "jog", "trial": 9, "subject": 99, "error": "ValueError: boom"}])
    ran, failed = run_all.split_failed(pd.concat([good, bad], ignore_index=True))
    assert len(failed) == 1 and len(ran) == len(good)
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_all.print_verdicts(ran)
    assert "VERDICT SUMMARY" in buf.getvalue()


# --- stage 4: what the null can and cannot show -----------------------------


def test_phase_randomised_null_does_not_preserve_step_locking():
    """Pins a LIMITATION so it is never read as a result: a pure stride-rate
    sinusoid locked to the step markers -- no laterality information by
    construction -- beats the sweep-null p95 on most runs, because phase
    randomisation destroys the step-lock the real signal has. 'Beats the
    null' therefore shows step-locked periodicity, not laterality."""
    fs, f = 50.0, 2.75
    t = _t(60.0, fs)
    beats = 0
    n_runs = 6
    for seed in range(n_runs):
        rng = np.random.default_rng(seed)
        omega = np.sin(2 * np.pi * (f / 2) * t + rng.uniform(0, 2 * np.pi)) + 0.5 * rng.normal(size=len(t))
        idx = np.round(np.cumsum(rng.normal(fs / f, 0.03 * fs / f, size=int(60 * f) - 2))).astype(int) + 10
        idx = idx[idx < len(t) - 30]
        gyro = np.column_stack([np.zeros_like(t), np.zeros_like(t), omega])
        lat = lateral.analyse(gyro, np.tile([0.0, 0.0, -1.0], (len(t), 1)), idx, fs, f, seed=seed)
        beats += int(lat["best_phase_beats_surrogate_p95"])
    assert beats >= 4, f"{beats}/{n_runs} -- if this drops, the null has changed; re-read REPORT.md section 5"
