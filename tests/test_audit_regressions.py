"""Regression tests for the findings of the September 2026 adversarial audit.

Each test is named for the failure it pins. The synthetic sessions reuse
`tests/test_integration.py`'s helpers so they go through the app's own file
format and the full pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import dsp, lateral, loader, orientation, pipeline, steps  # noqa: E402
from test_integration import (  # noqa: E402
    F_STEP_HZ,
    FS_HZ,
    synthetic_anatomical,
    write_logger_session,
)


def _still(seconds: float, seed: int = 3, noise_g: float = 0.02) -> dict:
    """Standing still: sensor noise on gravity, nothing else."""
    n = int(seconds * FS_HZ)
    rng = np.random.default_rng(seed)
    return {
        "accel": noise_g * rng.normal(size=(n, 3)),
        "gravity": np.tile([0.0, 0.0, -1.0], (n, 1)),
        "gyro": 0.01 * rng.normal(size=(n, 3)),
    }


def _join(blocks: list, cadence_spm: float = 60.0 * F_STEP_HZ) -> dict:
    out = {k: np.vstack([b[k] for b in blocks]) for k in ("accel", "gravity", "gyro")}
    out["t"] = np.arange(len(out["accel"])) / FS_HZ
    out["forward"] = np.array([1.0, 0.0, 0.0])
    out["up"] = np.array([0.0, 0.0, 1.0])
    out["cadence_spm"] = cadence_spm
    return out


# --- polarity ---------------------------------------------------------------


def test_loader_converts_coremotion_sign_to_kinematic(tmp_path):
    """The file holds CoreMotion's sign; `user_accel` is kinematic."""
    anat = synthetic_anatomical(seconds=20.0)
    R = dsp.random_rotation_matrix(4)
    folder = write_logger_session(tmp_path / "pol", anat, R, jitter_s=0.0)
    tr = loader.load_logger_session(folder)
    assert np.allclose(tr.user_accel, anat["accel"] @ R.T, atol=2e-6)
    assert np.allclose(tr.user_accel_as_recorded, -(anat["accel"] @ R.T), atol=2e-6)
    a_vert = orientation.vertical_component(tr.user_accel, tr.gravity)
    assert np.allclose(a_vert, anat["accel"][:, 2], atol=2e-6)


def test_vertical_reads_minus_one_g_in_free_fall_on_real_data():
    """FIXED: read +0.99 g on 48/48 jog trials before the sign was converted.

    |userAcceleration + gravity| is the hardware reading in Apple's sign;
    near zero it is free fall, when the body accelerates 1 g toward the
    earth, so the kinematic vertical component must be about -1 g there.
    """
    tested = 0
    for trial, subject in [(9, 1), (9, 4), (16, 5), (16, 24)]:
        tr = loader.load_trial("jog", trial, subject)
        seg = steps.steady_state_segment(np.linalg.norm(tr.user_accel, axis=1), tr.fs_hz)
        sl = slice(seg["start"], seg["stop"])
        raw = tr.user_accel_as_recorded[sl] + tr.gravity[sl]
        # |raw| < 0.35 g bounds the kinematic vertical to [-1.35, -0.65] g.
        near_free_fall = np.linalg.norm(raw, axis=1) < 0.35
        if near_free_fall.sum() < 10:
            continue  # a thigh is rarely in free fall; some trials have too few samples
        tested += 1
        a_vert = orientation.vertical_component(tr.user_accel[sl], tr.gravity[sl])
        assert a_vert[near_free_fall].mean() < -0.6, (trial, subject, a_vert[near_free_fall].mean())
    assert tested >= 2


# --- steady-state segmentation ---------------------------------------------


def test_standing_majority_record_yields_no_fabricated_steps(tmp_path):
    """CRITICAL, FIXED: 10 min standing + 5 min running used to pass as
    'ok' with ~1700 steps detected during the standing."""
    still_s, run_s = 120.0, 60.0
    joined = _join([_still(still_s), synthetic_anatomical(seconds=run_s, seed=2), _still(5.0, seed=5)])
    folder = write_logger_session(tmp_path / "standing", joined, np.eye(3), jitter_s=0.0)
    result = pipeline.run_session(folder)
    seg = result["segment"]
    assert seg["no_motion"] is False
    assert abs(seg["start"] / FS_HZ - still_s) < 1.5, "the standing was not cut"
    ts = result["detection"]["step_times_s"]
    assert (ts < still_s - 1.0).sum() == 0, "steps detected while standing"
    assert abs(result["cadence_summary"]["cadence_spm"] - joined["cadence_spm"]) < 1.0
    assert result["quality"]["verdict"] == "ok", result["quality"]["summary"]


def test_record_with_no_locomotion_is_insufficient(tmp_path):
    joined = _join([_still(60.0)])
    folder = write_logger_session(tmp_path / "nothing", joined, np.eye(3), jitter_s=0.0)
    result = pipeline.run_session(folder)
    assert result["segment"]["no_motion"] is True
    q = result["quality"]
    assert q["verdict"] == "insufficient"
    assert any("no sustained motion" in b for b in q["blockers"]), q["blockers"]


def test_steady_state_returns_the_same_keys_on_every_path():
    fs = 100.0
    keys = None
    for mag in (
        np.full(100, 0.05),  # too short to segment
        np.full(3000, 0.05),  # no motion
        np.r_[np.full(300, 0.05), np.full(3000, 1.0), np.full(300, 0.05)],  # one bout
    ):
        seg = steps.steady_state_segment(mag, fs)
        if keys is None:
            keys = set(seg) - {"window_rms"}
        assert set(seg) - {"window_rms"} == keys


# --- cadence statistics ----------------------------------------------------


def test_cadence_summary_is_parity_invariant():
    """FIXED: a perfectly regular 0.30/0.42 s alternation reported a median
    cadence of 166.7 or 200.0 spm and 0.00 or 0.49 irregular steps
    depending on whether the last interval was a short or a long one."""
    results = []
    for n_int in (40, 41, 42, 43):
        t = np.cumsum([0.0] + [0.30, 0.42] * (n_int // 2) + ([0.30] if n_int % 2 else []))
        cs = steps.cadence_summary(t)
        results.append((cs["cadence_spm"], cs["cadence_spm_median"], cs["irregular_stride_fraction"], cs["cadence_cv"]))
    for cadence, median, irregular, cv in results:
        assert abs(cadence - 60.0 / 0.36) < 0.01
        assert abs(median - 60.0 / 0.36) < 0.01
        assert irregular == 0.0
        assert cv < 1e-9, "stride CV must not see the step-interval alternation"
    # ...and the step-level alternation is still reported as a diagnostic.
    assert abs(steps.cadence_summary(np.cumsum([0.0] + [0.30, 0.42] * 20))["alternating_interval_asymmetry_abs_pct"] - 33.3) < 0.5


def test_bridged_standstill_does_not_bias_cadence():
    """FIXED: a 2.5 s standstill inside an 80 s bout read 163.6 for 168."""
    fs, f = 100.0, 2.8
    rng = np.random.default_rng(5)
    t1 = np.arange(int(40 * fs)) / fs
    run = np.sin(2 * np.pi * f * t1) + 0.4 * np.sin(2 * np.pi * 2 * f * t1)
    x = np.r_[run, np.zeros(250), run] + 0.03 * rng.normal(size=2 * len(run) + 250)
    det = steps.detect_steps(x, fs, f_step_hz=f)
    cs = steps.cadence_summary(det["step_times_s"])
    assert abs(cs["cadence_spm"] - 60 * f) < 0.5, cs["cadence_spm"]
    assert cs["cadence_spm_span"] < cs["cadence_spm"] - 3.0, "the span rate should still show the bias"
    assert cs["cadence_cv"] < 0.05


def test_cadence_series_edges_are_nan_not_sawtooth():
    times = np.cumsum([0.0] + [0.30, 0.42] * 20)
    df = steps.cadence_series(times, smooth_steps=6)
    sm = df["cadence_spm_smooth"].to_numpy()
    assert np.isnan(sm[:3]).all() and np.isnan(sm[-2:]).all()
    assert np.nanstd(sm) < 1e-6


def test_short_bout_with_nan_cadence_is_insufficient_not_partial(tmp_path):
    """FIXED: a 5.2 s bout clears MIN_STEADY_SECONDS, its step span does not,
    and the NaN cadence used to come out as a 'partial' caveat."""
    joined = _join([_still(10.0), synthetic_anatomical(seconds=5.2, seed=2), _still(10.0, seed=6)])
    folder = write_logger_session(tmp_path / "shortbout", joined, np.eye(3), jitter_s=0.0)
    result = pipeline.run_session(folder)
    assert not result["steady_too_short"]
    assert np.isnan(result["cadence_summary"]["cadence_spm"])
    q = result["quality"]
    assert q["verdict"] == "insufficient", q["summary"]
    assert any("no defensible cadence" in b for b in q["blockers"]), q["blockers"]


# --- bouts and gaits --------------------------------------------------------


def test_walk_merged_into_run_is_attributed_to_the_record_not_the_detector(tmp_path):
    """FIXED: a walking cool-down merged into the running bout produced a
    mixed cadence that was blamed on 'step detector failure'."""
    run = synthetic_anatomical(seconds=60.0, f_step_hz=2.8, seed=2)
    walk = synthetic_anatomical(seconds=60.0, f_step_hz=1.9, seed=4, amplitude=0.6)
    joined = _join([_still(5.0), run, walk, _still(5.0, seed=6)], cadence_spm=168.0)
    folder = write_logger_session(tmp_path / "walkrun", joined, np.eye(3), jitter_s=0.0)
    result = pipeline.run_session(folder)
    assert result["segment"]["n_segments"] == 1, "the walk must be inside the bout for this test"
    cd = result["cadence_diagnosis"]
    assert cd["mixed_gait"] is True
    assert cd["failure_attributed_to"] == "trial"
    q = result["quality"]
    assert q["verdict"] == "insufficient"
    assert any("more than one gait" in b for b in q["blockers"]), q["blockers"]
    assert not any("step detector failed" in b for b in q["blockers"]), q["blockers"]


def test_bouts_with_different_gaits_are_not_pooled_into_one_cadence(tmp_path):
    run = synthetic_anatomical(seconds=60.0, f_step_hz=2.8, seed=2)
    walk = synthetic_anatomical(seconds=60.0, f_step_hz=1.9, seed=4, amplitude=0.7)
    joined = _join([_still(5.0), run, _still(20.0, seed=7), walk, _still(5.0, seed=6)])
    folder = write_logger_session(tmp_path / "runstopwalk", joined, np.eye(3), jitter_s=0.0)
    result = pipeline.run_session(folder)
    pooled = result["cadence_summary_all_bouts"]
    assert pooled["n_bouts_analysed"] == 2
    assert pooled["bout_cadence_spread"] > steps.MIXED_CADENCE_SPREAD
    q = result["quality"]
    assert q["verdict"] == "insufficient"
    assert any("bouts differ in cadence" in b for b in q["blockers"]), q["blockers"]


def test_steady_running_is_not_called_mixed_gait(tmp_path):
    anat = synthetic_anatomical(seconds=90.0, step_interval_cv=0.04)
    folder = write_logger_session(tmp_path / "steady", anat, np.eye(3))
    result = pipeline.run_session(folder)
    cd = result["cadence_diagnosis"]
    assert cd["mixed_gait"] is False
    assert cd["cadence_spread"] < steps.MIXED_CADENCE_SPREAD
    assert result["quality"]["verdict"] == "ok", result["quality"]["summary"]


# --- spectral seed and sample rate -----------------------------------------


def test_harmonic_lock_on_is_flagged_not_attributed_to_the_runner():
    """FIXED: a slow jogger whose second harmonic beat the fundamental was
    reported at double cadence and the number attributed to the runner."""
    fs = 100.0
    rng = np.random.default_rng(9)
    t = np.arange(int(60 * fs)) / fs
    x = np.sin(2 * np.pi * 2.1 * t) + 1.2 * np.sin(2 * np.pi * 4.2 * t + 0.5) + 0.05 * rng.normal(size=t.size)
    spec = steps.estimate_step_frequency(x, fs)
    assert spec["harmonic_ambiguous"] is True
    assert abs(spec["alternative_f_step_hz"] - 2.1) < 0.1
    det = steps.detect_steps(x, fs, f_step_hz=spec["f_step_hz"])
    cs = steps.cadence_summary(det["step_times_s"])
    diag = steps.diagnose_cadence(
        cs["cadence_spm"], spec["cadence_spm"], 0.9, cs["irregular_stride_fraction"],
        harmonic_ambiguous=spec["harmonic_ambiguous"],
        subharmonic_power_ratio=spec["subharmonic_power_ratio"],
    )
    assert diag["failure_attributed_to"] == "algorithm"
    assert "ambiguous" in diag["diagnosis"]


def test_a_running_spectrum_is_not_called_ambiguous():
    fs = 100.0
    rng = np.random.default_rng(9)
    t = np.arange(int(60 * fs)) / fs
    x = (np.sin(2 * np.pi * 2.8 * t) + 0.4 * np.sin(2 * np.pi * 5.6 * t + 0.7)
         + 0.15 * np.sin(2 * np.pi * 1.4 * t) + 0.05 * rng.normal(size=t.size))
    spec = steps.estimate_step_frequency(x, fs)
    assert spec["harmonic_ambiguous"] is False
    assert spec["subharmonic_power_ratio"] < 0.1


def test_overstated_rate_below_the_spectral_blind_spot_is_not_blamed_on_the_runner():
    """FIXED: 50 Hz data claimed at 75 Hz passed the spectral rate check and
    the resulting 252 spm was attributed to the runner's cadence."""
    rng = np.random.default_rng(2)
    t = np.arange(int(60 * 50.0)) / 50.0
    x = np.sin(2 * np.pi * 2.8 * t) + 0.4 * np.sin(2 * np.pi * 5.6 * t + 0.7) + 0.05 * rng.normal(size=t.size)
    claimed = 75.0
    rc = steps.check_sample_rate(x, claimed)
    assert rc["sample_rate_plausible"] is True, "the blind spot this test documents has moved"
    spec = steps.estimate_step_frequency(x, claimed)
    det = steps.detect_steps(x, claimed, f_step_hz=spec["f_step_hz"])
    cs = steps.cadence_summary(det["step_times_s"])
    assert cs["cadence_spm"] > steps.PLAUSIBLE_CADENCE_MAX_SPM
    diag = steps.diagnose_cadence(
        cs["cadence_spm"], spec["cadence_spm"], 0.9, cs["irregular_stride_fraction"],
        sample_rate_plausible=rc["sample_rate_plausible"],
    )
    assert diag["failure_attributed_to"] == "sample_rate"


# --- orientation -----------------------------------------------------------


def test_symmetry_index_is_not_inflated_by_the_band_edge():
    """FIXED: the autocorrelation band's low edge sat ON the stride
    subharmonic and filtfilt removed half its power, so a 0.8-amplitude
    subharmonic read 0.74 where the raw signal reads 0.22."""
    fs, f = 100.0, 2.8
    t = np.arange(int(60 * fs)) / fs
    for amp, expected in ((0.5, 0.60), (0.8, 0.22)):
        x = np.cos(2 * np.pi * f * t) + amp * np.cos(np.pi * f * t + 0.3)
        per = orientation.verify_vertical_periodicity(x, fs)
        assert abs(per["step_symmetry_index"] - expected) < 0.05, (amp, per["step_symmetry_index"])
        assert per["stride_regularity"] > 0.95


def test_forward_sign_confidence_does_not_depend_on_the_null_impact_criterion():
    """FIXED: the gate required the impact statistic, which is ~0 by
    construction at the centre of mass, so 'confident' was unreachable at
    the production placement and 'criteria agree' was a coin flip."""
    anat = synthetic_anatomical()
    confident = []
    for seed in range(6):
        R = dsp.random_rotation_matrix(seed)
        frame = orientation.build_frame(anat["accel"] @ R.T, anat["gravity"] @ R.T, FS_HZ)
        confident.append(frame.diagnostics["forward_sign_confident"])
        assert dsp.angle_between_deg(frame.forward, R @ anat["forward"]) < 3.0
    assert all(confident)


def test_unresolved_forward_sign_is_a_caveat_when_the_axes_are_ok(tmp_path):
    anat = synthetic_anatomical()
    folder = write_logger_session(tmp_path / "sign", anat, np.eye(3))
    result = pipeline.run_session(folder)
    assert result["verify"]["verdict"] == "ok"
    assert result["quality"]["verdict"] == "ok"
    result["frame"].diagnostics["forward_sign_confident"] = False
    q = pipeline.assess_quality(result)
    assert q["verdict"] == "partial"
    assert any("undirected" in c for c in q["caveats"]), q["caveats"]


def test_autocorrelation_regularity_does_not_depend_on_record_length():
    fs, f = 100.0, 2.7
    for seconds in (10.0, 60.0):
        t = np.arange(int(seconds * fs)) / fs
        per = orientation.verify_vertical_periodicity(np.sin(2 * np.pi * f * t), fs)
        assert per["stride_regularity"] > 0.97, (seconds, per["stride_regularity"])
        assert abs(per["step_symmetry_index"] - 1.0) < 0.02, (seconds, per["step_symmetry_index"])


# --- stage 4 -----------------------------------------------------------------


def test_laterality_free_locked_signal_earns_no_side_verdict(tmp_path):
    """A stride-rate oscillation that is a pure function of the step times
    (no left/right content) beats the phase-randomised surrogate once the
    steps jitter like real ones. That is why there is no side verdict."""
    anat = synthetic_anatomical(seconds=90.0, step_interval_cv=0.12, seed=1)
    folder = write_logger_session(tmp_path / "locked", anat, np.eye(3))
    result = pipeline.run_session(folder)
    lat = result["lateral"]
    assert lat["alternation_consistency"] > 0.95
    assert lat["excess_over_surrogate"] > 0.10, "the finding this test documents no longer reproduces"
    q = result["quality"]
    assert q["side_classification"].startswith("not classifiable")
    assert "consistent" not in q["side_classification"]
    assert q["verdict"] in ("ok", "partial")


def test_alternation_rate_cannot_see_a_parity_slip_and_the_run_length_can_tell_it_from_a_glitch():
    """One wrong label and forty wrong labels (a parity slip at the midpoint)
    read the same alternation rate. The run-length diagnostic separates the
    two -- a glitch is a run of three, a slip a run of two -- but nothing
    signal-only can tell a slip from a correct sequence, which is why the
    labels are documented as a parity sequence that must never be
    aggregated per side."""
    base = np.array([1.0, -1.0] * 40)
    one = base.copy(); one[20] *= -1  # one wrong label
    forty = base.copy(); forty[20:60] *= -1  # forty wrong labels: a slip and a slip back
    assert lateral.alternation_rate(one) == lateral.alternation_rate(forty)
    assert lateral.longest_same_sign_run(base) == 1
    assert lateral.longest_same_sign_run(one) == 3
    assert lateral.longest_same_sign_run(forty) == 2
    s = np.sign(forty)
    assert int(np.sum(s[1:] * s[:-1] >= 0)) == 2


def test_surrogate_keeps_the_sign_of_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000) - 0.7
    s = lateral.phase_randomised_surrogate(x, rng)
    assert s.mean() < 0
    assert abs(s.mean() - x.mean()) < 1e-9
    assert np.allclose(np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(s)), rtol=1e-8)


def test_cluster_d_excess_is_zero_for_gaussian_noise():
    rng = np.random.default_rng(1)
    v = rng.normal(size=200000)
    a, b = v[v >= 0], v[v < 0]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = abs(a.mean() - b.mean()) / pooled
    assert abs(d - lateral.SIGN_SPLIT_D_GAUSSIAN) < 0.02


# --- integration: the app's own bookkeeping ----------------------------------


def test_rows_lost_to_disk_block_the_run(tmp_path):
    anat = synthetic_anatomical(seconds=30.0)
    folder = write_logger_session(tmp_path / "lost", anat, np.eye(3))
    meta = json.loads((folder / "session.json").read_text())
    meta["csvRowsLost"] = 500
    (folder / "session.json").write_text(json.dumps(meta))
    result = pipeline.run_session(folder)
    assert result["trial"].integrity["metadata_rows_lost"] == 500
    assert not result["trial"].integrity["clean"]
    q = result["quality"]
    assert q["verdict"] == "insufficient"
    assert any("never reached disk" in b for b in q["blockers"]), q["blockers"]


def test_sample_count_mismatch_blocks_the_run(tmp_path):
    anat = synthetic_anatomical(seconds=30.0)
    folder = write_logger_session(tmp_path / "count", anat, np.eye(3))
    meta = json.loads((folder / "session.json").read_text())
    meta["motionSampleCount"] += 300
    (folder / "session.json").write_text(json.dumps(meta))
    q = pipeline.run_session(folder)["quality"]
    assert q["verdict"] == "insufficient"
    assert any("truncated" in b for b in q["blockers"]), q["blockers"]


def test_hardware_clock_keeps_the_apps_origin_and_markers_align(tmp_path):
    """FIXED: t_hw was re-zeroed on the first sample and the offset thrown
    away, shifting derived times off the marker and GPS timeline."""
    anat = synthetic_anatomical(seconds=40.0)
    folder = write_logger_session(
        tmp_path / "marker", anat, np.eye(3), jitter_s=0.0,
        event_markers=[{"t": 30.0, "note": "cone"}],
    )
    # The first motion sample arrives 0.35 s after Start, as on a phone.
    lines = (folder / "motion.csv").read_text().splitlines()
    body = [",".join([f"{float(r.split(',')[0]) + 0.35:.6f}"] + r.split(",")[1:]) for r in lines[1:]]
    (folder / "motion.csv").write_text("\n".join([lines[0]] + body) + "\n")
    tr = loader.load_logger_session(folder)
    assert abs(tr.integrity["t_start_s"] - 0.35) < 1e-6
    assert abs(tr.df["t_hw"].iloc[0] - 0.35) < 1e-6
    assert tr.integrity["metadata_event_markers"][0]["t"] == 30.0
    result = pipeline.run_stages(tr)
    ts = result["detection"]["step_times_s"]
    assert result["detection"]["step_times_from_hardware_clock"] is True
    # The marker at t=30 on the app's clock lands within one step of a
    # detected step on the same clock; on the re-zeroed index it would
    # sit 0.35 s (a step period) off.
    assert np.min(np.abs(ts - 30.0)) < 1.0 / F_STEP_HZ


def test_dropped_samples_do_not_drift_step_times(tmp_path):
    """Step times come from the hardware clock: 2% of samples dropped must
    leave the cadence right, where the index grid would read 2% high."""
    anat = synthetic_anatomical(seconds=60.0)
    folder = write_logger_session(tmp_path / "drift", anat, np.eye(3), jitter_s=0.0)
    lines = (folder / "motion.csv").read_text().splitlines()
    # Every 50th row (2%), never two adjacent, so each drop is a 2x delta
    # -- below the gap threshold -- and the record loads.
    drop = set(range(200, len(lines) - 200, 50))
    kept = [l for i, l in enumerate(lines) if i == 0 or i not in drop]
    (folder / "motion.csv").write_text("\n".join(kept) + "\n")
    tr = loader.load_logger_session(folder)
    assert tr.integrity["n_dropped_estimate"] > 50
    result = pipeline.run_stages(tr)
    cadence = result["cadence_summary"]["cadence_spm"]
    assert abs(cadence - anat["cadence_spm"]) < 1.0, cadence
    # Grid-based times would be compressed by the drops.
    grid = steps.detect_steps(result["a_vertical"], tr.fs_hz, f_step_hz=result["detection"]["f_step_hz_used"])
    assert steps.cadence_summary(grid["step_times_s"])["cadence_spm"] > anat["cadence_spm"] + 2.0
    # The quality gate still refuses a record losing this many samples.
    assert result["quality"]["verdict"] == "insufficient"


def test_duration_comes_from_the_hardware_clock_when_there_is_one(tmp_path):
    anat = synthetic_anatomical(seconds=30.0)
    folder = write_logger_session(tmp_path / "dur", anat, np.eye(3), jitter_s=0.0, gap_at_s=15.0, gap_len_s=2.0)
    tr = loader.load_logger_session(folder, on_gap="longest")
    assert abs(tr.duration_s - tr.integrity["duration_s"]) < 1e-9
    assert tr.duration_s < 16.0


def test_invalid_speed_accuracy_rows_are_dropped(tmp_path):
    anat = synthetic_anatomical(seconds=20.0)
    folder = write_logger_session(tmp_path / "spd", anat, np.eye(3), stale_gps_fix=False)
    lines = (folder / "gps.csv").read_text().splitlines()
    cols = lines[0].split(",")
    assert cols == loader.LOGGER_GPS_COLUMNS
    row = lines[5].split(",")
    row[cols.index("speedAccuracy")] = "-1.000"
    lines[5] = ",".join(row)
    (folder / "gps.csv").write_text("\n".join(lines) + "\n")
    tr = loader.load_logger_session(folder)
    assert tr.integrity["gps_dropped_invalid"] == 1
    assert tr.integrity["gps_kept"] == len(lines) - 2


def test_coarse_gps_is_a_caveat(tmp_path):
    anat = synthetic_anatomical()
    folder = write_logger_session(tmp_path / "coarse", anat, np.eye(3), gps_accuracy_m=1500.0)
    result = pipeline.run_session(folder)
    assert result["trial"].integrity["gps_accuracy_median_m"] > loader.GPS_COARSE_ACCURACY_M
    q = result["quality"]
    assert q["verdict"] == "partial", q["summary"]
    assert any("GPS fixes are coarse" in c for c in q["caveats"])
