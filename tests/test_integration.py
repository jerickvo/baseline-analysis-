"""Integration: the app's own output through the pipeline, against ground truth.

The MotionSense tests check the pipeline on real pocket data where nothing
is known for certain. These tests check it on SYNTHETIC data where
everything is known -- the true anatomical axes, the true cadence, the true
sample rate, the true gaps -- written in the exact CSV format BaselineLogger
produces, so the two halves of the system are proven to connect.

Run with `pytest tests/`. Does not need the MotionSense dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import dsp, loader, orientation, pipeline, steps  # noqa: E402

FS_HZ = 100.0
F_STEP_HZ = 2.8
DURATION_S = 90.0


# --- a synthetic session in the logger's exact format ---------------------


def synthetic_anatomical(
    fs_hz: float = FS_HZ,
    seconds: float = DURATION_S,
    f_step_hz: float = F_STEP_HZ,
    seed: int = 0,
) -> dict:
    """Body-frame signals a lower-back sensor would see, plus their truth.

    The fore-aft channel is built with the centre-of-mass phase relation:
    braking (negative) while vertical acceleration rises, propulsion while it
    falls -- fore-aft leads vertical by about a quarter cycle. Mediolateral
    sway is at the stride rate, half the step rate. Angular velocity about
    vertical alternates sign every step.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs_hz)
    t = np.arange(n) / fs_hz
    w = 2 * np.pi * f_step_hz
    a_vert = 1.2 * np.sin(w * t) + 0.3 * np.sin(2 * w * t) + 0.05 * rng.normal(size=n)
    a_fwd = -0.5 * np.sin(w * t + 1.0) + 0.05 * rng.normal(size=n)
    a_ml = 0.2 * np.sin(w * t / 2) + 0.05 * rng.normal(size=n)
    omega_v = 1.5 * np.sin(w * t / 2) + 0.05 * rng.normal(size=n)
    return {
        "t": t,
        "accel": np.c_[a_fwd, a_ml, a_vert],  # columns: forward, ML (left), up
        "gravity": np.tile([0.0, 0.0, -1.0], (n, 1)),
        "gyro": np.c_[np.zeros(n), np.zeros(n), omega_v],
        "forward": np.array([1.0, 0.0, 0.0]),
        "up": np.array([0.0, 0.0, 1.0]),
        "cadence_spm": 60.0 * f_step_hz,
    }


def write_logger_session(
    folder: Path,
    anat: dict,
    rotation: np.ndarray,
    jitter_s: float | None = None,
    gap_at_s: float | None = None,
    gap_len_s: float = 0.0,
    stale_gps_fix: bool = True,
    seed: int = 1,
) -> Path:
    """Rotate the anatomical signals into an arbitrary device frame and
    write motion.csv / accel_raw.csv / gps.csv / session.json exactly as
    SessionRecorder.swift does (same headers, same %.6f formatting)."""
    rng = np.random.default_rng(seed)
    R = np.asarray(rotation, float)
    n = len(anat["t"])
    # Hardware timestamp jitter scales with the sample period; 2% of it is
    # generous for CoreMotion, which sits well under 1%.
    period = float(np.median(np.diff(anat["t"])))
    if jitter_s is None:
        jitter_s = 0.02 * period
    t = anat["t"] + rng.normal(0, jitter_s, n)
    t = np.maximum.accumulate(t)  # hardware clocks do not run backwards
    if gap_at_s is not None:
        t[t >= gap_at_s] += gap_len_s
    dev_a = anat["accel"] @ R.T
    dev_g = anat["gravity"] @ R.T
    dev_w = anat["gyro"] @ R.T
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "motion.csv", "w") as f:
        f.write("t,ax,ay,az,gx,gy,gz,rx,ry,rz,qw,qx,qy,qz\n")
        for i in range(n):
            f.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (
                t[i], *dev_a[i], *dev_g[i], *dev_w[i], 1.0, 0.0, 0.0, 0.0))
    with open(folder / "accel_raw.csv", "w") as f:
        f.write("t,ax,ay,az\n")
    with open(folder / "gps.csv", "w") as f:
        f.write("t,latitude,longitude,speed,horizontalAccuracy,altitude\n")
        if stale_gps_fix:
            # CoreLocation's cached last-known fix, stamped before the session.
            f.write("-812.301000,37.42000000,-122.08000000,-1.000,65.00,10.00\n")
        for k in range(int(anat["t"][-1])):
            f.write("%.6f,%.8f,%.8f,%.3f,%.2f,%.2f\n" % (k, 37.42 + k * 1e-5, -122.08, 3.3, 5.0, 10.0))
    meta = {
        "id": "synthetic", "label": "synthetic 3mi",
        "startTime": "2026-09-03T12:00:00Z", "endTime": "2026-09-03T12:01:30Z",
        "durationSeconds": float(t[-1] - t[0]),
        "motionSampleCount": n, "accelSampleCount": 0, "gpsSampleCount": int(anat["t"][-1]) + int(stale_gps_fix),
        "achievedMotionHz": n / float(t[-1] - t[0]), "achievedAccelHz": 0.0,
        "deviceModel": "synthetic", "iosVersion": "16.0",
        "motionGapCount": int(gap_at_s is not None), "accelGapCount": 0,
        "largestGapSeconds": float(gap_len_s), "eventMarkers": [], "inProgress": False,
    }
    with open(folder / "session.json", "w") as f:
        json.dump(meta, f, indent=1)
    return folder


def _rotations():
    """Arbitrary, upside-down, and yawed-180 placements."""
    yield "identity", np.eye(3)
    yield "upside_down", dsp.rotation_matrix_from_axis_angle([1, 0, 0], np.pi)
    yield "yaw_180", dsp.rotation_matrix_from_axis_angle([0, 0, 1], np.pi)
    yield "yaw_90", dsp.rotation_matrix_from_axis_angle([0, 0, 1], np.pi / 2)
    for seed in range(4):
        yield f"random_{seed}", dsp.random_rotation_matrix(seed)


# --- the loader ------------------------------------------------------------


def test_loader_ingests_the_apps_own_format(tmp_path):
    anat = synthetic_anatomical()
    folder = write_logger_session(tmp_path / "20260903T120000Z", anat, np.eye(3))
    tr = loader.load_logger_session(folder)
    assert tr.ident.session_id == "20260903T120000Z"
    assert tr.ident.label == "synthetic 3mi"
    assert list(tr.df.columns[:12]) == loader.DEVICE_MOTION_COLUMNS
    assert "t_hw" in tr.df.columns
    assert tr.n_samples == len(anat["t"])
    assert tr.integrity["clean"], tr.integrity["problems"]


def test_loader_measures_the_rate_instead_of_assuming_it(tmp_path):
    for fs in (50.0, 100.0, 200.0):
        anat = synthetic_anatomical(fs_hz=fs, seconds=30.0)
        folder = write_logger_session(tmp_path / f"fs{int(fs)}", anat, np.eye(3))
        tr = loader.load_logger_session(folder)
        assert abs(tr.fs_hz - fs) / fs < 0.002, f"measured {tr.fs_hz} for true {fs}"
        assert tr.integrity["has_timestamp_column"] is True
        assert tr.integrity["timestamp_irregularities_detectable"] is True
        assert tr.integrity["timestamps_uniform"] is True
        assert 0.0 < tr.integrity["jitter_ms"] < 1.0


def test_loader_refuses_a_gapped_session_by_default(tmp_path):
    anat = synthetic_anatomical()
    folder = write_logger_session(tmp_path / "gapped", anat, np.eye(3), gap_at_s=40.0, gap_len_s=0.8)
    with pytest.raises(ValueError, match="gap"):
        loader.load_logger_session(folder)


def test_loader_can_keep_the_longest_clean_stretch_and_says_what_it_cut(tmp_path):
    anat = synthetic_anatomical()
    folder = write_logger_session(tmp_path / "gapped", anat, np.eye(3), gap_at_s=40.0, gap_len_s=0.8)
    tr = loader.load_logger_session(folder, on_gap="longest")
    assert tr.integrity["n_gaps"] == 0
    # 90 s record, gap at 40 s: the longer piece is 50 s, ~40 s discarded.
    assert 38.0 < tr.integrity["discarded_for_gaps_s"] < 42.0
    assert 48.0 < tr.duration_s < 52.0
    assert not tr.integrity["clean"]  # a cut record is not a clean record


def test_loader_detects_duplicated_and_reordered_timestamps(tmp_path):
    anat = synthetic_anatomical(seconds=20.0)
    folder = write_logger_session(tmp_path / "dup", anat, np.eye(3), jitter_s=0.0)
    lines = (folder / "motion.csv").read_text().splitlines()
    # duplicate one row's timestamp onto the next
    a = lines[500].split(","); b = lines[501].split(",")
    b[0] = a[0]
    lines[501] = ",".join(b)
    (folder / "motion.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="non-monotonic"):
        loader.load_logger_session(folder)


def test_loader_counts_dropped_samples_the_gap_rule_cannot_see(tmp_path):
    """One missing sample is a 2x interval: below the 3x gap threshold."""
    anat = synthetic_anatomical(seconds=30.0)
    folder = write_logger_session(tmp_path / "dropped", anat, np.eye(3), jitter_s=0.0)
    lines = (folder / "motion.csv").read_text().splitlines()
    # remove five isolated rows: five single-sample drops, zero gaps
    for i in (2500, 1900, 1300, 800, 400):
        del lines[i]
    (folder / "motion.csv").write_text("\n".join(lines) + "\n")
    meta = json.loads((folder / "session.json").read_text())
    meta["motionSampleCount"] -= 5
    (folder / "session.json").write_text(json.dumps(meta))
    tr = loader.load_logger_session(folder)  # no gap, so it loads
    assert tr.integrity["n_gaps"] == 0
    assert tr.integrity["n_dropped_estimate"] == 5
    assert not tr.integrity["clean"]
    assert "dropped" in tr.integrity["problems"]


def test_loader_drops_the_stale_cached_gps_fix(tmp_path):
    anat = synthetic_anatomical(seconds=20.0)
    folder = write_logger_session(tmp_path / "gps", anat, np.eye(3), stale_gps_fix=True)
    tr = loader.load_logger_session(folder)
    assert tr.integrity["gps_dropped_stale"] == 1
    assert tr.gps is not None and (tr.gps["t"] >= 0).all()


def test_loader_flags_an_unfinished_session(tmp_path):
    anat = synthetic_anatomical(seconds=20.0)
    folder = write_logger_session(tmp_path / "unfinished", anat, np.eye(3))
    meta = json.loads((folder / "session.json").read_text())
    meta["inProgress"] = True
    (folder / "session.json").write_text(json.dumps(meta))
    tr = loader.load_logger_session(folder)
    assert tr.integrity["metadata_in_progress"] is True
    assert not tr.integrity["clean"]
    assert "in-progress" in tr.integrity["problems"]


def test_loader_cross_checks_the_apps_sample_count(tmp_path):
    anat = synthetic_anatomical(seconds=20.0)
    folder = write_logger_session(tmp_path / "count", anat, np.eye(3))
    assert loader.load_logger_session(folder).integrity["metadata_count_matches"] is True
    meta = json.loads((folder / "session.json").read_text())
    meta["motionSampleCount"] += 7
    (folder / "session.json").write_text(json.dumps(meta))
    assert loader.load_logger_session(folder).integrity["metadata_count_matches"] is False


def test_loader_rejects_the_wrong_format_loudly(tmp_path):
    folder = tmp_path / "wrong"
    folder.mkdir()
    (folder / "motion.csv").write_text("time,x,y,z\n0,1,2,3\n")
    with pytest.raises(ValueError, match="missing logger columns"):
        loader.load_logger_session(folder)


# --- frame recovery against ground truth ---------------------------------


@pytest.mark.parametrize("name,R", list(_rotations()))
def test_frame_recovers_the_true_axes_from_any_placement(name, R):
    """The whole point of stage 2, checked against a known answer.

    The device frame is the anatomical frame rotated by R, so a recovered
    axis expressed in device coordinates must equal R @ (true axis). Up is
    checked as a directed vector (gravity fixes its sign); forward is
    checked as a directed vector too, which is the sign-resolution test on
    a signal built with centre-of-mass phase relations.
    """
    anat = synthetic_anatomical()
    a = anat["accel"] @ R.T
    g = anat["gravity"] @ R.T
    frame = orientation.build_frame(a, g, FS_HZ)
    assert dsp.angle_between_deg(frame.up, R @ anat["up"]) < 0.5, name
    assert dsp.axis_angle_deg(frame.forward, R @ anat["forward"]) < 3.0, name
    assert frame.diagnostics["forward_sign_criteria_agree"], name
    assert frame.diagnostics["forward_sign_confident"], name
    assert dsp.angle_between_deg(frame.forward, R @ anat["forward"]) < 3.0, (
        f"{name}: forward sign resolved backwards"
    )
    # ML = up x forward points left; the synthetic ML sway is on +y.
    assert dsp.angle_between_deg(frame.mediolateral, R @ np.array([0.0, 1.0, 0.0])) < 3.0, name


def test_frame_verdict_is_ok_when_the_horizontal_split_really_holds():
    """With stride-rate ML and step-rate fore-aft, the cross-check passes."""
    anat = synthetic_anatomical()
    frame = orientation.build_frame(anat["accel"], anat["gravity"], FS_HZ)
    v = orientation.verify_frame(anat["accel"], anat["gravity"], frame, FS_HZ)
    assert v["verdict"] == "ok", v["reasons"]
    assert v["mediolateral_check"]["step_stride_axis_angle_deg"] > 80.0


def test_resolved_vertical_matches_the_true_vertical_channel():
    anat = synthetic_anatomical()
    R = dsp.random_rotation_matrix(5)
    a_vert = orientation.vertical_component(anat["accel"] @ R.T, anat["gravity"] @ R.T)
    assert np.allclose(a_vert, anat["accel"][:, 2], atol=1e-9)


# --- end to end ------------------------------------------------------------


@pytest.mark.parametrize("name,R", list(_rotations())[:5])
def test_end_to_end_cadence_from_a_logger_session(tmp_path, name, R):
    anat = synthetic_anatomical()
    folder = write_logger_session(tmp_path / name, anat, R)
    tr = loader.load_logger_session(folder)
    frame = orientation.build_frame(tr.user_accel, tr.gravity, tr.fs_hz)
    v = orientation.verify_frame(tr.user_accel, tr.gravity, frame, tr.fs_hz)
    det = steps.detect_steps(v["a_vertical"], tr.fs_hz, v["periodicity"]["f_step_hz"])
    cadence = steps.cadence_summary(det["step_times_s"])["cadence_spm"]
    assert abs(cadence - anat["cadence_spm"]) < 1.0, f"{name}: {cadence}"
    rate = steps.check_sample_rate(v["a_vertical"], tr.fs_hz)
    assert rate["sample_rate_plausible"] is True


# --- steady-state data loss is reported, not hidden ---------------------


def test_steady_state_reports_the_bout_it_discards():
    fs = 100.0
    rng = np.random.default_rng(1)

    def bout(sec):
        t = np.arange(int(sec * fs)) / fs
        return 1.5 * np.abs(np.sin(2 * np.pi * 2.8 * t)) + 0.3 + 0.05 * rng.normal(size=t.size)

    def still(sec):
        return 0.05 + 0.02 * rng.normal(size=int(sec * fs))

    mag = np.r_[still(5), bout(600), still(20), bout(400), still(5)]
    seg = steps.steady_state_segment(mag, fs)
    assert seg["n_segments"] == 2
    assert abs((seg["stop"] - seg["start"]) / fs - 600) < 2
    assert abs(seg["discarded_steady_s"] - 400) < 2, "the second bout is not being accounted for"


def test_single_bout_discards_nothing():
    fs = 100.0
    t = np.arange(int(60 * fs)) / fs
    mag = np.r_[np.full(300, 0.05), 1.5 * np.abs(np.sin(2 * np.pi * 2.8 * t)) + 0.3, np.full(300, 0.05)]
    seg = steps.steady_state_segment(mag, fs)
    assert seg["n_segments"] == 1
    assert seg["discarded_steady_s"] == 0.0


# --- the quality roll-up ----------------------------------------------------


def test_quality_is_ok_for_a_clean_session(tmp_path):
    anat = synthetic_anatomical()
    folder = write_logger_session(tmp_path / "clean", anat, dsp.random_rotation_matrix(2))
    result = pipeline.run_session(folder)
    q = result["quality"]
    assert q["verdict"] == "ok", q["summary"]
    assert q["blockers"] == [] and q["caveats"] == []
    row = pipeline.flatten(result)
    assert row["quality_verdict"] == "ok"
    assert abs(row["cadence_spm"] - anat["cadence_spm"]) < 1.0


def test_quality_is_insufficient_for_a_cut_record(tmp_path):
    anat = synthetic_anatomical()
    folder = write_logger_session(tmp_path / "gapped", anat, np.eye(3), gap_at_s=40.0, gap_len_s=0.8)
    result = pipeline.run_session(folder, on_gap="longest")
    assert result["quality"]["verdict"] == "insufficient"
    assert any("cut at gaps" in b for b in result["quality"]["blockers"])


def test_quality_is_insufficient_for_an_unfinished_session(tmp_path):
    anat = synthetic_anatomical(seconds=30.0)
    folder = write_logger_session(tmp_path / "unfinished", anat, np.eye(3))
    meta = json.loads((folder / "session.json").read_text())
    meta["inProgress"] = True
    (folder / "session.json").write_text(json.dumps(meta))
    result = pipeline.run_session(folder)
    assert result["quality"]["verdict"] == "insufficient"
    assert any("in-progress" in b for b in result["quality"]["blockers"])


def test_quality_is_partial_for_a_fragmented_run(tmp_path):
    """Two bouts separated by a stop: analysed, but flagged as a fraction."""
    a = synthetic_anatomical(seconds=60.0, seed=0)
    b = synthetic_anatomical(seconds=40.0, seed=1)
    n_still = int(20 * FS_HZ)
    still = {
        "accel": 0.02 * np.random.default_rng(3).normal(size=(n_still, 3)),
        "gravity": np.tile([0.0, 0.0, -1.0], (n_still, 1)),
        "gyro": np.zeros((n_still, 3)),
    }
    joined = {
        "accel": np.vstack([a["accel"], still["accel"], b["accel"]]),
        "gravity": np.vstack([a["gravity"], still["gravity"], b["gravity"]]),
        "gyro": np.vstack([a["gyro"], still["gyro"], b["gyro"]]),
        "forward": a["forward"], "up": a["up"], "cadence_spm": a["cadence_spm"],
    }
    joined["t"] = np.arange(len(joined["accel"])) / FS_HZ
    folder = write_logger_session(tmp_path / "fragmented", joined, np.eye(3))
    result = pipeline.run_session(folder)
    q = result["quality"]
    assert q["verdict"] == "partial", q["summary"]
    assert any("fragmented" in c for c in q["caveats"]), q["caveats"]
    assert result["segment"]["n_segments"] == 2
    assert 38.0 < result["segment"]["discarded_steady_s"] < 42.0


def test_quality_is_insufficient_when_the_rate_is_lied_about(tmp_path):
    """A logger session's rate is measured, so lie in the file itself."""
    anat = synthetic_anatomical(seconds=60.0)
    folder = write_logger_session(tmp_path / "lied", anat, np.eye(3), jitter_s=0.0)
    lines = (folder / "motion.csv").read_text().splitlines()
    head, body = lines[0], lines[1:]
    # rewrite t as if the phone had run at 400 Hz: the gait is now at 11 Hz
    body = [",".join([f"{i / 400.0:.6f}"] + row.split(",")[1:]) for i, row in enumerate(body)]
    (folder / "motion.csv").write_text("\n".join([head] + body) + "\n")
    result = pipeline.run_session(folder)
    assert result["cadence_diagnosis"]["failure_attributed_to"] == "sample_rate"
    assert result["quality"]["verdict"] == "insufficient"


def test_side_classification_is_never_a_gate(tmp_path):
    anat = synthetic_anatomical()
    folder = write_logger_session(tmp_path / "side", anat, np.eye(3))
    q = pipeline.run_session(folder)["quality"]
    # A stride-rate sinusoid alternates by construction, and so does its
    # surrogate: no information beyond the spectrum, so "unreliable" is the
    # honest label -- and it must not have blocked the run.
    assert q["side_classification"] == "unreliable"
    assert q["verdict"] == "ok"


# --- steady-state: dips are bridged, stops are not --------------------------


def _bouts(*parts, fs=100.0, seed=7):
    """parts: ('run', seconds) | ('still', seconds) -> acceleration magnitude."""
    rng = np.random.default_rng(seed)
    out = []
    for kind, sec in parts:
        n = int(sec * fs)
        if kind == "run":
            t = np.arange(n) / fs
            out.append(1.5 * np.abs(np.sin(2 * np.pi * 2.8 * t)) + 0.3 + 0.05 * rng.normal(size=n))
        else:
            out.append(0.05 + 0.02 * rng.normal(size=n))
    return np.concatenate(out)


def test_a_one_second_dip_does_not_split_a_continuous_run():
    """Regression: 14/48 MotionSense trials were halved by exactly this."""
    mag = _bouts(("still", 3), ("run", 40), ("still", 1), ("run", 40), ("still", 3))
    seg = steps.steady_state_segment(mag, 100.0)
    assert seg["n_segments"] == 1, seg["segments"]
    assert seg["discarded_steady_s"] == 0.0
    assert abs((seg["stop"] - seg["start"]) / 100.0 - 81) < 1.5


def test_a_two_second_dip_is_still_bridged():
    mag = _bouts(("still", 3), ("run", 40), ("still", 2), ("run", 40), ("still", 3))
    seg = steps.steady_state_segment(mag, 100.0)
    assert seg["n_segments"] == 1


def test_a_real_stop_still_splits_the_run():
    mag = _bouts(("still", 3), ("run", 40), ("still", 20), ("run", 30), ("still", 3))
    seg = steps.steady_state_segment(mag, 100.0)
    assert seg["n_segments"] == 2
    assert abs(seg["discarded_steady_s"] - 30) < 1.5


def test_leading_and_trailing_transients_are_never_bridged():
    """A 1 s handling blip at the start must still be trimmed, not kept."""
    mag = _bouts(("still", 1), ("run", 40), ("still", 1))
    seg = steps.steady_state_segment(mag, 100.0)
    assert seg["trimmed_start_s"] >= 1.0 - 1e-9
    assert seg["trimmed_end_s"] >= 1.0 - 1e-9
