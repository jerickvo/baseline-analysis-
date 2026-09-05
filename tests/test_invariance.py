"""Tests that the pipeline really is orientation- and rate-agnostic.

These are the checks that turn "designed to be placement- and rate-agnostic"
into a claim with evidence behind it:

* **Orientation invariance.** Rotate every sensor vector of a real trial by
  an arbitrary rotation -- exactly what a differently-oriented phone in the
  same pocket would produce -- and the pipeline must return the same steps,
  the same cadence and the same alternation. Anything that leaks the raw
  sensor axes into a result shows up here.
* **Sample-rate invariance.** Resample a real trial to other rates and the
  cadence must not move. Anything that hardcodes 50 Hz shows up here.

Run with `pytest tests/` or directly: `python tests/test_invariance.py`.
Set MOTIONSENSE_ROOT first (see scripts/fetch_data.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal as _signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import dsp, lateral, loader, orientation, pipeline, steps  # noqa: E402

REFERENCE = ("jog", 9, 1)

# Rotation invariance should hold to floating point. 1e-6 spm leaves room
# for an eigen-decomposition resolving a rotated matrix in a different
# order, without being loose enough to hide a real axis leak.
ROTATION_CADENCE_TOL_SPM = 1e-6
# Resampling genuinely changes the signal (anti-alias filtering, a new
# sample grid), so cadence may move by 1%: ~1.6 spm at 165 spm, far below
# the 40 spm width of the validation band.
RESAMPLE_CADENCE_TOL_FRACTION = 0.01
# Rates to stress-test. 25 Hz halves the real data (a genuine information
# loss); 100 and 200 Hz upsample, which adds nothing but proves no code
# path assumes 50.
TEST_RATES_HZ = (25.0, 40.0, 100.0, 200.0)


def _load(fs_hz: float = loader.DEFAULT_FS_HZ) -> loader.Trial:
    return loader.load_trial(*REFERENCE, fs_hz=fs_hz)


def _analyse_arrays(accel, gravity, gyro, fs_hz):
    """Stages 2-4 on raw arrays, so tests can feed transformed signals in."""
    seg = steps.steady_state_segment(np.linalg.norm(accel, axis=1), fs_hz)
    sl = slice(seg["start"], seg["stop"])
    a, g, w = accel[sl], gravity[sl], gyro[sl]
    frame = orientation.build_frame(a, g, fs_hz)
    a_vert = orientation.vertical_component(a, g, mode=frame.mode)
    f_step = steps.estimate_step_frequency(a_vert, fs_hz)["f_step_hz"]
    det = steps.detect_steps(a_vert, fs_hz, f_step_hz=f_step)
    summary = steps.cadence_summary(det["step_times_s"])
    lat = lateral.analyse(w, g, det["step_indices"], fs_hz, f_step)
    return {
        "n_steps": det["n_steps"],
        "step_times_s": det["step_times_s"],
        "cadence_spm": summary["cadence_spm"],
        "f_step_hz": f_step,
        "a_vert": a_vert,
        "alternation": lat["alternation_consistency"],
        "frame": frame,
        "segment_start": seg["start"],
    }


# --- orientation invariance ----------------------------------------------


def test_orientation_invariance():
    """An arbitrarily rotated phone must give identical results."""
    tr = _load()
    base = _analyse_arrays(tr.user_accel, tr.gravity, tr.rotation_rate, tr.fs_hz)

    for seed in range(5):
        R = dsp.random_rotation_matrix(seed)
        rot = _analyse_arrays(
            tr.user_accel @ R.T, tr.gravity @ R.T, tr.rotation_rate @ R.T, tr.fs_hz
        )
        assert rot["n_steps"] == base["n_steps"], (
            f"seed {seed}: step count changed under rotation "
            f"({base['n_steps']} -> {rot['n_steps']})"
        )
        assert np.allclose(rot["step_times_s"], base["step_times_s"], atol=1e-9), (
            f"seed {seed}: step timestamps moved under rotation"
        )
        assert abs(rot["cadence_spm"] - base["cadence_spm"]) < ROTATION_CADENCE_TOL_SPM, (
            f"seed {seed}: cadence changed under rotation "
            f"({base['cadence_spm']:.9f} -> {rot['cadence_spm']:.9f} spm)"
        )
        assert abs(rot["alternation"] - base["alternation"]) < 1e-9, (
            f"seed {seed}: alternation consistency changed under rotation"
        )
        # The vertical channel itself must be identical, not merely similar.
        assert np.allclose(rot["a_vert"], base["a_vert"], atol=1e-12), (
            f"seed {seed}: resolved vertical acceleration changed under rotation"
        )


def test_frame_axes_rotate_with_the_sensor():
    """The recovered axes must follow the rotation, not stay put."""
    tr = _load()
    frame = orientation.build_frame(tr.user_accel, tr.gravity, tr.fs_hz)
    for seed in range(3):
        R = dsp.random_rotation_matrix(seed)
        rotated = orientation.build_frame(
            tr.user_accel @ R.T, tr.gravity @ R.T, tr.fs_hz
        )
        # up expressed in the rotated sensor frame should be R @ up.
        assert dsp.angle_between_deg(rotated.up, R @ frame.up) < 1e-6
        assert dsp.axis_angle_deg(rotated.forward, R @ frame.forward) < 1e-3


# --- sample-rate invariance ----------------------------------------------


def _resample(x: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
    """Polyphase resample along axis 0, with an integer rate ratio."""
    from fractions import Fraction

    r = Fraction(fs_to / fs_from).limit_denominator(100)
    return _signal.resample_poly(x, r.numerator, r.denominator, axis=0)


def test_sample_rate_invariance():
    """Cadence must not depend on the rate the data is presented at."""
    tr = _load()
    base = _analyse_arrays(tr.user_accel, tr.gravity, tr.rotation_rate, tr.fs_hz)

    for fs_to in TEST_RATES_HZ:
        accel = _resample(tr.user_accel, tr.fs_hz, fs_to)
        gravity = _resample(tr.gravity, tr.fs_hz, fs_to)
        gyro = _resample(tr.rotation_rate, tr.fs_hz, fs_to)
        # Resampling denormalises gravity slightly; renormalise so it stays
        # a direction, which is all stage 2 uses it for.
        gravity = gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
        out = _analyse_arrays(accel, gravity, gyro, fs_to)
        rel = abs(out["cadence_spm"] - base["cadence_spm"]) / base["cadence_spm"]
        assert rel < RESAMPLE_CADENCE_TOL_FRACTION, (
            f"{fs_to} Hz: cadence moved {100 * rel:.2f}% "
            f"({base['cadence_spm']:.2f} -> {out['cadence_spm']:.2f} spm)"
        )


def test_detection_band_scales_with_rate_and_cadence():
    """Cutoffs are multiples of step frequency, not fixed numbers."""
    x = np.random.default_rng(0).normal(size=4000)
    for fs_hz in (50.0, 100.0, 200.0):
        for f_step in (2.0, 3.0):
            _, (lo, hi) = steps.bandpass_for_steps(x, fs_hz, f_step)
            assert abs(lo - steps.HIGHPASS_STEP_FREQ_MULTIPLE * f_step) < 1e-9
            assert abs(hi - steps.LOWPASS_STEP_FREQ_MULTIPLE * f_step) < 1e-9


def test_band_refuses_impossible_rate():
    """Too low a sample rate for the cadence must raise, not silently clamp."""
    x = np.random.default_rng(0).normal(size=500)
    try:
        steps.bandpass_for_steps(x, fs_hz=6.0, f_step_hz=3.0)
    except ValueError as exc:
        assert "sample rate" in str(exc).lower()
    else:
        raise AssertionError("expected a ValueError at an impossible rate")


# --- guards and contracts -------------------------------------------------


def test_cadence_smoothing_window_must_be_even():
    try:
        steps.cadence_series(np.arange(10) * 0.36, smooth_steps=5)
    except ValueError as exc:
        assert "even" in str(exc)
    else:
        raise AssertionError("odd smoothing window should be rejected")


def test_loader_reports_no_timestamps():
    """The dataset ships no timestamps and the record must say so."""
    tr = _load()
    assert tr.integrity["has_timestamp_column"] is False
    assert tr.integrity["timestamp_irregularities_detectable"] is False
    assert tr.integrity["index_contiguous"] is True


def test_time_index_follows_fs():
    for fs_hz in (50.0, 100.0):
        tr = _load(fs_hz)
        assert np.isclose(tr.t[1] - tr.t[0], 1.0 / fs_hz)
        assert np.isclose(tr.duration_s, tr.n_samples / fs_hz)


def test_lateral_never_claims_accuracy():
    result = pipeline.run_trial(*REFERENCE)
    lat = result["lateral"]
    assert lat["ground_truth_available"] is False
    assert "no left/right foot labels" in lat["accuracy_claim"]


def test_frame_is_orthonormal_and_right_handed():
    tr = _load()
    frame = orientation.build_frame(tr.user_accel, tr.gravity, tr.fs_hz)
    assert frame.diagnostics["orthonormal_residual"] < 1e-9
    assert frame.diagnostics["right_handed"]
    assert abs(float(frame.forward @ frame.up)) < 1e-12
    assert abs(float(frame.mediolateral @ frame.up)) < 1e-12


def test_tracking_frame_is_orthonormal_at_every_sample():
    """The per-sample tracking triad must be a real rotation, not just close.

    If it drifts from orthonormal, the resolved components stop being a
    decomposition of the vector and start scaling it -- silently.
    """
    tr = _load()
    frame = orientation.build_frame(tr.user_accel, tr.gravity, tr.fs_hz, mode="tracking")
    resolved = orientation.resolve(tr.user_accel, frame)
    assert np.allclose(
        np.linalg.norm(resolved, axis=1),
        np.linalg.norm(tr.user_accel, axis=1),
        atol=1e-12,
    ), "tracking frame does not preserve vector magnitude"


def test_vertical_component_matches_resolved_third_axis():
    """The shortcut used by stages 3 and 4 must equal the full resolution."""
    tr = _load()
    for mode in ("static", "tracking"):
        frame = orientation.build_frame(tr.user_accel, tr.gravity, tr.fs_hz, mode=mode)
        full = orientation.resolve(tr.user_accel, frame)[:, 2]
        short = orientation.vertical_component(tr.user_accel, tr.gravity, mode=mode)
        assert np.allclose(full, short, atol=1e-12), f"{mode}: vertical channels disagree"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
