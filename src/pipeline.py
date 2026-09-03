"""End-to-end glue: run stages 1-4 on one trial and return everything.

`run_trial` is what the notebook and the batch script both call, so the
walkthrough and the 48-trial table cannot drift apart.
"""

from __future__ import annotations

import numpy as np

from . import lateral, loader, orientation, steps


def run_trial(
    activity: str = "jog",
    trial: int = 9,
    subject: int = 1,
    root=None,
    fs_hz: float = loader.DEFAULT_FS_HZ,
    frame_mode: str = "tracking",
    forward_method: str = "step_band",
    trim_to_steady_state: bool = True,
    seed: int = 0,
) -> dict:
    """Run every stage on one trial.

    Returns a dict with the loaded `Trial`, the stage-2 frame and its
    verification, stage-3 detections and cadence, and the stage-4
    exploratory analysis. Nothing is suppressed on failure: if stage 2
    reports `verdict == "failed"` the later stages still run, and the
    verdict travels with the result so a caller can refuse to use it.
    """
    tr = loader.load_trial(activity, trial, subject, root, fs_hz)
    fs = tr.fs_hz

    # --- stage 3a: trim handling transients. Done on rotation-invariant
    # acceleration magnitude, so it is independent of stage 2.
    if trim_to_steady_state:
        seg = steps.steady_state_segment(np.linalg.norm(tr.user_accel, axis=1), fs)
    else:
        seg = {
            "start": 0, "stop": tr.n_samples, "segmented": False,
            "trimmed_start_s": 0.0, "trimmed_end_s": 0.0, "kept_fraction": 1.0,
            "n_windows": 0,
        }
    sl = slice(seg["start"], seg["stop"])
    accel = tr.user_accel[sl]
    gravity = tr.gravity[sl]
    gyro = tr.rotation_rate[sl]
    t0 = seg["start"] / fs
    steady_s = (seg["stop"] - seg["start"]) / fs
    too_short = steady_s < steps.MIN_STEADY_SECONDS

    # --- stage 2: orientation
    frame = orientation.build_frame(
        accel, gravity, fs, mode=frame_mode, forward_method=forward_method
    )
    verify = orientation.verify_frame(accel, gravity, frame, fs)
    a_vert = verify["a_vertical"]
    resolved = orientation.resolve(accel, frame)

    # --- stage 3b: steps and cadence
    f_step = verify["periodicity"]["f_step_hz"]
    det = steps.detect_steps(a_vert, fs, f_step_hz=f_step, t0_s=t0)
    summary = steps.cadence_summary(det["step_times_s"])
    cadence = steps.cadence_series(det["step_times_s"])
    spectral = steps.estimate_step_frequency(a_vert, fs)
    # `fs` is an input assertion, so test it against the data before any
    # cadence is attributed to the runner: two estimators sharing a wrong
    # rate agree with each other while both being wrong.
    rate_check = steps.check_sample_rate(a_vert, fs)
    diag = steps.diagnose_cadence(
        detected_spm=summary["cadence_spm"],
        spectral_spm=spectral["cadence_spm"],
        stride_regularity=verify["periodicity"]["stride_regularity"],
        irregular_step_fraction=summary["irregular_step_fraction"],
        sample_rate_plausible=rate_check["sample_rate_plausible"],
        out_of_band_peak_hz=rate_check["out_of_band_peak_hz"],
    )
    # `cadence_summary` already enforces MIN_STEADY_SECONDS on the step
    # span; this adds the segment-level context to the message.
    if too_short and rate_check["sample_rate_plausible"]:
        diag = dict(diag)
        diag["flagged"] = True
        diag["failure_attributed_to"] = "trial"
        diag["diagnosis"] = (
            f"only {steady_s:.1f}s of steady motion (< "
            f"{steps.MIN_STEADY_SECONDS}s): too short for a stable cadence; "
            + diag["diagnosis"]
        )

    # --- stage 4: exploratory L/R
    lat = lateral.analyse(gyro, gravity, det["step_indices"], fs, f_step, mode=frame_mode, seed=seed)

    return {
        "trial": tr,
        "segment": seg,
        "steady_seconds": steady_s,
        "steady_too_short": bool(too_short),
        "frame": frame,
        "verify": verify,
        "resolved": resolved,
        "a_vertical": a_vert,
        "detection": det,
        "cadence_summary": summary,
        "cadence_series": cadence,
        "spectral": spectral,
        "cadence_diagnosis": diag,
        "sample_rate_check": rate_check,
        "lateral": lat,
    }


def flatten(result: dict) -> dict:
    """One flat row per trial for the cross-subject tables."""
    tr = result["trial"]
    v = result["verify"]
    per = v["periodicity"]
    stab = v["stability"]
    fr = result["frame"]
    det = result["detection"]
    cs = result["cadence_summary"]
    cd = result["cadence_diagnosis"]
    lat = result["lateral"]
    return {
        # stage 1
        "activity": tr.ident.activity,
        "trial": tr.ident.trial,
        "subject": tr.ident.subject,
        "fs_hz": tr.fs_hz,
        "n_samples": tr.n_samples,
        "duration_s": tr.duration_s,
        "file_clean": tr.integrity["clean"],
        "trimmed_start_s": result["segment"]["trimmed_start_s"],
        "trimmed_end_s": result["segment"]["trimmed_end_s"],
        "steady_s": result["steady_seconds"],
        # stage 2
        "frame_mode": fr.mode,
        "frame_verdict": v["verdict"],
        "frame_reasons": " | ".join(v["reasons"]),
        "forward_eig_ratio": fr.diagnostics["forward_eigenvalue_ratio"],
        "forward_well_conditioned": fr.diagnostics["forward_well_conditioned"],
        "forward_sign_confident": fr.diagnostics["forward_sign_confident"],
        "vertical_tilt_median_deg": stab["vertical_tilt_median_deg"],
        "vertical_tilt_p95_deg": stab["vertical_tilt_p95_deg"],
        "forward_drift_p95_deg": stab["forward_drift_p95_deg"],
        "static_frame_valid": stab["frame_static_valid"],
        "step_stride_axis_angle_deg": v["mediolateral_check"]["step_stride_axis_angle_deg"],
        "ml_independently_supported": v["mediolateral_check"]["ml_independently_supported"],
        "fundamental_power_fraction": per["fundamental_power_fraction"],
        "step_regularity": per["step_regularity"],
        "stride_regularity": per["stride_regularity"],
        "step_symmetry_index": per["step_symmetry_index"],
        "periodicity_ok": per["periodicity_ok"],
        # stage 3
        "band_low_hz": det["band_hz"][0],
        "band_high_hz": det["band_hz"][1],
        "n_steps": cs["n_steps"],
        "cadence_spm": cs["cadence_spm"],
        "cadence_spm_spectral": result["spectral"]["cadence_spm"],
        "cadence_cv": cs["cadence_cv"],
        "irregular_step_fraction": cs["irregular_step_fraction"],
        "alternating_interval_asymmetry_abs_pct": cs["alternating_interval_asymmetry_abs_pct"],
        "cadence_in_band": cd["cadence_in_expected_band"],
        "detector_spectral_ratio": cd["detector_spectral_ratio"],
        "cadence_flagged": cd["flagged"],
        "cadence_failure_cause": cd["failure_attributed_to"],
        "sample_rate_plausible": cd["sample_rate_plausible"],
        "span_too_short": cs["span_too_short"],
        "cadence_diagnosis": cd["diagnosis"],
        # stage 4 (exploratory; no ground truth)
        "n_labelled_steps": lat["n_labelled_steps"],
        "contact_window_samples": lat["contact_window_samples"],
        "alternation_consistency": lat["alternation_consistency"],
        "alternation_surrogate_null": lat["null_surrogate_mean"],
        "alternation_random_null": lat["null_random_mean"],
        "alternation_mid_step": lat["null_mid_step_rate"],
        "excess_over_surrogate": lat["excess_over_surrogate"],
        "best_phase_alternation": lat["best_phase_alternation"],
        "best_phase_offset": lat["best_phase_offset_step_fraction"],
        "best_phase_excess_over_surrogate": lat["best_phase_excess_over_surrogate"],
        "best_phase_surrogate_max_mean": lat["best_phase_surrogate_max_mean"],
        "best_phase_beats_surrogate_p95": lat["best_phase_beats_surrogate_p95"],
        "alternation_phase_range": lat["alternation_phase_range"],
        "omega_v_rms_rad_s": lat["omega_v_rms_rad_s"],
        "omega_stride_power_fraction": lat["stride_power_fraction"],
        "omega_stride_over_step_power": lat["stride_over_step_power"],
        "cluster_separation_d": lat["cluster_separation_d"],
        "ground_truth_available": lat["ground_truth_available"],
    }
