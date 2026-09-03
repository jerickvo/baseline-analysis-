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
    """Run every stage on one MotionSense trial. See `run_stages`."""
    tr = loader.load_trial(activity, trial, subject, root, fs_hz)
    return run_stages(tr, frame_mode, forward_method, trim_to_steady_state, seed)


def run_session(
    folder,
    on_gap: str = "raise",
    frame_mode: str = "tracking",
    forward_method: str = "step_band",
    trim_to_steady_state: bool = True,
    seed: int = 0,
) -> dict:
    """Run every stage on one BaselineLogger session folder.

    The sample rate is measured from the session's own timestamps, so the
    same stages run at whatever rate the phone delivered. See
    `loader.load_logger_session` for `on_gap`.
    """
    tr = loader.load_logger_session(folder, on_gap=on_gap)
    return run_stages(tr, frame_mode, forward_method, trim_to_steady_state, seed)


def run_stages(
    tr: loader.Trial,
    frame_mode: str = "tracking",
    forward_method: str = "step_band",
    trim_to_steady_state: bool = True,
    seed: int = 0,
) -> dict:
    """Stages 2-4 on an already-loaded `Trial`, plus the quality roll-up.

    Returns a dict with the `Trial`, the stage-2 frame and its
    verification, stage-3 detections and cadence, the stage-4 exploratory
    analysis, and a single `quality` verdict. Nothing is suppressed on
    failure: if stage 2 reports `verdict == "failed"` the later stages
    still run, and the verdict travels with the result so a caller can
    refuse to use it.
    """
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

    result = {
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
    result["quality"] = assess_quality(result)
    return result


# Fraction of steady running that may sit outside the analysed bout before
# the run is called fragmented. 0.2 allows a short warm-up or cool-down
# bout to be dropped without comment, while a run split by a mid-run stop
# -- which drops 25-50% -- is flagged, because a summary built from half a
# run is not a summary of the run.
MAX_DISCARDED_STEADY_FRACTION = 0.20
# Dropped samples (below the gap threshold) as a fraction of all samples,
# above which the record is refused rather than caveated. Each dropped
# sample shifts everything after it by one period on the uniform grid the
# filters assume, so the error accumulates: at 0.1% of a 30-minute 100 Hz
# run that is 180 samples, 1.8 s of drift by the end, 0.1% of cadence --
# harmless for a rate, but the point where phase-sensitive quantities
# (stage 4, any future contact timing) can no longer be trusted. A single
# dropped sample in 180,000 must not condemn a run; a sample every few
# seconds must.
MAX_DROPPED_SAMPLE_FRACTION = 1e-3


def assess_quality(result: dict) -> dict:
    """One verdict for the run: "ok", "partial" or "insufficient".

    This is the gate between the numbers and anyone reading them. Every
    stage already reports its own diagnostics; this collects the ones that
    decide whether the run should be summarised at all, so that a
    beautiful report cannot be produced from a record that does not
    support one.

    "insufficient": do not report mechanics. The record is broken (gaps,
      dropped samples beyond the drift budget, reordered samples, an
      unfinished session, non-finite data), too short, recorded at an
      implausible rate, or shows no gait periodicity.
    "partial": report with the stated caveats. The run was fragmented and
      only its longest bout was analysed, the horizontal frame is
      unverified so only vertical-axis quantities are trustworthy, or a
      few samples were dropped (cadence unaffected, phase not).
    "ok": every check passed.

    Side classification is reported as information only -- it is never a
    gate, because nothing downstream is allowed to depend on it yet.
    """
    tr = result["trial"]
    integ = tr.integrity
    v = result["verify"]
    cd = result["cadence_diagnosis"]
    seg = result["segment"]
    lat = result["lateral"]

    blockers: list[str] = []
    caveats: list[str] = []

    # File-level defects are blockers whatever the source.
    if integ.get("n_nan", 0) or integ.get("n_inf", 0):
        blockers.append("non-finite samples in the record")
    if integ.get("flatline_suspect", False):
        blockers.append(f"stuck sensor: {integ.get('longest_flatline_s', 0):.2f}s flatline")
    if integ.get("index_contiguous") is False:
        blockers.append("row index not contiguous")

    if integ.get("has_timestamp_column", False):
        # Logger session: reason from the measured timestamp facts.
        if integ.get("n_gaps", 0) or integ.get("discarded_for_gaps_s", 0.0) > 0:
            blockers.append(
                f"{integ.get('n_gaps', 0)} gap(s); "
                f"{integ.get('discarded_for_gaps_s', 0.0):.1f}s cut at gaps"
            )
        if integ.get("n_nonmonotonic", 0):
            blockers.append(f"{integ['n_nonmonotonic']} duplicated/reordered sample(s)")
        if integ.get("metadata_in_progress", False):
            blockers.append("session.json marked in-progress: the app did not finish the session")
        if not integ.get("timestamps_uniform", True) and not integ.get("n_gaps", 0) and not integ.get("n_nonmonotonic", 0):
            blockers.append("timing jitter too large for the filters")
        dropped = int(integ.get("n_dropped_estimate", 0))
        if dropped:
            frac = dropped / max(tr.n_samples, 1)
            msg = f"~{dropped} sample(s) dropped below the gap threshold ({100 * frac:.3f}% of samples)"
            if frac > MAX_DROPPED_SAMPLE_FRACTION:
                blockers.append(msg + ": timing drift too large")
            else:
                caveats.append(msg + ": cadence unaffected, phase-sensitive quantities are not")
    elif not integ.get("clean", False):
        blockers.append(f"record not clean: {integ.get('problems', '')}")
    if result["steady_too_short"]:
        blockers.append(
            f"only {result['steady_seconds']:.1f}s of steady motion (< {steps.MIN_STEADY_SECONDS:g}s)"
        )
    if not cd["sample_rate_plausible"]:
        blockers.append("sample rate is probably wrong (see cadence diagnosis)")
    if v["verdict"] == "failed":
        blockers.append("vertical acceleration shows no repeating gait structure")
    if cd["failure_attributed_to"] == "algorithm":
        blockers.append(f"step detector failed: {cd['diagnosis']}")

    kept = result["steady_seconds"]
    discarded = float(seg.get("discarded_steady_s", 0.0))
    total_steady = kept + discarded
    if total_steady > 0 and discarded / total_steady > MAX_DISCARDED_STEADY_FRACTION:
        caveats.append(
            f"fragmented run: {seg.get('n_segments', 1)} bouts, only the longest "
            f"({kept:.0f}s) analysed, {discarded:.0f}s of steady motion left out"
        )
    if v["verdict"] == "vertical_only":
        caveats.append("horizontal frame unverified: forward/mediolateral quantities are not trustworthy")
    if cd["failure_attributed_to"] == "trial" and not result["steady_too_short"]:
        caveats.append(f"cadence outside the expected band: {cd['diagnosis']}")

    side = (
        "unreliable"
        if not (lat["alternation_consistent"] and lat["excess_over_surrogate"] > 0.10)
        else "consistent (not validated: no ground truth)"
    )

    verdict = "insufficient" if blockers else ("partial" if caveats else "ok")
    return {
        "verdict": verdict,
        "blockers": blockers,
        "caveats": caveats,
        "side_classification": side,
        "summary": "; ".join(blockers + caveats) if (blockers or caveats) else "all checks passed",
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
        "n_steady_segments": result["segment"].get("n_segments", 1),
        "discarded_steady_s": result["segment"].get("discarded_steady_s", 0.0),
        # stage 2
        "frame_mode": fr.mode,
        "frame_verdict": v["verdict"],
        "frame_reasons": " | ".join(v["reasons"]),
        "forward_eig_ratio": fr.diagnostics["forward_eigenvalue_ratio"],
        "forward_well_conditioned": fr.diagnostics["forward_well_conditioned"],
        "forward_sign_confident": fr.diagnostics["forward_sign_confident"],
        "forward_sign_criteria_agree": fr.diagnostics["forward_sign_criteria_agree"],
        "forward_phase_effect_size": fr.diagnostics["forward_phase_effect_size"],
        "forward_impact_effect_size": fr.diagnostics["forward_impact_effect_size"],
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
        # roll-up
        "quality_verdict": result["quality"]["verdict"],
        "quality_summary": result["quality"]["summary"],
        "side_classification": result["quality"]["side_classification"],
    }
