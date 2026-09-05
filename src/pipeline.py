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
            "n_windows": 0, "segments": [(0, tr.n_samples)], "n_segments": 1,
            "discarded_steady_s": 0.0, "threshold_g": float("nan"), "no_motion": False,
        }
    sl = slice(seg["start"], seg["stop"])
    accel = tr.user_accel[sl]
    gravity = tr.gravity[sl]
    gyro = tr.rotation_rate[sl]
    t0 = seg["start"] / fs
    steady_s = (seg["stop"] - seg["start"]) / fs
    too_short = steady_s < steps.MIN_STEADY_SECONDS

    # --- stage 2: orientation, on the longest bout
    frame = orientation.build_frame(
        accel, gravity, fs, mode=frame_mode, forward_method=forward_method
    )
    verify = orientation.verify_frame(accel, gravity, frame, fs)
    a_vert = verify["a_vertical"]
    resolved = orientation.resolve(accel, frame)

    # --- stage 3b: steps and cadence, on the longest bout
    f_step = verify["periodicity"]["f_step_hz"]
    t_hw = tr.sample_times_s
    det = steps.detect_steps(
        a_vert, fs, f_step_hz=f_step, t0_s=t0,
        sample_times_s=t_hw[sl] if t_hw is not None else None,
    )
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
        irregular_stride_fraction=summary["irregular_stride_fraction"],
        sample_rate_plausible=rate_check["sample_rate_plausible"],
        out_of_band_peak_hz=rate_check["out_of_band_peak_hz"],
        cadence_spread=steps.cadence_spread(cadence),
        harmonic_ambiguous=spectral["harmonic_ambiguous"],
        subharmonic_power_ratio=spectral["subharmonic_power_ratio"],
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

    # --- stage 3c: the other bouts. A run split by a stop is still one
    # run; its cadence is pooled over every bout long enough to carry one.
    # Only the vertical channel is needed here, and that needs only
    # gravity, so no frame is re-estimated per bout.
    bouts, unanalysed_s = analyse_bouts(tr, seg, det, summary, frame_mode)
    pooled = steps.cadence_summary_pooled([b["step_times_s"] for b in bouts])
    bout_cadences = [b["cadence_spm"] for b in bouts if np.isfinite(b["cadence_spm"])]
    if len(bout_cadences) >= 2 and pooled["cadence_spm"] > 0:
        bout_spread = float((max(bout_cadences) - min(bout_cadences)) / pooled["cadence_spm"])
    else:
        bout_spread = 0.0
    pooled.update({
        "n_bouts_analysed": int(len(bouts)),
        "bout_cadences_spm": bout_cadences,
        "bout_cadence_spread": bout_spread,
        "steady_s_all_bouts": float(sum(b["duration_s"] for b in bouts)),
        "unanalysed_steady_s": float(unanalysed_s),
    })

    # --- stage 4: exploratory L/R, on the longest bout
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
        "bouts": bouts,
        "cadence_summary_all_bouts": pooled,
        "lateral": lat,
    }
    result["quality"] = assess_quality(result)
    return result


def analyse_bouts(
    tr: loader.Trial, seg: dict, primary_det: dict, primary_summary: dict, frame_mode: str
) -> tuple[list, float]:
    """Stage 3 on every bout that can carry a cadence.

    Returns one record per analysed bout (the longest bout reuses the
    detection already made) and the seconds of steady motion in bouts too
    short to analyse, which the quality gate reports.
    """
    fs = tr.fs_hz
    primary = (seg["start"], seg["stop"])
    bouts = []
    unanalysed = 0.0
    for a, b in seg.get("segments", [primary]):
        duration = (b - a) / fs
        if (a, b) == primary:
            bouts.append({
                "start_s": a / fs, "stop_s": b / fs, "duration_s": duration, "primary": True,
                "n_steps": primary_det["n_steps"], "step_times_s": primary_det["step_times_s"],
                "cadence_spm": primary_summary["cadence_spm"],
                "f_step_hz": primary_det["f_step_hz_used"], "error": None,
            })
            continue
        if duration < steps.MIN_STEADY_SECONDS:
            unanalysed += duration
            continue
        a_vert = orientation.vertical_component(tr.user_accel[a:b], tr.gravity[a:b], mode=frame_mode)
        t_hw = tr.sample_times_s
        try:
            f_step = steps.estimate_step_frequency(a_vert, fs)["f_step_hz"]
            det = steps.detect_steps(
                a_vert, fs, f_step_hz=f_step, t0_s=a / fs,
                sample_times_s=t_hw[a:b] if t_hw is not None else None,
            )
        except ValueError as exc:  # a bout that cannot be band-passed is reported, not hidden
            unanalysed += duration
            bouts.append({
                "start_s": a / fs, "stop_s": b / fs, "duration_s": duration, "primary": False,
                "n_steps": 0, "step_times_s": np.empty(0), "cadence_spm": np.nan,
                "f_step_hz": np.nan, "error": str(exc),
            })
            continue
        cs = steps.cadence_summary(det["step_times_s"])
        bouts.append({
            "start_s": a / fs, "stop_s": b / fs, "duration_s": duration, "primary": False,
            "n_steps": det["n_steps"], "step_times_s": det["step_times_s"],
            "cadence_spm": cs["cadence_spm"], "f_step_hz": f_step, "error": None,
        })
    return bouts, unanalysed


# Fraction of steady running that may sit in bouts too short to analyse
# (under `steps.MIN_STEADY_SECONDS`) before that loss is called out. 0.2
# allows a few seconds of stop-and-go at a crossing to pass without
# comment; a run/walk session whose running bouts are mostly too short to
# carry a cadence is flagged, because the pooled number then describes a
# fraction of the run.
MAX_UNANALYSED_STEADY_FRACTION = 0.20
# Fraction of irregular strides -- missed steps, double detections, pauses
# inside a bout -- above which per-step quantities carry a caveat. On the
# 48 MotionSense trials the fraction is 0.00 at the median and 0.05 at the
# maximum after the stride-based definition; one broken stride in ten
# means the detector is missing something systematically, not
# occasionally, and anything per-step (stage 4, any future contact
# timing) should be read with that in mind. Cadence itself is unaffected,
# because irregular strides are excluded from it.
MAX_IRREGULAR_STRIDE_FRACTION = 0.10
# Dropped samples (below the gap threshold) as a fraction of all samples,
# above which the record is refused rather than caveated. Step times are
# read from the hardware clock when the record has one, so a drop no longer
# shifts every later step time by one period and cadence is immune to the
# accumulated drift. What a drop still does is put a discontinuity into
# the uniform grid the filters run on -- a one-period jump inside every
# band-pass and every stability window that spans it -- and say that the
# device was not keeping up. At 0.1% that is a discontinuity every 10 s at
# 100 Hz, inside every 8 s stability window: the point where the record
# stops being a faithful uniform sampling of anything. A single dropped
# sample in 180,000 must not condemn a run; a sample every few seconds must.
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
      unfinished session, non-finite data), contains no locomotion, is too
      short, was recorded at an implausible rate, shows no gait
      periodicity, holds more than one gait, or yields no cadence.
    "partial": report with the stated caveats. Too much of the run sits in
      bouts too short to analyse, the horizontal frame is unverified so
      only vertical-axis quantities are trustworthy, the forward sign is
      unresolved so left and right are undirected, a few samples were
      dropped (cadence unaffected, phase not), or one stride in ten is
      irregular.
    "ok": every check passed.

    Stage 4 never gates anything and never yields a side verdict: its
    alternation statistic is gait-locked by construction (see
    `lateral`), so no signal-only test can turn it into left/right.
    """
    tr = result["trial"]
    integ = tr.integrity
    v = result["verify"]
    cd = result["cadence_diagnosis"]
    seg = result["segment"]
    cs = result["cadence_summary"]
    pooled = result.get("cadence_summary_all_bouts", {})

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
        if integ.get("metadata_rows_lost"):
            blockers.append(
                f"the app reports {integ['metadata_rows_lost']} row(s) never reached disk: the file "
                f"is shorter than the session"
            )
        if integ.get("metadata_count_matches") is False:
            blockers.append(
                f"the app counted {integ.get('metadata_sample_count')} motion samples but the file "
                f"holds {tr.n_samples}: truncated or corrupted record"
            )
        if not integ.get("timestamps_uniform", True) and not integ.get("n_gaps", 0) and not integ.get("n_nonmonotonic", 0):
            blockers.append("timing jitter too large for the filters")
        dropped = int(integ.get("n_dropped_estimate", 0))
        if dropped:
            frac = dropped / max(tr.n_samples, 1)
            msg = f"~{dropped} sample(s) dropped below the gap threshold ({100 * frac:.3f}% of samples)"
            if frac > MAX_DROPPED_SAMPLE_FRACTION:
                blockers.append(msg + ": the uniform grid the filters assume is broken too often")
            else:
                caveats.append(
                    msg + ": step times come from the hardware clock so cadence is unaffected; "
                    "the filtered waveform has a discontinuity at each drop"
                )
    elif not integ.get("clean", False):
        blockers.append(f"record not clean: {integ.get('problems', '')}")
    if seg.get("no_motion", False):
        blockers.append(
            f"no sustained motion: no 1 s window reached {seg.get('threshold_g', float('nan')):.2f} g RMS, "
            f"so there is no locomotion to analyse"
        )
    elif result["steady_too_short"]:
        blockers.append(
            f"only {result['steady_seconds']:.1f}s of steady motion (< {steps.MIN_STEADY_SECONDS:g}s)"
        )
    if cd["failure_attributed_to"] == "sample_rate":
        blockers.append(f"sample rate is probably wrong: {cd['diagnosis']}")
    if v["verdict"] == "failed":
        blockers.append("vertical acceleration shows no repeating gait structure")
    if cd["failure_attributed_to"] == "algorithm":
        blockers.append(f"step detector failed: {cd['diagnosis']}")
    if cd.get("mixed_gait", False):
        blockers.append(f"more than one gait in the bout: {cd['diagnosis']}")
    elif (
        not np.isfinite(cs["cadence_spm"])
        and not result["steady_too_short"]
        and not seg.get("no_motion", False)
    ):
        # Reachable when the bout clears MIN_STEADY_SECONDS but the step
        # span inside it does not (a 5.2 s bout holds a ~4.9 s span).
        blockers.append(f"no defensible cadence: {cd['diagnosis']}")
    if pooled.get("bout_cadence_spread", 0.0) > steps.MIXED_CADENCE_SPREAD:
        cads = ", ".join(f"{c:.0f}" for c in pooled.get("bout_cadences_spm", []))
        blockers.append(
            f"bouts differ in cadence ({cads} spm, spread "
            f"{100 * pooled['bout_cadence_spread']:.0f}% of the pooled value): more than "
            f"one gait in the run, so no single cadence describes it"
        )

    unanalysed = float(pooled.get("unanalysed_steady_s", 0.0))
    analysed = float(pooled.get("steady_s_all_bouts", result["steady_seconds"]))
    total_steady = analysed + unanalysed
    if total_steady > 0 and unanalysed / total_steady > MAX_UNANALYSED_STEADY_FRACTION:
        caveats.append(
            f"fragmented run: {unanalysed:.0f}s of steady motion sits in bouts shorter than "
            f"{steps.MIN_STEADY_SECONDS:g}s and carries no cadence; {analysed:.0f}s in "
            f"{pooled.get('n_bouts_analysed', 1)} bout(s) analysed"
        )
    if v["verdict"] == "vertical_only":
        caveats.append("horizontal frame unverified: forward/mediolateral quantities are not trustworthy")
    elif not result["frame"].diagnostics.get("forward_sign_confident", False):
        # The axes are supported but not their direction: a 180-degree
        # error in forward maps left onto right exactly.
        caveats.append("forward sign unresolved: forward/back and left/right labels are undirected")
    if cd["failure_attributed_to"] == "trial" and not result["steady_too_short"] and not cd.get("mixed_gait", False):
        caveats.append(f"cadence outside the expected band: {cd['diagnosis']}")
    gps_acc = integ.get("gps_accuracy_median_m", np.nan)
    if np.isfinite(gps_acc) and gps_acc > loader.GPS_COARSE_ACCURACY_M:
        caveats.append(
            f"GPS fixes are coarse (median horizontal accuracy {gps_acc:.0f} m; Precise Location "
            f"off?): nothing here uses them yet, but no pace could be built on them"
        )
    irregular = cs.get("irregular_stride_fraction", np.nan)
    if np.isfinite(irregular) and irregular > MAX_IRREGULAR_STRIDE_FRACTION:
        caveats.append(
            f"{100 * irregular:.0f}% of strides irregular (missed or doubled steps, or pauses): "
            f"cadence excludes them, per-step quantities do not"
        )

    verdict = "insufficient" if blockers else ("partial" if caveats else "ok")
    return {
        "verdict": verdict,
        "blockers": blockers,
        "caveats": caveats,
        # There is no side verdict, and there cannot be one from this
        # data. The field stays so nothing that reads it breaks, and so
        # that what it says is the same on every record.
        "side_classification": "not classifiable: no ground truth, and alternation is gait-locked by construction",
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
    pooled = result.get("cadence_summary_all_bouts", {})
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
        "steady_threshold_g": result["segment"].get("threshold_g", np.nan),
        "no_motion": result["segment"].get("no_motion", False),
        "n_steady_segments": result["segment"].get("n_segments", 1),
        "discarded_steady_s": result["segment"].get("discarded_steady_s", 0.0),
        "n_bouts_analysed": pooled.get("n_bouts_analysed", 1),
        "steady_s_all_bouts": pooled.get("steady_s_all_bouts", result["steady_seconds"]),
        "unanalysed_steady_s": pooled.get("unanalysed_steady_s", 0.0),
        "bout_cadence_spread": pooled.get("bout_cadence_spread", 0.0),
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
        # stage 3. `cadence_spm` is pooled over every analysed bout of the
        # run; the longest bout's own value is beside it. On a one-bout
        # record they are identical.
        "band_low_hz": det["band_hz"][0],
        "band_high_hz": det["band_hz"][1],
        "n_steps": pooled.get("n_steps", cs["n_steps"]),
        "n_steps_primary_bout": cs["n_steps"],
        "cadence_spm": pooled.get("cadence_spm", cs["cadence_spm"]),
        "cadence_spm_primary_bout": cs["cadence_spm"],
        "cadence_spm_span": cs["cadence_spm_span"],
        "cadence_spm_spectral": result["spectral"]["cadence_spm"],
        "cadence_cv": pooled.get("cadence_cv", cs["cadence_cv"]),
        "irregular_stride_fraction": pooled.get("irregular_stride_fraction", cs["irregular_stride_fraction"]),
        "alternating_interval_asymmetry_abs_pct": cs["alternating_interval_asymmetry_abs_pct"],
        "cadence_in_band": cd["cadence_in_expected_band"],
        "detector_spectral_ratio": cd["detector_spectral_ratio"],
        "cadence_spread_within_bout": cd["cadence_spread"],
        "mixed_gait": cd["mixed_gait"],
        "harmonic_ambiguous": cd["harmonic_ambiguous"],
        "subharmonic_power_ratio": result["spectral"]["subharmonic_power_ratio"],
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
        "n_non_flips": lat["n_non_flips"],
        "longest_same_sign_run": lat["longest_same_sign_run"],
        "contact_margin_median": lat["contact_margin_median"],
        "contact_margin_below_quarter_fraction": lat["contact_margin_below_quarter_fraction"],
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
        "cluster_separation_d_excess": lat["cluster_separation_d_excess"],
        "ground_truth_available": lat["ground_truth_available"],
        # roll-up
        "quality_verdict": result["quality"]["verdict"],
        "quality_summary": result["quality"]["summary"],
        "side_classification": result["quality"]["side_classification"],
    }
