#!/usr/bin/env python3
"""Run the full pipeline over every jog trial and write the report tables.

    python scripts/run_all.py [--root PATH] [--fs 50] [--frame-mode tracking]

Writes to reports/:
    trial_integrity.csv   stage 1 -- per-trial file and sampling facts
    pipeline_results.csv  stages 2-4 -- one row per trial
    figures/*.png         walkthrough figures for one reference subject
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import lateral, loader, pipeline, plotting, steps  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports"

# Number of sampling offsets `lateral.phase_sweep` tries. Read from the
# module rather than repeated, so the chance rates below stay correct if the
# sweep resolution changes.
N_PHASE_OFFSETS = lateral.phase_sweep.__defaults__[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="dataset data/ directory")
    ap.add_argument("--fs", type=float, default=loader.DEFAULT_FS_HZ)
    ap.add_argument("--activity", default="jog")
    ap.add_argument("--frame-mode", default="tracking", choices=["tracking", "static"])
    ap.add_argument("--forward-method", default="step_band", choices=["step_band", "pca"])
    ap.add_argument("--figures-subject", type=int, default=1)
    ap.add_argument("--figures-trial", type=int, default=9)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    print(f"[stage 1] summarising {args.activity} trials ...")
    integrity = loader.summarize_trials(args.activity, root=args.root, fs_hz=args.fs)
    integrity.to_csv(OUT / "trial_integrity.csv", index=False)
    print(f"  {len(integrity)} trials, all clean: {integrity['clean'].all()}")

    rows = []
    idents = loader.discover_trials(args.root, args.activity)
    for i, ident in enumerate(idents, 1):
        try:
            res = pipeline.run_trial(
                ident.activity, ident.trial, ident.subject,
                root=args.root, fs_hz=args.fs,
                frame_mode=args.frame_mode, forward_method=args.forward_method,
            )
            rows.append(pipeline.flatten(res))
        except Exception as exc:  # a stage that cannot run is recorded, not hidden
            rows.append({
                "activity": ident.activity, "trial": ident.trial,
                "subject": ident.subject, "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  !! {ident}: {type(exc).__name__}: {exc}")
        print(f"  [{i}/{len(idents)}] {ident}", end="\r")
    print()

    table = pd.DataFrame(rows).sort_values(["trial", "subject"]).reset_index(drop=True)
    table.to_csv(OUT / "pipeline_results.csv", index=False)
    print(f"[stages 2-4] wrote {OUT / 'pipeline_results.csv'} ({len(table)} rows)")

    table, failed = split_failed(table)

    print("[figures] rendering walkthrough for "
          f"{args.activity}_{args.figures_trial}/sub_{args.figures_subject} ...")
    res = pipeline.run_trial(
        args.activity, args.figures_trial, args.figures_subject,
        root=args.root, fs_hz=args.fs, frame_mode=args.frame_mode,
    )
    figs = {
        "01_raw_signals": plotting.plot_raw_signals(res["trial"]),
        "02_trial_summary": plotting.plot_trial_summary(integrity),
        "03_steady_state": plotting.plot_steady_state(res),
        "04_frame": plotting.plot_frame(res),
        "05_vertical_verification": plotting.plot_vertical_verification(res),
        "06_steps": plotting.plot_steps(res),
        "07_lateral": plotting.plot_lateral(res),
        "08_cadence_all_subjects": plotting.plot_cadence_across_subjects(table),
    }
    for name, fig in figs.items():
        fig.savefig(OUT / "figures" / f"{name}.png", dpi=110, bbox_inches="tight")
    print(f"  wrote {len(figs)} figures to {OUT / 'figures'}")

    print_verdicts(table)
    if len(failed):
        print(f"\n[FAILED] {len(failed)} trial(s) raised and are NOT in the counts above:")
        for _, r in failed.iterrows():
            print(f"    {r['activity']}_{r['trial']}/sub_{r['subject']}: {r['error']}")
    return 0


def split_failed(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate trials that raised from trials that ran.

    A trial that raised has only its identity and an `error` column; every
    other column is NaN. Passing those rows into the boolean masks in
    `print_verdicts` crashed it ("cannot mask with non-boolean array
    containing NA"), so one bad file killed the whole summary after the
    CSVs were written. Returns (ran, failed).
    """
    if "error" in table:
        failed = table[table["error"].notna()].reset_index(drop=True)
        ran = table[table["error"].isna()].reset_index(drop=True)
        return ran, failed
    return table, table.iloc[0:0]


def print_verdicts(t: pd.DataFrame) -> None:
    """Stage-by-stage verdict, printed so a run is self-documenting."""
    n = len(t)
    print("\n" + "=" * 72)
    print(f"VERDICT SUMMARY  ({n} trials)")
    print("=" * 72)

    print("\n[stage 1] loader")
    print(f"  files clean                      {int(t['file_clean'].sum())}/{n}")
    print(f"  duration                         {t['duration_s'].min():.1f}-{t['duration_s'].max():.1f} s")
    print(f"  trimmed as handling transient    {t['trimmed_start_s'].sum() + t['trimmed_end_s'].sum():.0f} s total")
    print("  timestamp irregularities         NOT DETECTABLE (no timestamp column in dataset)")

    print("\n[stage 2] orientation")
    for verdict, cnt in t["frame_verdict"].value_counts().items():
        print(f"  verdict={verdict:<15s}          {cnt}/{n}")
    print(f"  vertical tilt within trial       median {t['vertical_tilt_median_deg'].median():.1f} deg, "
          f"p95 {t['vertical_tilt_p95_deg'].median():.1f} deg")
    print(f"  single static rotation valid     {int(t['static_frame_valid'].sum())}/{n}")
    print(f"  forward axis well conditioned    {int(t['forward_well_conditioned'].sum())}/{n}")
    print(f"  forward axis stable (p95<=10deg) "
          f"{int((t['forward_drift_p95_deg'] <= 10).sum())}/{n} "
          f"(median p95 {t['forward_drift_p95_deg'].median():.0f} deg)")
    print(f"  forward-axis band == verifier    {int((~t['frame_f_step_mismatch']).sum())}/{n}")
    for state, cnt in t["ml_check_state"].value_counts().items():
        print(f"  ML check: {state:<24s} {cnt}/{n}")
    print(f"  stride regularity                median {t['stride_regularity'].median():.2f}")
    print(f"  step symmetry index              median {t['step_symmetry_index'].median():.2f} "
          f"(range {t['step_symmetry_index'].min():.2f}-{t['step_symmetry_index'].max():.2f})")

    print("\n[stage 3] steps and cadence")
    print(f"  cadence                          {t['cadence_spm'].min():.1f}-{t['cadence_spm'].max():.1f} spm "
          f"(median {t['cadence_spm'].median():.1f})")
    print(f"  inside 150-190 spm               {int(t['cadence_in_band'].sum())}/{n}")
    print(f"  detector vs spectral estimate    ratio {t['detector_spectral_ratio'].min():.3f}-"
          f"{t['detector_spectral_ratio'].max():.3f}  (band-coupled: bounds within-band miscounting only)")
    print(f"  detection gaps (>{steps.DETECTION_GAP_STEP_PERIODS:g} step periods) {int(t['detection_gap'].sum())}/{n}")
    print(f"  longest active segment kept      median {100 * t['kept_fraction_of_active'].median():.0f}% of active time, "
          f"min {100 * t['kept_fraction_of_active'].min():.0f}%")
    flagged = t[t["cadence_flagged"]]
    print(f"  flagged                          {len(flagged)}/{n}")
    for _, r in flagged.iterrows():
        print(f"    {r['activity']}_{r['trial']}/sub_{r['subject']}: {r['cadence_spm']:.1f} spm "
              f"-> cause={r['cadence_failure_cause']}: {r['cadence_diagnosis']}")

    print("\n[stage 4] left/right (EXPLORATORY -- no ground truth in this dataset)")
    print(f"  alternation at detected contact  median {t['alternation_consistency'].median():.3f}")
    print(f"  phase-randomised surrogate null  median {t['alternation_surrogate_null'].median():.3f}")
    print(f"  beats that null by >0.10         {int((t['excess_over_surrogate'] > 0.10).sum())}/{n}")
    print(f"  best-phase alternation           median {t['best_phase_alternation'].median():.3f}")
    print(f"  ... vs max-over-phase null       median {t['best_phase_surrogate_max_mean'].median():.3f}")
    print(f"  ... exceeds that null's p95      {int(t['best_phase_beats_surrogate_p95'].sum())}/{n}")
    print("      (phase-randomised null does NOT preserve step-locking: a pure stride-rate")
    print("       sinusoid locked to the steps, with no laterality, beats it on 22/24 synthetic")
    print("       runs -- so this line shows step-locked periodicity, not laterality)")
    print(f"  alternation range over phase     median {t['alternation_phase_range'].median():.3f}")
    print("  NO ACCURACY IS CLAIMED: consistency is not correctness.")

    if "subject" in t and t["trial"].nunique() > 1:
        piv = t.pivot_table(index="subject", columns="trial", values="cadence_spm")
        cols = list(piv.columns)
        d = (piv[cols[0]] - piv[cols[1]]).abs()
        print(f"\n[reproducibility] same subject, trial {cols[0]} vs {cols[1]}")
        print(f"  |cadence difference|             median {d.median():.1f} spm, max {d.max():.1f} spm")

        # Best-phase agreement, reported at two tolerances.
        #
        # Exact-bin equality alone understates reproducibility badly: the
        # sweep quantises the step period into N_PHASE_OFFSETS bins, so two
        # genuinely matching phases that straddle a bin boundary score as a
        # miss. The within-one-bin figure is the honest headline, and both
        # are printed against their chance rates so neither can be read as
        # impressive or as dismal without the baseline beside it.
        po = t.pivot_table(index="subject", columns="trial", values="best_phase_offset")
        n_sub = len(po)
        delta = (po[cols[0]] - po[cols[1]]).abs()
        circular = np.minimum(delta, 1.0 - delta)  # the phase axis wraps
        bin_w = 1.0 / N_PHASE_OFFSETS
        exact = int((circular < 1e-9).sum())
        within = int((circular <= bin_w + 1e-9).sum())
        print(f"  best-phase offset, exact bin     {exact}/{n_sub} subjects "
              f"(chance {n_sub / N_PHASE_OFFSETS:.1f})")
        print(f"  ... within one bin (+/-{bin_w:.1f} step) {within}/{n_sub} subjects "
              f"(chance {3 * n_sub / N_PHASE_OFFSETS:.1f})")
        print(f"  median circular disagreement     {circular.median():.2f} of a step "
              f"(one bin = {bin_w:.2f}, the sweep's resolution floor)")


if __name__ == "__main__":
    sys.exit(main())
