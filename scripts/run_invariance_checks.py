#!/usr/bin/env python3
"""Sweep orientation and sample-rate invariance across every jog trial.

The unit tests in tests/ check one reference trial. This runs the same two
transformations over all 48 and writes a table, so the claim "placement- and
rate-agnostic" is backed by 48 measurements rather than one.

    python scripts/run_invariance_checks.py [--root PATH]

Writes reports/invariance.csv.
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as _signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import dsp, lateral, loader, orientation, steps  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports"

# 25 Hz halves the real data (genuine information loss and the interesting
# case); 100 and 200 Hz upsample, which adds no information but exercises
# every rate-dependent code path at the rates the production system will use.
RATES_HZ = (25.0, 100.0, 200.0)
N_ROTATIONS = 3


def _resample(x, fs_from, fs_to):
    r = Fraction(fs_to / fs_from).limit_denominator(100)
    return _signal.resample_poly(x, r.numerator, r.denominator, axis=0)


def _cadence(accel, gravity, gyro, fs_hz):
    seg = steps.steady_state_segment(np.linalg.norm(accel, axis=1), fs_hz)
    sl = slice(seg["start"], seg["stop"])
    a, g, w = accel[sl], gravity[sl], gyro[sl]
    frame = orientation.build_frame(a, g, fs_hz)
    a_vert = orientation.vertical_component(a, g, mode=frame.mode)
    f_step = steps.estimate_step_frequency(a_vert, fs_hz)["f_step_hz"]
    det = steps.detect_steps(a_vert, fs_hz, f_step_hz=f_step)
    cs = steps.cadence_summary(det["step_times_s"])
    lat = lateral.analyse(w, g, det["step_indices"], fs_hz, f_step)
    return cs["cadence_spm"], det["n_steps"], lat["alternation_consistency"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--activity", default="jog")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    rows = []
    idents = loader.discover_trials(args.root, args.activity)
    for i, ident in enumerate(idents, 1):
        tr = loader.load_trial(ident.activity, ident.trial, ident.subject, args.root)
        base_cad, base_n, base_alt = _cadence(
            tr.user_accel, tr.gravity, tr.rotation_rate, tr.fs_hz
        )
        row = {
            "trial": ident.label, "subject": ident.subject,
            "cadence_spm": base_cad, "n_steps": base_n,
        }

        worst_rot = 0.0
        for seed in range(N_ROTATIONS):
            R = dsp.random_rotation_matrix(seed)
            c, n, a = _cadence(
                tr.user_accel @ R.T, tr.gravity @ R.T, tr.rotation_rate @ R.T, tr.fs_hz
            )
            worst_rot = max(worst_rot, abs(c - base_cad))
            row[f"rot{seed}_step_count_match"] = bool(n == base_n)
            row[f"rot{seed}_alternation_delta"] = abs(a - base_alt)
        row["max_rotation_cadence_delta_spm"] = worst_rot

        for fs_to in RATES_HZ:
            g = _resample(tr.gravity, tr.fs_hz, fs_to)
            g = g / np.linalg.norm(g, axis=1, keepdims=True)
            try:
                c, n, a = _cadence(
                    _resample(tr.user_accel, tr.fs_hz, fs_to), g,
                    _resample(tr.rotation_rate, tr.fs_hz, fs_to), fs_to,
                )
                row[f"cadence_at_{fs_to:g}Hz"] = c
                row[f"rel_err_at_{fs_to:g}Hz"] = abs(c - base_cad) / base_cad
            except Exception as exc:
                row[f"cadence_at_{fs_to:g}Hz"] = np.nan
                row[f"rel_err_at_{fs_to:g}Hz"] = np.nan
                row[f"error_at_{fs_to:g}Hz"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(f"  [{i}/{len(idents)}] {ident}", end="\r")
    print()

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "invariance.csv", index=False)

    rot_cols = [c for c in table if c.endswith("_step_count_match")]
    print(f"\nwrote {OUT / 'invariance.csv'}")
    print(f"orientation invariance: step count preserved in "
          f"{int(table[rot_cols].to_numpy().sum())}/{table[rot_cols].size} rotated runs")
    print(f"  max cadence change under rotation: "
          f"{table['max_rotation_cadence_delta_spm'].max():.2e} spm")
    for fs_to in RATES_HZ:
        col = f"rel_err_at_{fs_to:g}Hz"
        print(f"sample rate {fs_to:g} Hz: median |cadence error| "
              f"{100 * table[col].median():.3f}%, worst {100 * table[col].max():.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
