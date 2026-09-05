#!/usr/bin/env python3
"""Run the pipeline on one BaselineLogger session folder and print the verdict.

    python scripts/run_session.py <session_folder> [--on-gap raise|longest] [--json]

The session folder is what the app writes: motion.csv, accel_raw.csv,
gps.csv, session.json. The sample rate is measured from the timestamps.
Exit status is 0 for "ok", 1 for "partial", 2 for "insufficient", so the
verdict can gate a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import pipeline  # noqa: E402

EXIT = {"ok": 0, "partial": 1, "insufficient": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--on-gap", default="raise", choices=["raise", "longest"])
    ap.add_argument("--json", action="store_true", help="print the flat row as JSON")
    args = ap.parse_args()

    try:
        result = pipeline.run_session(args.folder, on_gap=args.on_gap)
    except (FileNotFoundError, ValueError) as exc:
        print(f"INSUFFICIENT: {exc}")
        return 2

    row = pipeline.flatten(result)
    q = result["quality"]
    tr = result["trial"]
    if args.json:
        print(json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}, indent=1, default=str))
        return EXIT[q["verdict"]]

    print(f"session      {tr.ident}  label='{tr.ident.label}'")
    print(f"sampling     {tr.fs_hz:.2f} Hz measured, jitter {tr.integrity['jitter_ms']:.2f} ms, "
          f"gaps {tr.integrity['n_gaps']}, dropped ~{tr.integrity['n_dropped_estimate']}, "
          f"duration {tr.duration_s:.1f} s")
    print(f"steady       {row['steady_s_all_bouts']:.1f} s of running in {row['n_bouts_analysed']} bout(s) "
          f"(longest {result['steady_seconds']:.1f} s); {row['unanalysed_steady_s']:.1f} s in bouts too short to analyse")
    print(f"frame        {row['frame_verdict']}  (forward sign confident: {row['forward_sign_confident']})")
    print(f"cadence      {row['cadence_spm']:.1f} spm pooled over the bouts  (longest bout "
          f"{row['cadence_spm_primary_bout']:.1f}, spectral {row['cadence_spm_spectral']:.1f})  "
          f"cause={row['cadence_failure_cause']}")
    print(f"alternation  {row['alternation_consistency']:.2f} at the step marker vs {row['alternation_surrogate_null']:.2f} "
          f"for its surrogate -- parity of an unanchored sign, NOT a left/right result")
    print(f"QUALITY      {q['verdict'].upper()}")
    for b in q["blockers"]:
        print(f"  blocker  {b}")
    for c in q["caveats"]:
        print(f"  caveat   {c}")
    return EXIT[q["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
