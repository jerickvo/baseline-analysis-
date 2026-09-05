# baseline-analysis — running gait signal core

Signal-processing core for a running gait analysis system, prototyped in plain Python
(numpy / scipy / pandas / matplotlib) against the MotionSense dataset and against sessions
recorded by the companion iOS logger, [`jerickvo/baseline-ios`](https://github.com/jerickvo/baseline-ios).

This is a **prototyping repo**, not an app. Scripts and notebooks over CSV. No mobile code.

**Read [`REPORT.md`](REPORT.md) for the findings** — including what the September 2026
audit found and fixed in two rounds (§0), which stages work, which are shaky, which failed,
where pocket placement and 50 Hz are doing load-bearing work, and what this data
fundamentally cannot answer.

## What exists, and what does not

Built: loader (MotionSense **and** BaselineLogger sessions), orientation resolution, step
detection and cadence (pooled over every running bout of a record), exploratory left/right
diagnostics, a data-quality gate. The app-format ingestion is proven on synthetic sessions
written in the app's documented format; **no session recorded by a device has been run
through it yet.**

Not built, deliberately: ground contact time, asymmetry, vertical oscillation, braking
impulse, loading rate (they need true initial contact and sample rates MotionSense lacks);
the per-runner pace model, residuals, within-run trajectory, changepoint detection (they need
multi-session per-runner data at known pace); any report UI.

## Setup

```bash
pip install numpy scipy pandas matplotlib jupyter pytest
python scripts/fetch_data.py            # clones MotionSense (~400 MB) and unpacks it
export MOTIONSENSE_ROOT=$PWD/data/motion-sense/data   # or leave unset; the loader finds it
```

## Run

```bash
python scripts/run_session.py <session_folder>   # one BaselineLogger session -> quality verdict
python scripts/run_all.py                          # all 48 MotionSense trials -> reports/*.csv + verdict summary
python scripts/run_invariance_checks.py            # rotation + resample sweep
python -m pytest tests/                            # 147 tests, incl. ground-truth frame recovery
jupyter notebook notebooks/gait_pipeline_walkthrough.ipynb
```

`run_session.py` exits 0 for `ok`, 1 for `partial`, 2 for `insufficient`, so the verdict can
gate a script. A session folder is what the app writes: `motion.csv`, `accel_raw.csv`,
`gps.csv`, `session.json`.

## Layout

```
src/
  loader.py       stage 1  MotionSense trials and BaselineLogger sessions; integrity reporting
  orientation.py  stage 2  sensor -> anatomical frame, and the checks that can refute it
  steps.py        stage 3  running-state, band-pass, step detection, cadence, attribution
  lateral.py      stage 4  exploratory left/right, with its null models
  dsp.py                   shared primitives
  pipeline.py              run_trial / run_session / run_stages, and assess_quality
  plotting.py              figures for every stage
scripts/
  fetch_data.py            fetch and unpack MotionSense
  run_session.py           the pipeline on one app session
  run_all.py               batch over MotionSense, write report tables and figures
  run_invariance_checks.py orientation and sample-rate invariance across all 48 trials
notebooks/
  gait_pipeline_walkthrough.ipynb   one subject end to end, plots at every stage
tests/
  test_invariance.py       rotation and rate invariance on real data
  test_failure_modes.py    degenerate, short and lying input
  test_integration.py      the app's format through the pipeline, against ground truth
  test_audit_regressions.py one test per audit finding (polarity, fabricated steps, parity, gaits, ...)
  test_gaptracker_port.py  Python port of the app's GapTracker, and where the two gap rules diverge
reports/                   generated CSVs and figures
REPORT.md                  findings
```

## Design rules the code holds itself to

- **Sample rate is a parameter everywhere.** For logger sessions it is *measured* from the
  timestamps, and step times are read from the hardware clock. The number 50 appears once
  in the package, as a default for MotionSense.
- **Polarity is kinematic.** CoreMotion's `userAcceleration` is the negative of the
  kinematic acceleration; the loader converts it once, and positive vertical means the body
  accelerating away from the earth everywhere downstream.
- **Filter cutoffs are multiples of the estimated step frequency**, never fixed Hz.
- **Every cutoff, threshold and window length has a comment saying why that value.**
- **Orientation invariance is tested, not asserted**: 0.00 spm change across 144 rotated runs,
  and true axes recovered to < 3° from arbitrary, upside-down and yawed synthetic placements.
- **A failing stage fails loudly, and the run gets one verdict.** Stage 2 returns
  `ok` / `vertical_only` / `failed`; stage 3 attributes a flagged cadence to
  `sample_rate` / `algorithm` / `trial`; `assess_quality` rolls everything into
  `ok` / `partial` / `insufficient` with the reasons. A record with no locomotion, more than
  one gait, an ambiguous spectral seed or no defensible cadence is `insufficient`.
- **Stage 4 never yields a side verdict.** Its alternation statistic is gait-locked by
  construction, so no signal-only null can turn it into left/right; it is reported with its
  controls and never gated on.
- **Data loss is reported, never silent.** Bouts too short to analyse, samples dropped below
  the gap threshold, seconds cut at gaps, rows the app lost to disk, stale and coarse GPS
  fixes — all columns.

## Headline results (MotionSense, 48 trials)

| | |
|---|---|
| Cadence (regular strides, pooled over bouts) | 149.1–183.8 spm; **47/48** inside 150–190 spm |
| Detector vs spectral seed | within 4.1 % on every trial (ratio 0.965–1.041) |
| Stride-interval CV | median 0.039, max 0.067 |
| Flagged | 1 (`jog_9/sub_4`, 149.1 spm) — attributed to the **trial** |
| Stage-2 verdicts | 2 `ok`, 46 `vertical_only`, 0 `failed` |
| Quality roll-up | 2 `ok`, 46 `partial` (horizontal frame unverified at the pocket) |
| Forward sign confident (phase criterion) | **37/48** — a centre-of-mass criterion, an inference at the pocket |
| Static trial-constant frame valid | **0/48** |
| Orientation invariance | exact: 0.00 spm, 144/144 |
| Rate invariance (25 / 100 / 200 Hz) | worst 0.33 % / 0.24 % / 0.27 % |
| Side classification | none, by design |
