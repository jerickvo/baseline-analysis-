# baseline-analysis — running gait signal core

Signal-processing core for a running gait analysis system, prototyped in plain Python
(numpy / scipy / pandas / matplotlib) against the MotionSense dataset.

This is a **prototyping repo**, not an app. Scripts and notebooks over CSV. No mobile code.

**Read [`REPORT.md`](REPORT.md) for the findings** — which stages work, which are shaky,
which failed, where pocket placement and 50 Hz are doing load-bearing work, and what this
dataset fundamentally cannot answer.

## The data, and what it is for

[MotionSense](https://github.com/mmalekzadeh/motion-sense) `A_DeviceMotion_data`, jog
trials only (trials 9 and 16), 24 subjects, iPhone 6s in the front trouser pocket, 50 Hz.

The production target is the **lower back at 100 Hz fused / 200 Hz raw**. This dataset is
used only to develop and stress-test algorithms that must be *placement- and rate-agnostic*.
Anything that silently depends on the pocket or on 50 Hz is a bug, and the code is built to
flag it: stage 2 returns a three-valued verdict rather than a number, stage 3 attributes
every out-of-band cadence to either the algorithm or the trial, and stage 4 reports null
models alongside every result.

## Setup

```bash
pip install numpy scipy pandas matplotlib jupyter pytest
python scripts/fetch_data.py            # clones MotionSense (~400 MB) and unpacks it
export MOTIONSENSE_ROOT=$PWD/data/motion-sense/data   # or leave unset; the loader finds it
```

## Run

```bash
python scripts/run_all.py                 # all 48 trials -> reports/*.csv, figures, verdict summary
python scripts/run_invariance_checks.py   # rotation + resample sweep -> reports/invariance.csv
python -m pytest tests/                   # 10 tests, incl. orientation and rate invariance
jupyter notebook notebooks/gait_pipeline_walkthrough.ipynb   # one subject, every stage, plots
```

## Layout

```
src/
  loader.py       stage 1  dataset access, reconstructed time index, integrity reporting
  orientation.py  stage 2  sensor -> anatomical frame, and the checks that can refute it
  steps.py        stage 3  band-pass, footstrike detection, cadence, failure attribution
  lateral.py      stage 4  exploratory left/right, with its null models
  dsp.py                   shared primitives (spectra, filters, autocorrelation, rotations)
  pipeline.py              end-to-end glue for one trial
  plotting.py              figures for every stage
scripts/
  fetch_data.py            fetch and unpack the dataset
  run_all.py               batch over all jog trials, write report tables and figures
  run_invariance_checks.py orientation and sample-rate invariance across all 48 trials
notebooks/
  gait_pipeline_walkthrough.ipynb   one subject end to end, plots at every stage
tests/
  test_invariance.py       the tests behind the "placement- and rate-agnostic" claim
reports/                   generated CSVs and figures
REPORT.md                  findings
```

## Design rules the code holds itself to

- **Sample rate is a parameter everywhere.** The number 50 appears once in the package
  (`loader.DEFAULT_FS_HZ`) and only as a default argument. Verified by resampling all 48
  trials to 25 / 100 / 200 Hz: cadence moves by a median of 0.03–0.05%, worst case 0.79%.
- **Filter cutoffs are multiples of the estimated step frequency**, never fixed Hz, so the
  filter is identical in relative terms at any cadence and any rate.
- **Every cutoff, threshold and window length has a comment saying why that value.** Where
  a value came from a sweep, the comment says what was swept and what the alternatives cost.
- **Orientation invariance is tested, not asserted.** Rotating every sensor vector by an
  arbitrary rotation changes cadence by exactly 0.00 spm across 144 rotated runs.
- **A failing stage fails loudly.** Stage 2 will return `vertical_only` or `failed` and name
  its reasons; stage 3 attributes flagged cadences to the algorithm or the trial; stage 4
  carries `ground_truth_available: False` in every record.

## Headline results

| | |
|---|---|
| Trials analysed | 48 (24 subjects × 2 jog trials), all files clean |
| Cadence | 148.5–183.7 spm; **47/48** inside 150–190 spm |
| Detector vs independent spectral estimate | agrees within 4% on every trial |
| Flagged trials | 1 (`jog_9/sub_4`, 148.5 spm) — attributed to the **trial**, not the algorithm |
| Stage-2 verdicts | 2 `ok`, 46 `vertical_only`, 0 `failed` |
| Static trial-constant frame valid | **0/48** — the pocket sensor tilts 15° median, 30° p95 within a trial |
| Mediolateral cross-check | passes **2/48** — the ML axis is unverified at this placement |
| Left/right alternation at detected contact | median 0.485, *below* its own surrogate null (0.561) |
| Orientation invariance | exact: 0.00 spm cadence change, 144/144 runs |

## Not built, deliberately

No iOS or Swift code. No vertical oscillation, braking impulse, or loading rate — they need
sample rates this dataset does not have. No pace normalisation or changepoint detection —
they need multi-session per-runner data this dataset does not contain.
