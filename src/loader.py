"""Stage 1 -- loader and explorer for the MotionSense DeviceMotion dataset.

Layout was verified against the dataset's own README (not assumed)::

    <root>/A_DeviceMotion_data/<activity>_<trial>/sub_<subject>.csv

Each CSV carries an unnamed integer row-index column followed by the 12
DeviceMotion features. There is **no timestamp column anywhere in this
dataset** -- see `check_integrity` for what that costs us.

Trial codes come from the README's ``TRIAL_CODES`` table; jogging is trials
9 and 16 for each of the 24 subjects (48 trials total).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --- dataset facts -------------------------------------------------------

# Nominal rate stated by the dataset README. This is the ONLY place the
# number 50 appears in this package, and it is only ever used as a default
# argument -- every function that consumes a rate takes `fs_hz` explicitly.
DEFAULT_FS_HZ = 50.0

ATTITUDE_COLUMNS = ["attitude.roll", "attitude.pitch", "attitude.yaw"]
GRAVITY_COLUMNS = ["gravity.x", "gravity.y", "gravity.z"]
ROTATION_RATE_COLUMNS = ["rotationRate.x", "rotationRate.y", "rotationRate.z"]
USER_ACCEL_COLUMNS = [
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
]
DEVICE_MOTION_COLUMNS = (
    ATTITUDE_COLUMNS + GRAVITY_COLUMNS + ROTATION_RATE_COLUMNS + USER_ACCEL_COLUMNS
)

# From the README's TRIAL_CODES dict.
ACTIVITY_TRIALS: dict[str, tuple[int, ...]] = {
    "dws": (1, 2, 11),
    "ups": (3, 4, 12),
    "wlk": (7, 8, 15),
    "jog": (9, 16),
    "std": (6, 14),
    "sit": (5, 13),
}
JOG_TRIALS = ACTIVITY_TRIALS["jog"]
N_SUBJECTS = 24

DEVICE_MOTION_DIRNAME = "A_DeviceMotion_data"
SUBJECT_INFO_FILENAME = "data_subjects_info.csv"

# A run of identical consecutive samples this long in userAcceleration would
# indicate a stuck sensor / dropped-sample hold. 0.1 s is ~5 samples at
# 50 Hz and ~20 at 200 Hz -- far longer than any real dwell in a signal
# quantised at ~1e-6 g, but short enough to catch a genuine hold quickly.
FLATLINE_SECONDS = 0.1


# --- identifiers and containers ------------------------------------------


@dataclass(frozen=True)
class TrialID:
    """Addresses one CSV: one subject performing one trial of one activity."""

    activity: str
    trial: int
    subject: int

    def __str__(self) -> str:  # e.g. "jog_9/sub_3"
        return f"{self.activity}_{self.trial}/sub_{self.subject}"

    @property
    def label(self) -> str:
        return f"{self.activity}_{self.trial}"


@dataclass
class Trial:
    """A loaded trial: signals on a time index, plus integrity findings."""

    ident: TrialID
    fs_hz: float
    df: pd.DataFrame
    path: Path
    integrity: dict = field(default_factory=dict)

    # -- convenience views. Each returns an (n, 3) float array in the
    #    *sensor* frame; stage 2 is what rotates them into anatomy.
    @property
    def n_samples(self) -> int:
        return len(self.df)

    @property
    def duration_s(self) -> float:
        """Duration implied by sample count and rate.

        Note this is *derived*, not measured: with no timestamps in the file
        we cannot distinguish a 97.2 s trial from a 100 s trial that dropped
        140 samples.
        """
        return self.n_samples / self.fs_hz

    @property
    def t(self) -> np.ndarray:
        return self.df.index.to_numpy()

    @property
    def gravity(self) -> np.ndarray:
        return self.df[GRAVITY_COLUMNS].to_numpy(float)

    @property
    def user_accel(self) -> np.ndarray:
        return self.df[USER_ACCEL_COLUMNS].to_numpy(float)

    @property
    def rotation_rate(self) -> np.ndarray:
        return self.df[ROTATION_RATE_COLUMNS].to_numpy(float)

    @property
    def attitude(self) -> np.ndarray:
        return self.df[ATTITUDE_COLUMNS].to_numpy(float)


# --- path resolution -----------------------------------------------------


def default_data_root() -> Path:
    """Where to look for the dataset if the caller does not say.

    Order: ``$MOTIONSENSE_ROOT``, then ``<repo>/data/motion-sense/data``,
    then ``~/motion-sense/data``. ``scripts/fetch_data.py`` populates the
    second of these.
    """
    env = os.environ.get("MOTIONSENSE_ROOT")
    if env:
        return Path(env).expanduser()
    repo = Path(__file__).resolve().parent.parent
    candidates = [
        repo / "data" / "motion-sense" / "data",
        Path.home() / "motion-sense" / "data",
    ]
    for c in candidates:
        if (c / DEVICE_MOTION_DIRNAME).is_dir():
            return c
    return candidates[0]


def resolve_data_root(root: str | Path | None = None) -> Path:
    """Validate that `root` really contains the DeviceMotion folder."""
    p = Path(root).expanduser() if root is not None else default_data_root()
    if not (p / DEVICE_MOTION_DIRNAME).is_dir():
        raise FileNotFoundError(
            f"{p / DEVICE_MOTION_DIRNAME} not found. Point MOTIONSENSE_ROOT at the "
            f"dataset's `data/` directory, or run scripts/fetch_data.py."
        )
    return p


def trial_dir(root: str | Path | None, activity: str, trial: int) -> Path:
    return resolve_data_root(root) / DEVICE_MOTION_DIRNAME / f"{activity}_{trial}"


def trial_path(
    root: str | Path | None, activity: str, trial: int, subject: int
) -> Path:
    return trial_dir(root, activity, trial) / f"sub_{subject}.csv"


def list_subjects(root: str | Path | None, activity: str, trial: int) -> list[int]:
    """Subject codes actually present on disk for one activity/trial."""
    d = trial_dir(root, activity, trial)
    subs = []
    for f in d.glob("sub_*.csv"):
        try:
            subs.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(subs)


def discover_trials(
    root: str | Path | None = None,
    activity: str = "jog",
    trials: "tuple[int, ...] | None" = None,
) -> list[TrialID]:
    """Enumerate every (activity, trial, subject) present on disk."""
    if trials is None:
        trials = ACTIVITY_TRIALS[activity]
    out: list[TrialID] = []
    for tr in trials:
        for sub in list_subjects(root, activity, tr):
            out.append(TrialID(activity, tr, sub))
    return out


def load_subject_info(root: str | Path | None = None) -> pd.DataFrame:
    """Demographics table (code, weight, height, age, gender)."""
    p = resolve_data_root(root) / SUBJECT_INFO_FILENAME
    # utf-8-sig: the shipped file starts with a BOM, which otherwise turns
    # the first column name into "﻿code".
    return pd.read_csv(p, encoding="utf-8-sig")


# --- integrity -----------------------------------------------------------


def check_integrity(
    raw: pd.DataFrame, fs_hz: float, index_column: str | None
) -> dict:
    """Everything we can honestly say about this trial's sampling.

    The headline finding is negative and applies to the whole dataset:
    **there is no timestamp column**, so no wall-clock jitter, no dropped
    samples and no rate drift can be detected. The row index is a dense
    counter written at export time; it is contiguous by construction and
    proves nothing about acquisition. Everything below is therefore a check
    on *file* integrity, not on *sampling* integrity.
    """
    n = len(raw)
    out: dict = {
        "n_samples": n,
        "duration_s": n / fs_hz,
        "fs_hz_assumed": fs_hz,
        # This is the load-bearing caveat, carried in the record itself.
        "has_timestamp_column": False,
        "timestamp_irregularities_detectable": False,
        "index_column": index_column,
    }

    if index_column is not None:
        idx = raw[index_column].to_numpy()
        expected = np.arange(n)
        out["index_contiguous"] = bool(np.array_equal(idx, expected))
        d = np.diff(idx) if n > 1 else np.array([], dtype=idx.dtype)
        out["index_step_min"] = int(d.min()) if d.size else None
        out["index_step_max"] = int(d.max()) if d.size else None
        out["index_gaps"] = int((d != 1).sum()) if d.size else 0
    else:
        out["index_contiguous"] = None
        out["index_gaps"] = None

    sig = raw[DEVICE_MOTION_COLUMNS].to_numpy(float)
    out["n_nan"] = int(np.isnan(sig).sum())
    out["n_inf"] = int(np.isinf(sig).sum())
    out["n_duplicate_rows"] = int(raw.duplicated().sum())

    # Gravity magnitude. CoreMotion reports gravity in g and normalises it as
    # part of the fusion output, so this is ~1.000000 by construction. It is a
    # corruption check only -- it is NOT evidence that the fusion is tracking.
    gmag = np.linalg.norm(raw[GRAVITY_COLUMNS].to_numpy(float), axis=1)
    out["gravity_mag_mean"] = float(gmag.mean())
    out["gravity_mag_std"] = float(gmag.std())
    out["gravity_mag_min"] = float(gmag.min())
    out["gravity_mag_max"] = float(gmag.max())

    ua = raw[USER_ACCEL_COLUMNS].to_numpy(float)
    out["user_accel_rms_g"] = float(np.sqrt((ua**2).sum(axis=1).mean()))
    rr = raw[ROTATION_RATE_COLUMNS].to_numpy(float)
    out["rotation_rate_rms_rad_s"] = float(np.sqrt((rr**2).sum(axis=1).mean()))

    # Stuck-sensor / sample-hold detection: longest run of bit-identical
    # consecutive userAcceleration triples.
    if n > 1:
        same = np.all(np.isclose(ua[1:], ua[:-1], rtol=0, atol=0), axis=1)
        longest = _longest_true_run(same) + 1 if same.any() else 1
    else:
        longest = n
    out["longest_flatline_samples"] = int(longest)
    out["longest_flatline_s"] = float(longest / fs_hz)
    out["flatline_suspect"] = bool(longest / fs_hz >= FLATLINE_SECONDS)

    # attitude.yaw is a wrapped angle in (-pi, pi]; count the wraps so a
    # consumer who differentiates it knows to unwrap first.
    yaw = raw["attitude.yaw"].to_numpy(float)
    out["yaw_wraps"] = int((np.abs(np.diff(yaw)) > np.pi).sum()) if n > 1 else 0

    problems = []
    if out["n_nan"] or out["n_inf"]:
        problems.append("non-finite samples")
    if out["index_contiguous"] is False:
        problems.append("row index not contiguous")
    if out["flatline_suspect"]:
        problems.append(f"flatline {out['longest_flatline_s']:.2f}s")
    if not (0.99 < out["gravity_mag_mean"] < 1.01):
        problems.append("gravity magnitude off unit")
    out["problems"] = "; ".join(problems)
    out["clean"] = not problems
    return out


def _longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a boolean array."""
    if mask.size == 0:
        return 0
    # Difference-of-cumsum trick over run boundaries.
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    if edges.size == 0:
        return 0
    runs = edges[1::2] - edges[0::2]
    return int(runs.max()) if runs.size else 0


# --- loading -------------------------------------------------------------


def load_trial(
    activity: str = "jog",
    trial: int = 9,
    subject: int = 1,
    root: str | Path | None = None,
    fs_hz: float = DEFAULT_FS_HZ,
) -> Trial:
    """Load one trial onto a seconds-valued time index derived from `fs_hz`.

    The index is `arange(n) / fs_hz`, i.e. *reconstructed* uniform time. It
    is the only time base available: see `check_integrity`.
    """
    path = trial_path(root, activity, trial, subject)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)

    # The first column is an unnamed export counter ("Unnamed: 0"). Identify
    # it positionally rather than by name so a re-export that names it
    # differently still loads.
    index_column = None
    if raw.columns[0] not in DEVICE_MOTION_COLUMNS:
        index_column = raw.columns[0]

    missing = [c for c in DEVICE_MOTION_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")

    if fs_hz <= 0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")

    integrity = check_integrity(raw, fs_hz, index_column)

    df = raw[DEVICE_MOTION_COLUMNS].astype(float).copy()
    df.index = pd.Index(np.arange(len(df), dtype=float) / fs_hz, name="t_s")

    return Trial(
        ident=TrialID(activity, trial, subject),
        fs_hz=float(fs_hz),
        df=df,
        path=path,
        integrity=integrity,
    )


def load_all(
    activity: str = "jog",
    trials: "tuple[int, ...] | None" = None,
    root: str | Path | None = None,
    fs_hz: float = DEFAULT_FS_HZ,
):
    """Yield every trial for an activity, in (trial, subject) order."""
    for ident in discover_trials(root, activity, trials):
        yield load_trial(ident.activity, ident.trial, ident.subject, root, fs_hz)


def summarize_trials(
    activity: str = "jog",
    trials: "tuple[int, ...] | None" = None,
    root: str | Path | None = None,
    fs_hz: float = DEFAULT_FS_HZ,
) -> pd.DataFrame:
    """One row per trial: duration, sample count, integrity findings."""
    rows = []
    for tr in load_all(activity, trials, root, fs_hz):
        row = {
            "activity": tr.ident.activity,
            "trial": tr.ident.trial,
            "subject": tr.ident.subject,
        }
        row.update(tr.integrity)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["trial", "subject"]).reset_index(drop=True)
