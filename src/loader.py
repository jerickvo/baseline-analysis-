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


# CoreMotion's `userAcceleration` is the NEGATIVE of the kinematic
# acceleration, in g. Apple's accelerometer convention reports the specific
# force with the sign that makes a device at rest read its own `gravity`
# vector -- (0, 0, -1) lying face up -- and a device in free fall read 0.
# `userAcceleration` is defined as that reading minus `gravity`, so in free
# fall, when the body is accelerating at 1 g TOWARD the earth, it equals
# -gravity and points AWAY from the earth. Every stage here reasons in
# kinematic polarity (positive = accelerating away from the earth: the
# stance-phase push, the footstrike deceleration), so the sign is converted
# once, at this boundary, and `Trial.df` keeps the file's own values.
#
# Verified on MotionSense rather than taken from documentation: at samples
# where |userAcceleration + gravity| < 0.25 g -- the hardware reading near
# zero, i.e. the flight phase -- the uncorrected vertical component read
# +0.99 g on 48 of 48 jog trials, where physics requires -1 g. The same
# API writes the logger's motion.csv, so the same conversion applies there.
USER_ACCEL_KINEMATIC_SIGN = -1.0


# --- identifiers and containers ------------------------------------------


@dataclass(frozen=True)
class TrialID:
    """Addresses one recording.

    MotionSense trials are (activity, trial, subject). A BaselineLogger
    session carries its folder name in `session_id` instead; the three
    MotionSense fields are then placeholders so every consumer sees one
    type.
    """

    activity: str
    trial: int
    subject: int
    session_id: str | None = None

    def __str__(self) -> str:  # e.g. "jog_9/sub_3" or "20260828T134502Z"
        if self.session_id is not None:
            return self.session_id
        return f"{self.activity}_{self.trial}/sub_{self.subject}"

    @property
    def label(self) -> str:
        if self.session_id is not None:
            return self.activity  # the session label typed on the phone
        return f"{self.activity}_{self.trial}"


@dataclass
class Trial:
    """A loaded trial: signals on a time index, plus integrity findings."""

    ident: TrialID
    fs_hz: float
    df: pd.DataFrame
    path: Path
    integrity: dict = field(default_factory=dict)
    # Present only for BaselineLogger sessions: cleaned GPS fixes and the
    # session.json contents. MotionSense has neither.
    gps: pd.DataFrame | None = None
    metadata: dict | None = None

    # -- convenience views. Each returns an (n, 3) float array in the
    #    *sensor* frame; stage 2 is what rotates them into anatomy.
    @property
    def n_samples(self) -> int:
        return len(self.df)

    @property
    def duration_s(self) -> float:
        """Duration of the record.

        Measured from the hardware timestamps when the record has them (a
        logger session), otherwise *derived* from sample count and rate:
        with no timestamps in the file we cannot distinguish a 97.2 s trial
        from a 100 s trial that dropped 140 samples.
        """
        if "t_hw" in self.df.columns and self.n_samples > 1:
            t_hw = self.df["t_hw"].to_numpy(float)
            return float(t_hw[-1] - t_hw[0])
        return self.n_samples / self.fs_hz

    @property
    def sample_times_s(self) -> np.ndarray | None:
        """Per-sample hardware times on the app's origin, or None.

        Stage 3 timestamps detected steps from this when it exists, so a
        dropped sample cannot shift every later step time by one period."""
        if "t_hw" in self.df.columns:
            return self.df["t_hw"].to_numpy(float)
        return None

    @property
    def t(self) -> np.ndarray:
        return self.df.index.to_numpy()

    @property
    def gravity(self) -> np.ndarray:
        return self.df[GRAVITY_COLUMNS].to_numpy(float)

    @property
    def user_accel(self) -> np.ndarray:
        """User acceleration in the sensor frame, in **kinematic** polarity.

        `df` keeps the columns exactly as CoreMotion recorded them; this
        view negates them. See `USER_ACCEL_KINEMATIC_SIGN` for why.
        """
        return USER_ACCEL_KINEMATIC_SIGN * self.df[USER_ACCEL_COLUMNS].to_numpy(float)

    @property
    def user_accel_as_recorded(self) -> np.ndarray:
        """The same columns in CoreMotion's own sign, for anyone comparing
        against the file. Nothing in the pipeline uses this."""
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


# --- BaselineLogger sessions ----------------------------------------------
#
# The iOS logger (jerickvo/baseline-ios) writes one folder per session:
#
#     <folder>/motion.csv      t,ax,ay,az,gx,gy,gz,rx,ry,rz,qw,qx,qy,qz
#     <folder>/accel_raw.csv   t,ax,ay,az            (200 Hz raw, gravity in)
#     <folder>/gps.csv         t,latitude,longitude,speed,horizontalAccuracy,
#                              altitude,speedAccuracy,verticalAccuracy
#                              (six columns in files from before the two
#                              accuracy columns existed; loaded by name)
#     <folder>/session.json    counts, achieved rates, gap and drop statistics,
#                              rows lost to disk, markers
#
# Unlike MotionSense, `t` is a REAL per-sample hardware timestamp (seconds
# since session start on the monotonic clock), so sampling irregularities
# are measurable here and are measured. The vector columns are the same
# CoreMotion quantities as MotionSense's, in the same device (body) frame,
# so once the columns are renamed every later stage runs unchanged.

LOGGER_MOTION_FILENAME = "motion.csv"
LOGGER_ACCEL_FILENAME = "accel_raw.csv"
LOGGER_GPS_FILENAME = "gps.csv"
SESSION_JSON_FILENAME = "session.json"
LOGGER_TIME_COLUMN = "t"
# gps.csv header as SessionRecorder.swift writes it today. The test fixture
# imports this so the Python side cannot drift from what it claims to read.
LOGGER_GPS_COLUMNS = [
    "t", "latitude", "longitude", "speed", "horizontalAccuracy", "altitude",
    "speedAccuracy", "verticalAccuracy",
]
LOGGER_MOTION_COLUMNS = [
    "t", "ax", "ay", "az", "gx", "gy", "gz", "rx", "ry", "rz", "qw", "qx", "qy", "qz",
]
LOGGER_TO_DEVICE_MOTION = {
    "ax": "userAcceleration.x", "ay": "userAcceleration.y", "az": "userAcceleration.z",
    "gx": "gravity.x", "gy": "gravity.y", "gz": "gravity.z",
    "rx": "rotationRate.x", "ry": "rotationRate.y", "rz": "rotationRate.z",
}
# A sample interval more than this multiple of the measured median interval
# is a gap. The same multiple as the app's GapTracker, so the two agree on
# what a gap is on a steady stream (tests/test_gaptracker_port.py checks
# that on single drops). They are not the same algorithm: the app measures
# its median over a sliding 128-delta window, this loader over the whole
# record, so across a mid-session change in delivered rate the two counts
# differ (the app reports the 65-128 deltas until it recalibrates, this
# loader every delta of the slower section), and the loader's is the one
# any analysis decision uses.
GAP_INTERVAL_MULTIPLE = 3.0
# GPS fixes coarser than this (median horizontalAccuracy, metres) are
# flagged. A phone with Precise Location on reports 5-10 m in the open;
# with it off, CoreLocation reports fixes in the thousands of metres with a
# positive accuracy that passes every sign test. 20 m is the coarsest fix a
# pace-per-kilometre model could still use.
GPS_COARSE_ACCURACY_M = 20.0
# session.json fields the loader carries into the integrity record, with
# the integrity key they land under. Absent in files from before the field
# existed, in which case the value is None.
SESSION_JSON_INTEGRITY_FIELDS = {
    "motionSampleCount": "metadata_sample_count",
    "motionGapCount": "metadata_gap_count",
    "accelGapCount": "metadata_accel_gap_count",
    "largestGapSeconds": "metadata_largest_gap_s",
    "motionDroppedSampleEstimate": "metadata_dropped_estimate",
    "motionNonMonotonicCount": "metadata_nonmonotonic",
    "csvRowsLost": "metadata_rows_lost",
    "gpsStaleFixesSkipped": "metadata_gps_stale_skipped",
    "achievedMotionHz": "metadata_achieved_hz",
}
# A gap is not the only way to lose samples. One dropped sample makes a 2x
# interval and two make exactly 3x: the gap rule never counts either, so a
# stream can read "no gaps" while shedding a sample every few seconds. Any
# interval beyond 1.5x the median is therefore also converted into an
# estimate of samples missing -- round(interval / median) - 1 -- and that
# count is reported separately from gaps. 1.5x sits halfway between a
# normal interval (1x, plus jitter of a few percent) and one dropped sample
# (2x).
DROPPED_SAMPLE_INTERVAL_MULTIPLE = 1.5
# Jitter, as a fraction of the median interval, above which the record is
# no longer "uniform enough" for the filters. A tenth of the sample period
# is 1 ms at 100 Hz; the phase error a 1 ms timing wobble puts on a 3 Hz
# fundamental is 0.02 rad, far below anything the band-pass or the peak
# picker can resolve. Hardware timestamps from CoreMotion sit an order of
# magnitude under this.
UNIFORM_JITTER_FRACTION = 0.10


def quaternion_to_euler(qw, qx, qy, qz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll, pitch, yaw (radians) from a unit quaternion, ZYX convention.

    Provided so a logger session carries the same 12 columns as MotionSense.
    Nothing analytical consumes attitude -- every stage works from gravity,
    userAcceleration and rotationRate -- and CoreMotion's own roll/pitch/yaw
    may differ from this standard aerospace convention in axis order and
    sign. Treat these three columns as display-only.
    """
    qw, qx, qy, qz = (np.asarray(v, float) for v in (qw, qx, qy, qz))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def check_timestamps(t: np.ndarray, gap_multiple: float = GAP_INTERVAL_MULTIPLE) -> dict:
    """Everything the timestamp column says about how the data was sampled.

    This is the measurement MotionSense could never support. The median
    interval defines the achieved rate; jitter is the robust spread around
    it; a gap is an interval beyond `gap_multiple` times the median; and
    non-monotonic intervals (duplicated or reordered samples) are counted
    separately because the app's tracker does not see them at all.
    """
    t = np.asarray(t, float)
    n = len(t)
    if n and not np.all(np.isfinite(t)):
        # A NaN timestamp makes two NaN deltas, which every comparison
        # below silently excludes: the gap, drop and monotonicity counts
        # would all pass and the session would load as clean.
        n_bad = int((~np.isfinite(t)).sum())
        raise ValueError(
            f"{n_bad} non-finite timestamp(s) of {n}: the sample clock is broken, "
            f"no rate or gap can be measured"
        )
    if n < 2:
        return {
            "n_samples": n, "measured_fs_hz": np.nan, "median_interval_s": np.nan,
            "jitter_ms": np.nan, "n_gaps": 0, "largest_gap_s": 0.0, "gaps": [],
            "n_nonmonotonic": 0, "t_start_s": float(t[0]) if n else np.nan,
            "t_end_s": float(t[-1]) if n else np.nan, "uniform": False,
        }
    d = np.diff(t)
    positive = d[d > 0]
    median = float(np.median(positive)) if positive.size else np.nan
    fs = 1.0 / median if median and np.isfinite(median) else np.nan
    # 1.4826 * MAD, in ms: robust to the gaps it is measured alongside.
    jitter_ms = float(1.4826 * np.median(np.abs(positive - median)) * 1000.0) if positive.size else np.nan
    threshold = gap_multiple * median if np.isfinite(median) else np.inf
    gap_idx = np.flatnonzero(d > threshold)
    gaps = [(int(i), float(t[i]), float(d[i])) for i in gap_idx]  # (index, at_s, length_s)
    if np.isfinite(median):
        long_ = d[d > DROPPED_SAMPLE_INTERVAL_MULTIPLE * median]
        n_dropped = int(np.sum(np.round(long_ / median) - 1)) if long_.size else 0
    else:
        n_dropped = 0
    n_nonmonotonic = int((d <= 0).sum())
    jitter_ok = np.isfinite(jitter_ms) and (jitter_ms / 1000.0) < UNIFORM_JITTER_FRACTION * median
    return {
        "n_samples": int(n),
        "measured_fs_hz": float(fs),
        "median_interval_s": median,
        "jitter_ms": jitter_ms,
        "n_gaps": int(len(gaps)),
        "largest_gap_s": float(d[gap_idx].max()) if gap_idx.size else 0.0,
        "gaps": gaps,
        # Samples estimated missing from sub-gap-threshold long intervals.
        # Independent of `n_gaps`, and the number the app's own tracker
        # cannot see.
        "n_dropped_estimate": n_dropped,
        "n_nonmonotonic": n_nonmonotonic,
        "t_start_s": float(t[0]),
        "t_end_s": float(t[-1]),
        # "Uniform enough" for the filters: no gaps, no reordering, and
        # jitter under UNIFORM_JITTER_FRACTION of the sample period.
        "uniform": bool(len(gaps) == 0 and n_nonmonotonic == 0 and jitter_ok),
    }


def load_logger_gps(folder: str | Path) -> tuple[pd.DataFrame, dict]:
    """gps.csv with the rows no analysis should see removed, and a count of them.

    CoreLocation commonly delivers a *cached* last-known fix first, stamped
    minutes or hours before the session started. The app now skips those
    (and counts them as `gpsStaleFixesSkipped`); files written before it did
    carry them with a large negative `t`, so the filter stays. Speed is -1
    when CoreLocation could not compute it and horizontalAccuracy is
    negative when the fix is invalid. All are dropped here, and counted,
    rather than silently entering a pace estimate.

    A fix can also be *valid and coarse*: with Precise Location off the
    accuracy is positive and enormous. The median and 95th-percentile
    horizontalAccuracy of the kept fixes are reported so the quality gate
    can say so (`GPS_COARSE_ACCURACY_M`).
    """
    empty = {
        "gps_rows": 0, "gps_dropped_stale": 0, "gps_dropped_invalid": 0, "gps_kept": 0,
        "gps_accuracy_median_m": np.nan, "gps_accuracy_p95_m": np.nan,
    }
    path = Path(folder) / LOGGER_GPS_FILENAME
    if not path.is_file():
        return pd.DataFrame(), empty
    g = pd.read_csv(path)
    n0 = len(g)
    if n0 == 0:
        return g, empty
    stale = g["t"] < 0
    invalid = (g["speed"] < 0) | (g["horizontalAccuracy"] < 0)
    # Newer sessions carry speedAccuracy / verticalAccuracy (negative when
    # the corresponding value is invalid); older files predate the columns.
    if "speedAccuracy" in g.columns:
        invalid |= g["speedAccuracy"] < 0
    kept = g[~stale & ~invalid].reset_index(drop=True)
    acc = kept["horizontalAccuracy"].to_numpy(float) if len(kept) else np.empty(0)
    return kept, {
        "gps_rows": int(n0),
        "gps_dropped_stale": int(stale.sum()),
        "gps_dropped_invalid": int((invalid & ~stale).sum()),
        "gps_kept": int(len(kept)),
        "gps_accuracy_median_m": float(np.median(acc)) if acc.size else np.nan,
        "gps_accuracy_p95_m": float(np.percentile(acc, 95)) if acc.size else np.nan,
    }


def load_logger_session(
    folder: str | Path,
    on_gap: str = "raise",
    gap_multiple: float = GAP_INTERVAL_MULTIPLE,
) -> Trial:
    """Load one BaselineLogger session folder as a `Trial`.

    The sample rate is **measured from the timestamps**, never assumed, and
    the returned `Trial.fs_hz` is that measurement. Every later stage takes
    the rate as a parameter, so the pipeline runs at whatever rate the phone
    actually delivered.

    Because every filter downstream assumes a uniform grid, a session with a
    gap cannot be handed on as one continuous record: the uniform index
    would compress time across the gap and put every later timestamp in the
    wrong place. `on_gap` decides what happens:

    ``"raise"`` (default)
        Refuse. Matches the app's own policy -- its summary screen says to
        distrust a gapped session -- and matches how this package treats
        every other unusable input.
    ``"longest"``
        Keep the longest gap-free stretch and report what was cut, the same
        trade `steps.steady_state_segment` makes for handling transients.
        The dropped seconds land in `integrity["discarded_for_gaps_s"]`.

    Small timing jitter (sub-millisecond at 100 Hz) is measured and
    reported, and the record is placed on a uniform grid at the measured
    rate; the original timestamps are kept in the `t_hw` column.
    """
    if on_gap not in ("raise", "longest"):
        raise ValueError(f"on_gap must be 'raise' or 'longest', got {on_gap!r}")
    folder = Path(folder)
    motion_path = folder / LOGGER_MOTION_FILENAME
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    raw = pd.read_csv(motion_path)
    missing = [c for c in LOGGER_MOTION_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{motion_path}: missing logger columns {missing}")

    metadata = None
    json_path = folder / SESSION_JSON_FILENAME
    if json_path.is_file():
        import json
        with open(json_path, encoding="utf-8") as fh:
            metadata = json.load(fh)

    t_hw = raw[LOGGER_TIME_COLUMN].to_numpy(float)
    n_rows_written = len(raw)
    ts = check_timestamps(t_hw, gap_multiple)
    if not np.isfinite(ts["measured_fs_hz"]):
        raise ValueError(f"{motion_path}: cannot measure a sample rate from its timestamps")

    discarded_for_gaps = 0.0
    if ts["n_gaps"] or ts["n_nonmonotonic"]:
        if on_gap == "raise":
            raise ValueError(
                f"{folder.name}: {ts['n_gaps']} gap(s) (largest {ts['largest_gap_s']:.3f}s) and "
                f"{ts['n_nonmonotonic']} non-monotonic interval(s) in motion.csv. The filters "
                f"assume uniform sampling, so this session cannot be analysed as one record. "
                f"Pass on_gap='longest' to keep the longest clean stretch."
            )
        # Split at every gap or reversal and keep the longest piece.
        d = np.diff(t_hw)
        breaks = np.flatnonzero((d > gap_multiple * ts["median_interval_s"]) | (d <= 0)) + 1
        edges = np.r_[0, breaks, len(t_hw)]
        pieces = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        a, b = max(pieces, key=lambda p: p[1] - p[0])
        discarded_for_gaps = float((t_hw[-1] - t_hw[0]) - (t_hw[b - 1] - t_hw[a]))
        raw = raw.iloc[a:b].reset_index(drop=True)
        t_hw = t_hw[a:b]
        ts = check_timestamps(t_hw, gap_multiple)

    fs_hz = float(ts["measured_fs_hz"])
    df = pd.DataFrame(index=pd.Index(np.arange(len(raw), dtype=float) / fs_hz, name="t_s"))
    roll, pitch, yaw = quaternion_to_euler(raw["qw"], raw["qx"], raw["qy"], raw["qz"])
    df["attitude.roll"], df["attitude.pitch"], df["attitude.yaw"] = roll, pitch, yaw
    for src, dst in LOGGER_TO_DEVICE_MOTION.items():
        df[dst] = raw[src].to_numpy(float)
    df = df[DEVICE_MOTION_COLUMNS].copy()
    # Hardware timestamps stay on the APP's origin (session start), not on
    # the first sample: the first CMDeviceMotion sample arrives some hundred
    # milliseconds after Start, and the app's event markers and gps.csv are
    # on that origin. Re-zeroing here shifted every derived time off the
    # marker and GPS timeline by an amount that was then thrown away -- and
    # by minutes, under on_gap="longest", when a later piece won. The
    # uniform index `t_s` still starts at 0 for the kept record; the offset
    # between the two is `integrity["t_start_s"]`.
    df["t_hw"] = t_hw

    # File-level checks reuse the MotionSense machinery, then the timestamp
    # facts replace the "not measurable" placeholders it writes.
    integrity = check_integrity(df[DEVICE_MOTION_COLUMNS].reset_index(drop=True), fs_hz, None)
    integrity.update({
        "has_timestamp_column": True,
        "timestamp_irregularities_detectable": True,
        # Hardware time of the first kept sample on the app's origin: add it
        # to `t_s` to land on the marker / gps.csv timeline.
        "t_start_s": float(t_hw[0]),
        "measured_fs_hz": fs_hz,
        "jitter_ms": ts["jitter_ms"],
        "n_gaps": ts["n_gaps"],
        "largest_gap_s": ts["largest_gap_s"],
        "n_dropped_estimate": ts["n_dropped_estimate"],
        "n_nonmonotonic": ts["n_nonmonotonic"],
        "timestamps_uniform": ts["uniform"],
        "discarded_for_gaps_s": discarded_for_gaps,
        "duration_s": float(t_hw[-1] - t_hw[0]) if len(t_hw) > 1 else 0.0,
    })
    if metadata is not None:
        # Cross-check the app's own bookkeeping against the file it wrote.
        for src_key, dst_key in SESSION_JSON_INTEGRITY_FIELDS.items():
            integrity[dst_key] = metadata.get(src_key)
        n_meta = metadata.get("motionSampleCount")
        integrity["metadata_count_matches"] = (int(n_meta) == n_rows_written) if n_meta is not None else None
        integrity["metadata_in_progress"] = bool(metadata.get("inProgress", False))
        integrity["metadata_event_markers"] = metadata.get("eventMarkers") or []
    problems = [p for p in [integrity["problems"]] if p]
    if metadata is not None:
        rows_lost = integrity.get("metadata_rows_lost") or 0
        if rows_lost:
            problems.append(f"app reports {rows_lost} row(s) never reached disk")
        if integrity.get("metadata_count_matches") is False:
            problems.append(
                f"app counted {integrity['metadata_sample_count']} motion samples, file holds "
                f"{n_rows_written}"
            )
    if ts["n_gaps"]:
        problems.append(f"{ts['n_gaps']} gap(s)")
    if discarded_for_gaps > 0:
        problems.append(f"cut at gaps: {discarded_for_gaps:.1f}s discarded")
    if ts["n_dropped_estimate"]:
        problems.append(f"~{ts['n_dropped_estimate']} sample(s) dropped below the gap threshold")
    if ts["n_nonmonotonic"]:
        problems.append(f"{ts['n_nonmonotonic']} non-monotonic")
    if integrity.get("metadata_in_progress"):
        problems.append("session.json marked in-progress (app did not finish the session)")
    integrity["problems"] = "; ".join(problems)
    integrity["clean"] = not problems

    gps, gps_info = load_logger_gps(folder)
    integrity.update(gps_info)

    label = (metadata or {}).get("label") or "session"
    return Trial(
        ident=TrialID(activity=label, trial=0, subject=0, session_id=folder.name),
        fs_hz=fs_hz,
        df=df,
        path=motion_path,
        integrity=integrity,
        gps=gps,
        metadata=metadata,
    )

