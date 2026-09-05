"""A line-for-line Python port of BaselineLogger/Recording/GapTracker.swift,
and the tests that pin what the Swift and Python gap rules do and do not
share.

The port exists because no Swift toolchain runs here. It mirrors the struct's
state machine -- warm-up, ring window, median recalibration, the 3x gap rule,
the 1.5x dropped-sample estimate, the non-monotonic guard -- so a change to
one side can be checked against the other. Keep it in step with the Swift
file by hand; nothing enforces that automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import loader  # noqa: E402


class GapTrackerPort:
    """Port of GapTracker.swift (window 128, recalibrate every 64)."""

    WARMUP_DELTAS = 128
    WINDOW_SIZE = 128
    RECALIBRATE_EVERY = 64
    GAP_MULTIPLE = 3.0
    DROPPED_MULTIPLE = 1.5

    def __init__(self, expected_interval: float):
        self.expected_interval = expected_interval
        self.gap_count = 0
        self.largest_gap_s = 0.0
        self.dropped_sample_estimate = 0
        self.non_monotonic_count = 0
        self.median_interval = expected_interval
        self.threshold_s = expected_interval * self.GAP_MULTIPLE
        self.calibrated = False
        self.deltas_seen = 0
        self._previous = None
        self._window: list[float] = []
        self._window_next = 0
        self._held: list[float] = []

    def record(self, timestamp: float) -> None:
        if self._previous is None:
            self._previous = timestamp
            return
        delta = timestamp - self._previous
        if delta <= 0:
            # A duplicated or reordered sample: counted, kept out of the
            # median, and the reference timestamp is NOT advanced.
            self.non_monotonic_count += 1
            return
        self._previous = timestamp
        self._remember(delta)
        if not self.calibrated:
            self._held.append(delta)
            if self.deltas_seen >= self.WARMUP_DELTAS:
                self._calibrate()
                for held in self._held:
                    self._classify(held)
                self._held = []
            return
        self._classify(delta)
        if self.deltas_seen % self.RECALIBRATE_EVERY == 0:
            self._calibrate()

    def _remember(self, delta: float) -> None:
        self.deltas_seen += 1
        if len(self._window) < self.WINDOW_SIZE:
            self._window.append(delta)
        else:
            self._window[self._window_next] = delta
            self._window_next = (self._window_next + 1) % self.WINDOW_SIZE

    def _calibrate(self) -> None:
        if not self._window:
            return
        median = sorted(self._window)[len(self._window) // 2]
        if median <= 0:
            return
        self.median_interval = median
        self.threshold_s = median * self.GAP_MULTIPLE
        self.calibrated = True

    def _classify(self, delta: float) -> None:
        if delta > self.DROPPED_MULTIPLE * self.median_interval:
            self.dropped_sample_estimate += max(int(round(delta / self.median_interval)) - 1, 0)
        if delta <= self.threshold_s:
            return
        self.gap_count += 1
        self.largest_gap_s = max(self.largest_gap_s, delta)


def _run(ts: np.ndarray, expected: float = 0.01) -> GapTrackerPort:
    g = GapTrackerPort(expected)
    for t in ts:
        g.record(float(t))
    return g


def _steady(n: int, dt: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 12 * 86400.0 + np.arange(n) * dt + rng.normal(0, 2e-4, n)


def test_clean_stream_has_no_gaps_no_drops_in_either_implementation():
    ts = _steady(6000)
    port = _run(ts)
    py = loader.check_timestamps(ts)
    assert port.gap_count == py["n_gaps"] == 0
    assert port.dropped_sample_estimate == py["n_dropped_estimate"] == 0
    assert port.non_monotonic_count == py["n_nonmonotonic"] == 0


def test_single_drops_and_a_real_gap_agree_between_app_and_loader():
    """The thresholds are the same, so on a steady stream the counts are."""
    ts = _steady(6000, seed=1)
    ts = np.delete(ts, [1500, 4200])  # two single dropped samples (2x deltas)
    ts[3000:] += 0.5  # one real half-second outage
    port = _run(ts)
    py = loader.check_timestamps(ts)
    assert port.gap_count == py["n_gaps"] == 1
    assert abs(port.largest_gap_s - py["largest_gap_s"]) < 1e-9
    # 2 from the single drops + round(0.51 / 0.01) - 1 = 50 from the outage
    assert port.dropped_sample_estimate == py["n_dropped_estimate"] == 52


def test_duplicated_and_reordered_samples_are_counted_and_do_not_fake_a_gap():
    ts = _steady(2000, seed=2)
    ts = np.insert(ts, 900, ts[899])  # exact duplicate
    ts[1200], ts[1201] = ts[1201], ts[1200]  # one swapped pair
    port = _run(ts)
    py = loader.check_timestamps(ts)
    assert port.non_monotonic_count == 2
    assert py["n_nonmonotonic"] == 2
    assert port.gap_count == 0
    assert py["n_gaps"] == 0


def test_the_documented_divergence_on_a_mid_session_rate_change():
    """The two implementations diverge across a >3x change in delivered rate.

    Pinned so the README paragraph describing this stays true: the app's
    sliding median recalibrates 65-128 deltas after the change and reports
    only that lag window as gaps, while the loader's single whole-record
    median makes every delta of the slow section a gap. Both flag the
    session; neither count is the other's, and neither is "the" answer.
    """
    dt = 0.01
    base = 12 * 86400.0
    ts = np.concatenate([
        base + np.arange(3000) * dt,  # 30 s at 100 Hz
        base + 30 + np.arange(750) * 0.04,  # 30 s at 25 Hz
        base + 60 + np.arange(3000) * dt,  # 30 s at 100 Hz
    ])
    port = _run(ts)
    py = loader.check_timestamps(ts)
    assert 60 <= port.gap_count <= 128, port.gap_count
    assert py["n_gaps"] == 750
    # Neither side calls this stream continuous.
    assert port.dropped_sample_estimate > 0
    assert py["n_dropped_estimate"] > 0
    assert py["uniform"] is False


def test_loader_refuses_a_nan_timestamp():
    """FIXED: a NaN made two NaN deltas that every comparison excluded."""
    ts = _steady(500)
    ts[250] = np.nan
    try:
        loader.check_timestamps(ts)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("a NaN timestamp loaded as a clean record")
