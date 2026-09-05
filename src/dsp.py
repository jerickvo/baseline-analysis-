"""Shared signal-processing primitives.

Not a pipeline stage -- infrastructure used by stages 2, 3 and 4. Kept
separate so the stage modules do not import each other.

Every function here takes `fs_hz` explicitly. Nothing assumes 50 Hz.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# Welch segment length, in seconds of signal. 16 s gives 0.0625 Hz raw bin
# spacing -- ~2% of a 3 Hz running step frequency -- while still averaging
# several segments over a 90 s trial. Short trials fall back to using the
# whole record (see `_nperseg`). Expressed in seconds, not samples, so the
# resolution is identical at 50 Hz and at 200 Hz.
WELCH_SEGMENT_SECONDS = 16.0

# Plausible running/walking *step* frequency search band, in Hz.
# 1.2 Hz = 72 steps/min (slower than any sustained jog; a floor, not a
# tuning knob) and 4.5 Hz = 270 steps/min (faster than elite sprint
# cadence). Deliberately far wider than the 150-190 spm validation band so
# that a trial landing outside that band is *detected*, not clamped into it.
STEP_FREQ_SEARCH_BAND_HZ = (1.2, 4.5)


def _nperseg(n: int, fs_hz: float, seconds: float = WELCH_SEGMENT_SECONDS) -> int:
    """Welch segment length in samples, chosen so the segments TILE the record.

    `scipy.signal.welch` silently drops every trailing sample that does not
    fill a segment. With a fixed 16 s length and 50% overlap that discarded
    up to 7 s -- 30% of a short trial, > 4 s on 15 of the 48 -- and the
    spectral cadence is supposed to audit the detector over the same span
    the detector saw. So take the number of 16 s segments that fit, then
    stretch the segment length so that many segments at 50% overlap cover
    the whole record (k segments of length L cover L (k + 1) / 2 samples).
    Segments end up 16-24 s long; the parabolic peak refinement makes the
    small change in bin spacing immaterial.
    """
    target = int(max(8, min(n, round(seconds * fs_hz))))
    if target >= n:
        return target
    half = target // 2
    k = max(1, (n - half) // (target - half))
    # Even length, so the 50% overlap is exact and k segments cover exactly
    # L (k + 1) / 2 <= n samples: an odd L rounds the overlap down and can
    # lose a whole segment (n = 5535 gave L = 851, which fits only 11 of the
    # 12 intended segments and dropped 424 samples).
    return int(max(8, min(n, 2 * (n // (k + 1)))))


def robust_sigma(x: np.ndarray) -> float:
    """MAD-based standard-deviation estimate.

    Used instead of `np.std` for amplitude thresholds because a handful of
    hard footstrike impacts would otherwise inflate the scale that those
    same impacts are being compared against.

    Returns exactly 0.0 for a constant input, which is the mathematically
    correct answer and a trap for callers: a threshold built by scaling this
    value becomes 0, i.e. maximally *permissive*, at exactly the moment the
    signal carries no information. Callers that turn this into a detection
    threshold must treat a zero or near-zero result as "no signal" rather
    than as "accept everything" -- see `steps.detect_steps`.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)  # 1.4826 makes MAD consistent with sigma for Gaussians


def welch_psd(x: np.ndarray, fs_hz: float, seconds: float = WELCH_SEGMENT_SECONDS):
    """One-sided PSD of a mean-removed signal."""
    x = np.asarray(x, dtype=float)
    nps = _nperseg(len(x), fs_hz, seconds)
    return signal.welch(x - x.mean(), fs=fs_hz, nperseg=nps)


def spectral_peak(
    x: np.ndarray,
    fs_hz: float,
    band: tuple[float, float] = STEP_FREQ_SEARCH_BAND_HZ,
    seconds: float = WELCH_SEGMENT_SECONDS,
) -> dict:
    """Dominant frequency inside `band`, refined by parabolic interpolation.

    Interpolating the log-PSD across the peak bin recovers frequency to well
    below the bin spacing, which matters because a 0.0625 Hz bin error is a
    ~4 spm cadence error.

    Raises on non-finite input. `np.argmax` over an all-NaN array returns
    index 0, so without this guard an all-NaN signal would come back as a
    confident estimate sitting at the bottom edge of the search band. There
    is no defensible frequency for a signal that has no finite samples, so
    this refuses rather than inventing one.
    """
    x = np.asarray(x, dtype=float)
    if x.size and not np.all(np.isfinite(x)):
        n_bad = int((~np.isfinite(x)).sum())
        raise ValueError(
            f"spectral_peak: input has {n_bad} non-finite sample(s) of {x.size}; "
            f"no dominant frequency is defined. Clean or reject the segment first."
        )
    f, p = welch_psd(x, fs_hz, seconds)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        raise ValueError(f"search band {band} outside the spectrum (fs={fs_hz})")
    fb, pb = f[m], p[m]
    # Defence in depth: the guard above makes this unreachable for finite
    # input, but it protects the argmax regardless of how `p` was produced.
    if not np.any(np.isfinite(pb)):
        raise ValueError(
            f"spectral_peak: power spectrum is entirely non-finite in {band} Hz"
        )
    i = int(np.argmax(pb))
    f_peak = float(fb[i])
    if 0 < i < len(fb) - 1:
        # Parabolic interpolation on log power (a log-parabola fits a
        # windowed spectral peak better than a linear-power parabola).
        y0, y1, y2 = np.log(pb[i - 1 : i + 2] + 1e-300)
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            f_peak = float(fb[i] + delta * (fb[1] - fb[0]))
    total = float(p[(f >= band[0] / 2) & (f <= band[1] * 2)].sum())
    return {
        "f_peak_hz": f_peak,
        "peak_power": float(pb[i]),
        "bin_hz": float(f[1] - f[0]),
        "band_power_fraction": float(pb[i] / total) if total > 0 else np.nan,
        "freqs": f,
        "psd": p,
    }


def band_power(f: np.ndarray, p: np.ndarray, f_centre: float, rel_bw: float = 0.10) -> float:
    """Power within +/- `rel_bw` * f_centre of f_centre.

    A *relative* bandwidth keeps the measurement equivalent across cadences
    and sample rates; a fixed +/-0.25 Hz window would be a much tighter
    fractional window at 1.3 Hz than at 3 Hz.
    """
    bw = rel_bw * f_centre
    m = (f >= f_centre - bw) & (f <= f_centre + bw)
    return float(p[m].sum())


def bandpass(
    x: np.ndarray,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    * `sosfiltfilt`, not `lfilter`: filtfilt is zero-phase, so a detected
      peak keeps its true sample index. A causal filter would shift every
      footstrike by the filter's group delay (tens of ms here) and silently
      bias step timing.
    * Second-order sections rather than transfer-function coefficients: at
      the low cutoff/Nyquist ratios involved (0.04 at 50 Hz, 0.01 at
      200 Hz) a `ba` 4th-order design is numerically unstable.
    * order=4 per direction, so 8th-order effective. Steep enough to reject
      the neighbouring harmonics we care about, gentle enough that the
      passband stays flat across the cadence range.
    """
    nyq = fs_hz / 2.0
    if not (0 < low_hz < high_hz):
        raise ValueError(f"need 0 < low < high, got {low_hz}, {high_hz}")
    if high_hz >= nyq:
        raise ValueError(f"high cutoff {high_hz} Hz >= Nyquist {nyq} Hz")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=float))


def autocorrelation(x: np.ndarray, fs_hz: float, max_lag_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased-ish normalised autocorrelation, lag 0 .. max_lag_s."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    max_lag = int(min(n - 1, round(max_lag_s * fs_hz)))
    # FFT-based, zero-padded to avoid circular wrap.
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    fx = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(fx * np.conj(fx), nfft)[: max_lag + 1]
    denom = ac[0] if ac[0] != 0 else 1.0
    ac = ac / denom
    lags = np.arange(max_lag + 1) / fs_hz
    return lags, ac


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' formula: rotation by `angle_rad` about `axis`.

    Used by the tests to build specific placements (upside-down, yawed 180
    deg) where `random_rotation_matrix` gives only unnamed ones.
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def random_rotation_matrix(seed: int) -> np.ndarray:
    """A uniformly random 3-D rotation, reproducible from `seed`.

    Used to prove the pipeline is invariant to how the phone happens to be
    oriented -- the property the whole of stage 2 exists to provide.
    """
    rng = np.random.default_rng(seed)
    # QR of a Gaussian matrix, sign-fixed, gives Haar-uniform O(3); flip one
    # column if needed to land in SO(3).
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors, in degrees."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two *undirected* axes (sign-insensitive), in degrees.

    An axis recovered from an eigenvector has arbitrary sign, so comparing
    two of them must fold the answer into [0, 90].
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))
