"""Stage 2 -- resolve arbitrary sensor axes into an anatomical frame.

The phone sits at an unknown, arbitrary rotation. This module recovers::

    up            from the gravity vector
    forward       from the dominant horizontal acceleration axis in steady motion
    mediolateral  from the cross product (points to the runner's LEFT)

and builds the rotation that maps sensor-frame vectors into that triad.

It also carries the machinery to *disprove* its own output: frame stability
over the trial, an independent cross-check on the horizontal split, and a
periodicity test on the resulting vertical acceleration. Read
`verify_frame`'s verdict before trusting anything downstream.

Sign conventions
----------------
`up` points away from the earth. CoreMotion's `gravity` points *toward* the
earth (a device lying flat, screen up, reports (0, 0, -1)), so up = -g_hat.
`mediolateral = up x forward`, which for a right-handed triad points to the
runner's **left**.

Polarity of the projected components: every function here that projects
`user_accel` returns kinematic polarity -- positive vertical is the body
accelerating away from the earth -- **provided the input is kinematic**.
`loader.Trial.user_accel` delivers that; CoreMotion's raw
`userAcceleration` is the negative of it (see
`loader.USER_ACCEL_KINEMATIC_SIGN`). Feeding the raw columns in inverts
every polarity-dependent quantity while leaving cadence, spectra and axes
untouched, which is how that inversion went unnoticed until the free-fall
test in the loader caught it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dsp

# --- tunables, each with its reason ---------------------------------------

# Window for the sliding-window frame-stability check. 8 s spans >20 steps
# at running cadence, enough for a stable cross-spectral axis estimate,
# while still resolving drift over a 20 s trial into a few windows.
STABILITY_WINDOW_S = 8.0
STABILITY_HOP_S = 4.0  # 50% overlap: standard, halves the variance of the drift estimate

# A frame is called "stable" if the 95th-percentile tilt of instantaneous
# vertical away from the trial-mean vertical stays under this.
# 10 deg is where crosstalk starts to matter: a 10 deg frame error leaks
# sin(10 deg) = 17% of horizontal acceleration into the vertical channel.
# Fore-aft braking peaks are of the same order as the vertical waveform
# features we measure, so ~17% crosstalk is the point where the vertical
# channel stops being a clean vertical channel.
FRAME_STABLE_TILT_P95_DEG = 10.0

# Relative half-bandwidth used when integrating power around a harmonic.
HARMONIC_REL_BW = 0.10

# The horizontal axis estimate is called well-conditioned when the principal
# eigenvalue of the horizontal power matrix exceeds the secondary one by
# this factor. At 4:1 the axis direction is determined to roughly
# +/- 0.5 * atan(2 / (4 - 1)) ~ 17 deg by the eigenvalue spread alone;
# below that the "dominant" axis is not meaningfully dominant.
FORWARD_CONDITIONING_MIN_RATIO = 4.0

# The forward sign is called confident when the phase criterion's
# correlation is at least this strong. 0.1 is the level below which a
# correlation between two band-limited gait channels is indistinguishable
# from a chance alignment over a 20-90 s record; on the pocket data the
# criterion's median |statistic| is 0.20, on centre-of-mass synthetics it
# is > 0.9, so 0.1 separates "weak but present" from "noise".
FORWARD_SIGN_MIN_EFFECT = 0.1

# Low edge of the band the periodicity check autocorrelates, as a multiple
# of the step frequency. The stride subharmonic at 0.5 x f_step carries the
# left/right asymmetry that the symmetry index exists to measure. A 4th-
# order Butterworth run forward and backward has |H|^2 = 0.50 at its
# cutoff, so a cutoff AT 0.5 x f_step threw away half of that power and
# inflated the index (a synthetic with a 0.8-amplitude subharmonic read
# 0.74 instead of 0.22). At 0.25 x f_step the subharmonic passes with
# |H|^2 > 0.99 while drift below it is still removed.
SYMMETRY_BAND_LOW_MULTIPLE = 0.25

# Periodicity acceptance thresholds for the resolved vertical acceleration.
# The test is applied at the *stride* lag (two step periods), which is the
# true repeat period of gait: left and right steps are not required to be
# identical, and on this dataset they emphatically are not (see
# `step_symmetry_index`). Testing at one step period would fail a perfectly
# good frame purely because the runner's two legs differ.
# 0.30 is a deliberately weak floor -- it passes anything with visible
# repeating structure and fails only signals with no gait pattern at all,
# which is what this check is for. Broadband noise scores ~0.
PERIODICITY_MIN_AUTOCORR = 0.30
# ...and the step fundamental must hold at least this share of the power in
# [0.4 f, 4 f]. 0.20 admits signals with strong harmonics (a sharp impact
# spreads power upward) while rejecting broadband noise, where any single
# 10%-wide band holds only a few percent.
PERIODICITY_MIN_FUNDAMENTAL_FRACTION = 0.20


@dataclass
class AnatomicalFrame:
    """Sensor -> anatomical rotation, plus how much to trust it."""

    up: np.ndarray  # (3,) unit vector in sensor coordinates
    forward: np.ndarray  # (3,)
    mediolateral: np.ndarray  # (3,) = up x forward, points LEFT
    rotation: np.ndarray  # (3,3); rows [forward, mediolateral, up]
    mode: str  # "static" or "tracking"
    up_series: np.ndarray | None = None  # (n,3) per-sample up when tracking
    diagnostics: dict = field(default_factory=dict)

    @property
    def R(self) -> np.ndarray:
        return self.rotation


# --- vertical -------------------------------------------------------------


def unit_gravity(gravity: np.ndarray) -> np.ndarray:
    """Per-sample unit gravity direction, (n,3).

    Policy: unusable gravity **raises**, it never returns a flagged value.
    Both guards below follow that rule. The vertical axis is the one thing
    every later stage takes on trust -- steps, cadence and omega_v are all
    projections onto it -- so a frame that cannot be defined must stop the
    computation rather than seed NaN into every downstream number.

    The finiteness guard is not redundant with the zero-norm guard: a NaN
    norm compares False against 0, so all-NaN gravity would otherwise pass
    `n == 0` untouched and propagate NaN silently through the whole frame.
    """
    g = np.asarray(gravity, dtype=float)
    if g.size and not np.all(np.isfinite(g)):
        n_bad = int((~np.isfinite(g)).sum())
        raise ValueError(
            f"non-finite gravity: {n_bad} of {g.size} components are NaN or inf; "
            f"cannot define vertical"
        )
    n = np.linalg.norm(g, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-length gravity sample; cannot define vertical")
    return g / n


def vertical_axis(gravity: np.ndarray) -> np.ndarray:
    """Trial-constant up: the negated mean gravity direction, normalised.

    Averaging *before* normalising would let high-|g| samples dominate; here
    gravity is unit-norm by construction (CoreMotion normalises it), so mean
    then normalise is the correct spherical mean for small dispersions.
    """
    up = -unit_gravity(gravity).mean(axis=0)
    n = np.linalg.norm(up)
    if n < 1e-9:
        raise ValueError(
            "mean gravity direction is degenerate: the device tumbled through "
            "orientations that average to zero. No trial-constant vertical exists."
        )
    return up / n


def vertical_axis_series(gravity: np.ndarray) -> np.ndarray:
    """Per-sample up, (n,3). Tracks the sensor as it rotates on the body."""
    return -unit_gravity(gravity)


# --- horizontal plane -----------------------------------------------------


def horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An arbitrary orthonormal basis of the plane perpendicular to `up`.

    The choice of seed is arbitrary and cancels out: every quantity derived
    from this basis is later re-expressed in the estimated forward/ML axes.
    """
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(up @ seed)) > 0.9:  # avoid a near-parallel seed
        seed = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def project_horizontal(vectors: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Remove the `up` component from each row of an (n,3) array."""
    v = np.asarray(vectors, float)
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    return v - np.outer(v @ up, up)


def _principal_axis_2d(m: np.ndarray, rel_tol: float = 1e-12) -> tuple[np.ndarray, float]:
    """Principal eigenvector of a 2x2 symmetric matrix, plus eigenvalue ratio.

    A non-positive or numerically negligible secondary eigenvalue means the
    horizontal power matrix is rank-deficient: there is no second axis for
    the first to be dominant *over*. The ratio is then **undefined**, not
    infinite, and NaN is returned deliberately -- it compares False against
    every `>=`, so `well_conditioned` comes out False. Returning `inf` here
    inverted the semantics, passing the conditioning check for exactly the
    degenerate input that check exists to reject.

    `rel_tol` is relative to the leading eigenvalue, so the test scales with
    the signal's power and catches a secondary eigenvalue that is positive
    only through floating-point noise. An all-zero matrix makes both
    eigenvalues 0 and is caught by the same comparison.
    """
    w, v = np.linalg.eigh(m)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    if not np.isfinite(w).all() or w[1] <= rel_tol * abs(w[0]):
        return v[:, 0], float("nan")
    return v[:, 0], float(w[0] / w[1])


def forward_axis(
    user_accel: np.ndarray,
    up: np.ndarray,
    fs_hz: float,
    method: str = "step_band",
    f_step_hz: float | None = None,
) -> dict:
    """Estimate the horizontal axis of dominant acceleration.

    Two methods, both placement-agnostic in *formulation*:

    ``"pca"``
        Broadband principal component of horizontal acceleration -- the
        literal "dominant horizontal acceleration axis". Simple, and the
        baseline the spec asks for. On pocket data it is badly conditioned
        (see the eigenvalue ratio in the returned dict).

    ``"step_band"`` (default)
        Principal axis of the horizontal *cross-spectral* matrix restricted
        to a narrow band around the step frequency. Rationale: fore-aft
        acceleration is driven by braking and propulsion, which happen once
        per step, so it is concentrated at the step fundamental;
        stride-rate sway and low-frequency course changes are not. This
        rejects everything outside the band that broadband PCA is forced to
        fit, and is far better conditioned in practice.

    Returns the axis (unresolved sign), the eigenvalue ratio that says how
    dominant it really is, and the estimated step frequency used.
    """
    a = np.asarray(user_accel, float)
    up = np.asarray(up, float) / np.linalg.norm(up)
    e1, e2 = horizontal_basis(up)
    x1 = a @ e1
    x2 = a @ e2

    if f_step_hz is None:
        f_step_hz = dsp.spectral_peak(a @ up, fs_hz)["f_peak_hz"]

    if method == "pca":
        m = np.cov(np.vstack([x1, x2]))
    elif method == "step_band":
        from scipy import signal as _sig

        nps = dsp._nperseg(len(x1), fs_hz)
        f, p11 = _sig.welch(x1 - x1.mean(), fs=fs_hz, nperseg=nps)
        _, p22 = _sig.welch(x2 - x2.mean(), fs=fs_hz, nperseg=nps)
        _, p12 = _sig.csd(x1 - x1.mean(), x2 - x2.mean(), fs=fs_hz, nperseg=nps)
        bw = HARMONIC_REL_BW * f_step_hz
        sel = (f >= f_step_hz - bw) & (f <= f_step_hz + bw)
        if not sel.any():  # short record: widen to the nearest bin
            sel = np.zeros_like(f, dtype=bool)
            sel[int(np.argmin(np.abs(f - f_step_hz)))] = True
        # Real part only: the in-phase co-spectrum is what a fixed spatial
        # axis produces. The quadrature part describes elliptical (rotating)
        # motion and has no single axis to report.
        c = float(np.real(p12[sel]).sum())
        m = np.array([[float(p11[sel].sum()), c], [c, float(p22[sel].sum())]])
    else:
        raise ValueError(f"unknown method {method!r}")

    axis2, ratio = _principal_axis_2d(m)
    axis = axis2[0] * e1 + axis2[1] * e2
    axis /= np.linalg.norm(axis)
    return {
        "axis": axis,
        "eigenvalue_ratio": ratio,
        "well_conditioned": bool(ratio >= FORWARD_CONDITIONING_MIN_RATIO),
        "method": method,
        "f_step_hz": float(f_step_hz),
        "basis": (e1, e2),
    }


def resolve_forward_sign(
    user_accel: np.ndarray,
    up: np.ndarray,
    axis: np.ndarray,
    fs_hz: float,
    f_step_hz: float,
) -> dict:
    """Fix the 180-degree ambiguity of the forward axis.

    One criterion decides, one is reported. Both rest on the biomechanics
    of a stance phase: the ground reaction force brakes the runner early in
    stance and propels them late in stance, while vertical acceleration
    rises through early stance and falls through late stance.

    ``phase`` (decides the sign and the confidence)
        Braking coincides with *rising* vertical acceleration, propulsion
        with *falling*, so fore-aft acceleration is anti-correlated with the
        time derivative of vertical acceleration: pick the sign that makes
        corr(a_fwd, d a_vert/dt) negative. This holds at the centre of mass
        -- the lower back, where the product will run -- because there the
        vertical peak is midstance, the moment fore-aft force crosses zero.
        It is invariant to the polarity of the input (both channels flip
        together), and on synthetic centre-of-mass signals it recovers the
        true forward direction to < 1 deg under every rotation tried.

    ``impact`` (reported only; it does NOT gate confidence)
        Fore-aft acceleration should be negative *at* the vertical peak.
        That is valid where the peak is early stance, and at the centre of
        mass -- where the peak is midstance -- it samples fore-aft at its
        own zero crossing. Measured: on centre-of-mass synthetics its
        statistic is within +/-0.012 under 16 rotations, i.e. its sign is
        noise; on the pocket data it comes out negative on 24 of 48 trials,
        a coin flip. Gating confidence on agreement with a null statistic
        made "confident" unreachable at the production placement and turned
        a correct phase answer into "criteria disagree" at random, which is
        why the gate was removed.

    Confidence therefore means: the phase correlation is at least
    `FORWARD_SIGN_MIN_EFFECT` in magnitude and has the same sign on both
    halves of the record. On centre-of-mass data that is a validated
    criterion; on thigh-pocket data it is an inference from the same
    stance mechanics, not a validated one. Nothing in stages 3 or 4 depends
    on the result: it sets plot polarity and the left/right naming of the
    mediolateral axis, not any number computed.
    """
    a = np.asarray(user_accel, float)
    up = np.asarray(up, float) / np.linalg.norm(up)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)

    # Band-limit both channels to the step fundamental +/- one octave so the
    # phase comparison is not dominated by impact ringing or drift.
    lo, hi = 0.5 * f_step_hz, min(2.0 * f_step_hz, 0.4 * fs_hz)
    a_vert = dsp.bandpass(a @ up, fs_hz, lo, hi)
    a_fwd = dsp.bandpass(a @ axis, fs_hz, lo, hi)
    d_vert = np.gradient(a_vert) * fs_hz

    def _phase_stat(av_d: np.ndarray, af: np.ndarray) -> float:
        if len(af) < 4 or np.std(af) == 0 or np.std(av_d) == 0:
            return 0.0
        return float(np.corrcoef(af, av_d)[0, 1])

    def _impact_stat(av: np.ndarray, af: np.ndarray) -> float:
        # Top quartile of vertical acceleration: ~1/4 of each step, which at
        # 50 Hz is ~4 samples -- enough to be stable.
        if len(av) < 4:
            return 0.0
        scale = dsp.robust_sigma(af) or 1.0
        return float(af[av >= np.quantile(av, 0.75)].mean() / scale)

    phase = _phase_stat(d_vert, a_fwd)
    impact = _impact_stat(a_vert, a_fwd)

    # Split-half consistency of each criterion on its own.
    half = len(a_vert) // 2
    halves = (slice(0, half), slice(half, None))
    phase_votes = [np.sign(_phase_stat(d_vert[s], a_fwd[s])) for s in halves]
    impact_votes = [np.sign(_impact_stat(a_vert[s], a_fwd[s])) for s in halves]
    phase_consistent = bool(phase_votes[0] == phase_votes[1] != 0)
    impact_consistent = bool(impact_votes[0] == impact_votes[1] != 0)

    # Both criteria want a NEGATIVE statistic for a correctly oriented axis;
    # the phase criterion decides the flip.
    flip = phase > 0
    criteria_agree = bool(np.sign(phase) == np.sign(impact) != 0)
    strong = abs(phase) >= FORWARD_SIGN_MIN_EFFECT
    return {
        "flip": bool(flip),
        "axis": -axis if flip else axis,
        "phase_effect_size": float(phase),
        "impact_effect_size": float(impact),
        "split_half_consistent": phase_consistent,
        "impact_split_half_consistent": impact_consistent,
        # Diagnostic only: see the docstring for why agreement with the
        # impact statistic is not evidence at the centre of mass.
        "criteria_agree": criteria_agree,
        "sign_confident": bool(strong and phase_consistent),
    }


# --- frame construction ---------------------------------------------------


def build_frame(
    user_accel: np.ndarray,
    gravity: np.ndarray,
    fs_hz: float,
    mode: str = "tracking",
    forward_method: str = "step_band",
    f_step_hz: float | None = None,
) -> AnatomicalFrame:
    """Build the sensor -> anatomical rotation.

    `mode="static"` uses one rotation for the whole trial: the classic
    approach, and the one the spec describes.

    `mode="tracking"` (default) keeps the trial-constant *forward* estimate
    but takes `up` from instantaneous gravity at every sample, re-projecting
    forward into each sample's horizontal plane. On a segment that swings
    relative to the trunk -- a trouser pocket, say -- a single rotation is
    wrong by tens of degrees for most of the stride, and tracking removes
    that error without assuming anything about where the sensor is.
    """
    if mode not in ("static", "tracking"):
        raise ValueError(f"mode must be 'static' or 'tracking', got {mode!r}")

    a = np.asarray(user_accel, float)
    up = vertical_axis(gravity)

    fwd_est = forward_axis(a, up, fs_hz, method=forward_method, f_step_hz=f_step_hz)
    sign = resolve_forward_sign(a, up, fwd_est["axis"], fs_hz, fwd_est["f_step_hz"])
    forward = sign["axis"]
    # Re-orthogonalise against floating point drift; forward is already
    # horizontal by construction, this just cleans it.
    forward = forward - (forward @ up) * up
    forward /= np.linalg.norm(forward)
    ml = np.cross(up, forward)  # right-handed => points to the runner's LEFT
    ml /= np.linalg.norm(ml)

    R = np.vstack([forward, ml, up])

    up_series = vertical_axis_series(gravity) if mode == "tracking" else None

    diagnostics = {
        "forward_method": forward_method,
        "forward_eigenvalue_ratio": fwd_est["eigenvalue_ratio"],
        "forward_well_conditioned": fwd_est["well_conditioned"],
        "f_step_hz_used": fwd_est["f_step_hz"],
        "forward_sign_flipped": sign["flip"],
        "forward_phase_effect_size": sign["phase_effect_size"],
        "forward_impact_effect_size": sign["impact_effect_size"],
        "forward_sign_criteria_agree": sign["criteria_agree"],
        "forward_sign_confident": sign["sign_confident"],
        "forward_sign_split_half_consistent": sign["split_half_consistent"],
        "orthonormal_residual": float(np.abs(R @ R.T - np.eye(3)).max()),
        "right_handed": bool(np.linalg.det(R) > 0),
    }
    return AnatomicalFrame(
        up=up,
        forward=forward,
        mediolateral=ml,
        rotation=R,
        mode=mode,
        up_series=up_series,
        diagnostics=diagnostics,
    )


def resolve(vectors: np.ndarray, frame: AnatomicalFrame) -> np.ndarray:
    """Rotate (n,3) sensor-frame vectors into (n,3) [forward, ML, up].

    In `tracking` mode the up component uses the per-sample gravity
    direction and the horizontal axes are re-projected into each sample's
    horizontal plane, so the returned triad is orthonormal at every sample.

    Known limit of a gravity-only frame, stated rather than hidden: the
    re-projection corrects pitch and roll wobble of the sensor (synthetic
    error <= 0.002 g on every channel for 5 deg wobble) but cannot see a
    rotation ABOUT the vertical, because gravity does not change under it.
    A sensor that yaws relative to the direction of travel -- the pelvis
    rotates by roughly +/-5-10 deg in the transverse plane once per stride
    -- therefore leaks fore-aft acceleration into the mediolateral channel
    by sin(yaw): about 13% of the true mediolateral RMS on a pelvis-like
    synthetic, scaling with pace. Any mediolateral quantity derived here is
    a proxy contaminated by that term until the yaw is estimated from the
    gyro and removed.
    """
    v = np.asarray(vectors, float)
    if frame.mode == "static" or frame.up_series is None:
        return v @ frame.rotation.T

    up_t = frame.up_series  # (n,3)
    if len(up_t) != len(v):
        raise ValueError(
            f"tracking frame built from {len(up_t)} samples, got {len(v)} vectors"
        )
    comp_up = np.einsum("ij,ij->i", v, up_t)
    # Re-project the constant forward axis into each instantaneous horizontal
    # plane. Where the sensor's up happens to align with forward this is
    # ill-defined; guard and fall back to the static axis there.
    f_t = frame.forward - (up_t @ frame.forward)[:, None] * up_t
    norm = np.linalg.norm(f_t, axis=1, keepdims=True)
    degenerate = norm[:, 0] < 1e-6
    f_t = np.where(
        degenerate[:, None],
        frame.forward,
        f_t / np.where(degenerate[:, None], 1.0, norm),
    )
    ml_t = np.cross(up_t, f_t)
    ml_t /= np.linalg.norm(ml_t, axis=1, keepdims=True)
    comp_fwd = np.einsum("ij,ij->i", v, f_t)
    comp_ml = np.einsum("ij,ij->i", v, ml_t)
    return np.column_stack([comp_fwd, comp_ml, comp_up])


def vertical_component(
    vectors: np.ndarray, gravity: np.ndarray, mode: str = "tracking"
) -> np.ndarray:
    """Just the vertical component -- the one stages 3 and 4 actually use.

    Separate from `resolve` because it needs only gravity: it is available
    even when the horizontal split is untrustworthy, which on pocket data it
    is. Positive = away from the earth, for a kinematic-polarity input
    (`loader.Trial.user_accel`, or a rotation rate, which needs no
    conversion). See the module docstring.
    """
    v = np.asarray(vectors, float)
    if mode == "tracking":
        return -np.einsum("ij,ij->i", v, unit_gravity(gravity))
    return v @ vertical_axis(gravity)


# --- verification ---------------------------------------------------------


def frame_stability(
    user_accel: np.ndarray,
    gravity: np.ndarray,
    fs_hz: float,
    window_s: float = STABILITY_WINDOW_S,
    hop_s: float = STABILITY_HOP_S,
    forward_method: str = "step_band",
) -> dict:
    """Does the frame hold still over the trial?

    Two independent measurements:

    * **Vertical wander** -- angle between instantaneous up and trial-mean
      up, at every sample. This is a property of the *placement*: a sensor
      rigidly on the trunk barely moves, one on a swinging segment does.
    * **Forward drift** -- the forward axis re-estimated on overlapping
      windows, compared to the trial-level axis as an undirected axis.
    """
    g_hat = unit_gravity(gravity)
    up = vertical_axis(gravity)
    cos = np.clip(g_hat @ (-up), -1.0, 1.0)
    tilt = np.degrees(np.arccos(cos))

    a = np.asarray(user_accel, float)
    win = int(round(window_s * fs_hz))
    hop = int(round(hop_s * fs_hz))
    fwd_ref = forward_axis(a, up, fs_hz, method=forward_method)["axis"]
    drifts, ratios, starts = [], [], []
    if win >= int(round(4 * fs_hz)) and len(a) >= win:
        for s in range(0, len(a) - win + 1, max(hop, 1)):
            seg = a[s : s + win]
            gseg = gravity[s : s + win]
            try:
                up_w = vertical_axis(gseg)
                est = forward_axis(seg, up_w, fs_hz, method=forward_method)
            except (ValueError, np.linalg.LinAlgError):
                continue
            drifts.append(dsp.axis_angle_deg(est["axis"], fwd_ref))
            ratios.append(est["eigenvalue_ratio"])
            starts.append(s / fs_hz)

    drifts = np.asarray(drifts, float)
    out = {
        "vertical_tilt_median_deg": float(np.median(tilt)),
        "vertical_tilt_p95_deg": float(np.percentile(tilt, 95)),
        "vertical_tilt_max_deg": float(tilt.max()),
        "n_stability_windows": int(len(drifts)),
        "forward_drift_median_deg": float(np.median(drifts)) if drifts.size else np.nan,
        "forward_drift_p95_deg": float(np.percentile(drifts, 95)) if drifts.size else np.nan,
        "forward_drift_max_deg": float(drifts.max()) if drifts.size else np.nan,
        "forward_window_ratio_median": float(np.median(ratios)) if ratios else np.nan,
        "tilt_series_deg": tilt,
        "window_start_s": np.asarray(starts, float),
        "window_drift_deg": drifts,
    }
    out["frame_static_valid"] = bool(
        out["vertical_tilt_p95_deg"] <= FRAME_STABLE_TILT_P95_DEG
    )
    return out


def mediolateral_cross_check(
    user_accel: np.ndarray,
    frame: AnatomicalFrame,
    fs_hz: float,
    f_step_hz: float,
) -> dict:
    """An independent test of the horizontal split, which can fail.

    Biomechanics predicts a clean frequency separation at the trunk:
    fore-aft acceleration is driven by braking/propulsion once per **step**,
    while mediolateral acceleration is driven by weight transfer to the
    stance side once per **stride** (= every two steps). So the horizontal
    axis carrying most stride-rate power should be roughly perpendicular
    (~90 deg) to the axis carrying most step-rate power.

    If the measured angle is near 0 deg instead, both harmonics ride on the
    same physical axis, the prediction does not hold at this placement, and
    the mediolateral axis is an unverified construction of the cross product.
    """
    a = np.asarray(user_accel, float)
    step_axis = forward_axis(a, frame.up, fs_hz, "step_band", f_step_hz)
    stride_axis = forward_axis(a, frame.up, fs_hz, "step_band", f_step_hz / 2.0)
    ang = dsp.axis_angle_deg(step_axis["axis"], stride_axis["axis"])
    return {
        "step_stride_axis_angle_deg": ang,
        "stride_axis_eigenvalue_ratio": stride_axis["eigenvalue_ratio"],
        # 90 deg is the prediction; allow +/- 30 deg for estimation noise.
        "ml_independently_supported": bool(ang >= 60.0),
    }


def verify_vertical_periodicity(a_vert: np.ndarray, fs_hz: float) -> dict:
    """Does the resolved vertical acceleration actually look like running?

    Checks for the footstrike structure the frame is supposed to expose:
    a dominant spectral peak in the plausible step band, and an
    autocorrelation that returns near its own scale one *stride* later.

    Three autocorrelation quantities are reported, following the standard
    trunk-accelerometry definitions (Moe-Nilssen & Helbostad 2004):

    ``step_regularity``    autocorrelation at one step period
    ``stride_regularity``  autocorrelation at two step periods
    ``step_symmetry_index``  their ratio, in [0, ~1]

    A value near 1 for the symmetry index means consecutive steps are
    interchangeable. A low value means the two steps of a stride produce
    genuinely different waveforms -- which is informative, not a failure,
    and is why acceptance is judged on stride regularity.

    Both regularities are read from the *unbiased* autocorrelation
    (`dsp.autocorrelation`) at the local maximum nearest each lag, so they
    do not depend on record length or on the exact value of `f_step`.
    """
    a_vert = np.asarray(a_vert, float)
    pk = dsp.spectral_peak(a_vert, fs_hz)
    f_step = pk["f_peak_hz"]
    f, p = pk["freqs"], pk["psd"]

    total = dsp.band_power(f, p, f_step * 2.2, rel_bw=0.82)  # ~[0.4f, 4f]
    fundamental = dsp.band_power(f, p, f_step, HARMONIC_REL_BW)
    second = dsp.band_power(f, p, 2 * f_step, HARMONIC_REL_BW)
    stride = dsp.band_power(f, p, f_step / 2, HARMONIC_REL_BW)

    step_period = 1.0 / f_step
    # Autocorrelate the band-limited signal, not the raw one. The raw trace
    # carries wideband sensor and impact-ringing noise that contributes only
    # to the lag-0 normaliser, deflating every other lag and turning the
    # index into a noise measurement rather than a periodicity measurement.
    # The band keeps the stride subharmonic (see SYMMETRY_BAND_LOW_MULTIPLE
    # for why the low edge sits well below it), the fundamental and three
    # harmonics -- ample waveform shape -- and is expressed as multiples of
    # f_step so it is identical at any sample rate or cadence.
    lo = SYMMETRY_BAND_LOW_MULTIPLE * f_step
    hi = min(4.0 * f_step, 0.4 * fs_hz)
    a_band = dsp.bandpass(a_vert, fs_hz, lo, hi) if hi > lo else a_vert
    lags, ac = dsp.autocorrelation(a_band, fs_hz, max_lag_s=3.0 * step_period)

    def _at(multiple: float) -> float:
        # The autocorrelation maximum nearest the expected lag, searched
        # within +/- HARMONIC_REL_BW of a step period, rather than the
        # single sample at exactly 1/f_step: the spectral estimate of
        # f_step carries a few percent of error, and reading one sample
        # off the peak of a 3 Hz autocorrelation at 50 Hz costs ~15% of
        # its value for no reason. This is how Moe-Nilssen's regularity
        # is defined -- the peak, not a fixed lag.
        target = multiple * step_period
        half = HARMONIC_REL_BW * step_period
        m = (lags >= target - half) & (lags <= target + half)
        if not m.any():
            m = np.zeros_like(lags, dtype=bool)
            m[int(np.argmin(np.abs(lags - target)))] = True
        return float(ac[m].max())

    step_reg = _at(1.0)
    stride_reg = _at(2.0)
    symmetry = float(step_reg / stride_reg) if stride_reg > 1e-9 else np.nan

    frac = float(fundamental / total) if total > 0 else np.nan
    passed = bool(
        stride_reg >= PERIODICITY_MIN_AUTOCORR
        and frac >= PERIODICITY_MIN_FUNDAMENTAL_FRACTION
    )
    return {
        "f_step_hz": float(f_step),
        "implied_cadence_spm": float(f_step * 60.0),
        "fundamental_power_fraction": frac,
        "second_harmonic_ratio": float(second / fundamental) if fundamental > 0 else np.nan,
        "stride_subharmonic_ratio": float(stride / fundamental) if fundamental > 0 else np.nan,
        "step_regularity": step_reg,
        "stride_regularity": stride_reg,
        "step_symmetry_index": symmetry,
        "periodicity_ok": passed,
        "lags_s": lags,
        "autocorr": ac,
        "freqs": f,
        "psd": p,
        "a_vertical_band": a_band,
    }


def verify_frame(
    user_accel: np.ndarray,
    gravity: np.ndarray,
    frame: AnatomicalFrame,
    fs_hz: float,
) -> dict:
    """Run every stage-2 check and return one verdict plus the evidence.

    `verdict` is deliberately three-valued. "ok" means every check passed.
    "vertical_only" means the vertical axis is sound and periodic but the
    horizontal split is not supported -- downstream code may use vertical
    acceleration and must not use forward/ML. "failed" means the vertical
    axis itself did not produce a periodic footstrike signal.
    """
    a_vert = vertical_component(user_accel, gravity, mode=frame.mode)
    per = verify_vertical_periodicity(a_vert, fs_hz)
    stab = frame_stability(user_accel, gravity, fs_hz, forward_method=frame.diagnostics["forward_method"])
    ml = mediolateral_cross_check(user_accel, frame, fs_hz, per["f_step_hz"])

    horizontal_ok = bool(
        frame.diagnostics["forward_well_conditioned"] and ml["ml_independently_supported"]
    )
    if not per["periodicity_ok"]:
        verdict = "failed"
    elif frame.mode == "static" and not stab["frame_static_valid"]:
        verdict = "vertical_only"
    elif horizontal_ok:
        verdict = "ok"
    else:
        verdict = "vertical_only"

    reasons = []
    if not per["periodicity_ok"]:
        reasons.append(
            f"vertical acceleration shows no repeating footstrike structure "
            f"(stride regularity={per['stride_regularity']:.2f}, "
            f"fundamental fraction={per['fundamental_power_fraction']:.2f})"
        )
    if not frame.diagnostics["forward_well_conditioned"]:
        reasons.append(
            f"horizontal axis not dominant "
            f"(eigenvalue ratio {frame.diagnostics['forward_eigenvalue_ratio']:.1f} "
            f"< {FORWARD_CONDITIONING_MIN_RATIO})"
        )
    if not ml["ml_independently_supported"]:
        reasons.append(
            f"step-rate and stride-rate horizontal axes are {ml['step_stride_axis_angle_deg']:.0f} deg "
            f"apart, not ~90 deg: mediolateral axis is unverified"
        )
    if not stab["frame_static_valid"]:
        if frame.mode == "static":
            reasons.append(
                f"vertical wanders {stab['vertical_tilt_p95_deg']:.0f} deg (p95) within the "
                f"trial and this frame is static: the rotation is wrong for most of the stride"
            )
        else:
            reasons.append(
                f"NOTE (not a failure): vertical wanders "
                f"{stab['vertical_tilt_p95_deg']:.0f} deg (p95) within the trial, so a "
                f"trial-constant rotation would be invalid here; tracking mode is in use, "
                f"which removes that error"
            )
    if not frame.diagnostics["forward_sign_confident"]:
        reasons.append("forward sign (front vs back) is not resolved with confidence")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "periodicity": per,
        "stability": stab,
        "mediolateral_check": ml,
        "a_vertical": a_vert,
    }
