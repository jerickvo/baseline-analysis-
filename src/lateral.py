"""Stage 4 -- exploratory left/right step discrimination. NO GROUND TRUTH.

Theory under test: the pelvis rotates in opposite directions during left and
right stance, so the sign of angular velocity about the vertical axis,
sampled at initial contact, should separate the two sides.

This dataset contains no left/right foot labels of any kind, and no
information about which trouser pocket held the phone. Nothing here is an
accuracy measurement and nothing here should be read as one. What is
measured is **alternation consistency**: how reliably the sign flips from
one step to the next.

Why consistency is not correctness
----------------------------------
Alternation is close to guaranteed by signal structure alone. If the
vertical angular velocity carries a strong component at the stride rate
(half the step rate) -- which it does -- then sampling it once per step
lands on alternating half-cycles of that component *whatever its physical
origin and whatever its phase*. A pure sinusoid at f_step/2 sampled at
intervals of 1/f_step alternates perfectly while encoding nothing.

`null_models` makes that concrete by measuring three controls that contain
no left/right information by construction:

``random``       independent random signs (expectation 0.5)
``surrogate``    phase-randomised omega_v, preserving its power spectrum
                 exactly while destroying any step-locked phase relationship
``mid_step``     the real signal sampled halfway between steps, where no
                 initial-contact event occurs

If the real alternation rate does not beat the surrogate and mid-step
controls, the sign sequence carries no more information than "omega_v has
stride-rate power".
"""

from __future__ import annotations

import numpy as np

from . import dsp, orientation

# Half-width of the averaging window at contact, as a fraction of the step
# period. 0.10 spans a fifth of the step. Kept relative so it scales with
# cadence and sample rate -- but note what it costs at 50 Hz: a 0.36 s step
# gives +/-36 ms, which is only about +/-2 samples. The returned
# `contact_window_samples` makes that visible; at 200 Hz the same window is
# ~14 samples and the estimate is far better conditioned.
CONTACT_WINDOW_STEP_FRACTION = 0.10

# Number of phase-randomised surrogate draws. 200 gives the null
# alternation rate to about +/-0.02 (1 sigma), well under any effect worth
# reporting.
N_SURROGATES = 200

# Alternation is called "consistent" above this. 0.90 means at most one
# non-flip in ten steps. This is a description of the sign sequence, NOT an
# accuracy: see the module docstring.
ALTERNATION_CONSISTENT_THRESHOLD = 0.90


def omega_vertical(
    rotation_rate: np.ndarray, gravity: np.ndarray, mode: str = "tracking"
) -> np.ndarray:
    """Angular velocity about the vertical axis, rad/s.

    Positive is a right-handed rotation about `up`, i.e. turning to the
    runner's left. Uses the same frame machinery as stage 2 so the vertical
    axis is identical to the one the steps were detected on.
    """
    return orientation.vertical_component(rotation_rate, gravity, mode=mode)


def sample_at_contacts(
    omega_v: np.ndarray,
    step_indices: np.ndarray,
    fs_hz: float,
    f_step_hz: float,
    window_fraction: float = CONTACT_WINDOW_STEP_FRACTION,
    offset_step_fraction: float = 0.0,
) -> dict:
    """Average omega_v in a short window around each detected step.

    `offset_step_fraction` shifts the sampling point by that fraction of a
    step period; 0.5 gives the mid-step control.
    """
    omega_v = np.asarray(omega_v, float)
    idx = np.asarray(step_indices, int)
    step_period_samples = fs_hz / f_step_hz
    half = int(max(1, round(window_fraction * step_period_samples)))
    shift = int(round(offset_step_fraction * step_period_samples))

    centres = idx + shift
    keep = (centres - half >= 0) & (centres + half + 1 <= len(omega_v))
    centres = centres[keep]
    kept = idx[keep]
    if centres.size:
        # (n_steps, window) gather, then row means. Vectorised because the
        # surrogate nulls call this thousands of times per trial.
        offsets = np.arange(-half, half + 1)
        values = omega_v[centres[:, None] + offsets[None, :]].mean(axis=1)
    else:
        values = np.empty(0, dtype=float)
    return {
        "values": np.asarray(values, float),
        "step_indices": np.asarray(kept, int),
        "contact_window_samples": int(2 * half + 1),
        "contact_window_s": float((2 * half + 1) / fs_hz),
        "offset_step_fraction": float(offset_step_fraction),
    }


def alternation_rate(values: np.ndarray) -> float:
    """Fraction of consecutive step pairs whose sign flips.

    Zero-valued samples are treated as non-flips (they cannot support a
    flip claim). Returns NaN with fewer than two samples.
    """
    v = np.asarray(values, float)
    if len(v) < 2:
        return float("nan")
    s = np.sign(v)
    return float(np.mean(s[1:] * s[:-1] < 0))


def label_alternating(values: np.ndarray) -> np.ndarray:
    """Label steps 'A'/'B' by the sign of omega_v at contact.

    Deliberately named A and B, not left and right. Assigning anatomical
    sides needs (a) ground truth this dataset does not contain and (b)
    knowledge of which pocket held the phone, which it also does not
    contain.
    """
    s = np.sign(np.asarray(values, float))
    return np.where(s >= 0, "A", "B")


def phase_randomised_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A signal with x's exact power spectrum and randomised phases."""
    x = np.asarray(x, float)
    n = len(x)
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    phases = rng.uniform(0, 2 * np.pi, size=mag.shape)
    phases[0] = 0.0  # keep DC real
    if n % 2 == 0:
        phases[-1] = 0.0  # keep Nyquist real
    return np.fft.irfft(mag * np.exp(1j * phases), n)


def null_models(
    omega_v: np.ndarray,
    step_indices: np.ndarray,
    fs_hz: float,
    f_step_hz: float,
    n_surrogates: int = N_SURROGATES,
    seed: int = 0,
    offset_step_fraction: float = 0.0,
) -> dict:
    """Alternation rates for three controls that contain no L/R information.

    The surrogate is sampled at the *same* phase offset as the measurement
    it is a null for, so the comparison is like-for-like.
    """
    rng = np.random.default_rng(seed)
    n_steps = len(step_indices)

    # 1. independent random signs
    random_rates = [
        alternation_rate(rng.choice([-1.0, 1.0], size=n_steps))
        for _ in range(n_surrogates)
    ]

    # 2. phase-randomised surrogate, sampled at the same step indices.
    # Because the surrogate's phase is random, averaging over draws gives
    # the alternation rate you would expect from omega_v's *spectrum alone*,
    # with no knowledge of the correct sampling phase.
    surrogate_rates = []
    for _ in range(n_surrogates):
        s = phase_randomised_surrogate(omega_v, rng)
        v = sample_at_contacts(
            s, step_indices, fs_hz, f_step_hz,
            offset_step_fraction=offset_step_fraction,
        )["values"]
        r = alternation_rate(v)
        if np.isfinite(r):
            surrogate_rates.append(r)

    # 3. the real signal sampled half a step away from every contact
    mid = sample_at_contacts(
        omega_v, step_indices, fs_hz, f_step_hz, offset_step_fraction=0.5
    )
    mid_rate = alternation_rate(mid["values"])

    return {
        "random_mean": float(np.mean(random_rates)),
        "random_p95": float(np.percentile(random_rates, 95)),
        "surrogate_mean": float(np.mean(surrogate_rates)) if surrogate_rates else np.nan,
        "surrogate_p95": float(np.percentile(surrogate_rates, 95)) if surrogate_rates else np.nan,
        "mid_step_rate": mid_rate,
        "n_surrogates": int(n_surrogates),
    }


def phase_sweep(
    omega_v: np.ndarray,
    step_indices: np.ndarray,
    fs_hz: float,
    f_step_hz: float,
    n_offsets: int = 10,
) -> dict:
    """Alternation consistency as a function of where in the step we sample.

    This exists because stage 3 delivers a step-rate *phase marker*, not a
    true initial contact (see `steps` module docstring). The theory under
    test names a specific instant -- initial contact -- that we cannot
    locate. Sweeping the sampling point across the whole step period shows
    how much the headline number depends on that unknown offset. A result
    that swings from chance to near-perfect across the sweep is a result
    about sampling phase, not about laterality.
    """
    offsets = np.arange(n_offsets) / n_offsets
    rates = []
    for off in offsets:
        v = sample_at_contacts(
            omega_v, step_indices, fs_hz, f_step_hz, offset_step_fraction=float(off)
        )["values"]
        rates.append(alternation_rate(v))
    rates = np.asarray(rates, float)
    best = int(np.nanargmax(rates)) if np.isfinite(rates).any() else 0
    return {
        "offsets_step_fraction": offsets,
        "alternation_by_offset": rates,
        "best_offset_step_fraction": float(offsets[best]),
        "best_alternation": float(rates[best]),
        "worst_alternation": float(np.nanmin(rates)),
        "alternation_range": float(np.nanmax(rates) - np.nanmin(rates)),
    }


def spectral_context(omega_v: np.ndarray, fs_hz: float, f_step_hz: float) -> dict:
    """How much of omega_v sits at the stride rate -- the trivial-alternation term.

    A high `stride_power_fraction` means alternation is mostly a
    consequence of sampling a stride-rate oscillation once per step, and is
    expected even from a sensor that encodes nothing about laterality.
    """
    f, p = dsp.welch_psd(omega_v, fs_hz)
    stride = dsp.band_power(f, p, f_step_hz / 2.0, rel_bw=0.10)
    step = dsp.band_power(f, p, f_step_hz, rel_bw=0.10)
    total = dsp.band_power(f, p, f_step_hz * 2.2, rel_bw=0.82)  # ~[0.4f, 4f]
    return {
        "omega_v_rms_rad_s": float(np.sqrt(np.mean(np.asarray(omega_v, float) ** 2))),
        "omega_v_p95_abs_rad_s": float(np.percentile(np.abs(omega_v), 95)),
        "stride_power_fraction": float(stride / total) if total > 0 else np.nan,
        "step_power_fraction": float(step / total) if total > 0 else np.nan,
        "stride_over_step_power": float(stride / step) if step > 0 else np.nan,
    }


def phase_sweep_null(
    omega_v: np.ndarray,
    step_indices: np.ndarray,
    fs_hz: float,
    f_step_hz: float,
    n_surrogates: int = N_SURROGATES,
    seed: int = 0,
    n_offsets: int = 10,
) -> dict:
    """Null distribution for the *maximum* alternation over the phase sweep.

    `phase_sweep` reports a maximum over `n_offsets` candidate offsets.
    Comparing a maximum against a mean null is selection-biased: picking the
    best of ten tries beats an average even when nothing is there. The
    honest null therefore runs the same sweep-and-take-the-max procedure on
    each phase-randomised surrogate, so like is compared with like.
    """
    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(n_surrogates):
        s = phase_randomised_surrogate(omega_v, rng)
        sw = phase_sweep(s, step_indices, fs_hz, f_step_hz, n_offsets=n_offsets)
        if np.isfinite(sw["best_alternation"]):
            maxima.append(sw["best_alternation"])
    if not maxima:
        return {"max_mean": np.nan, "max_p95": np.nan, "n": 0}
    return {
        "max_mean": float(np.mean(maxima)),
        "max_p95": float(np.percentile(maxima, 95)),
        "n": int(len(maxima)),
    }


def analyse(
    rotation_rate: np.ndarray,
    gravity: np.ndarray,
    step_indices: np.ndarray,
    fs_hz: float,
    f_step_hz: float,
    mode: str = "tracking",
    seed: int = 0,
) -> dict:
    """Full stage-4 analysis for one trial segment.

    Returns alternation consistency alongside the controls needed to
    interpret it, plus a separation measure between the two sign clusters.
    """
    omega_v = omega_vertical(rotation_rate, gravity, mode=mode)
    contact = sample_at_contacts(omega_v, step_indices, fs_hz, f_step_hz)
    values = contact["values"]
    labels = label_alternating(values)
    rate = alternation_rate(values)
    nulls = null_models(omega_v, contact["step_indices"], fs_hz, f_step_hz, seed=seed)
    sweep = phase_sweep(omega_v, contact["step_indices"], fs_hz, f_step_hz)
    # Null for the best-phase result. Uses the same sweep-and-maximise
    # procedure on surrogates, so the selection over 10 offsets that
    # produced `best_alternation` is present in the null too.
    sweep_null = phase_sweep_null(
        omega_v, contact["step_indices"], fs_hz, f_step_hz,
        n_surrogates=max(20, N_SURROGATES // 4), seed=seed + 1,
    )
    spec = spectral_context(omega_v, fs_hz, f_step_hz)

    a = values[labels == "A"]
    b = values[labels == "B"]
    if len(a) > 1 and len(b) > 1:
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
        # Cohen's d between the two sign groups. Note this is guaranteed to
        # be positive by construction -- the groups are *defined* by sign --
        # so it measures how far apart the clusters sit, never whether the
        # split is real.
        separation = float(abs(a.mean() - b.mean()) / pooled) if pooled > 0 else np.inf
    else:
        separation = np.nan

    excess = rate - nulls["surrogate_mean"] if np.isfinite(nulls["surrogate_mean"]) else np.nan

    return {
        "omega_v": omega_v,
        "contact_values_rad_s": values,
        "contact_step_indices": contact["step_indices"],
        "labels": labels,
        "n_labelled_steps": int(len(values)),
        "contact_window_samples": contact["contact_window_samples"],
        "contact_window_s": contact["contact_window_s"],
        "alternation_consistency": rate,
        "alternation_consistent": bool(
            np.isfinite(rate) and rate >= ALTERNATION_CONSISTENT_THRESHOLD
        ),
        "cluster_separation_d": separation,
        "balance_A_fraction": float(np.mean(labels == "A")) if len(labels) else np.nan,
        "excess_over_surrogate": float(excess) if np.isfinite(excess) else np.nan,
        "best_phase_alternation": sweep["best_alternation"],
        "best_phase_offset_step_fraction": sweep["best_offset_step_fraction"],
        "worst_phase_alternation": sweep["worst_alternation"],
        "alternation_phase_range": sweep["alternation_range"],
        "best_phase_surrogate_max_mean": sweep_null["max_mean"],
        "best_phase_surrogate_max_p95": sweep_null["max_p95"],
        "best_phase_excess_over_surrogate": (
            sweep["best_alternation"] - sweep_null["max_mean"]
            if np.isfinite(sweep_null["max_mean"]) else np.nan
        ),
        # The only defensible verdict for the best-phase result: does it
        # exceed what the same procedure extracts from a signal with an
        # identical spectrum and no laterality information at all?
        "best_phase_beats_surrogate_p95": bool(
            np.isfinite(sweep_null["max_p95"])
            and sweep["best_alternation"] > sweep_null["max_p95"]
        ),
        "phase_sweep": sweep,
        **{f"null_{k}": v for k, v in nulls.items()},
        **spec,
        # Repeated in every record so it cannot be dropped downstream.
        "ground_truth_available": False,
        "accuracy_claim": "none possible: this dataset has no left/right foot labels",
    }
