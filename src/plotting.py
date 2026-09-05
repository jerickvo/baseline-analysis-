"""Plots for every stage. Figure-returning, so callers choose show vs save."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import dsp, loader, orientation, steps

# A short window is used wherever individual steps must be visible: 6 s
# shows ~17 steps at running cadence, enough to see the pattern and few
# enough that single footstrikes are still resolvable on screen.
DETAIL_WINDOW_S = 6.0


def _detail_slice(n: int, fs_hz: float, seconds: float = DETAIL_WINDOW_S, start_s: float = 0.0):
    a = int(round(start_s * fs_hz))
    return slice(a, min(n, a + int(round(seconds * fs_hz))))


def plot_raw_signals(trial: loader.Trial, seconds: float = 20.0, start_s: float = 0.0):
    """Stage 1: the four sensor groups, as they come off disk."""
    sl = _detail_slice(trial.n_samples, trial.fs_hz, seconds, start_s)
    t = trial.t[sl]
    groups = [
        ("attitude (rad)", loader.ATTITUDE_COLUMNS),
        ("gravity (g)", loader.GRAVITY_COLUMNS),
        ("rotationRate (rad/s)", loader.ROTATION_RATE_COLUMNS),
        ("userAcceleration (g)", loader.USER_ACCEL_COLUMNS),
    ]
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 9), sharex=True)
    for ax, (title, cols) in zip(axes, groups):
        for c in cols:
            ax.plot(t, trial.df[c].to_numpy()[sl], lw=0.8, label=c.split(".")[-1])
        ax.set_ylabel(title, fontsize=9)
        ax.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"{trial.ident}  |  {trial.n_samples} samples @ {trial.fs_hz:g} Hz "
        f"= {trial.duration_s:.1f} s  (showing {seconds:g} s)"
    )
    fig.tight_layout()
    return fig


def plot_trial_summary(summary: "object"):
    """Stage 1: distribution of duration and signal scale across all trials."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for label, grp in summary.groupby("trial"):
        axes[0].hist(grp["duration_s"], bins=12, alpha=0.6, label=f"trial {label}")
    axes[0].set_xlabel("duration (s)")
    axes[0].set_ylabel("trials")
    axes[0].legend(fontsize=8)
    axes[0].set_title("trial duration")
    axes[1].scatter(summary["subject"], summary["user_accel_rms_g"], s=18)
    axes[1].set_xlabel("subject")
    axes[1].set_ylabel("userAccel RMS (g)")
    axes[1].set_title("signal scale by subject")
    axes[2].scatter(summary["subject"], summary["rotation_rate_rms_rad_s"], s=18, color="C1")
    axes[2].set_xlabel("subject")
    axes[2].set_ylabel("rotationRate RMS (rad/s)")
    axes[2].set_title("gyro scale by subject")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_steady_state(result: dict):
    """Stage 3a: what the transient trimmer kept and what it cut."""
    tr = result["trial"]
    seg = result["segment"]
    mag = np.linalg.norm(tr.user_accel, axis=1)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(tr.t, mag, lw=0.5, color="0.6", label="|userAcceleration|")
    if "window_rms" in seg:
        w = int(round(1.0 * tr.fs_hz))
        centres = (np.arange(len(seg["window_rms"])) + 0.5) * w / tr.fs_hz
        ax.plot(centres, seg["window_rms"], lw=1.6, color="C0", label="1 s window RMS")
        thr = seg.get("threshold_g", 0.5 * np.median(seg["window_rms"]))
        ax.axhline(
            thr, color="C3", ls="--", lw=1.2,
            label=f"threshold {thr:.2f} g (0.5 x median, floor {steps.STEADY_RMS_FLOOR_G:g} g)",
        )
    ax.axvspan(
        seg["start"] / tr.fs_hz, seg["stop"] / tr.fs_hz, color="C2", alpha=0.15,
        label="kept as steady state",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("g")
    ax.set_title(
        f"{tr.ident}: trimmed {seg['trimmed_start_s']:.1f}s from the start, "
        f"{seg['trimmed_end_s']:.1f}s from the end"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_frame(result: dict):
    """Stage 2: is the anatomical frame stable, and is it well posed?"""
    fr = result["frame"]
    stab = result["verify"]["stability"]
    tr = result["trial"]
    fs = tr.fs_hz
    accel = tr.user_accel[result["segment"]["start"] : result["segment"]["stop"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    tilt = stab["tilt_series_deg"]
    ax.plot(np.arange(len(tilt)) / fs, tilt, lw=0.6)
    ax.axhline(np.median(tilt), color="C1", label=f"median {np.median(tilt):.1f} deg")
    ax.axhline(
        np.percentile(tilt, 95), color="C3", ls="--",
        label=f"p95 {np.percentile(tilt, 95):.1f} deg",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tilt of instantaneous vs mean up (deg)")
    ax.set_title("vertical-axis wander")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if stab["n_stability_windows"]:
        ax.plot(stab["window_start_s"], stab["window_drift_deg"], "o-", ms=4)
    ax.set_xlabel("window start (s)")
    ax.set_ylabel("forward-axis deviation (deg)")
    ax.set_title("forward-axis drift across windows")
    ax.grid(alpha=0.3)

    ax = axes[2]
    e1, e2 = orientation.horizontal_basis(fr.up)
    ax.scatter(accel @ e1, accel @ e2, s=2, alpha=0.25, color="0.5")
    lim = float(np.abs(np.c_[accel @ e1, accel @ e2]).max()) * 1.05
    ml_name = "ML (left)" if fr.diagnostics.get("forward_sign_confident") else "ML (side unresolved)"
    fwd_name = "forward" if fr.diagnostics.get("forward_sign_confident") else "forward (sign unresolved)"
    for vec, name, col in ((fr.forward, fwd_name, "C0"), (fr.mediolateral, ml_name, "C3")):
        ax.arrow(
            0, 0, float(vec @ e1) * lim * 0.8, float(vec @ e2) * lim * 0.8,
            color=col, width=lim * 0.008, length_includes_head=True, label=name,
        )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("horizontal axis 1 (g)")
    ax.set_ylabel("horizontal axis 2 (g)")
    ax.set_title(
        f"horizontal accel & estimated axes\neigenvalue ratio "
        f"{fr.diagnostics['forward_eigenvalue_ratio']:.1f}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"{tr.ident}  |  stage-2 verdict: {result['verify']['verdict']}", fontsize=11
    )
    fig.tight_layout()
    return fig


def plot_vertical_verification(result: dict):
    """Stage 2: does vertical acceleration show periodic footstrike structure?"""
    per = result["verify"]["periodicity"]
    tr = result["trial"]
    fs = tr.fs_hz
    a = result["a_vertical"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    sl = _detail_slice(len(a), fs)
    axes[0].plot(np.arange(len(a))[sl] / fs, a[sl], lw=1.0)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("vertical accel (g)")
    axes[0].set_title("resolved vertical acceleration")
    axes[0].grid(alpha=0.3)

    f, p = per["freqs"], per["psd"]
    m = f <= min(12.0, 0.45 * fs)
    axes[1].semilogy(f[m], p[m])
    for mult, style in ((0.5, ":"), (1.0, "-"), (2.0, "--")):
        axes[1].axvline(
            per["f_step_hz"] * mult, color="C3", ls=style, lw=1.0,
            label=f"{mult:g} x f_step",
        )
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].set_title(
        f"spectrum: f_step = {per['f_step_hz']:.2f} Hz = {per['implied_cadence_spm']:.0f} spm"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(per["lags_s"], per["autocorr"])
    T = 1.0 / per["f_step_hz"]
    axes[2].axvline(T, color="C1", ls="--", label=f"1 step ({per['step_regularity']:.2f})")
    axes[2].axvline(2 * T, color="C2", ls="--", label=f"1 stride ({per['stride_regularity']:.2f})")
    axes[2].axhline(0, color="0.7", lw=0.8)
    axes[2].set_xlabel("lag (s)")
    axes[2].set_ylabel("autocorrelation")
    axes[2].set_title(f"symmetry index = {per['step_symmetry_index']:.2f}")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_steps(result: dict, start_s: float = 0.0):
    """Stage 3: detected steps on the filtered trace, and the cadence series."""
    det = result["detection"]
    tr = result["trial"]
    fs = tr.fs_hz
    cs = result["cadence_summary"]
    cd = result["cadence_diagnosis"]
    x = det["filtered"]
    t0 = result["segment"]["start"] / fs

    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
    sl = _detail_slice(len(x), fs, DETAIL_WINDOW_S, start_s)
    t = np.arange(len(x))[sl] / fs + t0
    axes[0].plot(t, result["a_vertical"][sl], lw=0.8, color="0.7", label="vertical accel")
    axes[0].plot(
        t, x[sl], lw=1.4, color="C0",
        label=f"band-pass {det['band_hz'][0]:.2f}-{det['band_hz'][1]:.2f} Hz",
    )
    inwin = (det["step_times_s"] >= t[0]) & (det["step_times_s"] <= t[-1])
    axes[0].plot(
        det["step_times_s"][inwin], x[det["step_indices"][inwin]], "v", color="C3",
        ms=8, label="detected step",
    )
    axes[0].set_ylabel("g")
    axes[0].set_title(f"{tr.ident}: step detection (detail)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(alpha=0.3)

    cad = result["cadence_series"]
    axes[1].plot(cad["t_s"], cad["cadence_spm"], ".", ms=3, alpha=0.4, label="instantaneous")
    axes[1].plot(cad["t_s"], cad["cadence_spm_smooth"], lw=1.8, color="C1", label="6-interval median")
    axes[1].axhline(cs["cadence_spm"], color="C2", ls="-", label=f"trial {cs['cadence_spm']:.1f} spm")
    axes[1].axhline(
        result["spectral"]["cadence_spm"], color="C4", ls=":",
        label=f"spectral {result['spectral']['cadence_spm']:.1f} spm",
    )
    axes[1].axhspan(150, 190, color="C2", alpha=0.10, label="expected 150-190 spm")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("cadence (spm)")
    axes[1].set_title(f"cadence  |  {cd['diagnosis']}", fontsize=9)
    axes[1].legend(fontsize=8, ncol=2, loc="lower right")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_lateral(result: dict, start_s: float = 0.0):
    """Stage 4: sign of omega_v at contact, the phase sweep, and the nulls."""
    lat = result["lateral"]
    tr = result["trial"]
    fs = tr.fs_hz
    t0 = result["segment"]["start"] / fs
    omega = lat["omega_v"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))

    sl = _detail_slice(len(omega), fs, DETAIL_WINDOW_S, start_s)
    t = np.arange(len(omega))[sl] / fs + t0
    axes[0].plot(t, omega[sl], lw=1.0, color="0.4")
    axes[0].axhline(0, color="0.7", lw=0.8)
    idx = lat["contact_step_indices"]
    inwin = (idx >= sl.start) & (idx < sl.stop)
    for lab, col in (("A", "C0"), ("B", "C3")):
        m = inwin & (lat["labels"] == lab)
        axes[0].plot(idx[m] / fs + t0, lat["contact_values_rad_s"][m], "o", color=col, ms=7, label=f"step {lab}")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("omega about vertical (rad/s)")
    axes[0].set_title("sign at detected contact")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    sw = lat["phase_sweep"]
    axes[1].plot(sw["offsets_step_fraction"], sw["alternation_by_offset"], "o-")
    axes[1].axvline(0, color="C3", ls="--", lw=1.2, label="detected step phase")
    axes[1].axhline(lat["null_surrogate_mean"], color="C1", ls=":", label="surrogate null")
    axes[1].axhline(0.5, color="0.6", ls=":", label="random null")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("sampling offset (fraction of a step)")
    axes[1].set_ylabel("alternation consistency")
    axes[1].set_title("alternation depends on sampling phase")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    names = ["measured\n(at contact)", "best\nphase", "mid-step\ncontrol", "phase-random\nsurrogate", "random\nsigns"]
    vals = [
        lat["alternation_consistency"], lat["best_phase_alternation"],
        lat["null_mid_step_rate"], lat["null_surrogate_mean"], lat["null_random_mean"],
    ]
    axes[2].bar(names, vals, color=["C0", "C0", "0.6", "C1", "0.7"])
    axes[2].axhline(0.5, color="0.4", ls=":")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("alternation consistency")
    axes[2].set_title("consistency vs its null models\n(NO ground truth: not accuracy)", fontsize=9)
    axes[2].tick_params(axis="x", labelsize=7)
    axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_cadence_across_subjects(table):
    """Stage 3 validation: every trial's cadence against the expected band."""
    fig, ax = plt.subplots(figsize=(13, 4.4))
    for i, (label, grp) in enumerate(table.groupby("trial")):
        ok = ~grp["cadence_flagged"]
        ax.scatter(grp["subject"][ok], grp["cadence_spm"][ok], s=42, marker="os"[i],
                   label=f"trial {label} (ok)", color=f"C{i}")
        ax.scatter(grp["subject"][~ok], grp["cadence_spm"][~ok], s=90, marker="X",
                   color="C3", label=f"trial {label} (flagged)", zorder=5)
    ax.axhspan(150, 190, color="C2", alpha=0.12, label="expected 150-190 spm")
    ax.set_xlabel("subject")
    ax.set_ylabel("detected cadence (spm)")
    ax.set_xticks(sorted(table["subject"].unique()))
    ax.set_title("detected cadence, all jog trials")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
