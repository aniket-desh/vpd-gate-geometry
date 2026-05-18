"""Goodfire-style plotting helpers.

Visual language:
- Off-white warm background (#FAFAF6) for both figure and axes.
- Single muted-violet primary (#6840E0) for everything "raw".
- Teal (#1FB6A7) as the analytic counterpoint (residual / lagged).
- Orange (#FF8C42) for token-identity / lexical structure.
- Dark slate text (#2C2A35) on subtle grid (#E8E4DD).
- No top/right spines; thin (#C4BFB7) left/bottom spines.
- Inter / Source Sans Pro if available, else fall back to DejaVu Sans.

This mirrors the Goodfire research/blog aesthetic: white-on-warm,
deliberate accent colours, sparse axis chrome, big readable titles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Palette: muted, low-saturation accent colours over warm off-white.
PALETTE: dict[str, str] = {
    "raw":       "#6840E0",   # Goodfire-ish indigo / violet
    "residual":  "#1FB6A7",   # teal counterpoint
    "token":     "#FF8C42",   # warm orange (lexical)
    "lagged":    "#43B581",   # mid green
    "baseline":  "#9B9B9B",   # neutral mid-gray
    "highlight": "#E94F64",   # accent red, used sparingly
    "ink":       "#2C2A35",   # primary text
    "muted":     "#6C6873",   # secondary text
    "axis":      "#C4BFB7",   # axis lines + spines
    "grid":      "#E8E4DD",   # very subtle grid
    "bg":        "#FAFAF6",   # warm off-white
}

# Diverging colormap matching the violet/orange accent pair.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "vpd_diverging",
    [PALETTE["raw"], "#FFFFFF", PALETTE["token"]],
    N=256,
)
# Sequential heatmap colormap: white -> violet.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "vpd_sequential",
    ["#FFFFFF", "#E8E0FF", PALETTE["raw"], "#3A1F8A"],
    N=256,
)


def _preferred_font() -> str:
    """Pick the nicest sans-serif we can find on this machine."""
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Inter", "Source Sans Pro", "Source Sans 3",
                 "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if cand in available:
            return cand
    return "DejaVu Sans"


GOODFIRE_RC: dict[str, object] = {
    "figure.dpi": 140,
    "savefig.dpi": 240,
    "savefig.bbox": "tight",
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["bg"],
    "axes.edgecolor":   PALETTE["axis"],
    "axes.linewidth":   0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelcolor":   PALETTE["ink"],
    "axes.labelsize":    10.5,
    "axes.titlesize":    13,
    "axes.titleweight":  "semibold",
    "axes.titlecolor":   PALETTE["ink"],
    "axes.titlepad":     10,
    "axes.titlelocation": "left",
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.7,
    "grid.alpha":        1.0,
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   9.5,
    "xtick.major.size":  0,
    "ytick.major.size":  0,
    "text.color":        PALETTE["ink"],
    "legend.frameon":    False,
    "legend.fontsize":   9.5,
    "lines.linewidth":   1.6,
    "patch.linewidth":   0.0,
    "font.size":         10.5,
}


def set_style() -> None:
    plt.rcParams["font.family"] = _preferred_font()
    plt.rcParams.update(GOODFIRE_RC)


def set_style() -> None:
    plt.rcParams.update(GOODFIRE_RC)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def save_close(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_gate_activity_histogram(mean_per_component: torch.Tensor, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    data = _to_np(mean_per_component)
    ax.hist(data, bins=80, color=PALETTE["raw"], alpha=0.92,
            edgecolor=PALETTE["bg"], linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("mean causal importance per component (g̅)")
    ax.set_ylabel("# components  (log scale)")
    ax.set_title("Component activity")
    ax.margins(x=0)
    save_close(fig, out_path)


def plot_spectrum(
    eigvals: torch.Tensor,
    out_path: Path,
    title: str = "Co-importance kernel eigenspectrum",
    extra: dict[str, torch.Tensor] | None = None,
    annotation: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(_to_np(eigvals), color=PALETTE["raw"], lw=1.8, label="raw", zorder=3)
    if extra is not None:
        for name, vals in extra.items():
            ax.plot(_to_np(vals), color=PALETTE.get(name, PALETTE["muted"]),
                    lw=1.8, label=name, zorder=3)
        ax.legend(loc="upper right", borderpad=0.4)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(title)
    ax.set_yscale("symlog", linthresh=1e-4)
    if annotation:
        ax.text(
            0.98, 0.96, annotation, transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, color=PALETTE["muted"],
            bbox=dict(boxstyle="round,pad=0.4", fc=PALETTE["bg"],
                      ec=PALETTE["axis"], lw=0.6),
        )
    save_close(fig, out_path)


def plot_kernel_heatmap(
    K: torch.Tensor,
    out_path: Path,
    order: torch.Tensor | None = None,
    title: str = "Co-importance kernel",
    vmax: float | None = None,
) -> None:
    arr = _to_np(K)
    if order is not None:
        idx = _to_np(order)
        arr = arr[np.ix_(idx, idx)]
    if vmax is None:
        vmax = float(np.percentile(np.abs(arr), 99.0)) + 1e-9
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(arr, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("component  (reordered)")
    ax.set_ylabel("component  (reordered)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    save_close(fig, out_path)


def plot_spectral_embedding(
    coords: torch.Tensor,
    out_path: Path,
    color: torch.Tensor | None = None,
    title: str = "Spectral embedding (top 2 eigenvectors)",
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    xy = _to_np(coords)
    c = _to_np(color) if color is not None else None
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], c=c, cmap=SEQUENTIAL_CMAP, s=14, alpha=0.9,
        edgecolor="white", linewidth=0.3,
    )
    if c is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="mean gate g̅")
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    ax.set_xlabel("e₁")
    ax.set_ylabel("e₂")
    ax.set_title(title)
    save_close(fig, out_path)


def plot_token_r2_histogram(r2: torch.Tensor, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    data = _to_np(r2)
    ax.hist(data, bins=60, color=PALETTE["token"], alpha=0.9,
            edgecolor=PALETTE["bg"], linewidth=0.6)
    med = float(np.median(data))
    ax.axvline(med, color=PALETTE["ink"], lw=1.1, ls="--", zorder=4)
    ax.text(med, ax.get_ylim()[1] * 0.92, f"  median = {med:.3f}",
            color=PALETTE["ink"], fontsize=9.5, ha="left", va="top")
    ax.set_xlabel("R²  explained by token identity  (per component)")
    ax.set_ylabel("# components")
    ax.set_title("Token-identity explained variance")
    ax.margins(x=0)
    save_close(fig, out_path)


def plot_spectrum_raw_vs_residual(
    eigvals_raw: torch.Tensor, eigvals_resid: torch.Tensor, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(_to_np(eigvals_raw), color=PALETTE["raw"], lw=1.8,
            label="raw", zorder=3)
    ax.plot(_to_np(eigvals_resid), color=PALETTE["residual"], lw=1.8,
            label="token-residualized", zorder=3)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Raw vs token-residualized spectrum")
    ax.legend(loc="upper right", borderpad=0.4)
    save_close(fig, out_path)


def plot_lag_profile(
    lags: list[int], values: torch.Tensor, out_path: Path, label: str = "max |K(τ)|"
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    vals = _to_np(values)
    ax.plot(lags, vals, marker="o", color=PALETTE["lagged"],
            lw=1.8, markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.axvline(0, color=PALETTE["muted"], lw=0.8, linestyle=":", zorder=2)
    # Annotate the peak.
    peak_i = int(np.argmax(vals))
    ax.scatter([lags[peak_i]], [vals[peak_i]], s=120, facecolor="none",
               edgecolor=PALETTE["highlight"], lw=1.4, zorder=4)
    ax.annotate(
        f"τ*={lags[peak_i]:+d}\n{vals[peak_i]:.3f}",
        xy=(lags[peak_i], vals[peak_i]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=9.5, color=PALETTE["ink"],
    )
    ax.set_xlabel("lag τ  (positions B − positions A)")
    ax.set_ylabel(label)
    ax.set_title("Lagged co-importance profile")
    save_close(fig, out_path)


def plot_top_lagged_pair_heatmap(
    pairs: list[dict[str, object]], out_path: Path
) -> None:
    if not pairs:
        return
    labels = [f"{p['key_a']}  ↔  {p['key_b']}" for p in pairs]
    taus = [int(p["best_tau"]) for p in pairs]  # type: ignore[arg-type]
    scores = [float(p["signed"]) for p in pairs]  # type: ignore[arg-type]
    n = len(pairs)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * n + 1.4))
    pos = np.arange(n)
    colors = [PALETTE["raw"] if s >= 0 else PALETTE["highlight"] for s in scores]
    ax.barh(pos, scores, color=colors, edgecolor=PALETTE["bg"], linewidth=0.4)
    for y, tau, s in zip(pos, taus, scores, strict=True):
        ax.text(
            (s + 0.005) if s >= 0 else (s - 0.005),
            y,
            f"τ={tau:+d}",
            va="center",
            ha="left" if s >= 0 else "right",
            fontsize=9, color=PALETTE["muted"],
        )
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=8.5, color=PALETTE["ink"])
    ax.invert_yaxis()
    ax.axvline(0, color=PALETTE["axis"], lw=0.7, zorder=1)
    ax.set_xlabel("signed correlation at best τ")
    ax.set_title("Top lagged component pairs")
    save_close(fig, out_path)
