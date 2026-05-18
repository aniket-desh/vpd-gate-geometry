"""Goodfire-style plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


GOODFIRE_RC = {
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "axes.facecolor": "#fbfaf7",
    "figure.facecolor": "#fbfaf7",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.frameon": False,
}

PALETTE = {
    "raw": "#6C5CE7",
    "residual": "#00A6A6",
    "token": "#E76F51",
    "lagged": "#2A9D8F",
    "baseline": "#8D99AE",
}


def set_style() -> None:
    plt.rcParams.update(GOODFIRE_RC)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def save_close(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_gate_activity_histogram(mean_per_component: torch.Tensor, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    data = _to_np(mean_per_component)
    ax.hist(data, bins=64, color=PALETTE["raw"], edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("mean causal importance per component")
    ax.set_ylabel("count (log)")
    ax.set_title("Component activity distribution")
    save_close(fig, out_path)


def plot_spectrum(
    eigvals: torch.Tensor,
    out_path: Path,
    title: str = "Co-importance kernel eigenspectrum",
    extra: dict[str, torch.Tensor] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(_to_np(eigvals), color=PALETTE["raw"], lw=1.4, label="raw")
    if extra is not None:
        for name, vals in extra.items():
            ax.plot(_to_np(vals), color=PALETTE.get(name, "k"), lw=1.4, label=name)
        ax.legend(fontsize=9)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(title)
    ax.set_yscale("symlog", linthresh=1e-4)
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
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("component (reordered)")
    ax.set_ylabel("component (reordered)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_close(fig, out_path)


def plot_spectral_embedding(
    coords: torch.Tensor,
    out_path: Path,
    color: torch.Tensor | None = None,
    title: str = "Spectral embedding (top 2 eigenvectors)",
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    xy = _to_np(coords)
    c = _to_np(color) if color is not None else None
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="viridis", s=10, alpha=0.85, edgecolor="none")
    if c is not None:
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="mean gate")
    ax.set_xlabel("e_1")
    ax.set_ylabel("e_2")
    ax.set_title(title)
    save_close(fig, out_path)


def plot_token_r2_histogram(r2: torch.Tensor, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(_to_np(r2), bins=48, color=PALETTE["token"], edgecolor="white", linewidth=0.4)
    ax.set_xlabel("R² explained by token identity (per component)")
    ax.set_ylabel("count")
    ax.set_title("Token-identity explained variance")
    save_close(fig, out_path)


def plot_spectrum_raw_vs_residual(
    eigvals_raw: torch.Tensor, eigvals_resid: torch.Tensor, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(_to_np(eigvals_raw), color=PALETTE["raw"], lw=1.4, label="raw")
    ax.plot(_to_np(eigvals_resid), color=PALETTE["residual"], lw=1.4, label="residualized")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Raw vs token-residualized spectrum")
    ax.legend(fontsize=9)
    save_close(fig, out_path)


def plot_lag_profile(
    lags: list[int], values: torch.Tensor, out_path: Path, label: str = "max |K(τ)|"
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(lags, _to_np(values), marker="o", color=PALETTE["lagged"], lw=1.4)
    ax.axvline(0, color=PALETTE["baseline"], lw=0.8, linestyle=":")
    ax.set_xlabel("lag τ")
    ax.set_ylabel(label)
    ax.set_title("Lagged co-importance profile")
    save_close(fig, out_path)


def plot_top_lagged_pair_heatmap(
    pairs: list[dict[str, object]], out_path: Path
) -> None:
    if not pairs:
        return
    labels = [f"{p['key_a']} ↔ {p['key_b']}" for p in pairs]
    taus = [int(p["best_tau"]) for p in pairs]  # type: ignore[arg-type]
    scores = [float(p["signed"]) for p in pairs]  # type: ignore[arg-type]
    n = len(pairs)
    fig, ax = plt.subplots(figsize=(5.5, 0.3 * n + 1.2))
    pos = np.arange(n)
    ax.barh(pos, scores, color=[PALETTE["lagged"] if s >= 0 else PALETTE["token"] for s in scores])
    for y, tau in zip(pos, taus, strict=True):
        ax.text(0, y, f"τ={tau:+d} ", va="center", ha="right", fontsize=8, color="#444")
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("signed correlation at best τ")
    ax.set_title("Top lagged component pairs")
    save_close(fig, out_path)
