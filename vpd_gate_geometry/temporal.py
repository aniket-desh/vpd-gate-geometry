"""Per-component temporal persistence on the gate field.

For each component c and each per-sequence binary trace
b_t(c) = 1[g_{b,t,c} > threshold], we compute:

    persistence(c) = P(b_{t+1} = 1 | b_t = 1)
                   = sum_{seq, t} (b_t b_{t+1}) / sum_{seq, t} b_t

If a component is "context-like" (stays on for runs of tokens), its
persistence is high (close to 1). If it is "trigger-like" (fires on
single tokens), its persistence is close to its baseline density.

We also report:

- mean on-run length per component, estimated as 1 / (1 - persistence)
  under a geometric assumption;
- the baseline (independence) prediction P(on) for each component,
  so we can show how much above baseline the persistence is.

CLI:

    python -m vpd_gate_geometry.temporal \\
        --cache outputs/gate_geometry/cache/pile4L_16x8x512.pt \\
        --output-dir outputs/gate_geometry/temporal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import cache_io, plotting
from .lagged import _sequence_views


def _compute_persistence(
    G: torch.Tensor, sequence_starts: list[int], sequence_ends: list[int],
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute per-component persistence and baseline density on GPU.

    Returns dict with:
        on_count    [C]   sum_{n} 1[g_n > thr]
        cont_count  [C]   sum_{seq,t} 1[g_t > thr and g_{t+1} > thr]
        prev_count  [C]   sum_{seq,t} 1[g_t > thr] over t<T-1 (denominator)
        T_total     int   total token positions in pairs (sum of (T-1) over seqs)
    """
    C = G.shape[1]
    dev = G.device
    on_count = torch.zeros(C, device=dev, dtype=torch.float64)
    cont_count = torch.zeros(C, device=dev, dtype=torch.float64)
    prev_count = torch.zeros(C, device=dev, dtype=torch.float64)
    T_total = 0
    for s, e in zip(sequence_starts, sequence_ends, strict=True):
        chunk = G[s:e]                # [T, C]
        T = chunk.shape[0]
        if T <= 1:
            on_count += (chunk > threshold).to(torch.float64).sum(dim=0)
            continue
        on = (chunk > threshold)
        on_count += on.to(torch.float64).sum(dim=0)
        # Pairs (t, t+1): we count co-on and previous-on for t in [0, T-2].
        prev_on = on[:-1]
        next_on = on[1:]
        cont_count += (prev_on & next_on).to(torch.float64).sum(dim=0)
        prev_count += prev_on.to(torch.float64).sum(dim=0)
        T_total += T - 1
    return {
        "on_count": on_count,
        "cont_count": cont_count,
        "prev_count": prev_count,
        "T_total": T_total,
        "total_rows": G.shape[0],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vpd_gate_geometry.temporal")
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binary 'on' threshold for g; default 0.5.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--alive-threshold", type=float, default=1e-4,
                   help="Drop dead components from the histogram (mean g <= this).")
    args = p.parse_args(argv)

    print(f"[temporal] loading {args.cache} ...", flush=True)
    gm, meta = cache_io.load_gate_matrix_cache(args.cache)
    G = gm.G.to(args.device) if args.device == "cuda" and torch.cuda.is_available() else gm.G
    print(f"[temporal] G={tuple(G.shape)} on {G.device}", flush=True)

    views = _sequence_views(gm)
    starts = [v[1] for v in views]
    ends = [v[2] for v in views]
    print(f"[temporal] {len(views)} sequences (per-sequence T = {ends[0]-starts[0]})", flush=True)

    stats = _compute_persistence(G, starts, ends, threshold=args.threshold)
    on = stats["on_count"]
    prev = stats["prev_count"]
    cont = stats["cont_count"]
    total_rows = stats["total_rows"]
    T_pairs = stats["T_total"]

    # Per-component persistence and baseline.
    base_p = (on / total_rows).cpu()                 # marginal P(on)
    cond_p = (cont / prev.clamp_min(1)).cpu()        # P(on_{t+1} | on_t)
    # Excess over independence: cond_p - base_p (positive = sticky).
    excess = cond_p - base_p
    # Geometric on-run length given on, under stationary assumption.
    # E[run | on] = 1 / (1 - persistence)
    run_len = 1.0 / (1.0 - cond_p.clamp(max=1.0 - 1e-6))

    # Filter to alive components for the plot/histograms.
    alive_mask = (G.mean(dim=0) > args.alive_threshold).cpu()

    summary = {
        "config": {
            "threshold": args.threshold,
            "alive_threshold": args.alive_threshold,
            "n_sequences": len(views),
            "total_rows": total_rows,
            "T_pairs": T_pairs,
        },
        "totals": {
            "C": int(on.numel()),
            "alive": int(alive_mask.sum().item()),
            "mean_persistence_alive": float(cond_p[alive_mask].mean().item()),
            "median_persistence_alive": float(cond_p[alive_mask].median().item()),
            "p95_persistence_alive": float(
                torch.quantile(cond_p[alive_mask], 0.95).item()
            ),
            "mean_baseline_alive": float(base_p[alive_mask].mean().item()),
            "median_run_len_alive": float(run_len[alive_mask].median().item()),
            "mean_run_len_alive": float(run_len[alive_mask].mean().item()),
            "p95_run_len_alive": float(
                torch.quantile(run_len[alive_mask], 0.95).item()
            ),
            # Decile breakdown: how many components have persistence > thresholds
            "n_persistence_above_0_3": int((cond_p[alive_mask] > 0.3).sum().item()),
            "n_persistence_above_0_5": int((cond_p[alive_mask] > 0.5).sum().item()),
            "n_persistence_above_0_8": int((cond_p[alive_mask] > 0.8).sum().item()),
            "n_persistence_above_0_95": int((cond_p[alive_mask] > 0.95).sum().item()),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    plotting.set_style()

    # Plot 1: histogram of P(on_{t+1} | on_t) over alive components.
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    vals = cond_p[alive_mask].numpy()
    ax.hist(vals, bins=50, color=plotting.PALETTE["raw"], alpha=0.9,
            edgecolor=plotting.PALETTE["bg"], linewidth=0.6)
    med = float(np.median(vals))
    ax.axvline(med, color=plotting.PALETTE["ink"], lw=1.1, ls="--", zorder=4)
    ax.text(
        med, ax.get_ylim()[1] * 0.92, f"  median = {med:.3f}",
        color=plotting.PALETTE["ink"], fontsize=9.5, ha="left", va="top",
    )
    ax.set_xlabel("P(on at t+1 | on at t)")
    ax.set_ylabel("# alive components")
    ax.margins(x=0)
    plotting.save_close(fig, args.output_dir / "01_persistence_hist.png")

    # Plot 2: scatter persistence vs baseline density.
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    x = base_p[alive_mask].numpy()
    y = cond_p[alive_mask].numpy()
    sc = ax.scatter(
        x, y, s=10, alpha=0.75, c=np.log10(np.clip(x, 1e-6, None)),
        cmap=plotting.SEQUENTIAL_CMAP, edgecolor=plotting.PALETTE["ink"],
        linewidth=0.2,
    )
    # Diagonal: independence line P(on|on)=P(on).
    lo, hi = max(x.min(), 1e-4), 1.0
    ax.plot([lo, hi], [lo, hi], color=plotting.PALETTE["muted"], lw=1.0,
            ls="--", label="independence")
    ax.set_xscale("log")
    ax.set_xlabel("baseline density  P(on)")
    ax.set_ylabel("persistence  P(on at t+1 | on at t)")
    ax.legend(loc="lower right", borderpad=0.4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log10  P(on)")
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=plotting.PALETTE["muted"], labelsize=9)
    plotting.save_close(fig, args.output_dir / "02_persistence_vs_density.png")

    print(f"[temporal] wrote summary + plots -> {args.output_dir}", flush=True)
    print(
        f"[temporal] {summary['totals']['alive']:,} alive components; "
        f"median persistence {summary['totals']['median_persistence_alive']:.3f}, "
        f"median on-run length {summary['totals']['median_run_len_alive']:.2f} tokens; "
        f"{summary['totals']['n_persistence_above_0_8']:,} components stay on with "
        f"prob > 0.8.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
