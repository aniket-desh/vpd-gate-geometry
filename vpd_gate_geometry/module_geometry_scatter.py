"""One-figure summary of Experiment 4: per-(layer, module) geometry.

Reads `per_layer/summary.json` and produces a scatter of alive
components vs effective rank for all 24 decomposed matrices, colored by
module type and labeled by layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import plotting


MODULE_ORDER = [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "mlp.c_fc",
    "mlp.down_proj",
]
MODULE_COLORS = {
    "attn.q_proj":   "#E94F64",   # highlight red
    "attn.k_proj":   "#6840E0",   # violet
    "attn.v_proj":   "#1FB6A7",   # teal
    "attn.o_proj":   "#3A1F8A",   # deep violet
    "mlp.c_fc":      "#FF8C42",   # orange
    "mlp.down_proj": "#43B581",   # green
}


def _parse_key(key: str) -> tuple[int, str]:
    parts = key.split(".", 2)
    return int(parts[1]), parts[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vpd_gate_geometry.module_geometry_scatter")
    p.add_argument("--summary", type=Path,
                   default=Path("outputs/gate_geometry/per_layer/summary.json"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/gate_geometry/per_layer/00_module_geometry_scatter.png"))
    args = p.parse_args(argv)

    summary = json.load(open(args.summary))
    plotting.set_style()
    import matplotlib.pyplot as plt

    points: dict[str, list[tuple[int, int, float]]] = {m: [] for m in MODULE_ORDER}
    labels: list[tuple[float, float, str]] = []
    for key, stats in summary.items():
        layer, module = _parse_key(key)
        alive = stats["C_alive"]
        eff_rank = stats["effective_rank"]
        points[module].append((layer, alive, eff_rank))

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot([1, 3000], [1, 3000], color=plotting.PALETTE["axis"], lw=0.7,
            ls=":", zorder=1, label="eff. rank = alive")

    for module in MODULE_ORDER:
        if not points[module]:
            continue
        layers, alives, ranks = zip(*sorted(points[module]))
        ax.scatter(alives, ranks, s=130,
                   color=MODULE_COLORS[module],
                   edgecolor=plotting.PALETTE["bg"], linewidth=1.2,
                   alpha=0.92, zorder=3, label=module)
        for layer, a, r in zip(layers, alives, ranks):
            labels.append((a, r, f"L{layer}"))

    for a, r, txt in labels:
        ax.annotate(
            txt, xy=(a, r), xytext=(5.5, 4.0),
            textcoords="offset points",
            fontsize=7.5, color=plotting.PALETTE["ink"], zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7, 2600)
    ax.set_ylim(4, 800)
    ax.set_xlabel("alive components  (log)")
    ax.set_ylabel("effective rank  (log)")
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=False,
                    handletextpad=0.4, borderpad=0.4, ncol=1,
                    title="module", title_fontsize=8.5)
    leg.get_title().set_color(plotting.PALETTE["muted"])

    ax.text(0.97, 0.05,
            "L1 attn.q/k corner: 10–46 alive,  ~5–7 effective dims\n"
            "L3 mlp.down corner: 1837 alive,  ~434 effective dims",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=plotting.PALETTE["muted"],
            linespacing=1.5)

    plotting.save_close(fig, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
