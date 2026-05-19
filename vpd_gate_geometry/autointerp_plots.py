"""Plots + top-components table for the autointerp results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import plotting


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vpd_gate_geometry.autointerp_plots")
    p.add_argument("--persistent-scores", type=Path, required=True)
    p.add_argument("--control-scores", type=Path, default=None,
                   help="Optional non-persistent matched control scores file.")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotting.set_style()
    import matplotlib.pyplot as plt

    persistent = [v for v in json.load(open(args.persistent_scores))
                  if not v.get("skipped")]
    control = []
    if args.control_scores and args.control_scores.exists():
        control = [v for v in json.load(open(args.control_scores))
                   if not v.get("skipped")]

    # Sort persistent by balanced accuracy.
    persistent.sort(key=lambda v: -v["scores"]["balanced_accuracy"])

    # ---- Plot 1: bar chart of balanced accuracy ----
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(persistent) + 1.3))
    labels = [f"{v['label'][:34]}  ({v['module'].replace('h.','L').replace('_proj','').replace('.attn.','.attn.').replace('.mlp.','.mlp.')}#{v['local_idx']})"
              for v in persistent]
    accs = np.array([v["scores"]["balanced_accuracy"] for v in persistent])
    rand = np.array([v["random_label_scores"]["balanced_accuracy"] for v in persistent])
    y = np.arange(len(persistent))

    # Bars: persistent (real)
    colors = [plotting.PALETTE["raw"] if a >= 0.7 else
              plotting.PALETTE["lagged"] if a >= 0.55 else
              plotting.PALETTE["baseline"] for a in accs]
    ax.barh(y, accs, color=colors, edgecolor=plotting.PALETTE["bg"], linewidth=0.4,
            zorder=3, label="real")
    # Random-label baseline as dots
    ax.scatter(rand, y, s=22, color=plotting.PALETTE["muted"],
               marker="x", linewidth=1.2, zorder=4, label="random-label shuffle")
    # 0.5 chance line
    ax.axvline(0.5, color=plotting.PALETTE["axis"], lw=0.7, zorder=1)

    # If control distribution available, overlay its mean ± std
    if control:
        c_accs = np.array([v["scores"]["balanced_accuracy"] for v in control])
        cmean = c_accs.mean()
        cstd = c_accs.std()
        ax.axvspan(cmean - cstd, cmean + cstd,
                   color=plotting.PALETTE["token"], alpha=0.18,
                   zorder=2, label=f"non-persistent control band  (n={len(c_accs)})")
        ax.axvline(cmean, color=plotting.PALETTE["token"], lw=1.1, alpha=0.6, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=plotting.PALETTE["ink"],
                       family="monospace")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("held-out balanced accuracy")
    ax.legend(loc="lower right", borderpad=0.4)
    plotting.save_close(fig, args.output_dir / "01_validation_scores.png")

    # ---- Plot 2: accuracy vs token-identity baseline ----
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = np.array([v["token_identity_baseline"]["top_token_fraction_of_positives"]
                  for v in persistent])
    sc = ax.scatter(
        x, accs, s=60, c=accs, cmap=plotting.SEQUENTIAL_CMAP,
        edgecolor=plotting.PALETTE["ink"], linewidth=0.4, zorder=3,
    )
    ax.axhline(0.5, color=plotting.PALETTE["axis"], lw=0.7, zorder=1)
    ax.set_xlabel("fraction of positives sharing the top token id")
    ax.set_ylabel("held-out balanced accuracy")
    ax.set_xlim(0, max(0.5, x.max() * 1.05))
    ax.set_ylim(0, 1.02)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="balanced accuracy")
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=plotting.PALETTE["muted"], labelsize=9)
    plotting.save_close(fig, args.output_dir / "02_accuracy_vs_token_identity.png")

    # ---- Markdown table ----
    rows = []
    rows.append("| component | module | persistence | mean gate | label | bal acc | random-label | top token (frac) |")
    rows.append("| --- | --- | ---: | ---: | --- | ---: | ---: | --- |")
    for v in persistent:
        tbl = v["token_identity_baseline"]
        rows.append(
            f"| `c{v['component_idx']}` | `{v['module']}#{v['local_idx']}` | "
            f"{v['persistence']:.3f} | {v['mean_gate']:.4f} | "
            f"{v['label']} | **{v['scores']['balanced_accuracy']:.3f}** | "
            f"{v['random_label_scores']['balanced_accuracy']:.3f} | "
            f"{tbl['top_token_str']!r} ({tbl['top_token_fraction_of_positives']:.2f}) |"
        )
    (args.output_dir / "top_components_table.md").write_text("\n".join(rows))

    summary = {
        "n_persistent": len(persistent),
        "n_control": len(control),
        "persistent_mean_bal_acc": float(accs.mean()),
        "persistent_median_bal_acc": float(np.median(accs)),
        "persistent_random_label_mean": float(rand.mean()),
        "persistent_n_above_0_7": int((accs >= 0.7).sum()),
        "persistent_n_above_0_6": int((accs >= 0.6).sum()),
        "control_mean_bal_acc": float(c_accs.mean()) if control else None,
        "control_median_bal_acc": float(np.median(c_accs)) if control else None,
        "control_n_above_0_7": int((c_accs >= 0.7).sum()) if control else None,
    }
    (args.output_dir / "scoring_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote plots + table + summary -> {args.output_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
