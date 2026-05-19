"""Per-layer / per-module-type same-position kernel comparison.

Reads a cached GateMatrix and computes a cosine kernel + spectrum for
each (layer, module_type) subset. Outputs:

    summary.json   ─ effective rank / participation ratio / top-10
                     eigvals / alive count per layer-module.
    figures/       ─ per-layer overlaid spectrum + per-module-type
                     overlaid spectrum.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from . import cache_io, gate_matrix as gm_mod
from . import plotting, spectral
from .plotting import PALETTE


def _layer_and_module_type(name: str) -> tuple[int, str]:
    # e.g. "h.2.attn.q_proj" -> (2, "attn.q_proj")
    parts = name.split(".")
    return int(parts[1]), ".".join(parts[2:])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vpd_gate_geometry.per_layer")
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--alive-threshold", type=float, default=1e-4)
    p.add_argument("--max-components-per-module", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args(argv)

    print(f"[per_layer] loading {args.cache} ...", flush=True)
    gm, meta = cache_io.load_gate_matrix_cache(args.cache)
    modules = meta["modules"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotting.set_style()

    summary: dict[str, dict] = {}
    spectra_by_layer: dict[int, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    spectra_by_module_type: dict[str, list[tuple[int, torch.Tensor]]] = defaultdict(list)

    for module_name in modules:
        cols = [i for i, k in enumerate(gm.keys) if k.module_name == module_name]
        if not cols:
            continue
        idx = torch.tensor(cols, dtype=torch.long)
        G_mod = gm.G[:, idx]
        stats = spectral.gate_stats(G_mod, alive_threshold=args.alive_threshold)
        alive = stats.alive_mask
        if int(alive.sum().item()) < 4:
            print(f"[per_layer] {module_name}: only {int(alive.sum().item())} alive; skip")
            continue
        G_alive = G_mod[:, alive]
        if G_alive.shape[1] > args.max_components_per_module:
            G_alive, _sel = spectral.restrict_to_top_components(
                G_alive, args.max_components_per_module
            )

        K = spectral.cosine_kernel_gpu(G_alive, device=args.device)
        eigvals, _ = spectral.spectral_decompose(K, device=args.device)
        eff_rank = spectral.effective_rank(eigvals)
        pr = spectral.participation_ratio(eigvals)

        summary[module_name] = {
            "C_module": int(idx.numel()),
            "C_alive": int(alive.sum().item()),
            "C_used_for_kernel": int(G_alive.shape[1]),
            "effective_rank": eff_rank,
            "participation_ratio": pr,
            "top10": [float(x) for x in eigvals[:10].tolist()],
            "mean_of_means": float(stats.mean.mean().item()),
        }
        layer, mod_type = _layer_and_module_type(module_name)
        spectra_by_layer[layer].append((mod_type, eigvals))
        spectra_by_module_type[mod_type].append((layer, eigvals))
        print(
            f"[per_layer] {module_name:25s} alive={int(alive.sum()):4d} "
            f"eff_rank={eff_rank:7.2f} pr={pr:7.2f}",
            flush=True,
        )

    with (args.output_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    # Per-layer overlay
    import matplotlib.pyplot as plt
    palette_cycle = ["#6C5CE7", "#00A6A6", "#E76F51", "#2A9D8F", "#8D99AE", "#F4A261"]
    for layer, items in sorted(spectra_by_layer.items()):
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for color, (mod_type, eigvals) in zip(palette_cycle, items, strict=False):
            ax.plot(eigvals.cpu().numpy(), color=color, lw=1.2, label=mod_type)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel("eigenvalue")
        plotting._set_title(ax, f"Layer {layer}: per-module cosine spectrum")
        ax.legend(fontsize=8, ncol=2)
        plotting.save_close(fig, args.output_dir / f"layer_{layer}_spectra.png")

    # Per-module-type overlay across layers
    layer_colors = ["#6C5CE7", "#E76F51", "#2A9D8F", "#F4A261"]
    for mod_type, items in spectra_by_module_type.items():
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for (layer, eigvals) in sorted(items):
            ax.plot(eigvals.cpu().numpy(),
                    color=layer_colors[layer % len(layer_colors)],
                    lw=1.2, label=f"layer {layer}")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel("eigenvalue")
        plotting._set_title(ax, f"{mod_type}: spectrum across layers")
        ax.legend(fontsize=8)
        plotting.save_close(
            fig, args.output_dir / f"module_{mod_type.replace('.', '_')}_across_layers.png"
        )

    print(f"[per_layer] wrote summary + spectra -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
