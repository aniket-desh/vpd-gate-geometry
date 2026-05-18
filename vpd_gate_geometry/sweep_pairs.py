"""Sweep lagged-coimportance over a list of module pairs.

Run *after* you have cached a gate matrix with the main CLI, e.g.

    python -m vpd_gate_geometry.run_analysis ... \\
        --cache-gate-matrix outputs/gate_geometry/cache/pile4L.pt

then

    python -m vpd_gate_geometry.sweep_pairs \\
        --cache outputs/gate_geometry/cache/pile4L.pt \\
        --pairs "h.2.attn.q_proj:h.2.attn.k_proj,h.2.attn.o_proj:h.2.attn.v_proj,h.0.attn.q_proj:h.0.attn.k_proj,h.0.attn.o_proj:h.0.attn.v_proj,h.0.mlp.down_proj:h.1.attn.q_proj,h.0.mlp.down_proj:h.1.attn.k_proj" \\
        --output-dir outputs/gate_geometry/lagged_sweep \\
        --max-lag 6 --top-k 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import lagged
from . import plotting


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vpd_gate_geometry.sweep_pairs")
    p.add_argument("--cache", type=str, required=True)
    p.add_argument(
        "--pairs", type=str, required=True,
        help='Comma-separated "modA:modB" specs.',
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-lag", type=int, default=6)
    p.add_argument("--top-k", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args(argv)

    print(f"[sweep] loading cache {args.cache} ...", flush=True)
    cached = torch.load(args.cache, weights_only=False)
    gm = cached["gm"]
    modules = cached["modules"]
    print(f"[sweep] gate matrix: G={tuple(gm.G.shape)} modules={len(modules)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotting.set_style()

    summary: dict[str, dict] = {}
    pairs = [tuple(s.split(":")) for s in args.pairs.split(",")]
    lags = list(range(-args.max_lag, args.max_lag + 1))

    for module_a, module_b in pairs:
        if module_a not in modules or module_b not in modules:
            print(f"[sweep] skipping {module_a}->{module_b} (module missing)")
            continue
        print(f"[sweep] {module_a} -> {module_b} ...", flush=True)
        kernels = lagged.lagged_kernel(
            gm,
            module_a=module_a,
            module_b=module_b,
            lags=lags,
            top_k=args.top_k,
            normalize=True,
            device=args.device,
        )
        top_pairs = lagged.top_lagged_pairs(kernels, n_top=15)
        lag_profile = torch.tensor(
            [float(kernels.K[tau].abs().max().item()) for tau in lags]
        )
        tag = f"{module_a}__{module_b}".replace(".", "_")
        summary[f"{module_a}->{module_b}"] = {
            "lags": lags,
            "max_abs_per_lag": [float(x) for x in lag_profile.tolist()],
            "top_pairs": top_pairs,
            "n_top_components_a": len(kernels.keys_a),
            "n_top_components_b": len(kernels.keys_b),
        }
        plotting.plot_lag_profile(
            lags, lag_profile, args.output_dir / f"{tag}__lag_profile.png",
            label=f"max |K(τ)|  {module_a}→{module_b}",
        )
        plotting.plot_top_lagged_pair_heatmap(
            top_pairs, args.output_dir / f"{tag}__top_pairs.png"
        )

    with (args.output_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[sweep] wrote summary -> {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
