"""Lagged co-importance sweep with null controls.

Run *after* you have cached a gate matrix:

    python -m vpd_gate_geometry.run_analysis ... \\
        --cache-gate-matrix outputs/gate_geometry/cache/pile4L.pt

then

    python -m vpd_gate_geometry.sweep_pairs \\
        --cache outputs/gate_geometry/cache/pile4L.pt \\
        --pairs "h.2.attn.q_proj:h.2.attn.k_proj,..." \\
        --output-dir outputs/gate_geometry/lagged_sweep \\
        --max-lag 6 --top-k 384 \\
        --n-null-runs 8 --null-kind circular

For each pair we compute:
- the real lagged Pearson correlation tensor K(τ);
- the same on `n_null_runs` independently shuffled copies of module B's
  gate matrix (default: per-sequence circular shift);
- distributional summaries (max, p99, p99.9, mean top-100, count>0.7);
- a real-vs-null lag profile plot.

The max-|r| is reported but treated as illustrative; the real claim is
that the *real* tail statistics separate from the null distribution.
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
    p.add_argument("--n-null-runs", type=int, default=8)
    p.add_argument("--null-kind", type=str, default="circular",
                   choices=["circular", "column", "none"])
    args = p.parse_args(argv)

    print(f"[sweep] loading cache {args.cache} ...", flush=True)
    cached = torch.load(args.cache, weights_only=False)
    gm = cached["gm"]
    modules = cached["modules"]
    provenance = cached.get("provenance", {})
    print(f"[sweep] gate matrix: G={tuple(gm.G.shape)} modules={len(modules)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotting.set_style()

    summary: dict[str, dict] = {
        "provenance": provenance,
        "config": {
            "max_lag": args.max_lag,
            "top_k": args.top_k,
            "n_null_runs": args.n_null_runs,
            "null_kind": args.null_kind,
        },
        "pairs": {},
    }
    pairs = [tuple(s.split(":")) for s in args.pairs.split(",")]
    lags_list = list(range(-args.max_lag, args.max_lag + 1))

    for module_a, module_b in pairs:
        if module_a not in modules or module_b not in modules:
            print(f"[sweep] skipping {module_a}->{module_b} (module missing)")
            continue
        print(f"[sweep] {module_a} -> {module_b}  (null={args.null_kind}, n_null={args.n_null_runs}) ...",
              flush=True)
        result = lagged.lagged_kernel_with_nulls(
            gm,
            module_a=module_a,
            module_b=module_b,
            lags=lags_list,
            top_k=args.top_k,
            normalize=True,
            device=args.device,
            n_null_runs=args.n_null_runs,
            null_kind=args.null_kind,
        )
        real = result["real"]
        real_stats = result["real_stats"]
        null_summary = result["null_summary"]
        top_pairs = lagged.top_lagged_pairs(real, n_top=15)

        lag_profile_real = torch.tensor([real_stats[tau]["mean_top_100"] for tau in lags_list])
        lag_profile_max  = torch.tensor([real_stats[tau]["max"] for tau in lags_list])
        if null_summary:
            lag_profile_null_mean = torch.tensor(
                [null_summary[tau]["mean_top_100_mean"] for tau in lags_list]
            )
            lag_profile_null_p95 = torch.tensor(
                [null_summary[tau]["mean_top_100_p95"] for tau in lags_list]
            )
        else:
            lag_profile_null_mean = None
            lag_profile_null_p95 = None

        tag = f"{module_a}__{module_b}".replace(".", "_")
        summary["pairs"][f"{module_a}->{module_b}"] = {
            "lags": lags_list,
            "real_stats": real_stats,
            "null_summary": null_summary,
            "top_pairs": top_pairs,
            "n_top_components_a": len(real.keys_a),
            "n_top_components_b": len(real.keys_b),
        }

        plotting.plot_lag_profile_with_null(
            lags_list,
            lag_profile_real,
            lag_profile_max=lag_profile_max,
            null_mean=lag_profile_null_mean,
            null_p95=lag_profile_null_p95,
            out_path=args.output_dir / f"{tag}__lag_profile.png",
            title=f"Lagged coimportance: {module_a} → {module_b}",
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
