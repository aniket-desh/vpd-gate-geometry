"""End-to-end gate-geometry analysis CLI.

Pipeline:
    1. iterate gate batches from the chosen backend;
    2. flatten into a GateMatrix;
    3. compute per-component stats + alive filter;
    4. build cosine kernel and its eigenspectrum on the top-K alive
       components;
    5. fit token-identity baseline and recompute the spectrum;
    6. compute lagged co-importance kernels and surface the top pairs;
    7. write plots + summary.json.

Run:
    python -m vpd_gate_geometry.run_analysis --backend mock \
        --output-dir outputs/gate_geometry/mock_smoke \
        --max-components 128 --max-lag 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

from . import gate_matrix as gm_mod
from . import plotting, residualize, spectral, lagged
from .config import AnalysisConfig, config_from_argv
from .extract_gates import iter_gate_batches


def _summarize_stats(stats: spectral.GateStats) -> dict[str, Any]:
    return {
        "C": int(stats.mean.numel()),
        "alive_count": int(stats.alive_mask.sum().item()),
        "mean_of_means": float(stats.mean.mean().item()),
        "median_support": float(stats.support.median().item()),
    }


def run(cfg: AnalysisConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plot_dir = out / "plots"

    print(f"[vpd-gate-geometry] backend={cfg.backend} output_dir={out}", flush=True)

    batches = list(iter_gate_batches(cfg))
    assert batches, "no batches yielded by backend"
    modules = list(batches[0].gates.keys())
    print(f"[vpd-gate-geometry] {len(batches)} batches, "
          f"modules={modules}", flush=True)

    gm = gm_mod.build_gate_matrix(batches, modules=modules)
    print(f"[vpd-gate-geometry] gate matrix: G={tuple(gm.G.shape)} "
          f"N_rows={gm.n_rows} C_total={gm.n_components}", flush=True)

    stats = spectral.gate_stats(gm.G, alive_threshold=cfg.alive_threshold)
    per_module_stats = spectral.all_module_stats(gm, alive_threshold=cfg.alive_threshold)

    alive_keep = stats.alive_mask
    gm_alive = gm.restrict_components(alive_keep)
    # Cap component count for the kernel step.
    if gm_alive.n_components > cfg.max_components:
        G_top, sel = spectral.restrict_to_top_components(gm_alive.G, cfg.max_components)
        keys_top = [gm_alive.keys[int(i)] for i in sel.tolist()]
        gm_top = gm_mod.GateMatrix(
            G=G_top,
            keys=keys_top,
            token_ids=gm_alive.token_ids,
            batch_idx=gm_alive.batch_idx,
            pos_idx=gm_alive.pos_idx,
        )
    else:
        gm_top = gm_alive
    print(f"[vpd-gate-geometry] alive={int(alive_keep.sum().item())} "
          f"kernel C={gm_top.n_components}", flush=True)

    K_raw = spectral.cosine_kernel(gm_top.G)
    eigvals_raw, _ = spectral.spectral_decompose(K_raw)
    eff_rank_raw = spectral.effective_rank(eigvals_raw)
    pr_raw = spectral.participation_ratio(eigvals_raw)
    order = spectral.cluster_order_from_kernel(K_raw, n_clusters=min(8, gm_top.n_components // 4 + 1))
    embed = spectral.spectral_embedding(K_raw, dim=2)

    baseline = residualize.fit_token_baseline(
        gm_top.G,
        gm_top.token_ids,
        max_vocab_tokens=cfg.residualize_max_vocab_tokens,
        min_count=cfg.residualize_min_count,
        shrinkage=cfg.residualize_shrinkage,
    )
    G_resid = residualize.residualize_gates(gm_top.G, gm_top.token_ids, baseline)
    r2 = residualize.explained_variance_by_token(gm_top.G, G_resid)
    K_resid = spectral.cosine_kernel(G_resid)
    eigvals_resid, _ = spectral.spectral_decompose(K_resid)

    # Lagged kernel: prefer user-specified pair, otherwise default to a
    # meaningful within-layer Q/K pair if present, else the first two modules.
    module_a = cfg.lagged_module_a or next(
        (m for m in modules if m.endswith(".attn.q_proj")), modules[0]
    )
    module_b = cfg.lagged_module_b or next(
        (m for m in modules
         if m.endswith(".attn.k_proj") and m.rsplit(".", 2)[0] == module_a.rsplit(".", 2)[0]),
        modules[1] if len(modules) > 1 else modules[0],
    )
    lags = list(range(-cfg.max_lag, cfg.max_lag + 1))
    lagged_kernels = lagged.lagged_kernel(
        gm,
        module_a=module_a,
        module_b=module_b,
        lags=lags,
        top_k=cfg.lagged_top_k,
        normalize=True,
    )
    top_pairs = lagged.top_lagged_pairs(lagged_kernels, n_top=15)
    lag_profile = torch.tensor(
        [float(lagged_kernels.K[tau].abs().max().item()) for tau in lags]
    )

    summary: dict[str, Any] = {
        "backend": cfg.backend,
        "n_batches": len(batches),
        "modules": modules,
        "module_a": module_a,
        "module_b": module_b,
        "gate_matrix": {
            "N": gm.n_rows,
            "C_total": gm.n_components,
            "C_alive": int(alive_keep.sum().item()),
            "C_top_for_kernel": gm_top.n_components,
        },
        "per_module_stats": {m: _summarize_stats(s) for m, s in per_module_stats.items()},
        "spectrum_raw": {
            "effective_rank": eff_rank_raw,
            "participation_ratio": pr_raw,
            "top10": [float(x) for x in eigvals_raw[:10].tolist()],
        },
        "spectrum_residual": {
            "top10": [float(x) for x in eigvals_resid[:10].tolist()],
            "effective_rank": spectral.effective_rank(eigvals_resid),
        },
        "token_residualization": {
            "vocab_size_effective": int(baseline.token_count.numel() - 1),
            "median_r2": float(r2.median().item()),
            "mean_r2": float(r2.mean().item()),
        },
        "lagged": {
            "lags": lags,
            "max_abs_per_lag": [float(x) for x in lag_profile.tolist()],
            "top_pairs": top_pairs,
        },
        "config": {
            "max_components": cfg.max_components,
            "max_lag": cfg.max_lag,
            "alive_threshold": cfg.alive_threshold,
            "lagged_top_k": cfg.lagged_top_k,
            "residualize_max_vocab_tokens": cfg.residualize_max_vocab_tokens,
            "residualize_min_count": cfg.residualize_min_count,
            "residualize_shrinkage": cfg.residualize_shrinkage,
        },
    }

    with (out / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    if not cfg.skip_plots:
        plotting.set_style()
        plotting.plot_gate_activity_histogram(stats.mean, plot_dir / "01_gate_activity.png")
        plotting.plot_spectrum(
            eigvals_raw[: min(2 * cfg.max_components, eigvals_raw.numel())],
            plot_dir / "02_spectrum_raw.png",
            title="Cosine kernel spectrum (raw)",
        )
        plotting.plot_kernel_heatmap(
            K_raw, plot_dir / "03_kernel_clustered_heatmap.png", order=order,
            title="Cosine kernel (clustered)",
        )
        plotting.plot_spectral_embedding(
            embed, plot_dir / "04_spectral_embedding.png",
            color=gm_top.G.mean(dim=0),
        )
        plotting.plot_token_r2_histogram(r2, plot_dir / "05_token_r2.png")
        plotting.plot_spectrum_raw_vs_residual(
            eigvals_raw[: min(2 * cfg.max_components, eigvals_raw.numel())],
            eigvals_resid[: min(2 * cfg.max_components, eigvals_resid.numel())],
            plot_dir / "06_spectrum_raw_vs_residual.png",
        )
        plotting.plot_lag_profile(lags, lag_profile, plot_dir / "07_lag_profile.png")
        plotting.plot_top_lagged_pair_heatmap(top_pairs, plot_dir / "08_top_lagged_pairs.png")

    print(f"[vpd-gate-geometry] wrote summary -> {out / 'summary.json'}", flush=True)
    if not cfg.skip_plots:
        print(f"[vpd-gate-geometry] wrote plots  -> {plot_dir}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_argv(argv)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
