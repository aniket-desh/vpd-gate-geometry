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

    provenance: dict[str, Any] = {}
    if cfg.load_gate_matrix:
        cached = torch.load(cfg.load_gate_matrix, weights_only=False)
        gm = cached["gm"]
        modules = cached["modules"]
        provenance = cached.get("provenance") or {
            "n_batches": cached.get("n_batches"),
            "backend": "unknown (cache had no provenance dict)",
        }
        print(f"[vpd-gate-geometry] loaded cached gate matrix from "
              f"{cfg.load_gate_matrix} G={tuple(gm.G.shape)}", flush=True)
    else:
        batches = list(iter_gate_batches(cfg))
        assert batches, "no batches yielded by backend"
        modules = list(batches[0].gates.keys())
        print(f"[vpd-gate-geometry] {len(batches)} batches, "
              f"modules={modules}", flush=True)
        gm = gm_mod.build_gate_matrix(batches, modules=modules)
        provenance = {
            "backend": cfg.backend,
            "run_path": cfg.run_path,
            "target_model_path": cfg.target_model_path,
            "dataset_name": cfg.dataset_name,
            "dataset_split": cfg.dataset_split,
            "dataset_column": cfg.dataset_column,
            "n_batches": len(batches),
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "sampling": cfg.sampling,
            "seed": cfg.seed,
        }
        print(f"[vpd-gate-geometry] gate matrix: G={tuple(gm.G.shape)} "
              f"N_rows={gm.n_rows} C_total={gm.n_components}", flush=True)
        if cfg.cache_gate_matrix:
            from pathlib import Path as _P
            _P(cfg.cache_gate_matrix).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"gm": gm, "modules": modules, "provenance": provenance},
                cfg.cache_gate_matrix,
            )
            print(f"[vpd-gate-geometry] cached gate matrix -> "
                  f"{cfg.cache_gate_matrix}", flush=True)

    # Stage onto GPU once; analysis stays on GPU until the small final stats.
    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    work_device = "cuda" if use_cuda else "cpu"
    print(f"[vpd-gate-geometry] analysis device: {work_device}", flush=True)

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
    if use_cuda:
        gm_top = gm_mod.GateMatrix(
            G=gm_top.G.to(work_device, dtype=torch.float32),
            keys=gm_top.keys,
            token_ids=gm_top.token_ids.to(work_device),
            batch_idx=gm_top.batch_idx,
            pos_idx=gm_top.pos_idx,
        )
    print(f"[vpd-gate-geometry] alive={int(alive_keep.sum().item())} "
          f"kernel C={gm_top.n_components} on {work_device}", flush=True)

    K_raw = spectral.cosine_kernel_gpu(gm_top.G, device=work_device)
    eigvals_raw, _ = spectral.spectral_decompose(K_raw, device=work_device)
    eff_rank_raw = spectral.effective_rank(eigvals_raw)
    pr_raw = spectral.participation_ratio(eigvals_raw)
    order = spectral.cluster_order_from_kernel(
        K_raw, n_clusters=min(8, gm_top.n_components // 4 + 1), device=work_device
    )
    embed = spectral.spectral_embedding(K_raw, dim=2)

    # Centered Pearson kernel on the same gm_top — addresses the "is the
    # cosine spectrum just shared-activity / base-rate alignment?" critique.
    K_corr = spectral.correlation_kernel(
        gm_top.G if not use_cuda else gm_top.G
    )
    K_corr_cpu = K_corr.cpu() if K_corr.is_cuda else K_corr
    eigvals_corr, _ = spectral.spectral_decompose(K_corr, device=work_device)

    # Column-wise row-shuffled null on the cosine kernel: each column
    # is independently re-shuffled along the row axis, which preserves
    # every per-component marginal distribution while destroying all
    # cross-component co-activation. This is the "ambient" baseline the
    # real spectrum has to beat.
    perm = torch.randperm(gm_top.G.shape[0], device=gm_top.G.device)
    G_shuf = gm_top.G[perm]
    for c in range(0, gm_top.G.shape[1], 1024):
        slab = G_shuf[:, c:c + 1024]
        col_perm = torch.argsort(torch.rand(slab.shape, device=slab.device), dim=0)
        G_shuf[:, c:c + 1024] = torch.gather(slab, 0, col_perm)
    K_shuf = spectral.cosine_kernel_gpu(G_shuf, device=work_device)
    eigvals_shuf, _ = spectral.spectral_decompose(K_shuf, device=work_device)
    del G_shuf, perm
    if use_cuda:
        torch.cuda.empty_cache()

    baseline = residualize.fit_token_baseline(
        gm_top.G,
        gm_top.token_ids,
        max_vocab_tokens=cfg.residualize_max_vocab_tokens,
        min_count=cfg.residualize_min_count,
        shrinkage=cfg.residualize_shrinkage,
    )
    G_resid = residualize.residualize_gates(gm_top.G, gm_top.token_ids, baseline)
    r2 = residualize.explained_variance_by_token(gm_top.G, G_resid).cpu()
    K_resid = spectral.cosine_kernel_gpu(G_resid, device=work_device)
    eigvals_resid, _ = spectral.spectral_decompose(K_resid, device=work_device)

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
    lagged_result = lagged.lagged_kernel_with_nulls(
        gm,
        module_a=module_a,
        module_b=module_b,
        lags=lags,
        top_k=cfg.lagged_top_k,
        normalize=True,
        device=work_device,
        n_null_runs=cfg.n_null_runs,
        null_kind=cfg.null_kind,
    )
    lagged_kernels = lagged_result["real"]
    real_stats = lagged_result["real_stats"]
    null_summary = lagged_result["null_summary"]
    top_pairs = lagged.top_lagged_pairs(lagged_kernels, n_top=15)
    lag_profile_max = torch.tensor(
        [real_stats[tau]["max"] for tau in lags]
    )
    lag_profile_top100 = torch.tensor(
        [real_stats[tau]["mean_top_100"] for tau in lags]
    )
    if null_summary:
        null_top100_mean = torch.tensor(
            [null_summary[tau]["mean_top_100_mean"] for tau in lags]
        )
        null_top100_p95 = torch.tensor(
            [null_summary[tau]["mean_top_100_p95"] for tau in lags]
        )
    else:
        null_top100_mean = None
        null_top100_p95 = None

    summary: dict[str, Any] = {
        "backend": cfg.backend,
        "provenance": provenance,
        "modules": modules,
        "module_a": module_a,
        "module_b": module_b,
        "gate_matrix": {
            "N": gm.n_rows,
            "C_total": gm.n_components,
            "C_alive": int(alive_keep.sum().item()),
            "C_top_for_kernel": gm_top.n_components,
            "kernel_subset_label": (
                f"top-{gm_top.n_components} alive components "
                f"(threshold g̅>{cfg.alive_threshold}, ranked by mean activity)"
            ),
        },
        "per_module_stats": {m: _summarize_stats(s) for m, s in per_module_stats.items()},
        "spectrum_raw_cosine": {
            "effective_rank": eff_rank_raw,
            "participation_ratio": pr_raw,
            "top10": [float(x) for x in eigvals_raw[:10].tolist()],
        },
        "spectrum_pearson": {
            "effective_rank": spectral.effective_rank(eigvals_corr),
            "participation_ratio": spectral.participation_ratio(eigvals_corr),
            "top10": [float(x) for x in eigvals_corr[:10].tolist()],
        },
        "spectrum_shuffled_null_cosine": {
            "effective_rank": spectral.effective_rank(eigvals_shuf),
            "top10": [float(x) for x in eigvals_shuf[:10].tolist()],
        },
        "spectrum_residual_cosine": {
            "top10": [float(x) for x in eigvals_resid[:10].tolist()],
            "effective_rank": spectral.effective_rank(eigvals_resid),
        },
        "token_residualization": {
            "vocab_size_effective": int(baseline.token_count.numel() - 1),
            "median_r2": float(r2.median().item()),
            "mean_r2": float(r2.mean().item()),
            "scope": (
                "fitted on the same top-K alive subset used for the kernel "
                "(not the full 38,912 atoms)"
            ),
        },
        "lagged": {
            "lags": lags,
            "module_a": module_a,
            "module_b": module_b,
            "real_stats": real_stats,
            "null_kind": lagged_result["null_kind"],
            "n_null_runs": lagged_result["n_null_runs"],
            "null_summary": null_summary,
            "top_pairs_caveat": (
                "max |r| over hundreds of thousands of pairs has heavy "
                "extreme-value bias. Compare real_stats.mean_top_100 to "
                "null_summary.mean_top_100_p95 for a less fragile signal."
            ),
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
            "n_null_runs": cfg.n_null_runs,
            "null_kind": cfg.null_kind,
        },
    }

    with (out / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    if not cfg.skip_plots:
        plotting.set_style()
        plotting.plot_gate_activity_histogram(stats.mean, plot_dir / "01_gate_activity.png")
        n_show = min(2 * cfg.max_components, eigvals_raw.numel())
        plotting.plot_spectrum(
            eigvals_raw[:n_show],
            plot_dir / "02_spectrum_raw.png",
            title="Cosine kernel spectrum (raw)",
            extra={
                "token-residualized": eigvals_resid[:n_show],
                "shuffled null": eigvals_shuf[:n_show],
            },
            label_palette={
                "raw": "raw",
                "token-residualized": "residual",
                "shuffled null": "null",
            },
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
            eigvals_raw[:n_show],
            eigvals_resid[:n_show],
            plot_dir / "06_spectrum_raw_vs_residual.png",
        )
        # Real-vs-null lag profile (replaces old plot_lag_profile).
        plotting.plot_lag_profile_with_null(
            lags,
            lag_profile_top100,
            lag_profile_max=lag_profile_max,
            null_mean=null_top100_mean,
            null_p95=null_top100_p95,
            out_path=plot_dir / "07_lag_profile.png",
            title=f"Lagged co-importance: {module_a} → {module_b}",
        )
        plotting.plot_top_lagged_pair_heatmap(top_pairs, plot_dir / "08_top_lagged_pairs.png")
        # Kernel-variant overlay: cosine raw, Pearson, shuffled null.
        plotting.plot_spectrum(
            eigvals_raw[:n_show],
            plot_dir / "09_kernel_variants.png",
            title="Spectrum: cosine vs Pearson vs shuffled null",
            extra={
                "Pearson (centered)": eigvals_corr[:n_show],
                "shuffled null": eigvals_shuf[:n_show],
            },
            label_palette={
                "raw": "raw",
                "Pearson (centered)": "pearson",
                "shuffled null": "null",
            },
        )
        # Side-by-side real-vs-null kernel heatmaps under the same row order.
        # The dense block in the real panel should disappear under the null.
        plotting.plot_kernel_real_vs_null(
            K_raw, K_shuf,
            plot_dir / "10_kernel_real_vs_null.png",
            order=order,
            title="Co-importance kernel: real vs shuffled null",
        )

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
