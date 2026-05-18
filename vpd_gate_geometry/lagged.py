"""Lagged co-importance kernels for cross-position circuits.

For two component subsets A, B and lag τ:

    K^{A,B}(τ)_{c,c'} = corr(g^A_{b,t,c}, g^B_{b,t+τ,c'})

We restrict to the top-`k` components by mean activity by default to
keep the cross-product manageable; even at 38,912 atoms the per-module
top-256 × top-256 = 65,536 entries per lag is tiny.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gate_matrix import ComponentKey, GateMatrix
from .spectral import restrict_to_top_components


@dataclass
class LaggedKernels:
    lags: list[int]
    K: dict[int, torch.Tensor]                # lag -> [Ca, Cb] correlation
    keys_a: list[ComponentKey]
    keys_b: list[ComponentKey]
    module_a: str | None
    module_b: str | None


def _sequence_views(gm: GateMatrix) -> list[tuple[int, int, int]]:
    """Return [(seq_idx, row_start, row_end), ...] one entry per token sequence.

    `build_gate_matrix` flattens [B, T, C] to [B*T, C] in row-major
    order, so position 0 starts each sequence. We detect those starts
    via `pos_idx == 0`, which is robust to any batch packing.
    """
    pos = gm.pos_idx
    if pos.numel() == 0:
        return []
    starts = torch.nonzero(pos == 0, as_tuple=False).squeeze(-1).tolist()
    ends = starts[1:] + [pos.numel()]
    return [(i, int(s), int(e)) for i, (s, e) in enumerate(zip(starts, ends))]


def _module_columns(gm: GateMatrix, module: str | None) -> torch.Tensor:
    if module is None:
        return torch.arange(gm.n_components, dtype=torch.long)
    cols = [i for i, k in enumerate(gm.keys) if k.module_name == module]
    return torch.tensor(cols, dtype=torch.long)


def lagged_kernel(
    gm: GateMatrix,
    *,
    module_a: str | None = None,
    module_b: str | None = None,
    lags: list[int] | None = None,
    top_k: int = 256,
    normalize: bool = True,
    device: str = "cpu",
) -> LaggedKernels:
    """Compute lagged correlations between components in module_a and module_b.

    `module_a = module_b = None` means "all modules" on both sides, which
    is only safe for very small C (e.g. the mock backend's defaults).
    When `device="cuda"` the per-sequence cross-products are accumulated
    on GPU; the result is moved back to CPU before returning.
    """
    lags = lags if lags is not None else [-2, -1, 0, 1, 2]
    cols_a = _module_columns(gm, module_a)
    cols_b = _module_columns(gm, module_b)
    G_a = gm.G[:, cols_a]
    G_b = gm.G[:, cols_b]

    G_a_top, sel_a = restrict_to_top_components(G_a, top_k)
    G_b_top, sel_b = restrict_to_top_components(G_b, top_k)
    if device != "cpu" and torch.cuda.is_available():
        G_a_top = G_a_top.to(device, dtype=torch.float32)
        G_b_top = G_b_top.to(device, dtype=torch.float32)
    keys_a = [gm.keys[int(cols_a[int(i)])] for i in sel_a.tolist()]
    keys_b = [gm.keys[int(cols_b[int(i)])] for i in sel_b.tolist()]

    # Per-sequence views so cross-position pairs stay within a sequence.
    views = _sequence_views(gm)

    Ca = G_a_top.shape[1]
    Cb = G_b_top.shape[1]
    accum_dev = G_a_top.device
    # Pearson correlation accumulated per-lag from sufficient statistics:
    #     E[ab] - E[a]E[b]
    #     ─────────────────
    #     std_a · std_b
    # where the expectations are over the τ-valid slice (a_part, b_part).
    sum_ab: dict[int, torch.Tensor] = {tau: torch.zeros((Ca, Cb), device=accum_dev) for tau in lags}
    sum_a:  dict[int, torch.Tensor] = {tau: torch.zeros(Ca, device=accum_dev) for tau in lags}
    sum_b:  dict[int, torch.Tensor] = {tau: torch.zeros(Cb, device=accum_dev) for tau in lags}
    sum_a2: dict[int, torch.Tensor] = {tau: torch.zeros(Ca, device=accum_dev) for tau in lags}
    sum_b2: dict[int, torch.Tensor] = {tau: torch.zeros(Cb, device=accum_dev) for tau in lags}
    n_pairs: dict[int, int] = {tau: 0 for tau in lags}

    for _, start, end in views:
        A = G_a_top[start:end]
        B = G_b_top[start:end]
        T = A.shape[0]
        for tau in lags:
            if tau >= 0:
                if T - tau <= 0:
                    continue
                a_part = A[: T - tau]
                b_part = B[tau:]
            else:
                if T + tau <= 0:
                    continue
                a_part = A[-tau:]
                b_part = B[: T + tau]
            sum_ab[tau] += a_part.T @ b_part
            sum_a[tau]  += a_part.sum(dim=0)
            sum_b[tau]  += b_part.sum(dim=0)
            sum_a2[tau] += (a_part * a_part).sum(dim=0)
            sum_b2[tau] += (b_part * b_part).sum(dim=0)
            n_pairs[tau] += a_part.shape[0]

    K_out: dict[int, torch.Tensor] = {}
    for tau in lags:
        n = max(n_pairs[tau], 1)
        mean_a = sum_a[tau] / n
        mean_b = sum_b[tau] / n
        cov = sum_ab[tau] / n - torch.outer(mean_a, mean_b)
        if normalize:
            var_a = (sum_a2[tau] / n - mean_a * mean_a).clamp_min(1e-12)
            var_b = (sum_b2[tau] / n - mean_b * mean_b).clamp_min(1e-12)
            denom = torch.sqrt(torch.outer(var_a, var_b))
            K_out[tau] = (cov / denom).clamp_(-1.0, 1.0).cpu()
        else:
            K_out[tau] = cov.cpu()

    return LaggedKernels(
        lags=lags,
        K=K_out,
        keys_a=keys_a,
        keys_b=keys_b,
        module_a=module_a,
        module_b=module_b,
    )


def lagged_distributional_stats(kernels: LaggedKernels) -> dict[int, dict[str, float]]:
    """Tail statistics of |r| per lag.

    Max is fragile under extreme-value bias, especially when comparing
    hundreds of thousands of component pairs across many τ. We surface
    several less-fragile summaries:

    - p99_9 / p99 / p95 percentiles of |r|;
    - mean of the top-100 |r|;
    - counts of pairs above |r| ∈ {0.5, 0.7, 0.9}.

    Use these alongside (not instead of) `top_lagged_pairs`.
    """
    out: dict[int, dict[str, float]] = {}
    for tau in kernels.lags:
        absK = kernels.K[tau].abs().flatten().float()
        k_top = min(100, int(absK.numel()))
        out[tau] = {
            "max": float(absK.max().item()),
            "p99_9": float(torch.quantile(absK, 0.999).item()),
            "p99": float(torch.quantile(absK, 0.99).item()),
            "p95": float(torch.quantile(absK, 0.95).item()),
            "mean_top_100": float(torch.topk(absK, k=k_top).values.mean().item()),
            "n_above_0_5": int((absK > 0.5).sum().item()),
            "n_above_0_7": int((absK > 0.7).sum().item()),
            "n_above_0_9": int((absK > 0.9).sum().item()),
            "n_pairs": int(absK.numel()),
        }
    return out


def lagged_kernel_with_nulls(
    gm: GateMatrix,
    *,
    module_a: str,
    module_b: str,
    lags: list[int] | None = None,
    top_k: int = 256,
    normalize: bool = True,
    device: str = "cpu",
    n_null_runs: int = 8,
    null_kind: str = "circular",
    null_seed_base: int = 0,
) -> dict[str, object]:
    """Run `lagged_kernel` on real data and on `n_null_runs` randomized copies.

    The `null_kind` flag selects which permutation breaks structure:
    - "circular" → independent per-sequence circular shift of module B
                   (the canonical null for cross-position alignment);
    - "column"   → shuffle module B's columns uniformly at random
                   (tests whether the high-r pairs are component-specific
                    vs. surface from any same-module activity-matching);
    - "none"     → no null, return real-only.

    Returns:
        {
            "real": LaggedKernels (real data),
            "real_stats": {τ: distributional stats},
            "null_stats_per_run": [{τ: stats} × n_null_runs],
            "null_summary": {τ: per-stat mean/p95 over null runs},
            "null_kind": ...,
        }
    """
    from . import nulls as nulls_mod

    real = lagged_kernel(
        gm,
        module_a=module_a,
        module_b=module_b,
        lags=lags,
        top_k=top_k,
        normalize=normalize,
        device=device,
    )
    real_stats = lagged_distributional_stats(real)

    null_stats_per_run: list[dict[int, dict[str, float]]] = []
    if null_kind in ("circular", "column") and n_null_runs > 0:
        for run_idx in range(n_null_runs):
            seed = null_seed_base + run_idx
            if null_kind == "circular":
                gm_null = nulls_mod.circular_shift_per_sequence_b(gm, module_b, seed=seed)
            else:
                gm_null = nulls_mod.permute_components_b(gm, module_b, seed=seed)
            kernels_null = lagged_kernel(
                gm_null,
                module_a=module_a,
                module_b=module_b,
                lags=lags,
                top_k=top_k,
                normalize=normalize,
                device=device,
            )
            null_stats_per_run.append(lagged_distributional_stats(kernels_null))

    # Summarize: per-τ mean and 95th percentile of each stat across null runs.
    null_summary: dict[int, dict[str, float]] = {}
    if null_stats_per_run:
        for tau in real.lags:
            agg: dict[str, list[float]] = {}
            for s in null_stats_per_run:
                for k, v in s[tau].items():
                    agg.setdefault(k, []).append(v)
            null_summary[tau] = {}
            for k, vs in agg.items():
                vs_t = torch.tensor(vs, dtype=torch.float32)
                null_summary[tau][f"{k}_mean"] = float(vs_t.mean().item())
                null_summary[tau][f"{k}_p95"] = float(torch.quantile(vs_t, 0.95).item())

    return {
        "real": real,
        "real_stats": real_stats,
        "null_stats_per_run": null_stats_per_run,
        "null_summary": null_summary,
        "null_kind": null_kind,
        "n_null_runs": n_null_runs,
    }


def top_lagged_pairs(
    kernels: LaggedKernels, n_top: int = 20, exclude_tau0: bool = True
) -> list[dict[str, object]]:
    """Sort (c_a, c_b, τ) triples by |K(τ)| and return the top entries."""
    best_score: dict[tuple[int, int], tuple[int, float]] = {}
    for tau in kernels.lags:
        if exclude_tau0 and tau == 0:
            continue
        K = kernels.K[tau]
        absK = K.abs()
        flat = absK.flatten()
        top = torch.topk(flat, k=min(n_top * 4, flat.numel())).indices
        for f in top.tolist():
            i, j = divmod(int(f), K.shape[1])
            cur = best_score.get((i, j))
            score = float(absK[i, j].item())
            if cur is None or score > cur[1]:
                best_score[(i, j)] = (tau, score)

    ordered = sorted(best_score.items(), key=lambda kv: kv[1][1], reverse=True)
    out: list[dict[str, object]] = []
    for (i, j), (tau, score) in ordered[:n_top]:
        out.append({
            "key_a": str(kernels.keys_a[i]),
            "key_b": str(kernels.keys_b[j]),
            "best_tau": tau,
            "score": score,
            "signed": float(kernels.K[tau][i, j].item()),
        })
    return out
