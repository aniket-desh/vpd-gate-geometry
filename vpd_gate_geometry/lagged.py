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
    """Return [(batch_idx, row_start, row_end), ...] sorted by batch.

    Within each batch the rows are guaranteed contiguous in `gm.G`
    (build_gate_matrix concatenates in batch order), so we can index
    them as a single 2D [T, C] slice.
    """
    out: list[tuple[int, int, int]] = []
    batch = gm.batch_idx
    n = batch.numel()
    if n == 0:
        return out
    start = 0
    cur = int(batch[0].item())
    for i in range(1, n):
        b = int(batch[i].item())
        if b != cur:
            out.append((cur, start, i))
            start = i
            cur = b
    out.append((cur, start, n))
    return out


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
) -> LaggedKernels:
    """Compute lagged correlations between components in module_a and module_b.

    `module_a = module_b = None` means "all modules" on both sides, which
    is only safe for very small C (e.g. the mock backend's defaults).
    """
    lags = lags if lags is not None else [-2, -1, 0, 1, 2]
    cols_a = _module_columns(gm, module_a)
    cols_b = _module_columns(gm, module_b)
    G_a = gm.G[:, cols_a]
    G_b = gm.G[:, cols_b]

    G_a_top, sel_a = restrict_to_top_components(G_a, top_k)
    G_b_top, sel_b = restrict_to_top_components(G_b, top_k)
    keys_a = [gm.keys[int(cols_a[int(i)])] for i in sel_a.tolist()]
    keys_b = [gm.keys[int(cols_b[int(i)])] for i in sel_b.tolist()]

    # Per-sequence views so cross-position pairs stay within a sequence.
    views = _sequence_views(gm)

    if normalize:
        mu_a = G_a_top.mean(dim=0)
        mu_b = G_b_top.mean(dim=0)
        std_a = G_a_top.std(dim=0, unbiased=False).clamp_min(1e-8)
        std_b = G_b_top.std(dim=0, unbiased=False).clamp_min(1e-8)

    Ca = G_a_top.shape[1]
    Cb = G_b_top.shape[1]
    K_out: dict[int, torch.Tensor] = {tau: torch.zeros((Ca, Cb)) for tau in lags}
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
            if normalize:
                a_part = (a_part - mu_a[None, :]) / std_a[None, :]
                b_part = (b_part - mu_b[None, :]) / std_b[None, :]
            K_out[tau] += a_part.T @ b_part
            n_pairs[tau] += a_part.shape[0]

    for tau in lags:
        if n_pairs[tau] > 0:
            K_out[tau] /= n_pairs[tau]

    return LaggedKernels(
        lags=lags,
        K=K_out,
        keys_a=keys_a,
        keys_b=keys_b,
        module_a=module_a,
        module_b=module_b,
    )


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
