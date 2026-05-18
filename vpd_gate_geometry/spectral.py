"""Spectral / covariance / kernel statistics over the gate matrix."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gate_matrix import GateMatrix


@dataclass
class GateStats:
    mean: torch.Tensor          # [C]
    var: torch.Tensor           # [C]
    support: torch.Tensor       # [C] fraction of rows with g > threshold
    alive_mask: torch.Tensor    # [C] bool


def gate_stats(G: torch.Tensor, alive_threshold: float = 1e-3) -> GateStats:
    N = G.shape[0]
    mean = G.mean(dim=0)
    var = G.var(dim=0, unbiased=False)
    support = (G > alive_threshold).float().mean(dim=0)
    alive_mask = mean > alive_threshold
    assert mean.shape == (G.shape[1],)
    _ = N
    return GateStats(mean=mean, var=var, support=support, alive_mask=alive_mask)


def restrict_to_top_components(
    G: torch.Tensor, k: int, score: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the top-k columns of G by `score` (defaults to mean activity)."""
    C = G.shape[1]
    if score is None:
        score = G.mean(dim=0)
    k = min(k, C)
    top = torch.topk(score, k=k).indices
    top, _ = torch.sort(top)
    return G[:, top].contiguous(), top


def cosine_kernel(G: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """K_{c,c'} = <G_c, G_c'> / (||G_c|| ||G_c'||)."""
    norms = G.norm(dim=0).clamp_min(eps)
    Gn = G / norms[None, :]
    return Gn.T @ Gn


def correlation_kernel(G: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation across rows."""
    Gc = G - G.mean(dim=0, keepdim=True)
    norms = Gc.norm(dim=0).clamp_min(eps)
    Gn = Gc / norms[None, :]
    return Gn.T @ Gn


def gram_then_cosine(G: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Equivalent to `cosine_kernel` but routed through the Gram matrix.

    Useful when G is very tall: we materialize `G^T @ G` once and reuse
    its diagonal for the normalization step.
    """
    S = G.T @ G
    q = S.diagonal().clamp_min(eps)
    return S / torch.sqrt(torch.outer(q, q))


def spectral_decompose(
    K: torch.Tensor, k: int | None = None, device: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition of a symmetric kernel; returns (eigvals desc, eigvecs).

    By default runs on whatever device K is on. Pass `device="cuda"` to
    force the eigendecomp onto GPU for big (>=1024 dim) kernels — torch
    eigh on H200 is much faster than CPU at that scale.
    """
    use_dev = device if device is not None else str(K.device)
    K_sym = 0.5 * (K + K.T)
    if use_dev != str(K.device):
        K_sym = K_sym.to(use_dev)
    # eigh on cuda needs float32 / float64
    eigvals, eigvecs = torch.linalg.eigh(K_sym)
    eigvals = eigvals.flip(0).cpu()
    eigvecs = eigvecs.flip(1).cpu()
    if k is not None:
        eigvals = eigvals[:k]
        eigvecs = eigvecs[:, :k]
    return eigvals, eigvecs


def effective_rank(eigvals: torch.Tensor, eps: float = 1e-12) -> float:
    p = eigvals.clamp_min(0) / eigvals.clamp_min(0).sum().clamp_min(eps)
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum()))


def participation_ratio(eigvals: torch.Tensor, eps: float = 1e-12) -> float:
    s1 = eigvals.clamp_min(0).sum()
    s2 = (eigvals.clamp_min(0) ** 2).sum().clamp_min(eps)
    return float((s1 * s1) / s2)


def spectral_embedding(
    K: torch.Tensor, dim: int = 2
) -> torch.Tensor:
    """Top-`dim` eigenvectors scaled by sqrt(eigenvalue)."""
    eigvals, eigvecs = spectral_decompose(K, k=dim)
    return eigvecs * eigvals.clamp_min(0).sqrt()[None, :]


def cluster_order_from_kernel(
    K: torch.Tensor, n_clusters: int = 8, max_n_for_agglom: int = 1024
) -> torch.Tensor:
    """Permutation that groups similar components together.

    Strategy:
        - For K with up to `max_n_for_agglom` nodes, run sklearn's
          AgglomerativeClustering on precomputed distance `1-K`.
        - Above that threshold, agglomerative becomes prohibitively
          slow (O(n^3)); fall back to lexicographic ordering by the
          sign-clustered top eigenvector and then by mean activity.
          This still produces a clean block layout in the heatmap
          while staying linear in n.
    """
    n = K.shape[0]
    if n > max_n_for_agglom:
        eigvals, eigvecs = spectral_decompose(K, k=min(8, n))
        # Quantize the top eigenvectors into buckets, then sort lex.
        v1 = eigvecs[:, 0]
        v2 = eigvecs[:, 1] if eigvecs.shape[1] > 1 else torch.zeros_like(v1)
        # Sort by sign of v1, then by v2 value, then by v1 magnitude.
        key = torch.stack([torch.sign(v1), v2, v1.abs()], dim=1)
        order = torch.argsort(key[:, 1])  # secondary key first
        order = order[torch.argsort(key[order, 0], stable=True)]  # primary key
        return order.to(torch.long)

    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        _, eigvecs = spectral_decompose(K, k=1)
        return torch.argsort(eigvecs[:, 0])

    D = (1.0 - K).clamp_min(0.0).cpu().numpy()
    D = 0.5 * (D + D.T)
    for i in range(D.shape[0]):
        D[i, i] = 0.0
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    labels = clusterer.fit_predict(D)
    order = sorted(range(len(labels)), key=lambda i: (labels[i], -float(K[i, i])))
    return torch.tensor(order, dtype=torch.long)


def cosine_kernel_gpu(
    G: torch.Tensor, device: str = "cuda", eps: float = 1e-8
) -> torch.Tensor:
    """Cosine kernel computed on `device` then returned on CPU.

    Useful when G is large (tens of thousands of rows × thousands of
    columns) and CPU BLAS would be the bottleneck. The result tensor
    is moved back to CPU before return so downstream eigendecomp /
    plotting don't have to care.
    """
    if not torch.cuda.is_available() or device == "cpu":
        return cosine_kernel(G, eps=eps)
    G_dev = G.to(device, dtype=torch.float32)
    norms = G_dev.norm(dim=0).clamp_min(eps)
    Gn = G_dev / norms[None, :]
    K = (Gn.T @ Gn).cpu()
    del G_dev, Gn
    torch.cuda.empty_cache()
    return K


def all_module_stats(
    gm: GateMatrix, alive_threshold: float = 1e-3
) -> dict[str, GateStats]:
    """GateStats per source module, useful for layerwise tables."""
    out: dict[str, GateStats] = {}
    keys = gm.keys
    module_to_cols: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        module_to_cols.setdefault(key.module_name, []).append(i)
    for m, cols in module_to_cols.items():
        idx = torch.tensor(cols, dtype=torch.long)
        out[m] = gate_stats(gm.G[:, idx], alive_threshold=alive_threshold)
    return out
