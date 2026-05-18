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
    K: torch.Tensor, k: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition of a symmetric kernel; returns (eigvals desc, eigvecs)."""
    K_sym = 0.5 * (K + K.T)
    eigvals, eigvecs = torch.linalg.eigh(K_sym)
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)
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


def cluster_order_from_kernel(K: torch.Tensor, n_clusters: int = 8) -> torch.Tensor:
    """Permutation that groups similar components together.

    Uses agglomerative clustering on distance `1 - K` if scikit-learn is
    available; otherwise falls back to sorting by the leading eigenvector.
    """
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
