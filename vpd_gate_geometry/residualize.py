"""Token-identity residualization of the gate field.

We model

    g_{n,c} ≈ μ_c + a_{z(n),c}

with `z(n)` = token id at row n. The token-conditioned offset is
shrunk for rare tokens:

    â_{z,c} = (n_z / (n_z + λ)) (mean_z g_{z,c} - μ_c)

Tokens seen fewer than `min_count` times are pooled into a single "rare"
bucket with zero offset. The vocabulary is optionally capped at the top
`max_vocab_tokens` ids by frequency to keep the per-token table from
blowing up on a sparse run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TokenBaseline:
    mu: torch.Tensor              # [C]      global mean
    token_offset: torch.Tensor    # [V_eff, C]
    token_count: torch.Tensor     # [V_eff]
    id_to_row: dict[int, int]     # original token id -> row in token_offset
    rare_row: int                 # row reserved for "rare" tokens (offset = 0)


def fit_token_baseline(
    G: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    max_vocab_tokens: int = 4096,
    min_count: int = 16,
    shrinkage: float = 100.0,
) -> TokenBaseline:
    """Fit per-token means with shrinkage.

    Args:
        G:            [N, C] gate matrix.
        token_ids:    [N] integer token ids aligned with rows of G.
        max_vocab_tokens: keep at most this many distinct tokens (by
            frequency); the rest fall into the rare bucket.
        min_count:    tokens with fewer than this many rows fall into
            the rare bucket too.
        shrinkage:    λ in the shrinkage formula above.
    """
    N, C = G.shape
    assert token_ids.shape == (N,)
    device = G.device
    mu = G.mean(dim=0)

    unique, counts = torch.unique(token_ids, return_counts=True)
    if max_vocab_tokens < unique.numel():
        top = torch.topk(counts, k=max_vocab_tokens).indices
        unique = unique[top]
        counts = counts[top]
    keep = counts >= min_count
    unique = unique[keep]
    counts = counts[keep]

    # Build a dense (V_eff + 1, C) offset table; the last row is the rare bucket.
    V_eff = unique.numel()
    id_to_row = {int(t): i for i, t in enumerate(unique.tolist())}
    rare_row = V_eff
    token_count = torch.zeros(V_eff + 1, device=device)
    token_sum = torch.zeros(V_eff + 1, C, device=device)
    # Vectorized id -> row lookup: pre-fill with `rare_row`, then scatter the
    # kept token rows in one shot (avoids a Python loop with GPU syncs).
    max_tok = int(token_ids.max().item()) + 1
    row_lookup = torch.full((max_tok,), rare_row, dtype=torch.long, device=device)
    row_lookup[unique] = torch.arange(V_eff, device=device)
    rows = row_lookup[token_ids]
    token_sum.index_add_(0, rows, G)
    token_count.index_add_(0, rows, torch.ones_like(rows, dtype=G.dtype))
    raw_mean = token_sum / token_count.clamp_min(1.0)[:, None]
    shrink = token_count / (token_count + shrinkage)
    token_offset = shrink[:, None] * (raw_mean - mu[None, :])
    # Rare bucket gets zero offset by definition.
    token_offset[rare_row] = 0.0

    return TokenBaseline(
        mu=mu,
        token_offset=token_offset,
        token_count=token_count,
        id_to_row=id_to_row,
        rare_row=rare_row,
    )


def residualize_gates(G: torch.Tensor, token_ids: torch.Tensor, baseline: TokenBaseline) -> torch.Tensor:
    device = G.device
    rare = baseline.rare_row
    max_tok = int(token_ids.max().item()) + 1
    row_lookup = torch.full((max_tok,), rare, dtype=torch.long, device=device)
    # Vectorized scatter of kept token-ids.
    keys = torch.tensor(list(baseline.id_to_row.keys()), dtype=torch.long, device=device)
    vals = torch.tensor(list(baseline.id_to_row.values()), dtype=torch.long, device=device)
    row_lookup[keys] = vals
    rows = row_lookup[token_ids]
    mu = baseline.mu.to(device)
    offset = baseline.token_offset.to(device)
    return G - mu[None, :] - offset[rows]


def explained_variance_by_token(
    G: torch.Tensor, G_residual: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Per-component R² = 1 - SS_residual / SS_centered."""
    centered = G - G.mean(dim=0, keepdim=True)
    sse = (G_residual ** 2).sum(dim=0)
    sst = (centered ** 2).sum(dim=0).clamp_min(eps)
    return 1.0 - sse / sst
