"""Null generators for the gate field.

These are deliberately *structure-destroying* permutations: each one
breaks one specific axis of the gate tensor while preserving the
marginals along the others. They let us ask "would my statistic look
this extreme under no real cross-position structure?".

Two complementary nulls:

- `circular_shift_per_sequence_b`: roll the target module's gate
  matrix by a random per-sequence offset. Preserves each module's
  own within-sequence autocorrelation and marginal histogram; breaks
  cross-module same-position alignment. This is the canonical null
  for lagged co-importance: under it, every Pearson r at every τ
  should be ≈ 0 except for ambient base-rate matching.

- `permute_components_b`: shuffle the columns of one module's gate
  matrix uniformly at random. Preserves all marginal-per-position
  statistics, breaks component identity. Tests whether high-r pairs
  surface because of *which* component is which vs. because the
  module's overall activity is correlated.

Both functions return a *new* `GateMatrix` (do not mutate the input).
The B module's keys remain attached but their column data is shuffled;
downstream code that filters by `key.module_name` keeps working.
"""

from __future__ import annotations

import numpy as np
import torch

from .gate_matrix import GateMatrix
from .lagged import _sequence_views


def _module_columns(gm: GateMatrix, module: str) -> torch.Tensor:
    return torch.tensor(
        [i for i, k in enumerate(gm.keys) if k.module_name == module],
        dtype=torch.long,
    )


def circular_shift_per_sequence_b(
    gm: GateMatrix, target_module: str, seed: int = 0
) -> GateMatrix:
    """Roll the target module's gate matrix by a random per-sequence offset.

    Breaks cross-module position alignment while keeping the target
    module's within-sequence statistics intact.
    """
    rng = np.random.default_rng(seed)
    cols = _module_columns(gm, target_module).to(gm.G.device)
    views = _sequence_views(gm)

    G_new = gm.G.clone()
    G_target = gm.G.index_select(1, cols)  # [N, C_target] view-friendly copy
    G_target_shuffled = G_target.clone()
    for _, start, end in views:
        T = end - start
        if T <= 1:
            continue
        offset = int(rng.integers(1, T))
        G_target_shuffled[start:end] = torch.roll(
            G_target[start:end], shifts=offset, dims=0
        )
    # Write the shifted target columns back into a full [N, C_total] copy.
    G_new.index_copy_(1, cols, G_target_shuffled)

    return GateMatrix(
        G=G_new,
        keys=gm.keys,
        token_ids=gm.token_ids,
        batch_idx=gm.batch_idx,
        pos_idx=gm.pos_idx,
    )


def permute_components_b(
    gm: GateMatrix, target_module: str, seed: int = 0
) -> GateMatrix:
    """Shuffle the columns of the target module uniformly at random."""
    rng = np.random.default_rng(seed)
    cols = _module_columns(gm, target_module).to(gm.G.device)
    perm = torch.as_tensor(rng.permutation(int(cols.numel())), dtype=torch.long, device=gm.G.device)
    permuted_cols = cols[perm]

    G_new = gm.G.clone()
    G_target_permuted = gm.G.index_select(1, permuted_cols)
    G_new.index_copy_(1, cols, G_target_permuted)
    return GateMatrix(
        G=G_new,
        keys=gm.keys,
        token_ids=gm.token_ids,
        batch_idx=gm.batch_idx,
        pos_idx=gm.pos_idx,
    )
