"""Flatten the per-module gate dict into a (B*T, C_total) matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .extract_gates import GateBatch


@dataclass(frozen=True)
class ComponentKey:
    """Identifies one rank-one parameter atom across all modules."""

    module_name: str
    local_idx: int

    def __str__(self) -> str:
        return f"{self.module_name}#{self.local_idx}"


@dataclass
class GateMatrix:
    """Concatenated gate field with row/column provenance.

    Shapes:
        G:         Float[Tensor, "N C"]      where N = sum_b B_b * T_b
        token_ids: Int[Tensor,   "N"]
        batch_idx: Int[Tensor,   "N"]        global batch index per row
        pos_idx:   Int[Tensor,   "N"]        sequence position per row
        keys:      list[ComponentKey] of length C, aligned with G's columns
    """

    G: torch.Tensor
    keys: list[ComponentKey]
    token_ids: torch.Tensor
    batch_idx: torch.Tensor
    pos_idx: torch.Tensor

    @property
    def n_rows(self) -> int:
        return int(self.G.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.G.shape[1])

    def restrict_components(self, mask: torch.Tensor) -> "GateMatrix":
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        return GateMatrix(
            G=self.G[:, idx].contiguous(),
            keys=[self.keys[int(i)] for i in idx.tolist()],
            token_ids=self.token_ids,
            batch_idx=self.batch_idx,
            pos_idx=self.pos_idx,
        )


def flatten_gate_dict(
    gates: dict[str, torch.Tensor], modules: list[str] | None = None
) -> tuple[torch.Tensor, list[ComponentKey]]:
    """Concatenate per-module [B, T, C_m] tensors into a single [B, T, C] tensor."""
    keys: list[ComponentKey] = []
    pieces: list[torch.Tensor] = []
    iter_modules = modules if modules is not None else list(gates.keys())
    for m in iter_modules:
        t = gates[m]
        assert t.dim() == 3, f"expected [B, T, C], got {tuple(t.shape)} for {m}"
        pieces.append(t)
        keys.extend(ComponentKey(module_name=m, local_idx=i) for i in range(t.shape[-1]))
    flat = torch.cat(pieces, dim=-1)
    return flat, keys


def build_gate_matrix(
    batches: Iterable[GateBatch],
    modules: list[str] | None = None,
    dtype: torch.dtype = torch.float32,
) -> GateMatrix:
    """Stack the gate dicts from multiple batches into one GateMatrix.

    Modules must be present in every batch (we take the intersection if
    `modules` is None and any batch disagrees). All batches must share
    `C_module` per module.
    """
    rows: list[torch.Tensor] = []
    token_rows: list[torch.Tensor] = []
    batch_idx_rows: list[torch.Tensor] = []
    pos_idx_rows: list[torch.Tensor] = []
    keys: list[ComponentKey] | None = None
    chosen_modules = modules

    for bi, batch in enumerate(batches):
        if chosen_modules is None:
            chosen_modules = list(batch.gates.keys())
        flat, batch_keys = flatten_gate_dict(batch.gates, chosen_modules)
        if keys is None:
            keys = batch_keys
        else:
            assert keys == batch_keys, "component keys must be stable across batches"
        B, T, _ = flat.shape
        rows.append(flat.reshape(B * T, -1).to(dtype))
        token_rows.append(batch.token_ids.reshape(B * T))
        batch_idx_rows.append(torch.full((B * T,), bi, dtype=torch.long))
        pos_idx_rows.append(
            torch.arange(T).unsqueeze(0).expand(B, T).reshape(B * T)
        )

    assert keys is not None and rows, "build_gate_matrix received zero batches"
    return GateMatrix(
        G=torch.cat(rows, dim=0),
        keys=keys,
        token_ids=torch.cat(token_rows, dim=0),
        batch_idx=torch.cat(batch_idx_rows, dim=0),
        pos_idx=torch.cat(pos_idx_rows, dim=0),
    )
