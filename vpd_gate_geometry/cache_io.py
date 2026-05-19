"""Sparse COO storage for the gate-matrix cache.

The gate matrix is ~0.5% dense (most gate values are exactly 0 or below
the 10⁻⁴ noise floor), so dense fp32 storage wastes about 190× the disk
and IO needed. This module saves and loads the GateMatrix as a
COO-indexed fp16 tensor plus the small metadata vectors.

Storage shapes (for the canonical 16-batch × 8-seq × 512-tok run with
~13.3M NNZ at threshold 10⁻⁴):

    indices:  int32  [2, nnz]   ~106 MB
    values:   fp16   [nnz]       ~27 MB
    rest:                         <2 MB
    total:                       ~135 MB    (vs. 9.6 GB dense fp32)

Load happens in two stages: read the sparse arrays, then scatter them
into a freshly allocated dense fp32 tensor. Downstream code keeps
operating on dense G, so no analysis function needs to change.

Backwards-compatible: `load_gate_matrix_cache` detects the old dense
format and returns it unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .gate_matrix import GateMatrix


def save_gate_matrix_cache(
    gm: GateMatrix,
    modules: list[str],
    path: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
    threshold: float = 1e-5,
) -> dict[str, Any]:
    """Save `gm` to `path` in sparse COO fp16 format.

    Entries with `|g| <= threshold` are dropped. Returns a small
    summary dict (NNZ, density, file size) for logging.
    """
    G = gm.G
    if G.is_cuda:
        G = G.cpu()
    G = G.contiguous()

    mask = G > threshold
    nnz = int(mask.sum().item())
    indices = torch.nonzero(mask, as_tuple=False).T.to(torch.int32).contiguous()
    values = G[mask].to(torch.float16).contiguous()

    payload = {
        "format": "sparse_coo_fp16_v1",
        "indices": indices,
        "values": values,
        "shape": (int(G.shape[0]), int(G.shape[1])),
        "threshold": float(threshold),
        "modules": modules,
        "keys": gm.keys,
        "token_ids": gm.token_ids.contiguous(),
        "batch_idx": gm.batch_idx.contiguous(),
        "pos_idx": gm.pos_idx.contiguous(),
        "provenance": provenance or {},
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    size_bytes = out_path.stat().st_size
    return {
        "nnz": nnz,
        "density": nnz / (G.shape[0] * G.shape[1]),
        "file_size_mb": size_bytes / 1e6,
        "compression_ratio_vs_dense_fp32": (G.shape[0] * G.shape[1] * 4) / size_bytes,
    }


def load_gate_matrix_cache(path: str | Path) -> tuple[GateMatrix, dict[str, Any]]:
    """Load a cache file, transparently handling sparse-COO or legacy-dense.

    Returns `(gm, meta)` where `meta` contains the rest of the cache dict
    (modules, provenance, etc.) so callers can extract per-cache settings.
    """
    payload = torch.load(path, weights_only=False)

    if payload.get("format") == "sparse_coo_fp16_v1":
        N, C = payload["shape"]
        indices = payload["indices"].to(torch.long)
        values = payload["values"].to(torch.float32)
        G = torch.zeros((N, C), dtype=torch.float32)
        G[indices[0], indices[1]] = values
        gm = GateMatrix(
            G=G,
            keys=payload["keys"],
            token_ids=payload["token_ids"],
            batch_idx=payload["batch_idx"],
            pos_idx=payload["pos_idx"],
        )
        meta = {
            "modules": payload["modules"],
            "provenance": payload.get("provenance", {}),
            "threshold": payload["threshold"],
            "nnz": int(values.numel()),
            "format": payload["format"],
        }
        return gm, meta

    # Legacy dense format. Old cache files stored {"gm", "modules", "provenance", ...}.
    gm = payload["gm"]
    meta = {
        "modules": payload.get("modules", []),
        "provenance": payload.get("provenance", {}),
        "format": "dense_legacy",
    }
    return gm, meta
