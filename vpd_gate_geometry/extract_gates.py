"""Backends for obtaining the causal-importance gate field.

The contract is: every backend yields one or more `GateBatch` objects,
each containing a dict of per-module gates of shape [B, T, C_module]
together with the token ids and metadata. The downstream `gate_matrix`
module flattens these into a (B*T, C_total) matrix for spectral
analysis.

Three backends:

- `mock`     : synthetic structured-noise gates with planted same-token
               clusters, token-correlated activity, and a lagged pair.
               Useful for end-to-end pipeline validation without W&B.
- `repo`     : live `ComponentModel.calc_causal_importances` on a
               trained VPD run. Requires `param_decomp` to be importable
               and `WANDB_API_KEY` to be set if the run path points at
               wandb. See `docs/repo_readiness_report.md` for the exact
               call chain.
- `artifact` : load `component_correlations.pt` and `token_stats.pt`
               from a finished harvest sub-run. Sufficient for the
               same-position kernel + token residualization, but cannot
               support lagged kernels (those need raw per-sequence
               gates, which harvest does not persist).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from .config import AnalysisConfig


@dataclass
class GateBatch:
    """One batch of causal-importance gates.

    Shapes:
        gates[module_name]: Float[Tensor, "B T C_module"]
        token_ids:          Int[Tensor,   "B T"]
    `metadata` carries per-batch provenance (module list, layer indices,
    backend name, ...) that may or may not be useful downstream.
    """

    gates: dict[str, torch.Tensor]
    token_ids: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.token_ids.shape[1])


# ---------------------------------------------------------------- mock backend


def _mock_iter_batches(cfg: AnalysisConfig) -> Iterator[GateBatch]:
    """Synthetic gates with planted structure for pipeline validation.

    Structure we plant so the spectral / residualization / lagged
    analyses each have something nontrivial to find:

    1. Per-token base rate ~ Beta(0.5, 5) per component (sparse).
    2. A planted same-position cluster: two components per module fire
       together when a "trigger" token is present.
    3. A planted lagged pair across the first two modules: when the
       trigger fires at position t, a downstream component in the
       second module fires at position t+1.
    4. Mild token-identity bias: each component has a per-token offset
       that residualization should be able to subtract out.
    """
    g = torch.Generator().manual_seed(cfg.seed)
    modules: list[str] = []
    for layer in range(cfg.mock_n_layers):
        for m in cfg.mock_modules:
            modules.append(f"h.{layer}.{m}")
    C = cfg.mock_components_per_module
    B = cfg.mock_batch_size
    T = cfg.mock_seq_len
    V = cfg.mock_vocab_size
    trigger_tok = 7  # arbitrary id used as "co-activation cue"

    # Token-identity offsets: small per-(token, component) bias.
    token_offsets = {
        m: 0.05 * torch.rand((V, C), generator=g) for m in modules
    }

    for batch_idx in range(cfg.mock_n_batches):
        token_ids = torch.randint(0, V, (B, T), generator=g)

        gates: dict[str, torch.Tensor] = {}
        for module_name in modules:
            base_rate = 0.02 + 0.10 * torch.distributions.Beta(0.5, 5.0).sample((C,))
            ci = (torch.rand((B, T, C), generator=g) < base_rate[None, None, :]).float()
            ci = ci * (0.3 + 0.7 * torch.rand((B, T, C), generator=g))
            # Token-identity bias.
            ci = ci + token_offsets[module_name][token_ids]
            gates[module_name] = ci.clamp(0.0, 1.0)

        # Plant same-position cluster: components 0 and 1 fire on trigger.
        for module_name in modules:
            mask = (token_ids == trigger_tok).float().unsqueeze(-1)
            gates[module_name][..., 0:2] = (
                gates[module_name][..., 0:2] * 0.1 + 0.8 * mask
            ).clamp(0.0, 1.0)

        # Plant lagged pair: trigger at position t in modules[0] → component 3
        # of modules[1] fires at position t+1.
        if len(modules) >= 2:
            src, dst = modules[0], modules[1]
            trig = (token_ids == trigger_tok).float()
            shifted = torch.zeros_like(trig)
            shifted[:, 1:] = trig[:, :-1]
            gates[dst][..., 3] = (gates[dst][..., 3] * 0.1 + 0.7 * shifted).clamp(0.0, 1.0)

        yield GateBatch(
            gates=gates,
            token_ids=token_ids,
            metadata={
                "backend": "mock",
                "batch_idx": batch_idx,
                "modules": modules,
                "C_per_module": C,
            },
        )


# ---------------------------------------------------------------- repo backend


def _repo_iter_batches(cfg: AnalysisConfig) -> Iterator[GateBatch]:
    """Live `ComponentModel.calc_causal_importances` over a tokenized dataset.

    Mirrors the call chain used by upstream's harvest pipeline
    (see external/param-decomp/.../harvest_fn/param_decomp.py).

    The run config carried by `ParamDecompRunInfo` already specifies the
    dataset, tokenizer, sampling type, and seq_len. By default we use
    that run config verbatim; `cfg.dataset_name` / `cfg.seq_len` /
    `cfg.sampling` only override it if explicitly set.
    """
    try:
        from param_decomp.models.component_model import (
            ComponentModel,
            ParamDecompRunInfo,
        )
        from param_decomp.data import train_loader_and_tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "repo backend requires `param_decomp` to be importable. "
            "Install it from external/param-decomp/ (note: it pins "
            "Python==3.13.*). Steps:\n"
            "    cd external/param-decomp && uv sync && cd ../..\n"
            "Then run this CLI from inside that venv "
            "(`uv run --project external/param-decomp -m vpd_gate_geometry.run_analysis ...`)."
        ) from exc

    if not cfg.run_path:
        raise RuntimeError(
            "repo backend requires --run-path "
            "(e.g. wandb:goodfire/spd/runs/s-55ea3f9b)."
        )

    # Offline-safe W&B: ComponentModel internals only call wandb.Api() if the
    # pretrained_model_name is a wandb path, so we sidestep both by setting
    # WANDB_MODE=offline and pointing pretrained_model_name at a local file.
    import os as _os
    _os.environ.setdefault("WANDB_MODE", "offline")

    print(f"[repo] loading run info from {cfg.run_path!r} ...", flush=True)
    run_info = ParamDecompRunInfo.from_path(cfg.run_path)
    if cfg.target_model_path:
        print(f"[repo] overriding pretrained_model_name -> {cfg.target_model_path}", flush=True)
        run_info.config = run_info.config.model_copy(
            update={"pretrained_model_name": cfg.target_model_path}
        )
    config = run_info.config
    sampling = cfg.sampling or config.sampling
    print(f"[repo] sampling={sampling!r}, building ComponentModel ...", flush=True)
    model = ComponentModel.from_run_info(run_info).to(cfg.device).eval()

    # If the user overrode any dataset bits, splice them into a *new* task_config
    # (pydantic models are frozen, so we model_copy through to a new Config).
    task_overrides: dict[str, object] = {}
    if cfg.dataset_name:
        task_overrides["dataset_name"] = cfg.dataset_name
    if cfg.dataset_split:
        task_overrides["train_data_split"] = cfg.dataset_split
    if cfg.dataset_column:
        task_overrides["column_name"] = cfg.dataset_column
    if cfg.seq_len:
        task_overrides["max_seq_len"] = cfg.seq_len
    if task_overrides:
        new_task = config.task_config.model_copy(update=task_overrides)
        config = config.model_copy(update={"task_config": new_task})

    task_cfg = config.task_config
    print(
        f"[repo] dataset={task_cfg.dataset_name} split={task_cfg.train_data_split} "
        f"col={task_cfg.column_name} seq_len={task_cfg.max_seq_len} batch_size={cfg.batch_size}",
        flush=True,
    )
    loader, _tok = train_loader_and_tokenizer(config, batch_size=cfg.batch_size)

    import torch as _torch
    n_yielded = 0
    for batch in loader:
        if n_yielded >= cfg.n_batches:
            break
        batch_device = batch.to(cfg.device)
        with _torch.no_grad():
            out = model(batch_device, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache,
                sampling=sampling,
                detach_inputs=True,
            )
        # Bring gates back to CPU to keep VRAM bounded across many batches.
        gates_cpu = {k: v.detach().cpu() for k, v in ci.lower_leaky.items()}
        n_yielded += 1
        print(
            f"[repo] batch {n_yielded}/{cfg.n_batches} "
            f"shape={tuple(batch.shape)} modules={len(gates_cpu)} "
            f"C_per_module={list(gates_cpu.values())[0].shape[-1]}",
            flush=True,
        )
        yield GateBatch(
            gates=gates_cpu,
            token_ids=batch.detach().cpu(),
            metadata={
                "backend": "repo",
                "run_path": cfg.run_path,
                "batch_idx": n_yielded - 1,
                "sampling": sampling,
            },
        )


# ------------------------------------------------------------ artifact backend


def _artifact_iter_batches(cfg: AnalysisConfig) -> Iterator[GateBatch]:
    """Load precomputed harvest artifacts.

    The harvest pipeline persists `component_correlations.pt` and
    `token_stats.pt` at:

        $PARAM_DECOMP_OUT_DIR/harvest/<run_id>/h-<timestamp>/

    These suffice for the same-position kernel and a coarse token
    residualization, but they are *not* per-batch raw gates, so the
    lagged-kernel analysis cannot be reconstructed from them alone.

    Status: stub. Wiring this in requires deciding on a representation
    for "kernel-only mode" — likely a different code path that skips
    `gate_matrix` entirely. Out of scope for the first commit.
    """
    raise RuntimeError(
        "artifact backend is not yet implemented. The harvest artifacts "
        "(component_correlations.pt, token_stats.pt) need a kernel-only "
        "analysis path that bypasses `gate_matrix`. See the TODO in "
        "extract_gates._artifact_iter_batches."
    )


# ----------------------------------------------------------- dispatch


def iter_gate_batches(cfg: AnalysisConfig) -> Iterator[GateBatch]:
    if cfg.backend == "mock":
        yield from _mock_iter_batches(cfg)
    elif cfg.backend == "repo":
        yield from _repo_iter_batches(cfg)
    elif cfg.backend == "artifact":
        yield from _artifact_iter_batches(cfg)
    else:
        raise ValueError(f"unknown backend: {cfg.backend!r}")
