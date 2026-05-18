# Repo readiness report — VPD gate geometry mini-project

Written from `/workspace/aniket/vpd-gate-geometry` after a non-mutating
inspection of the upstream `goodfire-ai/param-decomp` repo (shallow-cloned
into `external/param-decomp` at commit `74146b5`).

This report is the answer to "what is the shortest path from zero to a real
gate-geometry analysis on the canonical VPD run?".

---

## 1. Environment summary

| Item | Value |
| --- | --- |
| GPU | NVIDIA H200, 144 GB VRAM, idle |
| RAM | 2.0 TiB |
| Workspace disk | 138 TB free out of 210 TB |
| Repo size on disk | 7.1 MB (this repo) + 125 MB (shallow upstream) |
| Driver / CUDA | 570.211.01 / CUDA 12.8 |
| OS Python | 3.12.3 (`/usr/local/bin/python`) |
| `uv` | 0.9.0 |
| `pip` | 25.2 |
| `pd-local` / `pd-run` / `pd-postprocess` | not on PATH (upstream not installed yet) |

Env vars (resolved before sourcing `.env`):

| Variable | Status (shell) | Defined in `.env`? |
| --- | --- | --- |
| `WANDB_API_KEY` | missing | yes |
| `WANDB_ENTITY` | missing | yes |
| `WANDB_PROJECT` | missing | yes |
| `HF_TOKEN` | set | yes |
| `HUGGING_FACE_HUB_TOKEN` | missing | yes (aliases `$HF_TOKEN`) |
| `OPENROUTER_API_KEY` | missing | no |
| `ANTHROPIC_API_KEY` | missing | yes |

Action: `source scripts/runpod_activate.sh` before any run; it exports `.env`.
Upstream postprocessing autointerp needs `OPENROUTER_API_KEY` (not required
for the gate-geometry analysis itself).

---

## 2. Upstream entry points (from `external/param-decomp/pyproject.toml`)

```
pd-run             param_decomp.scripts.run_cli:cli                  (SLURM launcher)
pd-local           param_decomp.scripts.run_local:cli                (single-node)
pd-pretrain        param_decomp.pretrain.scripts.run_slurm:cli
pd-harvest         param_decomp.harvest.scripts.run_slurm_cli:cli
pd-postprocess     param_decomp.postprocess.cli:cli                  (umbrella)
pd-autointerp      param_decomp.autointerp.scripts.run_slurm_cli:cli
pd-attributions    param_decomp.dataset_attributions.scripts.run_slurm_cli:cli
pd-clustering      param_decomp.clustering.scripts.run_pipeline:cli
pd-graph-interp    param_decomp.graph_interp.scripts.run_slurm_cli:cli
```

Install:

```bash
cd external/param-decomp
make install-dev   # uses uv sync; requires Python 3.13.*
```

**Compatibility note:** upstream pins `requires-python = "==3.13.*"`. We are
on 3.12.3. `uv python install 3.13` and `uv sync` will resolve a 3.13
interpreter under `.venv/` without touching system Python. If we want to
avoid that, our new `vpd_gate_geometry/` package itself only requires
torch/numpy/scipy/matplotlib and can live on 3.12 — the constraint only
applies if we want to call into `param_decomp.*` for live gate extraction.

Test / lint targets: `make check`, `make test`, `make type`, `make format`.

Canonical-run configs:

- Pile 4-layer (paper):
  `external/param-decomp/param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml`
- SimpleStories 2-layer (smaller fallback):
  `external/param-decomp/param_decomp/experiments/lm/ss_llama_simple_mlp-2L.yaml`
- Postprocess umbrella:
  `external/param-decomp/param_decomp/postprocess/pile.yaml`

Module patterns in the pile config (matches the paper's 38,912 atoms = 4
layers × 9,728/layer):

```
h.*.mlp.c_fc       C=3072
h.*.mlp.down_proj  C=3584
h.*.attn.q_proj    C= 512
h.*.attn.k_proj    C= 512
h.*.attn.v_proj    C=1024
h.*.attn.o_proj    C=1024
```

`sampling: continuous`, `sigmoid_type: leaky_hard`, `n_mask_samples: 1`.

---

## 3. Where causal importances live in the codebase

The gate field \(g^\ell_{b,t,c}\) is produced live by `ComponentModel`. It
is **not** persisted in W&B artifacts as a tensor; the standard route is to
recompute it on a forward pass.

Authoritative API (see `external/param-decomp/param_decomp/models/component_model.py:572`):

```python
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo

run_info = ParamDecompRunInfo.from_path("wandb:goodfire/spd/runs/s-55ea3f9b")
model = ComponentModel.from_run_info(run_info).to("cuda").eval()

# `batch` is Int[Tensor, "B T"] of input_ids
out = model(batch, cache_type="input")           # OutputWithCache
ci = model.calc_causal_importances(
    pre_weight_acts=out.cache,
    sampling=run_info.config.sampling,            # "continuous" for canonical run
    detach_inputs=True,
)
# ci.lower_leaky : dict[str, Float[Tensor, "B T C"]]  — the gate field g
# ci.upper_leaky : dict[str, Float[Tensor, "B T C"]]
# ci.pre_sigmoid : dict[str, Float[Tensor, "B T C"]]
```

The harvest pipeline uses exactly the same call chain
(`external/param-decomp/param_decomp/harvest/harvest_fn/param_decomp.py`),
storing `ci.lower_leaky` per layer as `"causal_importance"` and also
`component_activation = (V^T x) * ||U||` alongside it. Token IDs travel as
`HarvestBatch.tokens : Int[Tensor, "B T"]`.

Tensor-shape convention: gates dict is keyed by **target module path**
(e.g., `"h.0.mlp.c_fc"`); each tensor is `[B, T, C_module]`. Layer index
is the integer after `h.`; module type is the suffix
(`mlp.c_fc`, `attn.q_proj`, ...).

**What is in W&B for the canonical run** (`goodfire/spd/runs/s-55ea3f9b`):
trained model weights + run config. **Not** raw gates, **not**
`component_correlations.pt`, **not** `token_stats.pt` — those are produced
by a separate harvest invocation locally and live under
`$PARAM_DECOMP_OUT_DIR/harvest/<run_id>/h-<timestamp>/`. The canonical
harvested artifact set has not been published as a downloadable bundle
that I can see in the README/registry.

---

## 4. Artifact availability plan

Three viable paths, ordered by cost:

### Path A — live extraction from the canonical model (recommended)

1. Authenticate: `wandb login` (key already in `.env`).
2. Install upstream into a 3.13 venv inside `external/param-decomp/`.
3. From our analysis package, load `ComponentModel` and stream tokenized
   Pile batches through it, capturing `ci.lower_leaky` per module.
4. Persist gates in chunked `.pt` files (sparse-COO for the long-tail
   sparse modules, dense for the dense ones — see §6 of `docs/code.md`).

Cost: a few GB of disk per 10k sequences × 4 layers × 6 modules. H200 will
chew through Pile batches at ~100 ms/batch. The model itself is ~67 M
params; fits trivially in 144 GB VRAM.

### Path B — run the upstream harvest pipeline first

1. Install upstream.
2. Run `pd-harvest` on the canonical run for ~1 k–10 k batches.
3. Use the produced `component_correlations.pt` + `token_stats.pt` directly
   for same-position kernel + token residualization (no lagged kernels —
   harvest's correlations are τ=0 only).

Cost: similar to Path A but routed through upstream's pipeline. Less
flexible (no lagged correlations from this artifact set alone), but
gives us pre-computed same-position correlations and per-token
distributions for free.

### Path C — SimpleStories fallback

`ss_llama_simple_mlp-2L` is a 2-layer decomposition over
`SimpleStories/SimpleStories`. Smaller model, smaller `C`, smaller dataset.
Use if Path A surfaces issues (e.g., Pile streaming flakes, W&B artifact
gated). Same API; just swap the run path.

### Path D — bypass repo entirely with synthetic data

Implemented as the `mock` backend in `vpd_gate_geometry/extract_gates.py`.
It generates structured-noise gates with planted clusters and lagged
pairs so the analysis pipeline can be validated end-to-end without
hitting W&B or installing 1 GB of upstream deps. Always available.

### Decision

- Implement `mock` first (Phase 4); smoke-test the analysis pipeline.
- Provide a `repo` backend with the exact `ComponentModel` call chain
  stubbed and a clear `RuntimeError` if `param_decomp` is unimportable.
- The first real run targets the **SimpleStories** decomposition because
  its smaller `C` (~2 × 1,856 = 3,712 components) makes Day-2 iteration
  faster; promote to the Pile canonical run once the spectral/residual
  plots look sensible at the smaller scale.

---

## 5. Implementation plan

### Files to create (this commit)

```
pyproject.toml                                  # uv-managed; torch+numpy+scipy+matplotlib
vpd_gate_geometry/__init__.py
vpd_gate_geometry/config.py                     # AnalysisConfig dataclass + CLI parsing
vpd_gate_geometry/extract_gates.py              # GateBatch + mock|repo|artifact backends
vpd_gate_geometry/gate_matrix.py                # ComponentKey + GateMatrix + flatten
vpd_gate_geometry/spectral.py                   # alive filter, covariance, SVD, kernels
vpd_gate_geometry/residualize.py                # token-identity baseline + R²
vpd_gate_geometry/lagged.py                     # K^(τ) with chunking + top-k restriction
vpd_gate_geometry/plotting.py                   # Goodfire-style rcParams + 8 plot fns
vpd_gate_geometry/run_analysis.py               # CLI entry point
vpd_gate_geometry/README.md
docs/external_repos.md                          # SPD/CLT/BAE links + status
outputs/gate_geometry/.gitkeep
```

### Files NOT to modify

- Anything under `external/param-decomp/` (shallow clone; treat as upstream).
- The existing `scripts/runpod_*.sh` (bootstrap files).
- `docs/theory.md`, `docs/code.md`, `docs/writing.md` (planning docs).

### Memory-safety constraints baked into the implementation

- `spectral.build_cosine_kernel` accepts `max_components` and works on
  module subsets; never materializes a dense 38k × 38k kernel by default.
- `lagged.lagged_kernel` restricts to top-`k` components by mean activity
  before forming the cross-product, and accumulates in fp32 from bf16/fp16
  chunks.
- `residualize.token_baseline` uses shrinkage and a `min_count` cap on
  per-token estimates; large vocab is handled with `index_add_` not dense
  one-hot.

### First smoke test

```bash
# from /workspace/aniket/vpd-gate-geometry
source scripts/runpod_activate.sh
uv sync
uv run python -m vpd_gate_geometry.run_analysis \
    --backend mock \
    --output-dir outputs/gate_geometry/mock_smoke \
    --max-components 128 \
    --max-lag 4
```

Expected outputs:

- `outputs/gate_geometry/mock_smoke/summary.json` with shapes / alive
  counts / spectrum / lag-profile stats.
- 8 PNGs in `outputs/gate_geometry/mock_smoke/plots/`.

### First real-data attempt (after upstream install)

```bash
# install upstream first
cd external/param-decomp && uv python install 3.13 && uv sync && cd ../..
# load + run
uv run python -m vpd_gate_geometry.run_analysis \
    --backend repo \
    --run-path wandb:goodfire/spd/runs/s-55ea3f9b \
    --output-dir outputs/gate_geometry/canonical_vpd \
    --n-batches 32 --batch-size 8 --seq-len 256 \
    --max-components 2048 --max-lag 8
```

This is expected to fail with a precise error until upstream is installed
and `WANDB_API_KEY` is exported in the shell — that error is the
deliverable of this scaffold, not a regression.

### Risk register

- **Python 3.13 mismatch.** Mitigation: keep our package on a flat
  `pyproject.toml` that supports `>=3.11`; only the *repo backend* import
  requires the upstream venv.
- **Pile dataset streaming flakes.** Mitigation: SimpleStories Path C.
- **Disk fills on long harvest.** Mitigation: chunked sparse storage;
  workspace is 138 TB free, low risk.
- **W&B run gated.** Mitigation: check via `ParamDecompRunInfo.from_path`
  before extracting; surface a clean error to the user.
- **Component-count math.** The paper reports 38,912 atoms across 4
  layers × 6 modules. Each module's `C` is fixed in the config. Our
  `flatten_gate_dict` keys are `(module_name, local_idx)` so we never
  rely on a global ordering.

---

## 6. What is *not* in scope for this commit

- Retraining VPD (paper run or otherwise).
- Cloning `nn_decompositions` (CLT/PLT comparison) — referenced only.
- BAE / bilinear-MLP implementation — referenced only.
- Causal validation of clusters (ablation experiments).
- Autointerp labelling of clusters.

These are deferred to later commits if/when the spectral baseline shows
something worth investigating.
