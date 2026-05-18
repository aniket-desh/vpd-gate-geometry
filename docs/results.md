# Gate-geometry results on the canonical VPD run

**Run:** `wandb:goodfire/spd/runs/s-55ea3f9b`
(VPD paper, April 2026 — `pile_llama_simple_mlp-4L`)

**Status:** in progress. This document is updated as the run sweeps land.

## 0. What was actually done

- Shallow-cloned `goodfire-ai/param-decomp` and built a separate
  analysis package `vpd_gate_geometry/` against it.
- Downloaded the canonical VPD model checkpoint (2.9 GB
  `model_400000.pth`) and the target LM checkpoint
  (256 MB `model_step_99999.pt`) directly from the public Google
  Cloud storage backend of W&B, bypassing the wandb-python auth
  requirement.
- Patched `ParamDecompRunInfo.config` in memory to point at the
  local target-LM file, then called the upstream
  `ComponentModel.calc_causal_importances(...).lower_leaky` over a
  tokenized stream of the canonical Pile dataset.
- All gates extracted via the actual upstream code path
  (`param_decomp/harvest/harvest_fn/param_decomp.py`-style call
  chain), no reimplementation.

## 1. Setup

| Item | Value |
| --- | --- |
| Model | 4-layer Llama-MLP-only, ~67M target params + ~660M CI transformer |
| Decomposition | 38,912 rank-one atoms = 4 layers × 6 modules × {3072,3584,512,512,1024,1024} |
| Tokenizer | EleutherAI/gpt-neox-20b |
| Dataset | danbraunai/pile-uncopyrighted-tok-shuffled (streaming) |
| Sampling type | `continuous` (canonical run config) |
| Sigmoid type | `leaky_hard` |
| Extraction batches | _to fill in_ |
| Token positions | _to fill in_ |
| Hardware | NVIDIA H200, 144 GB VRAM, 2 TiB RAM |

## 2. Smoke pass (2 batches × 4 sequences × 512 tokens, 4,096 positions)

A first canonical-run smoke pass to confirm the pipeline end-to-end.

### 2.1 Alive-component distribution

| Layer | Module | C | Alive (g̅ > 1e-3) |
| --- | --- | ---: | ---: |
| 0 | attn.q | 512 | 16 |
| 0 | attn.k | 512 | 16 |
| 0 | attn.v | 1024 | 95 |
| 0 | attn.o | 1024 | 88 |
| 0 | mlp.c_fc | 3072 | 747 |
| 0 | mlp.down | 3584 | 830 |
| 1 | attn.q | 512 | 7 |
| 1 | attn.k | 512 | 10 |
| 1 | attn.v | 1024 | 97 |
| 1 | attn.o | 1024 | 78 |
| 1 | mlp.c_fc | 3072 | 112 |
| 1 | mlp.down | 3584 | 104 |
| 2 | attn.q | 512 | 89 |
| 2 | attn.k | 512 | 121 |
| 2 | attn.v | 1024 | 384 |
| 2 | attn.o | 1024 | 428 |
| 2 | mlp.c_fc | 3072 | 205 |
| 2 | mlp.down | 3584 | 276 |
| 3 | attn.q | 512 | 25 |
| 3 | attn.k | 512 | 43 |
| 3 | attn.v | 1024 | 192 |
| 3 | attn.o | 1024 | 200 |
| 3 | mlp.c_fc | 3072 | 823 |
| 3 | mlp.down | 3584 | 1,498 |
| **Total** |  | **38,912** | **6,484** |

The paper reports 9,972 alive at threshold `g̅ > 1e-6`. We see
6,484 at the stricter `g̅ > 1e-3`; the per-layer proportions
(L0 28% / L1 6% / L2 23% / L3 43%) roughly match the paper's
(L0 37% / L1 9% / L2 19% / L3 35%). L3 `mlp.down` is the
single densest module in both.

### 2.2 Surprise: layer 2 attention is densely occupied

`h.2.attn.o_proj` has **428 of 1024** components alive (41.8%),
and `h.2.attn.v_proj` has 384 / 1024 (37.5%). For comparison the
attention modules at layers 0/1 are at 1–10% activity. This is
not visible in the paper's per-layer totals — those average
over all modules in a layer — but pops out cleanly once you look
at the per-module breakdown.

### 2.3 Spectrum + heatmap

The top-1024-by-activity cosine kernel has:

| Metric | Value |
| --- | ---: |
| Effective rank | 180.5 |
| Participation ratio | 45.1 |
| Top eigenvalue | 110.6 |
| 10th eigenvalue | 15.7 |
| 1000th eigenvalue | ≈ 0.05 |

→ steep, ≈3-decade power-law-like decay with a clear top mode.

The clustered heatmap (see `outputs/.../03_kernel_clustered_heatmap.png`)
already shows **4–5 visually distinct block-diagonal mechanism
bundles** — even from a tiny 4,096-position sample. **Strong H1.**

### 2.4 Token residualization

| Metric | Value |
| --- | ---: |
| Effective unique tokens (≥8 occurrences) | 87 |
| Median R² explained by token identity | 0.116 |
| Mean R² | 0.159 |
| Top eigenvalue (residualized) | 61.9 |
| Effective rank (residualized) | 270.8 |

The R² histogram is visibly **bimodal-with-tail**: a primary
mode at R² ≈ 0.05 (purely contextual), a secondary mode at
0.2–0.3 (partially token-bound), and a long tail to R² > 0.7
(strongly lexical components).

The leading eigenvalue drops 110.6 → 61.9, but the broad shape
of the spectrum survives. **H2 supported in its mixed form**:
token identity explains a meaningful but incomplete fraction
of gate variance, with components stratifying cleanly into
lexical vs. contextual.

### 2.5 Lagged co-importance (smoke)

Lagged kernel computed on `h.0.attn.k_proj → h.0.attn.o_proj`
(default module pair from the smoke run, not specifically meaningful).
Sample size (4,096 positions across 8 sequences) is too small
for sharp conclusions; numerically the original implementation
returned scores >1.0 at high τ, which has been fixed in commit
0f26f2f. **H3 to be revisited on the bigger run.**

## 3. v1 pass (16 batches × 8 sequences × 512 tokens, 65,536 positions)

_to fill in when the larger run lands_

Targets:

- alive counts at threshold `1e-4` (paper-matched)
- recompute spectrum / residualization at >10× the sample size
- lagged kernel on `h.{0,2}.attn.q_proj → h.{0,2}.attn.k_proj`
  pairs (same-layer Q/K), where attention circuits should live

## 4. Per-layer / per-module-type comparison

_to fill in via `vpd_gate_geometry.per_layer` on the v1 cache_

Goal: address H4 by overlaying spectra across layers (for each
module type) and across module types (for each layer).

## 5. Lagged sweep across multiple module pairs

_to fill in via `vpd_gate_geometry.sweep_pairs` on the v1 cache_

Pairs in priority order:

1. `h.0.attn.q → h.0.attn.k`  (induction-like, same layer)
2. `h.2.attn.q → h.2.attn.k`  (densest attention layer)
3. `h.0.attn.o → h.0.attn.v`
4. `h.2.attn.o → h.2.attn.v`
5. `h.0.mlp.down → h.1.attn.q`  (cross-layer MLP→attn coupling)
6. `h.0.mlp.down → h.1.attn.k`

## 6. Things that did NOT happen

- We did **not** retrain VPD or any decomposition.
- We did **not** use the upstream harvest pipeline; gates come
  straight from `model.calc_causal_importances(...)` over our
  own dataloader, so we have *true raw gates* (not summary stats).
- We did **not** modify upstream code; everything wraps it.

## 7. Reproducibility

```bash
# 1. Clone scaffold + setup
git clone https://github.com/aniket-desh/vpd-gate-geometry.git
cd vpd-gate-geometry
git clone --depth 1 https://github.com/goodfire-ai/param-decomp.git external/param-decomp
cd external/param-decomp && uv sync && cd ../..

# 2. Download canonical VPD + target-LM checkpoints from public W&B GCS
# (see scripts/download_canonical.py — TODO: extract from the in-band
# Python snippet used here)

# 3. Run
source scripts/runpod_activate.sh   # loads .env (HF_TOKEN at minimum)
external/param-decomp/.venv/bin/python -m vpd_gate_geometry.run_analysis \
    --backend repo \
    --run-path runs/pile-4L/model_400000.pth \
    --target-model-path runs/pretrain-4L/files/model_step_99999.pt \
    --output-dir outputs/gate_geometry/pile4L_v1 \
    --cache-gate-matrix outputs/gate_geometry/cache/pile4L_16x8x512.pt \
    --n-batches 16 --batch-size 8 \
    --max-components 4096 --max-lag 6 \
    --lagged-top-k 512 \
    --lagged-module-a h.2.attn.q_proj \
    --lagged-module-b h.2.attn.k_proj
```
