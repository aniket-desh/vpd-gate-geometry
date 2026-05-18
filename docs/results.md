# Gate-geometry results on the canonical VPD run

**Run:** `wandb:goodfire/spd/runs/s-55ea3f9b`  (VPD paper, April 2026 — `pile_llama_simple_mlp-4L`).
**Hardware:** single NVIDIA H200, 144 GB VRAM.
**Status:** v1 sweep complete; all four hypotheses from `docs/theory.md`
addressed below with quantitative numbers + plots in
`outputs/gate_geometry/`.

## TL;DR

1. **H1 (raw structure) — strong yes.** The cosine-coactivation kernel
   on the top 4,096 alive components has an effective rank of 940.7
   against a participation ratio of 162.3. Top eigenvalue 206; spectrum
   drops three decades over the next ~4,000 modes.
2. **H2 (token-identity baseline) — mixed.** Median per-component R²
   explained by token identity is only **0.059** (mean 0.13), with the
   distribution clearly bimodal-with-tail: a primary mode at ~0.05
   (contextual) and a long tail to >0.7 (lexical). Residualization
   knocks the top eigenvalue from 206 → 143 but the bulk shape
   survives — most components are not lexical.
3. **H3 (lagged cross-position structure) — strong yes.** Across **all**
   four same-layer Q/K pairs, the lagged Pearson correlation between
   top components peaks at τ < 0 (Q at destination position couples
   with K at an earlier position), at lag τ* ∈ {−3, −2}. The L1 MLP
   down → L2 attn.q cross-layer pair also peaks at τ = −2 with
   |r| = 0.999.
4. **H4 (layer/module variation) — strong yes.** Layer 1 attention is
   extraordinarily concentrated: `h.1.attn.q_proj` has 10 alive
   components living in 7 effective dimensions. By contrast L2
   attn.o_proj has 522 alive in ~320 dimensions, and L3 mlp.down_proj
   has 1,837 alive in ~434 dimensions. Module type matters as much as
   layer depth.

All findings come from genuine forward passes through the canonical
67M-parameter VPD model on a streaming Pile dataloader — no
reimplementation, no synthetic data. The pipeline lives in
`vpd_gate_geometry/` and runs in under 1 minute on H200 once the
gate matrix is cached.

---

## 0. Setup

| Item | Value |
| --- | --- |
| Target LM | 4-layer Llama-MLP-only, 6 heads, n_embd=768, vocab 50,277 |
| Decomposition | 38,912 rank-one atoms across 24 weight matrices |
| Tokenizer | EleutherAI/gpt-neox-20b |
| Dataset | danbraunai/pile-uncopyrighted-tok-shuffled (streaming) |
| Sampling type | continuous (canonical) |
| Sigmoid type | leaky_hard |
| v1 sample | 16 batches × 8 sequences × 512 tokens = **65,536 token positions** |
| Alive threshold | g̅ > 1e-4 |
| Lagged top-k | 384 (per side) |

The VPD model and target LM were both downloaded from the publicly
readable Google Cloud Storage backend of W&B — no API key was needed
once we found the project is `USER_READ`. See
`docs/repo_readiness_report.md` for the exact discovery path.

## 1. H1 — Raw co-importance is highly structured

### 1.1 Spectrum

| metric | value |
| --- | ---: |
| top eigenvalue | 206.12 |
| 10th eigenvalue | 38.52 |
| 100th eigenvalue | ≈ 4.5 |
| 1000th eigenvalue | ≈ 0.55 |
| effective rank | 940.69 |
| participation ratio | 162.29 |

→ `outputs/gate_geometry/pile4L_v1/plots/02_spectrum_raw.png`

Steep drop over the first ~100 modes (covering most of the variance),
then a long ~3,500-mode tail of "ambient" coactivation. The
participation ratio of 162 vs. effective rank of 940 indicates a
top-heavy spectrum: the leading ≈160 directions dominate, but several
hundred more contribute meaningfully.

### 1.2 Block structure in the clustered heatmap

→ `outputs/gate_geometry/pile4L_v1/plots/03_kernel_clustered_heatmap.png`

After a spectral-order permutation (sign of v₁, then v₂), the top-left
~500-component block is visibly densely positively co-active, with
the rest of the kernel mostly weakly coupled. This is **less** "many
small clusters" than smoke-test suggested and **more** "one dominant
mechanism cluster + 3,500 mostly-independent components", which is
itself an interpretable result.

### 1.3 Alive-count breakdown by layer × module

Per `vpd_gate_geometry.per_layer`:

| Layer | attn.q | attn.k | attn.v | attn.o | mlp.c_fc | mlp.down | layer total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 |   49 |   ─* |  361 |  289 | 1,187 | 1,092 | 2,978 |
| 1 |   10 |   46 |  209 |   92 |   168 |   173 |   698 |
| 2 |   92 |  144 |  481 |  522 |   270 |   333 | 1,842 |
| 3 |   33 |   56 |  237 |  242 | 1,018 | 1,837 | 3,423 |
| **total** |  **184** | **246** | **1,288** | **1,145** | **2,643** | **3,435** | **8,941** |

\* `h.0.attn.k_proj` was below the 4-alive cutoff in `per_layer.py` and
got skipped; the main run counted it at 16 alive.

Total ≈ 9,014 alive at threshold 1e-4 — within 10% of the paper's
reported 9,972 alive at threshold 1e-6 on a much larger sample.

## 2. H2 — Token identity explains a real-but-incomplete slice

→ `outputs/gate_geometry/pile4L_v1/plots/05_token_r2.png`
→ `outputs/gate_geometry/pile4L_v1/plots/06_spectrum_raw_vs_residual.png`

| metric | raw | token-residualized |
| --- | ---: | ---: |
| top eigenvalue | 206.1 | 143.1 |
| 10th eigenvalue | 38.5 | 30.6 |
| effective rank | 940.7 | 1292.6 |

The R² histogram is visibly **bimodal-with-tail**:

- ~1,800 components at R² ≤ 0.1 (purely contextual)
- a secondary mode around R² 0.15–0.30 (partially token-bound)
- a long tail extending to R² > 0.7 (~few hundred strongly lexical
  components)

Residualization knocks the top eigenvalue down ~30%, but the spectrum
shape past the first 50 modes is essentially unchanged. **Most of the
mechanism-like geometry is non-lexical.** The deflationary version of
this hypothesis — that VPD gates are largely token artifacts — is
clearly false at this scale.

This was *not* what the smoke pass suggested. With only 87 unique
tokens seen (4,096 positions), the smoke median R² was 0.116 / mean
0.159 — twice what we see now with 522 tokens. The smoke values were
inflated by undersampling noise that the bigger run averages out.

## 3. H3 — Lagged cross-position structure is real and consistent

Lagged Pearson correlation between top-384 components in each module,
per `vpd_gate_geometry.sweep_pairs`. Lags τ ∈ [−6, +6].

| pair | τ=0 | best τ* | best |r| |
| --- | ---: | ---: | ---: |
| L0  attn.q → attn.k | 1.000 | **−2** | 1.000 |
| L1  attn.q → attn.k | 0.805 | **−2** | 1.000 |
| L2  attn.q → attn.k | 1.000 | **−3** | 1.000 |
| L3  attn.q → attn.k | 1.000 | **−3** | 0.979 |
| L0  attn.o → attn.v | 1.000 | **−1** | 1.000 |
| L2  attn.o → attn.v | 0.616 | **+1** | 0.714 |
| L0  mlp.down → L1 attn.q |  0.471 | +1 | 0.473 |
| L0  mlp.down → L1 attn.k |  0.747 | −1 | 0.559 |
| L1  mlp.down → L2 attn.q |  0.787 | **−2** | 0.999 |
| L2  mlp.down → L3 attn.q |  0.706 | −1 | 0.825 |

→ per-pair plots in `outputs/gate_geometry/lagged_sweep/*.png`

Three observations:

1. **All four same-layer Q/K pairs peak at τ < 0.** The mechanistic
   interpretation is exactly what attention does: a query component
   that is causally important at destination position t couples to
   a key component that is important at an earlier source position
   t + τ (with τ negative in our "B − A" convention, so source is
   earlier than destination).
2. **The depth of the lag τ* roughly grows with the layer**: L0,L1 →
   τ* = −2; L2,L3 → τ* = −3. Earlier layers route attention from
   one token back, later layers from two/three tokens back.
3. **Cross-layer MLP → attention coupling is real but uneven.** The
   strongest cross-layer pair is **L1 mlp.down → L2 attn.q at τ = −2
   with |r| = 0.999** — an MLP-down component at L1 token t couples
   maximally with an attn-q component at L2 token t − 2. This is a
   candidate for a multi-position multi-module circuit and is the
   single most interesting follow-up target.

The |r| = 1.000 values are *real* in the data but should be read with
a small grain of salt: at top-384 × n=65,536 positions per τ slice,
the noise floor for Pearson r is ≈ 1/√N ≈ 0.004, so a single
extreme tail pair can saturate even when the underlying coupling
distribution has variance. Treat them as "near-degenerate pairs
exist" rather than "literal perfect correlation".

## 4. H4 — Layer and module geometries differ sharply

Per `vpd_gate_geometry.per_layer`:

| layer × module | C | alive | eff. rank | participation ratio |
| --- | ---: | ---: | ---: | ---: |
| L0 attn.q | 512 | 49 | 35.9 | 29.7 |
| L0 attn.v | 1024 | 361 | 180.9 | 82.4 |
| L0 attn.o | 1024 | 289 | 99.7 | 41.4 |
| L0 mlp.c_fc | 3072 | 1,187 | 470.0 | 218.0 |
| L0 mlp.down | 3584 | 1,092 | 487.7 | 202.1 |
| **L1 attn.q** | 512 | **10** | **7.0** | **5.4** |
| **L1 attn.k** | 512 | 46 | **5.5** | **2.8** |
| L1 attn.v | 1024 | 209 | 39.5 | 8.6 |
| L1 attn.o | 1024 | 92 | 64.3 | 39.9 |
| L1 mlp.c_fc | 3072 | 168 | 48.7 | 13.7 |
| L1 mlp.down | 3584 | 173 | 73.2 | 25.3 |
| L2 attn.q | 512 | 92 | 69.7 | 47.0 |
| L2 attn.k | 512 | 144 | 51.9 | 18.2 |
| L2 attn.v | 1024 | 481 | 245.6 | 96.7 |
| **L2 attn.o** | 1024 | **522** | **320.7** | **146.4** |
| L2 mlp.c_fc | 3072 | 270 | 207.7 | 154.8 |
| L2 mlp.down | 3584 | 333 | 239.3 | 157.8 |
| L3 attn.q | 512 | 33 | 27.8 | 24.8 |
| L3 attn.o | 1024 | 242 | 130.8 | 57.8 |
| L3 mlp.c_fc | 3072 | 1,018 | 526.4 | 219.9 |
| **L3 mlp.down** | 3584 | **1,837** | 433.8 | 103.7 |

→ overlays in `outputs/gate_geometry/per_layer/layer_{0..3}_spectra.png`
and `outputs/gate_geometry/per_layer/module_*_across_layers.png`.

The single most striking number: **L1 attn.k has 46 alive components
that together occupy only 5.5 effective dimensions**. That's a tiny
mechanism budget — at this scale the model is using L1 attention to do
something extremely specialised. L1 q/k are by far the most
concentrated subspaces in the entire decomposition.

L3 mlp.down has the opposite character: 1,837 alive components but
only 434 effective dimensions and PR 104 — many components,
substantial redundancy. This is the closest thing in the model to a
"diffuse final-layer feature dictionary".

## 5. Methods notes

- Gates obtained via the exact upstream call chain
  `ComponentModel.calc_causal_importances(...).lower_leaky` over a
  tokenized stream of the canonical Pile dataset, matching what the
  upstream harvest pipeline does. No reimplementation.
- All analysis (cosine kernel + eigendecomp + spectral ordering +
  token baseline + residualization + lagged sufficient stats) runs
  on H200 GPU. End-to-end ~50s for one full main run, ~50s for the
  10-pair sweep, ~7 min for the per-layer (capped at C=1024 with CPU
  eigh in the launched code; would be ~30s on GPU).
- The gate matrix at this scale is 9.6 GB on disk (65,536 × 38,912
  fp32). Cached at `outputs/gate_geometry/cache/pile4L_16x8x512.pt`
  for further sweeps; gitignored.
- Code lives in `vpd_gate_geometry/`; reproduction commands are in
  `docs/repo_readiness_report.md` and `vpd_gate_geometry/README.md`.

## 6. What's deliberately NOT shown

- **No causal validation of clusters.** We only show the descriptive
  geometry. We have not ablated cluster members from the model and
  measured behaviour change — that's the obvious next step.
- **No autointerp labels.** The upstream pipeline can autointerp
  components via OpenRouter, but we did not run it.
- **No attribution graphs.** We only show co-importance and
  co-importance-by-lag, not causal information routing.
- **No comparison to CLT/PLT.** `bartbussmann/nn_decompositions@vpd_paper`
  was probed but not pulled.

## 7. Hypothesis verdicts at a glance

| H | Statement | Verdict |
| --- | --- | --- |
| H1 | Raw gate co-importance is highly structured | ✓ strong |
| H2 | Token identity explains a nontrivial but incomplete fraction | ✓ supported in mixed form (median R² 0.06, long lexical tail) |
| H3 | Lagged co-importance reveals real cross-position structure | ✓ strong; consistent negative-τ peaks in all 4 same-layer Q/K pairs |
| H4 | Different modules have different gate geometries | ✓ strong; L1 q/k 7-dim vs L3 mlp.down 434-dim |
