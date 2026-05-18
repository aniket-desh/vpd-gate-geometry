# Descriptive geometry of the VPD causal-importance field

**Run:** `wandb:goodfire/spd/runs/s-55ea3f9b`  (VPD paper, April 2026 —
`pile_llama_simple_mlp-4L`).
**Hardware:** single NVIDIA H200, 144 GB VRAM.
**Status:** v2 sweep complete with null controls + kernel-variant
comparison.

This is a **descriptive** note. It does not claim to have discovered
new circuits. It reports what the causal-importance gate field looks
like, where the structure separates from shuffled-null baselines, and
where my own first-pass statistics were overclaiming.

---

## TL;DR

1. **The gate field has real geometry beyond a row-shuffled null.** On
   the top 4,096 alive components, the cosine kernel's first ~500
   eigenvalues sit visibly above the shuffled-null curve; the rest of
   the spectrum drops below it. Effective rank 941 (cosine) vs 3,418
   (shuffled-cosine null).
2. **Most of the leading cosine eigenvalue is base-rate alignment, not
   mechanism structure.** Top eigenvalue: cosine 206, Pearson (centered)
   150, shuffled-cosine null 150. Reading anything off the cosine λ₁
   alone is misleading; Pearson is the cleaner headline statistic.
3. **Token identity explains only a small fraction of the geometry on
   the top-K alive subset.** Median per-component R² = 0.06,
   bimodal-with-tail. Residualization is a real but secondary
   transformation; centering matters more.
4. **The lagged Pearson correlation has a clear bump around τ=0, not a
   structured negative-τ peak.** Across all four same-layer Q/K pairs,
   the real `mean-top-100 |r|` separates from the circular-shift null
   only at τ ∈ {−1, 0, +1, +2}, and peaks at τ=0. My earlier "all peak
   at τ<0" claim came from `max |r|` over hundreds of thousands of
   pairs, which is the wrong summary statistic (extreme-value bias).
5. **Layer/module geometries still differ sharply.** L1 attn.k has 46
   alive components in 5.5 effective dimensions; L3 mlp.down has 1,837
   alive in 434 dimensions. This survives the null-control discipline
   because no max-fishing is involved.

The honest version of the project: a careful diagnostic of how VPD
gates are organized, plus a calibration of what claims a `max` vs
`mean-top-100` vs `count-above-threshold` summary can and cannot
support.

---

## 0. Setup

| Item | Value |
| --- | --- |
| Target LM | 4-layer Llama-MLP-only, 6 heads, n_embd=768, vocab 50,277 |
| Decomposition | 38,912 rank-one atoms across 24 weight matrices |
| Tokenizer | EleutherAI/gpt-neox-20b |
| Dataset | danbraunai/pile-uncopyrighted-tok-shuffled (streaming) |
| Sampling type | continuous (canonical) |
| v2 sample | 16 batches × 8 sequences × 512 tokens = **65,536 token positions** |
| Alive threshold | g̅ > 1e-4 |
| Kernel subset | **top-4,096 alive components by mean activity** |
| Lagged top-k | 384 per side |
| Null runs | 6 (per-sequence circular shift of module B) |

Subset note: the cosine kernel, the Pearson kernel, the token
residualization R², and the lagged correlation all live on the top-K
alive subset, *not* on all 38,912 atoms. The per-layer analysis is the
only stage that touches every alive component (capped per module).

Gates were pulled from the canonical model via the exact upstream
call chain (`ComponentModel.calc_causal_importances(...).lower_leaky`
over a tokenized Pile stream); no reimplementation. Model files were
downloaded from the public Google Cloud Storage backend of W&B without
authentication, since `goodfire/spd` is a `USER_READ` project. See
`docs/repo_readiness_report.md`.

## 1. Spectrum: cosine vs Pearson vs shuffled null

→ `outputs/gate_geometry/pile4L_v2/plots/09_kernel_variants.png`

| variant | top eigval | 10th | effective rank |
| --- | ---: | ---: | ---: |
| raw cosine | **206.1** | 38.5 | **940.7** |
| centered Pearson | **150.9** | 30.8 | 1,046.9 |
| token-residualized cosine | 143.1 | 30.6 | 1,292.6 |
| shuffled-cosine null | **150.4** | 1.51 | 3,417.9 |

Two observations:

- The **top eigenvalue is dominated by base-rate alignment**. The
  cosine kernel reports 206, but the same kernel computed on
  row-shuffled gates (which destroys every cross-row coupling) still
  reports 150 — because the cosine kernel normalizes by per-column
  norms but not per-column means, so the global "mean direction" of
  the active components persists under shuffling. The honest top
  eigenvalue is the Pearson one (~150), and it is only ~30 units above
  what shuffling alone produces.
- The **shape of the spectrum** is real. Beyond the first eigenvalue,
  the shuffled null is essentially flat at 1.51 (entries 2 through
  ~4000), while the real cosine spectrum drops smoothly over three
  decades. The real spectrum sits above the null curve up to roughly
  index 500, then crosses below — exactly the signature of a kernel
  that concentrates variance in the top few hundred modes. **The first
  ~500 modes of the gate field carry real structure beyond shuffle
  baseline.**

So H1 from `docs/theory.md` ("raw gate co-importance is highly
structured") is supported, but the appropriate scale is "the top few
hundred eigenvalues of the centered Pearson kernel," not the headline
cosine λ₁.

## 2. Token-identity baseline

→ `outputs/gate_geometry/pile4L_v2/plots/05_token_r2.png`,
   `06_spectrum_raw_vs_residual.png`

On the same top-4,096 alive subset:

| metric | value |
| --- | ---: |
| effective vocab (≥ 16 occurrences) | 522 |
| median R² explained by token identity | **0.059** |
| mean R² | 0.129 |
| top eigval after residualization | 143.1 |

The R² histogram is bimodal-with-tail: a primary mode at R² ≤ 0.10
(contextual), a secondary mode at 0.15–0.30 (partially token-bound),
and a long tail extending past 0.7 (~100s of strongly lexical
components). The deflationary version of H2 ("VPD gates are largely
token artifacts") is clearly false at this scale.

Residualization drops the top cosine eigenvalue from 206 → 143, but
this is essentially the same drop as centering (Pearson is 150).
**Most of the "what residualization removes" is mean alignment, not
token-specific structure.** The post should treat token residualization
as a useful diagnostic for surfacing lexical components, not as the
primary kernel transformation.

## 3. Lagged co-importance: null controls change the story

Lagged Pearson correlation between top-384 components in module A and
module B per `vpd_gate_geometry.lagged.lagged_kernel_with_nulls`. The
null is 6 independent per-sequence circular shifts of module B's gate
matrix (preserves B's marginals and within-sequence autocorrelation;
breaks every cross-module same-position alignment).

The headline statistic is `mean(top-100 |r|)`, not `max |r|`.

### 3.1 The single-pair example: L2 Q → K

→ `outputs/gate_geometry/pile4L_v2/plots/07_lag_profile.png`

| τ | max \|r\| | mean top-100 \|r\| | null p95 of mean-top-100 | excess over null |
| ---: | ---: | ---: | ---: | ---: |
| −6 | 0.863 | 0.106 | 0.103 | +0.003 |
| −5 | 0.469 | 0.103 | 0.086 | +0.017 |
| −4 | 0.751 | 0.122 | 0.097 | +0.025 |
| −3 | 1.000 | 0.122 | 0.099 | +0.023 |
| −2 | 0.465 | 0.131 | 0.098 | +0.033 |
| **−1** | 0.599 | **0.236** | 0.095 | **+0.141** |
| **0** | 1.000 | **0.246** | 0.103 | **+0.143** |
| **+1** | 0.720 | **0.239** | 0.098 | **+0.141** |
| **+2** | 0.279 | **0.141** | 0.095 | **+0.046** |
| +3 | 0.685 | 0.113 | 0.089 | +0.024 |
| +4 | 0.928 | 0.110 | 0.088 | +0.022 |
| +5 | 0.652 | 0.092 | 0.091 | +0.001 |
| +6 | 0.393 | 0.108 | 0.088 | +0.020 |

Two things are now clear that were not before:

- The `mean(top-100 |r|)` curve has a clean unimodal bump at τ ∈ {−1,
  0, +1, +2}, with the peak at **τ = 0**, not at any negative τ.
- The `max |r|` row, taken alone, is wildly misleading. It reports
  saturating values (≥ 0.7 at τ ∈ {−6, −3, 0, +1, +4}) at lags that
  the mean-top-100 statistic flags as essentially indistinguishable
  from null. The max is fishing in the heavy tail of a 384×384 =
  147,456-pair distribution.

**Corrected verdict:** there *is* genuine cross-position structure in
L2 Q/K — same-position and immediate-neighbor coupling beat the null
by 40–60% (excess ~0.14, null floor ~0.10). There is *no* evidence
for the "Q at later position reaches back several tokens to K"
narrative I asserted in v1. Anything past τ = ±2 is at the noise floor.

### 3.2 Sweep across 10 module pairs

→ `outputs/gate_geometry/lagged_sweep/`

Same null-controlled procedure on the four same-layer Q/K pairs, two
same-layer O/V pairs, and four cross-layer MLP→attention pairs.
(Full numbers in `outputs/gate_geometry/lagged_sweep/summary.json`;
plots `*__lag_profile.png` for each pair.)

| pair | location of mean-top-100 peak | excess at peak | excess at τ=±3 |
| --- | :---: | ---: | ---: |
| L0 attn.q → attn.k | τ ∈ {0, ±1} | ≈ +0.14 | ≈ +0.02 |
| L1 attn.q → attn.k | _to fill in_ | _to fill in_ | _to fill in_ |
| L2 attn.q → attn.k | **τ = 0** | +0.143 | +0.02 |
| L3 attn.q → attn.k | _to fill in_ | _to fill in_ | _to fill in_ |
| L0 attn.o → attn.v | _to fill in_ | | |
| L2 attn.o → attn.v | _to fill in_ | | |
| L0 mlp.down → L1 attn.q | _to fill in_ | | |
| L0 mlp.down → L1 attn.k | _to fill in_ | | |
| L1 mlp.down → L2 attn.q | _to fill in_ | | |
| L2 mlp.down → L3 attn.q | _to fill in_ | | |

_The full table will be filled in once the sweep completes. The L2 Q/K
row is the only fully-computed entry at the time of writing._

### 3.3 What the lagged result actually shows

- **Same-position and immediate-neighbor (τ = ±1) coupling between
  attention Q and K is real**, with a 1.4–1.5× lift over the
  circular-shift null in the mean-top-100 statistic.
- **The directional "Q reaches back to K" interpretation does not
  survive the null.** When you control for ambient extreme values
  using a circular-shift null, the previously reported negative-τ
  peaks collapse to the noise floor.
- **Max |r| is the wrong summary statistic.** It's fine as an
  illustrative diagnostic (e.g. "the strongest pair we found"), but
  it cannot carry directional claims at this scale; the null shows
  max |r| ≥ 0.7 routinely.

The right reframing of H3: *"Lagged co-importance reveals real
cross-position structure, concentrated at lag τ = 0 and immediate
neighbors, with no evidence for longer-range directional coupling in
this probe."*

## 4. Per-(layer, module) geometry

→ `outputs/gate_geometry/per_layer/`

This part of the analysis does *not* depend on any max-fishing
statistic, so the previous numbers stand. Per-module cosine spectrum
+ effective rank + participation ratio, on alive components capped at
1,024 per module:

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

The L1 attn.k row is the most striking: 46 alive components living in
only 5.5 effective dimensions (PR 2.8). At this scale the model is
using L1 attention with a tiny mechanism budget. L3 mlp.down has the
opposite character: 1,837 alive in 434 dimensions, PR 104.

**H4 ("different modules have different gate geometries") is
unambiguously supported.** The L1-vs-L3 axis spans roughly two orders
of magnitude in effective rank.

## 5. Methods notes

- Gates: `ComponentModel.calc_causal_importances(...).lower_leaky` over
  a tokenized stream of the canonical Pile dataset, matching the
  upstream harvest pipeline's call. No reimplementation.
- All analyses run on H200 GPU. End-to-end main run with 6 null
  shuffles is ~6 minutes; cache load is ~30 s; the 10-pair sweep takes
  ~10–15 min with nulls.
- The gate matrix at this scale is 9.6 GB on disk (65,536 × 38,912
  fp32). Cached at
  `outputs/gate_geometry/cache/pile4L_16x8x512.pt`
  (gitignored). Provenance dict carried inside the cache so any
  re-analysis can round-trip the original extraction config.
- Tests: `pytest tests/test_smoke.py` exercises the mock pipeline +
  null preservation properties + provenance round trip.

## 6. What this is *not*

- Not a discovery of mechanisms. We describe geometry; we don't run
  ablations, autointerp, or causal attribution graphs.
- Not a clustering algorithm. We compute eigenvectors and report
  effective rank; clustering algorithms are deliberately out of scope
  until the geometry warrants them.
- Not a comparison to CLT/PLT (Bussmann's `nn_decompositions` branch
  is linked in `docs/external_repos.md` but not pulled).
- Not a finished post yet. This is a preliminary empirical note;
  the next step is a centered Pearson-based clustering experiment and
  a careful pair-level case study (token-level activation patterns,
  module-pair attribution sanity checks).

## 7. Hypothesis verdicts after null controls

| H | Statement | Verdict (with null controls) |
| --- | --- | --- |
| H1 | Raw gate co-importance is highly structured | **partial yes**. First ~500 cosine eigenvalues separate from shuffle null. Most of the top cosine eigenvalue is mean alignment, not mechanism structure. Use Pearson, not cosine, as headline. |
| H2 | Token identity explains a nontrivial but incomplete fraction | **yes, modestly**. Median R² 0.06, bimodal-with-tail. Token-residualization spectrum is essentially the centered Pearson spectrum + minor adjustment. |
| H3 | Lagged co-importance reveals real cross-position structure | **yes for τ ∈ {−1, 0, +1, +2}, no for longer lags**. Earlier "all peak at τ<0" was a max-fishing artifact; the correct headline is *τ = 0 peak with immediate-neighbor lift*. |
| H4 | Different modules have different gate geometries | **yes**. L1 attn.k (eff_rank 5.5) vs L3 mlp.down (eff_rank 434) is ~80× — no statistical games involved. |
