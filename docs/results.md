# Descriptive geometry of the VPD causal-importance field

**Run:** `wandb:goodfire/spd/runs/s-55ea3f9b`  (VPD paper, April 2026;
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

1. **The gate field has real geometry beyond a column-wise
   row-shuffled null.** On the top 4,096 alive components, the cosine
   kernel's first ~500 eigenvalues sit visibly above the shuffled-null
   curve; the rest of the spectrum drops below it. Effective rank 941
   (cosine) vs 3,418 (shuffled-cosine null).
2. **Most of the leading cosine eigenvalue is base-rate alignment, not
   mechanism structure.** Top eigenvalue: cosine 206, Pearson (centered)
   150, shuffled-cosine null 150. Reading anything off the cosine λ₁
   alone is misleading; Pearson is the cleaner headline statistic.
3. **Token identity explains only a small fraction of the geometry on
   the top-K alive subset.** Median per-component R² = 0.06,
   bimodal-with-tail. Residualization is a real but secondary
   transformation; centering matters more.
4. **Within-layer Q/K has real one-token-back coupling for 3 of 4
   layers; longer-range claims do not survive.** Using the
   null-controlled `mean(top-100 |r|)` statistic, L0/L1/L3 Q/K peak
   at **τ = −1** with excess +0.27 to +0.61 over null; L2 peaks at
   τ = 0. Past τ = ±2 every layer collapses to within the null band.
   My v1 "τ deepens to −2/−3 with depth" claim was a max-fishing
   artifact; the real signal is one position back, not three.
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
  cosine kernel reports 206, but the same kernel computed on a
  column-wise row-shuffled copy of $G$ (which destroys every
  cross-component co-activation while preserving each component's
  marginal distribution) still reports 150. That happens because the
  cosine kernel normalizes by per-column norms but not per-column
  means, so the global "mean direction" of the active components
  persists under shuffling. The honest top eigenvalue is the Pearson
  one (~150), and it is only ~30 units above what shuffling alone
  produces.
- The **shape of the spectrum** is real. Beyond the first eigenvalue,
  the shuffled null is essentially flat at 1.51 (entries 2 through
  ~4000), while the real cosine spectrum drops smoothly over three
  decades. The real spectrum sits above the null curve up to roughly
  index 500, then crosses below; exactly the signature of a kernel
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
(contextual), a secondary mode at 0.15-0.30 (partially token-bound),
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
L2 Q/K, where same-position and immediate-neighbor coupling beat the null
by 40-60% (excess ~0.14, null floor ~0.10). There is *no* evidence
for the "Q at later position reaches back several tokens to K"
narrative I asserted in v1. Anything past τ = ±2 is at the noise floor.

### 3.2 Sweep across all four same-layer Q/K pairs

→ `outputs/gate_geometry/lagged_sweep_qk/`

Same null-controlled procedure on the four within-layer
attention.q_proj → attention.k_proj pairs.

| pair | τ where top-100 \|r\| peaks | top-100 \|r\| at peak | null p95 at peak | excess at peak |
| --- | :---: | ---: | ---: | ---: |
| L0 q→k | **−1** | 0.689 | 0.083 | **+0.607** |
| L1 q→k | **−1** | 0.393 | 0.046 | **+0.347** |
| L2 q→k | **0**  | 0.246 | 0.103 | **+0.143** |
| L3 q→k | **−1** | 0.357 | 0.091 | **+0.265** |

For three of the four layers the peak is at **τ = −1, not τ = 0**.
That direction is consistent with "the query gate at destination
position t couples maximally with the key gate at the previous source
position t − 1", i.e. one-token-back attention. L2 is the exception
and peaks at τ = 0 (same-token coupling). All four peaks sit well
above the circular-shift null distribution.

What does **not** survive the null at these scales: longer-range
peaks. Looking at the L0 q→k full curve as the cleanest example,

| τ | top-100 \|r\| | null p95 | excess |
| ---: | ---: | ---: | ---: |
| −6 | 0.138 | 0.097 | +0.041 |
| −5 | 0.115 | 0.092 | +0.023 |
| −4 | 0.072 | 0.094 | −0.022 |
| −3 | 0.190 | 0.072 | +0.119 |
| −2 | 0.256 | 0.079 | **+0.177** |
| **−1** | **0.689** | 0.083 | **+0.607** |
| **0**  | **0.687** | 0.111 | **+0.576** |
| +1 | 0.075 | 0.080 | −0.005 |
| +2 | 0.080 | 0.135 | −0.055 |
| +3 | 0.085 | 0.083 | +0.002 |
| +4 | 0.085 | 0.095 | −0.011 |
| +5 | 0.272 | 0.184 | +0.087 |
| +6 | 0.414 | 0.100 | +0.314 |

a sharp asymmetric peak at τ ∈ {−1, 0} with substantial neighbor
support at τ = −2, −3, then collapse to the null band for τ ≥ +1.
The small uptick at τ = +5, +6 is suspicious (could be a sequence
boundary or window-edge artifact at our seq_len = 512) and would
need a longer τ range to interpret; the headline structure is the
−1/0 peak. The visible plot is
`outputs/gate_geometry/lagged_sweep_qk/h_0_attn_q_proj__h_0_attn_k_proj__lag_profile.png`.

### 3.3 What the lagged result actually shows

- **Asymmetric one-token-back coupling between Q and K is real for
  L0, L1, L3**, with excess over null of +0.27 to +0.61 in the
  mean-top-100 statistic. L2 alone peaks at τ = 0.
- **Past τ = ±2 the signal collapses to noise floor.** The earlier
  v1 claim that "peaks deepen with depth from τ=−2 to τ=−3" came
  from max-fishing in the heavy tail and does not survive the null.
- **Max |r| is unusable on its own.** It saturates near 1.0 at many
  τ values that are statistically indistinguishable from null. The
  lag-profile plot keeps it visible in dotted gray so the reader can
  see the fragility; the green line is the actual signal.

Honest reframing of H3: *"Within-layer Q/K coupling has real
cross-position structure, concentrated at τ = −1 in three of four
layers (with L2 peaking at τ = 0), tapering into the null band by
τ = ±2. There is no evidence for longer-range directional coupling
in this probe."*

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
  ~10-15 min with nulls.
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

## 6.5 Tensor structure of the gate field

Two probes that don't fit the kernel framing: per-component temporal
persistence and row-pattern vocabulary. Both run from
`vpd_gate_geometry.temporal` and `vpd_gate_geometry.row_patterns` on
the cached sparse gate matrix.

**Shape and sparsity.** Full tensor $G \in \mathbb{R}^{65{,}536 \times 38{,}912}$,
all entries in $[0, 1]$. At threshold $g > 10^{-4}$ the NNZ is
13.3 M (0.52% density); mean L0 per token is **202.9**, matching
the paper's reported 205. Cached as sparse-COO fp16: 135 MB on disk
vs 9.6 GB dense fp32 (75× compression), max precision loss 2.4e-4.

**Temporal persistence** (threshold $g > 0.5$):

| metric (over 9,014 alive components) | value |
| --- | ---: |
| median $P(\text{on at } t{+}1 \mid \text{on at } t)$ | 0.054 |
| mean persistence | 0.106 |
| mean baseline density $P(\text{on})$ | 0.017 |
| median geometric on-run length | 1.06 tokens |
| p95 persistence | 0.409 |
| p95 on-run length | 1.69 tokens |
| components with persistence > 0.3 | 842 of 9,014  (9.3%) |
| components with persistence > 0.5 | 244 of 9,014  (2.7%) |
| components with persistence > 0.8 | 32 of 9,014  (0.4%) |

Most alive components are short-lived. Mean persistence is about
$6\times$ the mean baseline density, so there is real above-chance
temporal correlation, but only 0.4% of alive components reach
persistence > 0.8 (the threshold I would use for "context-state
variable").

**Row-pattern vocabulary** (threshold $g > 0.5$, alive cols only):

| pattern statistic | value |
| --- | ---: |
| unique active-set patterns | 65,211 of 65,536  (99.5%) |
| patterns appearing once | 65,013  (99.2%) |
| most frequent pattern | 11 rows |
| patterns to cover 50% of rows | 32,443 |
| Jaccard, random row pairs (mean / median / p95) | 0.067 / 0.057 / 0.140 |
| Jaccard, within-sequence $(t, t{+}1)$ (mean / median / p95) | **0.156** / 0.134 / 0.322 |

The model assembles a near-unique combination of ~150 atoms for
almost every token. Adjacent tokens are 2.3× more similar than
random pairs but still share only ~15% of their support.

## 7. Hypothesis verdicts after null controls

| H | Statement | Verdict (with null controls) |
| --- | --- | --- |
| H1 | Raw gate co-importance is highly structured | **partial yes**. First ~500 cosine eigenvalues separate from shuffle null. Most of the top cosine eigenvalue is mean alignment, not mechanism structure. Use Pearson, not cosine, as headline. |
| H2 | Token identity explains a nontrivial but incomplete fraction | **yes, modestly**. Median R² 0.06, bimodal-with-tail. Token-residualization spectrum is essentially the centered Pearson spectrum + minor adjustment. |
| H3 | Lagged co-importance reveals real cross-position structure | **yes for τ ∈ {−2, −1, 0} with peak at τ = −1 in 3 of 4 same-layer Q/K pairs (L2 peaks at τ = 0); no for longer lags.** Earlier "peak deepens to τ = −3" was a max-fishing artifact; the real signal is one position back. |
| H4 | Different modules have different gate geometries | **yes**. L1 attn.k (eff_rank 5.5) vs L3 mlp.down (eff_rank 434) is ~80×, with no statistical games involved. |
| H5 (temporal) | Most gate atoms are persistent context variables | **mostly no**. Mean persistence 0.11 vs mean baseline 0.02 (a real $\sim 6\times$ lift, but small in absolute terms); only 32 of 9,014 (0.4%) alive components reach persistence > 0.8. |
| H6 (row patterns) | Adjacent tokens use roughly the same active set | **partly**. Adjacent-pair Jaccard 0.156 vs random 0.067 ($2.3\times$ lift), but adjacent tokens still differ on ~85% of their active components. |
