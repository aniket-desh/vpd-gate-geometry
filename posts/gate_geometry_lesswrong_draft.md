# The Gate Geometry of Parameter Decompositions

*VPD decomposes model weights into rank-one subcomponents. I looked
at the geometry of when those subcomponents are causally important.*

TL;DR:

- VPD ([Bushnaq et al., 2026](https://www.goodfire.ai/research/interpreting-lm-parameters))
  produces a causal-importance gate field $g^\ell_{b,t,c}\in[0,1]$
  that says which rank-one parameter subcomponents are needed on each
  token. I treat that field as a geometric object on its own.
- On the canonical 4-layer Pile model, the cosine co-importance kernel
  on the top 4,096 alive subcomponents has visible structure beyond a
  row-shuffled null, but the leading raw-cosine eigenvalue is partly
  shared-base-rate alignment. The cleaner story is the centered Pearson
  kernel and the separation of the leading few hundred modes from the
  null spectrum.
- Token identity explains a modest fraction of the top-K gate field
  (median per-component $R^2 = 0.06$, mean 0.13), with a clear lexical
  tail. It is not the dominant story, but it is not negligible either.
- Lagged Q/K co-importance shows a real, *narrow* cross-position signal:
  in three of four layers the mean-top-100 $|r|$ peaks at $\tau = -1$
  (the query gate at position $t$ couples maximally with the key gate
  at the previous source position), with L2 peaking at $\tau = 0$. Past
  $\tau = \pm 2$, every layer collapses into the circular-shift null.
- Module geometries vary by roughly two orders of magnitude in effective
  rank within the same decomposition: L1 `attn.k_proj` lives in $\sim 5.5$
  effective dimensions, L3 `mlp.down_proj` in $\sim 434$.

Code, data, and full numbers:
[github.com/aniket-desh/vpd-gate-geometry](https://github.com/aniket-desh/vpd-gate-geometry).
Authoritative results: [`docs/results.md`](https://github.com/aniket-desh/vpd-gate-geometry/blob/main/docs/results.md).

## Why look at gates?

VPD gives us two things. The first is the obvious one: a decomposition
of model weights into rank-one parameter subcomponents. The second is
easier to read as just an implementation detail: a causal-importance
field $g^\ell_{b,t,c}$, saying which subcomponents are needed on which
token positions. This post is about the second object.

[Goodfire's writeup](https://www.goodfire.ai/research/interpreting-lm-parameters)
says directly that VPD's current clustering uses same-position
correlations between causal importances, and that this is blind to
multi-sequence-position circuits. (Their canonical example: query and
key subcomponents in an induction-like head might be causally
important at different token positions.) That's a natural invitation
to look at the gate field itself as a data matrix, and to ask what
geometry it actually contains.

I do not run ablations, autointerp, or attribution graphs here. The
contribution is descriptive: same-position co-importance kernels,
token-identity residualization, lagged co-importance with null
controls, and per-(layer, module) geometry. The first version of this
project also taught me a methodological lesson about max statistics
that I think is worth writing down explicitly.

## VPD in the minimum necessary detail

The target model is a 4-layer decoder-only Llama-MLP-only transformer
with 6 attention heads, $n_\text{embd} = 768$, vocab 50,277, trained
on an uncopyrighted Pile subset (Bushnaq et al., 2026). The
decomposition lives over 24 weight matrices, with 38,912 rank-one
subcomponents in total. About 9,972 of those are "alive" in the
paper's reported sense (threshold $\bar g > 10^{-6}$).

For each subcomponent VPD trains a causal-importance value
$g^\ell_{b,t,c}\in[0,1]$ which says how necessary that subcomponent is
on token $(b,t)$ in layer/module $\ell$. The training minimizes
importance subject to a faithfulness constraint that uses both
stochastic and adversarially selected ablation masks. The key
mechanistic interpretation, taken from the paper, is that
$g \to 0$ means the subcomponent should be ablatable on that token
without changing the model's output, and $g \to 1$ means it must be
preserved.

For the geometry analysis, the relevant fact is that $g$ is a
high-dimensional sparse-ish $[B, T, C]$ tensor that can be flattened
into a $[N, C]$ matrix where $N = B \cdot T$.

## The causal-importance field as a data object

Flatten the gate dict across modules:
$G \in \mathbb{R}^{N \times C}$, with $C = \sum_\ell C_\ell$ and
each row being a token instance. In my main run, $N = 65{,}536$ token
positions (16 batches of 8 sequences of 512 tokens, streamed from the
canonical Pile dataset), and after restricting to alive components I
work on the top-4,096 alive subset ranked by mean activity.

The four probes are:

1. **Same-position spectral structure.** Eigenvalues and clustering
   ordering of cosine, centered Pearson, and shuffled-null kernels on
   the top-K alive matrix.
2. **Token-identity residualization.** Per-component $R^2$ explained
   by a per-token-id baseline, and the resulting residualized
   spectrum.
3. **Lagged co-importance.** Pearson $r$ between top-K components of
   module $A$ at position $t$ and module $B$ at position $t + \tau$,
   compared to a per-sequence circular-shift null on module $B$.
4. **Per-(layer, module) spectra.** The same cosine kernel computed
   separately for each of the 24 decomposed matrices.

All four are descriptive. None of them tells us what a component
*does*; only how its causal-importance pattern relates to others.

## Experiment 1: same-position gate geometry

The first natural object is the cosine kernel of the top-4,096 alive
subset and its eigenspectrum. The raw numbers:

| variant | top eigenvalue | 10th | effective rank |
| --- | ---: | ---: | ---: |
| raw cosine | **206.1** | 38.5 | 940.7 |
| centered Pearson | **150.9** | 30.8 | 1{,}046.9 |
| token-residualized cosine | 143.1 | 30.6 | 1{,}292.6 |
| shuffled-cosine null | **150.4** | 1.51 | 3{,}417.9 |

![Spectrum of the top-4096 alive VPD components on three kernel variants and a shuffled null. The raw cosine and centered Pearson curves are nearly indistinguishable beyond their first eigenvalue; the null curve is flat beyond its own λ1 (≈150) and crosses above both real curves only deep in the tail.](../outputs/gate_geometry/pile4L_v2/plots/09_kernel_variants.png)

The plot says something subtle that the v1 cosine headline got wrong.
The raw cosine kernel reports a top eigenvalue of 206. That looks
impressive until you compute the same cosine kernel on a
row-permuted, column-permuted version of $G$, which destroys every
cross-row coupling. The shuffled cosine kernel still has a top
eigenvalue of 150. That's because the cosine kernel normalizes by
per-column $L^2$ norm but not per-column mean, so the global "mean
direction" of the active components survives row shuffling.

The honest claim is the **centered Pearson** top eigenvalue ($\approx 150$),
and even more importantly, the *shape* of the spectrum. Beyond the
first eigenvalue, the shuffled null collapses to roughly $1.51$ for
all subsequent indices (entries 2 through $\sim 4000$). The real
spectrum drops smoothly over three decades. The first few hundred
modes sit visibly above the null; the tail crosses below it. That
crossover is the signature of a structured kernel that concentrates
variance in the top eigenvectors at the expense of the bulk.

So the H1 claim survives the null in its weaker form: VPD gates have
real spectral structure on the top-K alive subset, concentrated in
roughly the leading few hundred modes of the *centered* kernel.

## Experiment 2: token identity is real, but not the whole story

How much of the same gate matrix is explained by token identity alone?
I fit a per-token-id baseline (with shrinkage for rare tokens) on the
top-K alive subset:
$g_{b,t,c} \approx \mu_c + a_{z(b,t),c}$,
and look at the per-component $R^2$.

![Histogram of per-component R² explained by token identity on the top-4096 alive subset. Median 0.06, with a bimodal-with-tail shape: a primary mode at R² ≤ 0.1, a secondary mode at 0.15-0.30, and a long tail past 0.7.](../outputs/gate_geometry/pile4L_v2/plots/05_token_r2.png)

Numbers:

- Effective vocab with at least 16 occurrences: 522.
- Median per-component $R^2 = 0.059$.
- Mean $R^2 = 0.129$.
- Top eigenvalue after token residualization: 143.1 (vs. raw cosine 206.1).

The histogram is bimodal-with-tail. Most components live near $R^2 = 0.05$:
their gate fields are not primarily token-identity driven. A
secondary mode at $0.15$ to $0.30$ is partly token-bound (these are
components that fire reliably on a few token types but also carry
context). The long tail past $R^2 > 0.7$ is the lexical population I
would expect VPD-style "support" components to live in.

Critically, residualization barely moves the bulk of the spectrum
beyond what *centering alone* already does (compare residualized top
eigval 143 to Pearson 151). Most of what residualization "removes" is
the same mean alignment that the centered Pearson kernel already
removes. The verdict on H2 is therefore mixed: token identity is a
real and stratifying axis, but not the dominant geometric story of
the gate field at this scale.

## Experiment 3: lagged co-importance and why max statistics fooled me

This is the experiment that I got wrong the first time, and the
correction is the most useful part of this writeup.

For each pair of modules $(A, B)$ I compute the Pearson correlation
between top-384 components of $A$ at position $t$ and top-384
components of $B$ at position $t + \tau$, for $\tau \in [-6, 6]$.
That gives a $384 \times 384$ correlation matrix per lag.

### The v1 attempt

I summarized each lag's correlation matrix by its **maximum absolute
entry**. That looked like a sharp result: across all four same-layer
Q$\to$K pairs, the max $|r|$ peaked at $\tau \in \{-2, -3\}$. I wrote
down a story about deeper layers reaching further back. That story
did not survive the first null control.

The problem: with 147,456 component pairs at each lag, the maximum of
the heavy tail saturates near 1.0 for *many* lags, including ones
that the rest of the distribution is indistinguishable from null. Max
$|r|$ is fine for finding a single illustrative pair. It is not a
summary statistic for whether the distribution at one lag is heavier
than at another.

### The fix

I switched to a less fragile statistic, the **mean of the top-100
$|r|$** at each lag, and added a per-sequence circular-shift null on
module $B$ (six independent shuffles per pair). Under that null,
module $B$'s within-sequence marginal histogram and autocorrelation
are preserved, but every cross-module position alignment is broken.
Comparing real $|r|$ to null is then a well-posed question.

![Lagged co-importance for h.2.attn.q_proj → h.2.attn.k_proj. The green line (mean of top-100 |r|) is the actual signal. The dotted gray line (max |r|) saturates at lags that the null distribution shows are noise. Real signal rises above null only for τ ∈ {−1, 0, +1, +2}.](../outputs/gate_geometry/pile4L_v2/plots/07_lag_profile.png)

The corrected within-layer Q$\to$K results:

| pair | $\tau$ at peak (mean top-100) | mean top-100 $|r|$ | null p95 | excess |
| --- | :---: | ---: | ---: | ---: |
| L0  q → k | **−1** | 0.689 | 0.083 | **+0.607** |
| L1  q → k | **−1** | 0.393 | 0.046 | **+0.347** |
| L2  q → k | **0**  | 0.246 | 0.103 | +0.143 |
| L3  q → k | **−1** | 0.357 | 0.091 | **+0.265** |

For three of the four layers the peak is at $\tau = -1$, not $\tau = 0$.
That direction is consistent with the mechanistic reading: the query
gate at destination position $t$ couples maximally with the key gate
at the previous source position $t - 1$, i.e. one-token-back attention.
L2 is the exception and peaks at $\tau = 0$ (same-token coupling).

![L0 within-layer Q→K lagged profile. Sharp asymmetric peak at τ=-1 with neighbours at τ=0 and τ=-2; collapses into the null band by τ=+1.](../outputs/gate_geometry/lagged_sweep_qk/h_0_attn_q_proj__h_0_attn_k_proj__lag_profile.png)

Past $\tau = \pm 2$, every same-layer Q$\to$K pair drops into the
null band. Anything I previously read off the $\tau \in \{-3, -6\}$
end of the curve was the heavy tail talking, not real cross-position
coupling.

So the methodological lesson, stated bluntly:

> Max $|r|$ over hundreds of thousands of component pairs is an
> attractive nuisance. It is good for finding examples; it is not a
> summary statistic.

I'm writing this out at length because I think it's a trap that's
easy to walk into for anyone analyzing VPD gates (or, frankly, any
high-dimensional activation matrix) without an explicit null. The
green line in the figure above is the one I would actually defend.

## Experiment 4: per-(layer, module) geometry

This is the most robust observation, in the sense that no
extreme-value statistic is involved.

For each of the 24 decomposed matrices I compute its own cosine kernel
on its alive components (capped at 1,024 per module, with the same
$\bar g > 10^{-4}$ threshold). The full table is in
[`docs/results.md`](https://github.com/aniket-desh/vpd-gate-geometry/blob/main/docs/results.md);
the highlights:

| layer × module | alive | effective rank | participation ratio |
| --- | ---: | ---: | ---: |
| **L1 attn.k** | 46 | **5.5** | **2.8** |
| **L1 attn.q** | 10 | 7.0 | 5.4 |
| L1 attn.v | 209 | 39.5 | 8.6 |
| L2 attn.o | 522 | 320.7 | 146.4 |
| L2 attn.v | 481 | 245.6 | 96.7 |
| L0 mlp.down | 1,092 | 487.7 | 202.1 |
| L3 mlp.c_fc | 1,018 | 526.4 | 219.9 |
| **L3 mlp.down** | **1,837** | **433.8** | 103.7 |

L1 attention is the standout: 46 alive components in `attn.k_proj`
that together occupy roughly 5.5 effective dimensions, and only 10
alive in `attn.q_proj` at 7.0 effective dimensions. That's the
model using L1 attention with a tiny mechanism budget. L3 `mlp.down`
is the opposite shape: a wide, redundant 1,837-component dictionary
in $\sim 434$ effective dimensions.

The L1-vs-L3 axis spans roughly two orders of magnitude in effective
rank within the same model and decomposition. H4 from the project
plan ("different modules have different gate geometries") is
unambiguously supported, and unlike H3 it does not need any null
discipline to defend.

## What survived the null controls?

A simple table:

| claim | held up? |
| --- | --- |
| Raw cosine kernel has real spectral structure | Partly. The cosine $\lambda_1$ is half-driven by base-rate alignment; the leading $\sim 500$ modes of the centered Pearson kernel are above the shuffled null. |
| Token identity is the dominant geometric axis | No. Median $R^2 = 0.06$. There's a lexical tail; most components are not primarily token-bound. |
| Q/K coupling reaches back two or three tokens | No. Real signal collapses to within the null band past $\tau = \pm 2$. |
| Q/K coupling is one-token-back in most layers | Yes, with excess of $+0.27$ to $+0.61$ over null p95 in L0/L1/L3. L2 peaks at $\tau = 0$. |
| Different modules have wildly different gate geometries | Yes. $\sim 80\times$ range in effective rank across modules in the same decomposition, no statistical games involved. |

## What this suggests for VPD-style clustering

I want to be careful not to oversell this. The post is descriptive,
not prescriptive. But three things follow that I think are worth
noting.

First, if a future VPD-clustering pipeline uses cosine kernels on
gates as a similarity, the leading component of that similarity is
partly shared base rate. Switching to centered Pearson is essentially
free and removes the inflation.

Second, the same-position vs cross-position story for attention Q/K
is real but narrow. Pairs that show coupling at $\tau = 0$ vs
$\tau = -1$ in three of four layers are not a "global cross-position
circuit" story; they're a "Q/K are tied to attention head structure"
story. For a clustering method that wants to group Q and K into the
same mechanism, this is actually good news: the relevant offset is
small and consistent within a layer.

Third, the L1 attention modules occupying $\sim 7$ effective
dimensions deserves a follow-up. Either L1 attention does very little
useful work and the alive components are largely vestigial, or it
does a very concentrated piece of work and a tiny number of
mechanisms suffice. Autointerp labels on the L1 alive components
would distinguish these.

## Limitations

This is a descriptive note. It does not establish what any component
does. In particular:

- Co-importance is not causal attribution. Two components that
  consistently fire on the same tokens may belong to the same
  mechanism, but they may also be independently triggered by the
  same lexical or syntactic feature, or be redundant copies, or be
  causally upstream/downstream rather than co-located.
- The kernel-level analyses (sections 1, 2, 3) all live on the
  top-4,096 alive subset, ranked by mean activity. They are not
  about all 38,912 atoms.
- The sample is 65,536 token positions from the canonical Pile
  stream. That's enough to bring the alive-count and per-layer
  proportions within 10% of the paper, but not the full training
  distribution.
- The lagged null uses 6 circular-shift runs per pair. Enough to
  catch the v1 max-fishing error and to establish the $\tau = -1$
  signal in L0/L1/L3, but not enough for a final p-value style
  analysis.
- No cluster ablations were run. No autointerp labels or attribution
  graphs were used. No comparison to CLT/PLT
  ([Bussmann's `nn_decompositions@vpd_paper`](https://github.com/bartbussmann/nn_decompositions/tree/vpd_paper))
  or activation-space baselines.

## Future work

In rough order of how interesting they look from the current data:

1. **Bilinear gate features.** The natural next step is to compute
   pairwise products $g_{b,t,c}\,g_{b,t,c'}$ and look for low-rank
   quadratic structure, in the spirit of bilinear autoencoders
   ([Dooms & Gauderis, 2025](https://arxiv.org/abs/2510.16820);
   [project page](https://tdooms.github.io/research/bae/)).
   The "L0 Q$\to$K coupling at $\tau = -1$" result is exactly the
   kind of pairwise structure such a method could surface
   automatically.
2. **Cluster-level case study.** Pick one of the dense blocks from
   the cosine-kernel heatmap and run autointerp on its members.
   This is the qualitative companion to the spectral picture.
3. **Compare to CLT/PLT gates on the same model.** The Bussmann fork
   provides matched decompositions; the analogous spectrum and lag
   profile would tell us whether what we see here is VPD-specific
   or generic to any rank-one parameter-decomposition method.
4. **Larger lag range.** The current $\tau \in [-6, 6]$ window may
   miss longer-range Q/K coupling that's genuinely there but was
   not visible in 6 lags either side of zero.

## Reproducibility

```bash
# 1. Clone scaffold + setup
git clone https://github.com/aniket-desh/vpd-gate-geometry.git
cd vpd-gate-geometry
git clone --depth 1 https://github.com/goodfire-ai/param-decomp.git external/param-decomp
cd external/param-decomp && uv sync && cd ../..

# 2. Download canonical VPD model + target LM from public W&B GCS
#    (the goodfire/spd project is USER_READ, no API key required).
#    See docs/repo_readiness_report.md for the exact GraphQL queries.

# 3. Run the main analysis with null controls
external/param-decomp/.venv/bin/python -m vpd_gate_geometry.run_analysis \
    --backend repo \
    --run-path runs/pile-4L/model_400000.pth \
    --target-model-path runs/pretrain-4L/files/model_step_99999.pt \
    --output-dir outputs/gate_geometry/pile4L_v2 \
    --cache-gate-matrix outputs/gate_geometry/cache/pile4L_16x8x512.pt \
    --n-batches 16 --batch-size 8 \
    --max-components 4096 --max-lag 6 \
    --lagged-top-k 512 \
    --alive-threshold 1e-4 \
    --n-null-runs 6 --null-kind circular \
    --device cuda
```

Concrete configuration used for the numbers in this post:

| | |
| --- | --- |
| canonical run | `wandb:goodfire/spd/runs/s-55ea3f9b` |
| target model | `wandb:goodfire/spd/runs/t-9d2b8f02` |
| decomposition | 4-layer Llama-MLP-only, 24 matrices, 38,912 rank-one atoms |
| dataset | `danbraunai/pile-uncopyrighted-tok-shuffled` (streaming) |
| sample | 16 batches × 8 sequences × 512 tokens = 65,536 token positions |
| alive threshold | $\bar g > 10^{-4}$ |
| kernel subset | top-4,096 alive components ranked by mean activity |
| lagged top-K | 512 (main run); 384 (Q/K sweep) |
| null | per-sequence circular shift of module $B$, 6 independent runs |
| hardware | single NVIDIA H200, 144 GB VRAM |

The full quantitative table including all 24 modules and the QK
sweep results lives in
[`docs/results.md`](https://github.com/aniket-desh/vpd-gate-geometry/blob/main/docs/results.md);
the per-module spectra and lag profiles for each pair are in
[`outputs/gate_geometry/`](https://github.com/aniket-desh/vpd-gate-geometry/tree/main/outputs/gate_geometry).

## Acknowledgments and citations

Goodfire's VPD paper is the central object this post depends on
(Bushnaq, Braun, Clive-Griffin, Bussmann, Hu, Ivanitskiy, Linsefors,
Sharkey, 2026: [paper](https://www.goodfire.ai/research/interpreting-lm-parameters),
[code](https://github.com/goodfire-ai/param-decomp)). The framing of
parameter decomposition as the right object for mechanistic
interpretability comes from the SPD/APD lineage
([Bushnaq, Braun, Sharkey, 2025](https://arxiv.org/abs/2506.20790)).
The pairwise / bilinear extension I gesture at in Future Work follows
the bilinear MLP and bilinear autoencoder line of work
([Pearce et al., 2025](https://arxiv.org/abs/2410.08417);
[Dooms & Gauderis, 2025](https://arxiv.org/abs/2510.16820)).

Thanks to whoever reads this and tells me what's wrong with it.

---

*Code: [github.com/aniket-desh/vpd-gate-geometry](https://github.com/aniket-desh/vpd-gate-geometry).
Tests: `pytest tests/test_smoke.py` covers the mock pipeline, null
preservation invariants, and the planted lagged-pair detection.*
