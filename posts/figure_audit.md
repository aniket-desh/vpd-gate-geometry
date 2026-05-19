# Figure audit for the LessWrong post

For each candidate figure: path, intended caption, label readability at
LessWrong column width, legend overlap, clipping, contrast on the
off-white background, and whether the figure communicates one claim.

The plots were re-rendered on 2026-05-19 after fixing three visual
bugs (`PALETTE` whites → soft tints, "residual" legend key for the
Pearson curve, monospaced + shortened top-pair labels). The audit
below reflects the re-rendered state.

---

## 0. `outputs/gate_geometry/pile4L_v2/plots/10_kernel_real_vs_null.png`  ★ tactile evidence

**Intended caption.** *Cosine kernel on the top-4,096 alive VPD
components (left) and on a row+column-shuffled version of the same
gate matrix (right). Both panels use the same vmin/vmax, the same
diverging colormap, and the same row ordering (sign-of-leading-eigvec
spectral order on the real kernel). The dense block in the upper-left
of the real panel is the same block that the spectrum and effective
rank diagnose; it is visibly absent from the shuffled-null panel.*

| check | status |
| --- | --- |
| labels readable | yes |
| legend overlap | n/a |
| clipping | no |
| contrast | yes; soft-lavender midpoint keeps zero-correlation cells distinct from the warm bg, while the dense block reads as a clear orange patch |
| communicates one point | yes: "the block structure isn't an artifact of cosine on positive sparse data; the shuffled control kills it" |

Verdict: **include as the headline visual.** This is what the
spectrum plot was claiming verbally; this plot makes that claim
visceral in one figure.

---

## 1. `outputs/gate_geometry/pile4L_v2/plots/09_kernel_variants.png`  ★ headline

**Intended caption.** *Top-4,096 alive VPD gate eigenspectra: raw
cosine kernel (purple), centered Pearson kernel (teal), and a
row+column-shuffled cosine null (gray). The raw cosine λ₁ is partly
shared-base-rate alignment (the null λ₁ is comparable); the cleaner
"real structure" claim is the separation of the leading few hundred
modes from the null across all three real curves.*

| check | status |
| --- | --- |
| labels readable at LessWrong column width | yes |
| legend overlap with data | no, top-right corner is empty |
| any text clipped | no |
| low-contrast / white-on-bg points | no — all three lines use saturated colours |
| communicates one point | yes: "cosine λ₁ is partly base-rate; Pearson is the cleaner kernel" |
| recent fix | legend now says **"Pearson (centered)"** and **"shuffled null"**, not "residual" / "baseline" |

Verdict: **include as the headline figure.**

---

## 2. `outputs/gate_geometry/pile4L_v2/plots/05_token_r2.png`

**Intended caption.** *Per-component R² explained by token identity
on the top-4,096 alive subset. The distribution is bimodal-with-tail:
a primary mode at R² ≤ 0.10 (contextual), a secondary mode at
0.15-0.30 (partially lexical), and a long tail past 0.7. Median R² =
0.06.*

| check | status |
| --- | --- |
| labels readable | yes |
| legend overlap | n/a (no legend) |
| clipping | no |
| contrast | yes, orange bars with subtle off-white edges |
| one clear point | yes: "most components are not primarily lexical" |

Verdict: **include.**

---

## 3. `outputs/gate_geometry/pile4L_v2/plots/07_lag_profile.png`  ★ correction

**Intended caption.** *Lagged Pearson correlation between top-384
components of `h.2.attn.q_proj` and `h.2.attn.k_proj`. The green line
is mean(top-100 |r|); the dotted gray line is max |r| over all 147k
pairs at each lag; the gray band is the per-sequence-circular-shift
null distribution (mean to 95th percentile across 6 shuffles). The
robust signal is the green bump at τ ∈ {−1, 0, +1, +2}; the wildly
fluctuating max |r| line is shown deliberately as a warning about the
statistic.*

| check | status |
| --- | --- |
| labels readable | yes |
| legend overlap | no, legend in upper-right |
| clipping | no |
| contrast | yes; green line on grey band on warm off-white |
| one clear point | yes: "real signal at τ=0 ± 2, max statistic is misleading" |

Verdict: **include as the methodology-honesty figure.** This is the
one that demonstrates the v1 correction.

---

## 4. `outputs/gate_geometry/lagged_sweep_qk/h_0_attn_q_proj__h_0_attn_k_proj__lag_profile.png`

**Intended caption.** *L0 within-layer Q→K lagged co-importance.
Same legend structure as Figure 3. Peak at τ = −1 with mean-top-100
|r| = 0.689 (excess +0.607 over null p95), tapering into the null band
by τ = +1.*

| check | status |
| --- | --- |
| labels readable | yes |
| legend overlap | no |
| clipping | no, peak annotation sits clearly above its point |
| contrast | yes |
| one clear point | yes: "one-token-back coupling in L0 Q/K survives the null" |

Verdict: **include as the per-pair example.** L1 is also fine but L0
has the largest signal and is visually clearest.

---

## 5. `outputs/gate_geometry/per_layer/layer_2_spectra.png` (or a module-across-layers)

**Intended caption.** *Per-module cosine kernel spectra at layer 2.
`attn.o_proj` and `attn.v_proj` are the high-rank modules at this
layer (eff. rank 320, 246); `attn.k_proj` collapses to rank 50.*

| check | status |
| --- | --- |
| labels readable | yes (6 modules, legend at top-right) |
| legend overlap | minor — legend at top right partly over the high-eigval region |
| clipping | no |
| contrast | yes, six muted colours separable |
| one clear point | yes: "module geometries differ within a single layer" |

Verdict: **include if space permits**, otherwise prefer a
`module_*_across_layers.png` to make the H4 point cleanly with
layer-to-layer variation.

---

## 6. `outputs/gate_geometry/pile4L_v2/plots/03_kernel_clustered_heatmap.png`  (after colormap fix)

**Intended caption.** *Cosine kernel of the top-4,096 alive
components, reordered by sign of the leading eigenvector. The dense
block in the upper-left corresponds to the "primary mechanism
cluster" of ~500-800 components.*

| check | status |
| --- | --- |
| labels readable | yes |
| legend overlap | n/a |
| clipping | no |
| contrast | improved — zero-cells now soft lavender, not pure white |
| one clear point | yes: "block structure exists, but is dominated by a single big cluster" |

Verdict: **include as an optional supplement.** The block structure
is more honest than v1 suggested: not 4–5 well-separated clusters,
but one dominant cluster plus a long uncoupled tail.

---

## 7. `outputs/gate_geometry/pile4L_v2/plots/04_spectral_embedding.png`  (after colormap fix)

**Intended caption.** *Top-2 spectral embedding of the cosine kernel,
coloured by per-component mean gate (log scale).*

| check | status |
| --- | --- |
| labels readable | yes |
| legend overlap | no |
| clipping | no |
| contrast | improved — lowest-activity points now light lavender, dark thin edge |
| one clear point | weak; mostly a sanity check that there is some 2D layout |

Verdict: **omit from the main post.** It's a nice sanity check but
doesn't carry a claim that the spectrum and heatmap don't already make.

---

## 8. `outputs/gate_geometry/pile4L_v2/plots/08_top_lagged_pairs.png`  (after crop + shortened labels)

**Intended caption.** *Top 8 lagged component pairs by max |r| for
`h.2.attn.q_proj → h.2.attn.k_proj`. Caveat: max |r| is the fragile
statistic; this figure is illustrative, not the headline.*

| check | status |
| --- | --- |
| labels readable | yes (short form `L2.attn.q#220 ↔ ...`, monospaced) |
| legend overlap | n/a |
| clipping | no, cropped to 8 rows |
| contrast | yes |
| one clear point | the strongest individual pairs, but reader has to read the caveat |

Verdict: **omit from the main post**, keep in the appendix as a
diagnostic of why max is the wrong headline.

---

## TODO / outstanding visual concerns

- The `per_layer/layer_2_spectra.png` legend partly overlaps the top
  decade of eigvals; consider `legend(loc="lower left")` or moving
  to a sidebar if I use this in the post.
- All curves are visible at 240 DPI when downscaled to ~700 px column
  width; haven't tested at 1.5x zoom in a browser. If the LW preview
  shows fuzziness, regenerate at 300 DPI.
- Inspect captions for em-dashes one more time before publishing.
