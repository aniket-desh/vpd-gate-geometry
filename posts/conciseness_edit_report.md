# Conciseness edit report

Editing pass on `posts/gate_geometry_lesswrong_draft.md`, applied only
to material from `## VPD in the minimum necessary detail` to the end
of the file. The TL;DR, lead figure, and `## Why look at gates?` were
left untouched per the brief.

## Word count

| scope | before | after | Δ |
| --- | ---: | ---: | ---: |
| from `## VPD in the minimum necessary detail` to end | 3,087 | 2,556 | **−17 %** |
| whole file | 3,522 | 2,991 | −15 % |

The 17 % cut sits in the requested 10–20 % band. No section was
shortened so much that a claim or caveat was dropped.

## Sections shortened most

Ranked by approximate compression (post / pre, by word):

| section | pre | post | Δ |
| --- | ---: | ---: | ---: |
| `## Reproducibility` | 296 | 187 | −37 % |
| `## VPD in the minimum necessary detail` | 199 | 121 | −39 % |
| `## The causal-importance field as a data object` | 199 | 156 | −22 % |
| `## Experiment 1: same-position gate geometry` | 463 | 343 | −26 % |
| `## What this suggests for VPD-style clustering` | 230 | 190 | −17 % |
| `## Experiment 2: token identity is real…` | 234 | 218 | −7 % |
| `## Experiment 3: lagged co-importance…` | 562 | 487 | −13 % |
| `## Experiment 4: per-(layer, module) geometry` | 213 | 184 | −14 % |
| `## Limitations` | 196 | 158 | −19 % |
| `## Future work` | 187 | 173 | −7 % |
| `## What survived the null controls?` | 144 | 124 | −14 % |
| `## Acknowledgments and citations` | 124 | 124 | 0 % |

Reproducibility was compressed by collapsing the full reproduction
command into an HTML `<details>` block while keeping the
configuration table inline; this is the only place where content was
moved rather than rewritten. Experiment 1 took the second largest
cut, mostly by merging the three places that previously explained the
same "cosine λ1 inflated by base rate" point.

## Claims I softened (no numbers changed)

- "i.e. one-token-back attention" → "one-token-back **Q/K gate
  coupling** rather than one-token-back attention, since we are still
  talking about gate-field correlation, not attribution." Done so the
  Experiment 3 prose does not overclaim mechanistic interpretation.
- "That direction is consistent with the mechanistic reading" → "That
  is consistent with the reading that…" (same intent, less assertive
  about the mechanism).
- "The L1 attention does very little useful work and the alive
  components are largely vestigial, or it does a very concentrated
  piece of work" → "Either these alive components implement a small
  concentrated behavior, or the decomposition has allocated a
  low-dimensional residual subspace there." Removed the "does very
  little useful work" phrasing per the brief.
- "Larger lag range. The current τ window may miss longer-range Q/K
  coupling that's genuinely there but was not visible…" → "Robustness
  checks over sequence length and lag window. I would want to verify
  that the small upticks at τ ∈ {+5, +6} visible in some pairs are
  window artifacts rather than real long-range structure. This is a
  check on the current conclusion, not a hint that I expect hidden
  signal." Removes the implicit suggestion that the corrected
  conclusion is wrong.

## Claims I removed

None. Every numerical result, every caveat, and every figure
reference in the original draft is still present.

## Internal-scaffolding language removed

- "the H1 claim survives the null in its weaker form"
- "the verdict on H2 is therefore mixed"
- "H4 from the project plan (\"different modules have different gate
  geometries\")"
- "and unlike H3 it does not need any null discipline"

These were replaced with prose that names the substantive claim
directly. The `## Summary: what survived the controls?` table (renamed
from `## What survived the null controls?`) makes the
hypothesis-status format explicit without using the H1/H2/H3/H4 labels
in running prose.

## Figures / tables moved or removed

- **No figures removed.** All four currently-used image links
  (`09_kernel_variants.png`, `10_kernel_real_vs_null.png`,
  `07_lag_profile.png`, the L0 lag profile from
  `lagged_sweep_qk/`) remain in their original positions.
- **No tables removed.** Three tables in the body (kernel variants,
  Q/K results, per-module geometry, summary, reproducibility config)
  are all unchanged in structure and numbers.
- **One block moved**: the exact reproduction shell command was
  wrapped in `<details><summary>Exact reproduction command</summary>`,
  so it collapses by default in renderers that respect raw HTML
  (LessWrong does). The reproducibility configuration table stays
  inline, since it is short and load-bearing.

## Other small changes

- Heading rename: `## What survived the null controls?` →
  `## Summary: what survived the controls?` per the brief.
- The four-probe list in `## The causal-importance field as a data
  object` was collapsed from four expanded paragraphs into four
  single-line bullets, with one new closing line:
  > These are all proxies. They tell us how gate patterns relate, not
  > what the components do.
  This consolidates the "descriptive, not causal" caveat that
  previously appeared near the top of multiple sections.
- The "Why look at gates?" section was not touched.
- En-dashes in numeric ranges (`0.15–0.30`) were normalized to
  hyphens (`0.15-0.30`) for safety.

## Style-rule confirmation

Verified by grep over the final file:

- **No em-dashes (`—`) anywhere.**
- **No en-dashes (`–`) anywhere.**
- **No `H1`/`H2`/`H3`/`H4` scaffolding markers** in prose.
- **No `$|r|$` math-with-pipe inside a table cell** (the table cell
  uses `\|r\|` to escape the pipes for GitHub's table parser; prose
  uses plain `|r|`).
- **Terminology consistent**: "column-wise row-shuffled null" used
  uniformly for the spectrum/heatmap null; "per-sequence
  circular-shift null" used uniformly for the lagged null; "top-4,096"
  used uniformly for the kernel subset.

Nothing in this pass touched the analysis code, the gate matrix, or
the figures.
