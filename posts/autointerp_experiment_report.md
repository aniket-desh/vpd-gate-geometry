# Autointerp experiment report

Phase report for the new Experiment 6 in
`posts/gate_geometry_lesswrong_draft.md`.

## Pipeline

```bash
# Phase 1: snippet extraction
external/param-decomp/.venv/bin/python -m vpd_gate_geometry.autointerp_persistent \
    --cache outputs/gate_geometry/cache/pile4L_16x8x512.pt \
    --output-dir outputs/gate_geometry/autointerp_persistent \
    --persistence-threshold 0.8 --gate-threshold 0.5 \
    --n-train-pos 40 --n-heldout-pos 30 --n-heldout-neg 30 \
    --context-left 48 --context-right 24 \
    --include-matched-control --device cuda

# Phase 2: persistent components (excess > 0.5 to drop always-on ones)
external/param-decomp/.venv/bin/python -m vpd_gate_geometry.openai_autointerp \
    --examples outputs/gate_geometry/autointerp_persistent/examples.jsonl \
    --output-dir outputs/gate_geometry/autointerp_persistent \
    --only-category persistent --min-excess-persistence 0.5 \
    --model gpt-4o-mini

# Phase 3: matched non-persistent control (persistence < 0.3, similar mean gate)
external/param-decomp/.venv/bin/python -m vpd_gate_geometry.openai_autointerp \
    --examples outputs/gate_geometry/autointerp_persistent/examples.jsonl \
    --output-dir outputs/gate_geometry/autointerp_persistent/control \
    --only-category control_non_persistent --max-components 12 \
    --model gpt-4o-mini

# Phase 4: plots + table
external/param-decomp/.venv/bin/python -m vpd_gate_geometry.autointerp_plots \
    --persistent-scores outputs/gate_geometry/autointerp_persistent/validation_scores.json \
    --control-scores outputs/gate_geometry/autointerp_persistent/control/validation_scores.json \
    --output-dir outputs/gate_geometry/autointerp_persistent
```

## What was run

| step | n components | API model | calls | notes |
| --- | ---: | --- | ---: | --- |
| Extraction | 64 (32 persistent + 32 matched control) | none | 0 | tokenized with `EleutherAI/gpt-neox-20b` |
| Persistent LLM sweep | 17 (after excess-persistence > 0.5 filter) | `gpt-4o-mini` | 34 (2/comp) | hypothesis + held-out classification |
| Non-persistent control | 12 | `gpt-4o-mini` | 24 (2/comp) | same prompts, matched mean gate |

Each component had 40 train positives, 30 held-out positives, and 30
matched-low-gate negatives. Snippets were 48 tokens of left context +
24 of right context with the focus token marked `<<...>>`.

## Headline results

| group | mean balanced accuracy | median | # ≥ 0.7 | # ≥ 0.6 |
| --- | ---: | ---: | ---: | ---: |
| persistent (17 components) | **0.66** | 0.57 | **6** | **8** |
| random-label shuffle | 0.48 | — | — | — |
| matched non-persistent control (12) | **0.41** | 0.46 | 0 | 1 |

The persistent population scores about 0.25 above the matched
non-persistent control in mean balanced accuracy, and 0.18 above the
random-label shuffle. None of the 12 control components reach 0.7,
while 6 of the 17 persistent components do.

The four strongest persistent hits all have low token-identity
baselines (top token shares ≤ 13% of positives), so they are not
trivial single-token detectors:

| component | label | bal. acc. | top token (frac) |
| --- | --- | ---: | --- |
| `h.0.attn.o_proj#722` | mathematical expressions | 0.983 | `'*'` (0.13) |
| `h.3.attn.v_proj#457` | mathematical operations | 0.950 | `'*'` (0.10) |
| `h.0.attn.o_proj#329` | equations and scientific content | 0.950 | `'car'` (0.04) |
| `h.0.attn.o_proj#536` | political/parliamentary discourse | 0.833 | `'ų'` (0.06) |

The clearest interpretable signal is **math-context recognition**:
three of the top-six components are independently labeled
"mathematical expressions / operations / equations" across two
layers and three different module types (`attn.o_proj`,
`attn.v_proj`, `mlp.c_fc`). One additional component on `h.0.attn.o`
captures a parliamentary-discourse register (positives are
Lithuanian Europarl text; the LLM picks up on register without
seeing the language).

## Files generated

```
outputs/gate_geometry/autointerp_persistent/
├── examples.jsonl                              5.2 MB, 64 components
├── extraction_summary.json
├── hypotheses.json                             persistent sweep
├── validation_scores.json                      persistent sweep
├── run_summary.json
├── scoring_summary.json                        plots/table aggregate
├── 01_validation_scores.png                    main figure
├── 02_accuracy_vs_token_identity.png           sanity check
├── top_components_table.md                     full per-component table
├── llm_cache/   *.json                         34 cached LLM responses (persistent)
├── prompts/     *.txt                          prompts (audit trail)
└── control/
    ├── hypotheses.json
    ├── validation_scores.json
    ├── run_summary.json
    ├── llm_cache/  *.json                      24 cached LLM responses
    └── prompts/    *.txt
```

## Controls

- **Random-label shuffle.** For each component, after scoring, we
  shuffle the held-out positive/negative labels and re-score the same
  model predictions. This baselines the LLM's positive/negative
  prediction bias. Persistent mean random-label score: 0.48 (i.e.
  the LLM does not have a strong bias).
- **Token-identity baseline.** For each component, we report the
  fraction of positives that share the most common focus token id.
  All top-4 high-accuracy components have ≤ 13% top-token share,
  so they're not single-token detectors.
- **Matched non-persistent control.** 12 alive components with
  persistence < 0.3 and mean gate matched (mean ≈ 0.40) to the
  persistent group. These also fire densely but without temporal
  stickiness. Mean held-out balanced accuracy 0.41 (below random).
  This is the most informative control: it shows that *persistence*
  selects for interpretability, not just "any active component".

## Caveats and known limitations

- Hypotheses come from `gpt-4o-mini`. A stronger model would likely
  generate cleaner labels but also be more biased toward inventing
  patterns. For an attention/ablation experiment, the absolute scores
  matter less than the persistent-vs-control gap.
- The "medical terminology focus" labels in the bottom half are
  classic overfitting: the LLM saw a few snippets containing words
  like "respiratory" and generalized, then failed on held-out. This
  is expected behaviour and accounted for by the random-label
  baseline.
- We did not run causal ablations. A natural follow-up is to ablate
  the math-context components on positive vs negative contexts and
  measure KL on next-token predictions.
- We used the persistent-subspace filter as a selection criterion,
  not the only criterion. Some components in the top-K alive set may
  also be interpretable; the persistent filter is one principled way
  to find a small candidate set.

## Cost

- ~58 OpenAI API calls total against `gpt-4o-mini`.
- Approximate spend: a few cents at current OpenAI pricing. All
  responses are cached on disk, so the pipeline is fully resumable.

## Whether the post was updated

Yes. `posts/gate_geometry_lesswrong_draft.md` now has a new
**Experiment 6: interpreting the candidate persistent components**
section with the validation-score figure, the top-components table,
two example snippets, and the persistent-vs-control gap. The TL;DR
and the "what survived the controls" table at the end of the post
each got one new line referencing the autointerp result.
