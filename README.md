# vpd-gate-geometry

[Blogpost: Gate Geometry of Parameter Decompositions](TODO: LessWrong link)

A geometric look at the causal-importance gate field `g[layer, batch, token, component] ∈ [0,1]` from Goodfire's [VPD / param-decomp](https://github.com/goodfire-ai/param-decomp). Not a reimplementation — a small analysis pipeline that treats `g` as a first-class data object.

## Methodology

Freeze the canonical 4-layer Llama-MLP VPD run (`goodfire/spd/runs/s-55ea3f9b`), stream 65,536 token positions through it, dump the gate field, and on the top-4,096 alive components run:

1. Cosine + centered Pearson kernels vs a column-wise row-shuffled null
2. Per-component R² explained by token identity
3. Lagged Pearson r per pair, with per-sequence circular-shift nulls (mean-top-100 statistic, not max)
4. Per-(layer, module) eigenspectra
5. Per-component persistence `P(on_{t+1} | on_t)` and within-sequence row-pattern Jaccard
6. Autointerp on 17 persistent components vs a matched non-persistent control (`gpt-4o-mini` hypothesis + held-out classification)

Gate field is cached as sparse-COO fp16 (135 MB on disk vs 9.6 GB dense).

## Results

See the blogpost. Headlines:

- Cosine spectrum separates from a row-shuffled null for ~500 modes; centered Pearson is the honest headline statistic.
- Median per-component R² explained by token identity is **0.06** — most components are not lexical.
- Q/K coupling peaks at **τ = −1** in L0/L1/L3 and τ = 0 in L2, collapses to the null band by τ = ±2. An earlier draft claiming a depth-dependent τ = −2/−3 peak was a max-fishing artifact.
- L1 `attn.k` has 46 alive components in 5.5 effective dimensions; L3 `mlp.down` has 1,837 in 434.
- 17 persistent components score **0.66** mean held-out balanced accuracy under LLM autointerp vs **0.41** for a matched non-persistent control.

## Configuration

```
# .env
HF_TOKEN=your_token         # streaming Pile from HuggingFace Datasets
WANDB_API_KEY=optional      # only if downloading runs through W&B
OPENAI_API_KEY=optional     # only for the autointerp pipeline (Experiment 6)
```

## Quick start

```bash
git clone https://github.com/aniket-desh/vpd-gate-geometry.git
cd vpd-gate-geometry
git clone --depth 1 https://github.com/goodfire-ai/param-decomp.git external/param-decomp
cd external/param-decomp && uv sync && cd ../..

# Mock smoke test (no model download, validates the pipeline)
uv run python -m vpd_gate_geometry.run_analysis \
    --backend mock --output-dir outputs/gate_geometry/mock_smoke

# Real analysis on the canonical pile-4L VPD run
external/param-decomp/.venv/bin/python -m vpd_gate_geometry.run_analysis \
    --backend repo \
    --run-path runs/pile-4L/model_400000.pth \
    --target-model-path runs/pretrain-4L/files/model_step_99999.pt \
    --cache-gate-matrix outputs/gate_geometry/cache/pile4L.pt \
    --output-dir outputs/gate_geometry/pile4L_v2 \
    --n-batches 16 --batch-size 8 --max-components 4096 --max-lag 6 \
    --lagged-top-k 512 --n-null-runs 6 --null-kind circular --device cuda
```

If you'd rather skip the ~13 min extraction step, the pipeline reads a sparse-COO fp16 cache via `--load-gate-matrix outputs/gate_geometry/cache/pile4L_16x8x512.pt`. The cache file is 135 MB and is produced by passing `--cache-gate-matrix <path>` on a first run.

Tested on a single H200 (144 GB VRAM). End-to-end with null controls: ~6 min after extraction is cached.
