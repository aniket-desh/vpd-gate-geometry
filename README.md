# vpd-gate-geometry

Spectral / tensor geometry of the VPD causal-importance gate field
`g[layer, batch, token, component] ∈ [0, 1]` from Goodfire's
[`param-decomp`](https://github.com/goodfire-ai/param-decomp)
(VPD paper, April 2026).

This is not a reimplementation. It's a small analysis pipeline that
treats the gate field as a first-class object and runs a few linear
baselines on it: same-position kernel + spectrum, token-identity
residualization, and lagged cross-position kernels.

## Results

See **[`docs/results.md`](docs/results.md)** for the full quantitative
write-up on the canonical paper run (`goodfire/spd/runs/s-55ea3f9b`).

Headlines, on 65,536 token positions from the canonical Pile stream
(top-4,096 alive subset for the kernel-level claims):

- **The cosine spectrum separates from a row-shuffled null** for its
  first ~500 modes; the rest crosses below the null curve. **Pearson
  (centered) is the honest headline statistic.** Top eigval 151,
  effective rank 1,047. The raw cosine λ₁ = 206 is partly base-rate
  alignment (shuffled-null λ₁ = 150 too).
- **Most components are not lexical.** Median per-component R²
  explained by token identity is 0.06; distribution is
  bimodal-with-tail. Token-residualization spectrum is essentially
  the centered Pearson spectrum.
- **Q/K coupling is one-token-back, not three.** Under a per-sequence
  circular-shift null, the real `mean(top-100 |r|)` for within-layer
  Q→K peaks at **τ = −1** in L0/L1/L3 (excess +0.27 to +0.61 over null)
  and at τ = 0 in L2. The signal collapses to the null band by τ = ±2.
  An earlier draft of this repo claimed "peak deepens to τ = −2/−3
  with depth"; that was a max-fishing artifact and is corrected in
  `docs/results.md`.
- **Layer/module geometries vary by ~80×.** L1 attn.k has 46 alive
  components in 5.5 effective dimensions; L3 mlp.down has 1,837 alive
  in 434 dimensions. No statistical games involved.

Plots and per-pair lag-profile-with-null curves live under
[`outputs/gate_geometry/`](outputs/gate_geometry).

## Quick start

```bash
# 1. Clone scaffold + setup
git clone https://github.com/aniket-desh/vpd-gate-geometry.git
cd vpd-gate-geometry
git clone --depth 1 https://github.com/goodfire-ai/param-decomp.git external/param-decomp
cd external/param-decomp && uv sync && cd ../..

# 2. Download canonical VPD model + target LM
#    (public W&B project: signed-GCS URLs work without an API key)
#    See "Reproducibility" section in docs/results.md for exact URLs.

# 3. Mock smoke test (no real data, just validates the pipeline)
uv run python -m vpd_gate_geometry.run_analysis \
    --backend mock \
    --output-dir outputs/gate_geometry/mock_smoke

# 4. Real analysis on the canonical run
external/param-decomp/.venv/bin/python -m vpd_gate_geometry.run_analysis \
    --backend repo \
    --run-path runs/pile-4L/model_400000.pth \
    --target-model-path runs/pretrain-4L/files/model_step_99999.pt \
    --cache-gate-matrix outputs/gate_geometry/cache/pile4L.pt \
    --output-dir outputs/gate_geometry/pile4L_v2 \
    --n-batches 16 --batch-size 8 \
    --max-components 4096 --max-lag 6 \
    --lagged-top-k 512 \
    --device cuda
```

## Layout

```
vpd_gate_geometry/        analysis package
  config.py               AnalysisConfig + CLI parsing
  extract_gates.py        GateBatch + mock|repo|artifact backends
  gate_matrix.py          ComponentKey + GateMatrix + flatten
  spectral.py             alive filter, GPU cosine kernel, eigvals
  residualize.py          token-identity baseline + R²
  lagged.py               Pearson r per τ, GPU-resident accumulation
  plotting.py             Goodfire-style palette + 8 plot helpers
  run_analysis.py         end-to-end CLI
  per_layer.py            per-(layer,module) spectrum comparison
  sweep_pairs.py          lagged-coimportance over many module pairs

docs/
  theory.md               theoretical framing (LessWrong target)
  code.md                 codebase plan
  writing.md              citations + style
  repo_readiness_report.md  environment + extraction path
  external_repos.md       SPD/CLT/BAE links + status
  results.md              quantitative results

outputs/gate_geometry/    summary.json + plots (gate dumps gitignored)
```

## Hardware

Designed for a single H200 (144 GB VRAM). The whole pipeline keeps
the gate matrix on GPU and routes cosine kernel + eigendecomp +
lagged-cross-product to CUDA.

Wall-clock on H200, after one-time extraction is cached:

- main analysis (no null controls): ~50s end-to-end.
- main analysis with 6 circular-shift null runs: ~6 min.
- 4-pair Q/K sweep with 6 null runs: ~10-15 min including a
  ~2-3 min cache load from the network filesystem.
- 10-pair sweep with null runs: ~25-35 min, dominated by the
  cross-layer MLP→attention pairs whose source modules have
  3,072-3,584 columns.

The 9.6 GB gate matrix is the bottleneck before caching. First
extraction is ~13 min, mostly HF tokenization + forward passes.
