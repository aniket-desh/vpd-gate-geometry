# Experimental and Codebase Plan

## Minimal functional codebase for the VPD gate-geometry project

**Project:** *The Gate Geometry of Parameter Decompositions*

**Goal:** Build a small, reproducible analysis codebase that extracts or loads VPD causal-importance values \(g^\ell_{b,t,c}\), computes gate-geometry diagnostics, and produces Goodfire-style plots for a LessWrong post.

**Non-goal:** Reimplement VPD from scratch. The Goodfire `param-decomp` repo already contains the method, a single-file `nano_param_decomp` implementation, experiment configs, postprocessing, and visualization tooling.

---

## 0. Recommended scope

### Default plan: analyze an existing/canonical VPD run

Use the Goodfire `param-decomp` repo and the canonical paper run exposed in the README:

- canonical paper run: `goodfire/spd/runs/s-55ea3f9b`
- paper experiment config: `pile_llama_simple_mlp-4L`
- target model checkpoint reference in config: `goodfire/spd/runs/t-9d2b8f02`
- dataset: `danbraunai/pile-uncopyrighted-tok-shuffled`
- tokenizer: `EleutherAI/gpt-neox-20b`

This is the best route for a 4--5 day post. You do not need to reproduce a 400k-step VPD training run to make the contribution. You need enough gate statistics to analyze the geometry of VPD's causal-importance field.

### Secondary plan: run the smaller SimpleStories decomposition

Use the repo's smaller experiment:

- experiment: `ss_llama_simple_mlp-2L`
- target model: `wandb:goodfire/spd/runs/gf6rbga0`
- dataset: `SimpleStories/SimpleStories`
- tokenizer: `SimpleStories/test-SimpleStories-gpt2-1.25M`

This is a safer training target if the full Pile model is too slow or if W&B artifacts are inconvenient.

### Stretch plan: exact full VPD run on H200

You have a RunPod with an H200, so you can plausibly run the full config, but you should not make the mini-project depend on completing it. The exact config has:

- 400,000 training steps;
- batch size 64;
- bf16 autocast enabled;
- a global shared transformer causal-importance function with `d_model=2048`, `n_blocks=8`, `n_heads=16`, `max_len=512`;
- persistent PGD reconstruction loss;
- streaming Pile subset;
- 24 decomposed matrices across the 4-layer model.

This may still be expensive, and runtime will depend on repo state, W&B artifact download, dataloader throughput, and whether you run single-GPU or adapt distributed assumptions. Treat full retraining as optional.

---

## 1. Hardware plan

### RunPod H200 environment

Recommended base image:

- CUDA 12.x PyTorch image;
- Python 3.11 or whatever `uv` resolves from the repo;
- enough disk for W&B artifacts, harvested data, and plots;
- persistent volume if possible.

Rough disk allocation:

- repo + dependencies: 5--20 GB;
- W&B/model artifacts: unknown, reserve 50--100 GB;
- harvested gate examples/correlation stats: reserve 50--100 GB for full run, much less for SimpleStories;
- plot outputs: negligible.

### What the H200 is useful for

The H200 is useful for:

1. harvesting gates over many Pile sequences;
2. computing correlations / coactivations at high throughput;
3. optionally training the SimpleStories or full Pile VPD decomposition;
4. running postprocessing over larger batches;
5. storing dense/sparse gate chunks in bf16/fp16.

The spectral analysis itself can usually be done on CPU/GPU depending on component count.

For the full model, there are ~38,912 subcomponents. A full dense \(C\times C\) kernel is about:

\[
38912^2 \approx 1.5\times 10^9
\]

entries. At fp32 this is ~6 GB; at fp16/bf16 ~3 GB. That is possible on H200, but you should usually analyze one layer/module at a time. Per-layer \(C=9728\) gives ~95M entries, much easier.

---

## 2. External code and resources

### Goodfire / VPD

- VPD paper/blog:  
  https://www.goodfire.ai/research/interpreting-lm-parameters
- VPD explainer:  
  https://www.goodfire.ai/research/vpd-explainer
- Goodfire `param-decomp` repo:  
  https://github.com/goodfire-ai/param-decomp
- Paper config:  
  https://github.com/goodfire-ai/param-decomp/blob/main/param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml
- SimpleStories config:  
  https://github.com/goodfire-ai/param-decomp/blob/main/param_decomp/experiments/lm/ss_llama_simple_mlp-2L.yaml
- Postprocessing config:  
  https://github.com/goodfire-ai/param-decomp/blob/main/param_decomp/postprocess/pile.yaml
- Harvest module docs:  
  https://github.com/goodfire-ai/param-decomp/blob/main/param_decomp/harvest/CLAUDE.md
- Comparison CLT/PLT repo from Goodfire README:  
  https://github.com/bartbussmann/nn_decompositions/tree/vpd_paper

### SPD predecessor

- SPD paper:  
  https://arxiv.org/abs/2506.20790
- SPD branch is now in the same repo:  
  https://github.com/goodfire-ai/param-decomp/tree/spd-paper

### Bilinear / polynomial bridge

- Bilinear MLPs paper:  
  https://arxiv.org/abs/2410.08417
- ICLR page:  
  https://iclr.cc/virtual/2025/poster/28827
- Earlier bilinear decomposition paper:  
  https://arxiv.org/abs/2406.03947
- Lee Sharkey bilinear technical note:  
  https://arxiv.org/abs/2305.03452
- Bilinear Autoencoders paper:  
  https://arxiv.org/abs/2510.16820
- BAE project page:  
  https://tdooms.github.io/research/bae/
- BAE code:  
  https://github.com/tdooms/bae
- Chi-net / compositional interpretable models:  
  https://arxiv.org/abs/2504.02667

---

## 3. Exact VPD paper details to mirror

From the VPD paper/blog and public config:

### Target LM architecture

- decoder-only transformer;
- 4 layers;
- residual dimension 768;
- MLP intermediate dimension 3072;
- 6 attention heads;
- head dimension 128;
- context length 512;
- vocabulary size 50,277;
- RoPE positional encoding;
- RMSNorm;
- GELU activation;
- standard multi-head attention;
- tied embeddings;
- ~28M non-embedding parameters;
- ~67M total parameters including embedding;
- trained on an uncopyrighted subset of The Pile.

### Decomposition scale

- 24 weight matrices decomposed;
- embedding/unembedding omitted;
- 38,912 total rank-one subcomponents;
- ~9,972 alive subcomponents with mean causal importance greater than \(10^{-6}\);
- average \(L_0\) of about 205 subcomponents per sequence position;
- about 2.1% of alive subcomponents active per sequence position.

### Per-layer subcomponent summary from paper

| Layer | C | Alive | Mean L0 | L0 / Alive |
|---|---:|---:|---:|---:|
| 0 | 9728 | 3709 | 44.6 | 0.012 |
| 1 | 9728 | 848 | 18.9 | 0.022 |
| 2 | 9728 | 1943 | 49.5 | 0.025 |
| 3 | 9728 | 3472 | 92.0 | 0.026 |
| Total | 38912 | 9972 | 205.0 | 0.021 |

### Config module capacities

From `pile_llama_simple_mlp-4L.yaml`, each layer has these module patterns and subcomponent counts:

| Module pattern | C per matching module |
|---|---:|
| `h.*.mlp.c_fc` | 3072 |
| `h.*.mlp.down_proj` | 3584 |
| `h.*.attn.q_proj` | 512 |
| `h.*.attn.k_proj` | 512 |
| `h.*.attn.v_proj` | 1024 |
| `h.*.attn.o_proj` | 1024 |

Per layer total: \(3072+3584+512+512+1024+1024=9728\). Across 4 layers: \(38912\).

### Config training choices

From the public config:

```yaml
wandb_project: param-decomp
seed: 0
autocast_bf16: true
n_mask_samples: 1
ci_config:
  mode: global
  fn_type: global_shared_transformer
  simple_transformer_ci_cfg:
    d_model: 2048
    n_blocks: 8
    mlp_hidden_dim: [8192]
    attn_config:
      n_heads: 16
      max_len: 512
      rope_base: 10000.0
sampling: continuous
sigmoid_type: leaky_hard
use_delta_component: true
steps: 400000
batch_size: 64
lr_schedule:
  start_val: 5.0e-05
  final_val_frac: 0.1
  fn_type: cosine
pretrained_model_class: param_decomp.pretrain.models.llama_simple_mlp.LlamaSimpleMLP
pretrained_model_name: goodfire/spd/runs/t-9d2b8f02
tokenizer_name: EleutherAI/gpt-neox-20b
task_config:
  task_name: lm
  max_seq_len: 512
  buffer_size: 1000
  dataset_name: danbraunai/pile-uncopyrighted-tok-shuffled
  column_name: input_ids
  train_data_split: train
  eval_data_split: val
  is_tokenized: true
  streaming: true
```

### Loss terms in the config

The exact config includes:

- `ImportanceMinimalityLoss`, coefficient `0.0002`, with p-annealing final p `0.4`;
- `StochasticReconSubsetLoss`, coefficient `0.5`;
- `PersistentPGDReconLoss`, coefficient `0.5`;
- `FaithfulnessLoss`, coefficient `10000000.0`;
- eval metrics including `CIHistograms`, `ComponentActivationDensity`, `CI_L0`, `CEandKLLosses`, `CIMeanPerComponent`, `StochasticHiddenActsReconLoss`, `CIHiddenActsReconLoss`, and `PGDReconLoss` with 20 steps.

For the mini-project, these are mostly provenance details. The analysis should reuse their trained run rather than alter these losses.

---

## 4. Proposed local codebase

Create a separate lightweight analysis repo so your post is not buried inside a fork of Goodfire's codebase.

```text
vpd-gate-geometry/
  README.md
  pyproject.toml
  configs/
    pile_canonical.yaml
    simplestories.yaml
    plot_style.yaml
  scripts/
    00_check_goodfire_install.sh
    01_harvest_or_load.py
    02_extract_gate_matrix.py
    03_compute_kernels.py
    04_token_residualize.py
    05_lagged_kernels.py
    06_cluster_components.py
    07_make_figures.py
    08_export_post_assets.py
  src/vpd_gate_geometry/
    __init__.py
    io.py
    gates.py
    kernels.py
    residualization.py
    lagged.py
    clustering.py
    plotting.py
    reports.py
    types.py
  notebooks/
    00_sanity_check.ipynb
    01_gate_spectra.ipynb
    02_token_residualization.ipynb
    03_lagged_coimportance.ipynb
    04_case_studies.ipynb
  data/
    README.md
    raw/                  # gitignored
    processed/            # gitignored
  figures/
    raw/
    final/
  posts/
    lesswrong_draft.md
```

### Dependencies

Use the Goodfire repo for VPD-specific loading/harvesting. Your analysis repo can depend on:

```toml
[project]
name = "vpd-gate-geometry"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "torch",
  "numpy",
  "scipy",
  "pandas",
  "scikit-learn",
  "matplotlib",
  "seaborn",
  "einops",
  "tqdm",
  "pyyaml",
  "wandb",
  "datasets",
  "transformers",
  "networkx",
  "plotly",
  "kaleido",
  "orjson",
]
```

Use `seaborn` for Goodfire-like clean heatmaps and line plots. Use Plotly only if you want interactive HTML appendices.

---

## 5. Installation and run setup

### 5.1 Clone Goodfire repo

```bash
mkdir -p ~/research
cd ~/research
git clone https://github.com/goodfire-ai/param-decomp.git
cd param-decomp
make install-dev
```

If `make install-dev` fails because of system differences, try:

```bash
pip install -e .
```

or use `uv` if the repo expects it.

### 5.2 Configure W&B

```bash
wandb login
```

Set output directories:

```bash
export PARAM_DECOMP_OUT_DIR=/workspace/param_decomp_outputs
mkdir -p $PARAM_DECOMP_OUT_DIR
```

### 5.3 Verify Goodfire experiment registry

```bash
pd-local --help
pd-run --help
pd-postprocess --help
pd-harvest --help
```

Then inspect registered experiments:

```bash
python - <<'PY'
from param_decomp.registry import EXPERIMENTS
print('\n'.join(sorted(EXPERIMENTS.keys())))
PY
```

If `EXPERIMENTS` import path differs, search:

```bash
rg "pile_llama_simple_mlp" param_decomp
rg "ss_llama_simple_mlp" param_decomp
```

---

## 6. Data extraction: what exactly we need

For each analyzed batch/sequence chunk, we need:

```python
GateBatch = {
    "input_ids": LongTensor[B, T],
    "tokens": optional list[list[str]],
    "position": LongTensor[T],
    "module_name": str,
    "layer": int,
    "component_ids": LongTensor[C],
    "gates": FloatTensor[B, T, C],  # causal importances
}
```

For full analysis, we want a standardized long-form or chunked format:

```text
gates/
  layer_0_attn_q.pt       # maybe sparse/chunked
  layer_0_attn_k.pt
  layer_0_attn_v.pt
  layer_0_attn_o.pt
  layer_0_mlp_fc.pt
  layer_0_mlp_down.pt
  ...
metadata/
  token_ids.pt
  token_strings.jsonl
  sequence_offsets.parquet
  component_table.parquet
```

### Dense vs sparse storage

Dense gates can be huge. Use sparse/top-k storage when possible.

Option A: dense chunked tensors:

```text
processed/gates_dense/layer_2_mlp_down/chunk_00000.pt
processed/gates_dense/layer_2_mlp_down/chunk_00001.pt
```

Option B: sparse COO per chunk:

```python
{
  "indices": LongTensor[nnz, 3],  # b,t,c within chunk
  "values": FloatTensor[nnz],
  "shape": (B, T, C),
  "threshold": 1e-6,
}
```

Option C: only store sufficient statistics:

- \(G^\top G\);
- \(\sum_n G_{n,c}\);
- \(\sum_n G_{n,c}^2\);
- token-conditioned sums;
- lagged cross-products.

For the post, sufficient statistics may be enough. But for qualitative examples, store top activation examples per component/cluster.

---

## 7. Stage 1: harvest or load gates

### 7.1 Use Goodfire harvest if possible

Goodfire's harvest pipeline already collects:

- activation examples;
- correlations;
- token statistics;
- component-level data;
- SQLite `harvest.db`;
- `component_correlations.pt`;
- `token_stats.pt`.

Run, depending on config availability:

```bash
cd ~/research/param-decomp
pd-postprocess param_decomp/postprocess/pile.yaml
```

or directly:

```bash
pd-harvest path/to/harvest_config.yaml
```

Non-SLURM single-GPU usage from their docs:

```bash
python -m param_decomp.harvest.scripts.run_worker \
  --config_json '{"method_config": {"wandb_path": "goodfire/spd/runs/s-55ea3f9b"}, "n_batches": 1000}'
```

The exact JSON schema may need inspection in `param_decomp/harvest/config.py`. Build `scripts/01_harvest_or_load.py` to create a valid config programmatically.

### 7.2 If full gate tensors are not exposed

The harvest module may expose component correlations and token stats without raw gates. That is still enough for a version of the project:

- use `component_correlations.pt` as same-position kernel;
- use `token_stats.pt` for token-identity analysis;
- skip lagged kernels unless raw/sequential gate examples are available;
- write the post as a partial first look.

But try to get raw \(g\), because lagged co-importance is the sharpest novel piece.

### 7.3 Extraction pseudocode

```python
# scripts/02_extract_gate_matrix.py
from pathlib import Path
import torch
from tqdm import tqdm

from vpd_gate_geometry.io import load_vpd_run, iter_gate_batches, save_gate_chunk


def main(cfg):
    run = load_vpd_run(cfg.wandb_path, cfg.goodfire_repo)
    for module_name in cfg.modules:
        out_dir = Path(cfg.out_dir) / "gates" / module_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(tqdm(iter_gate_batches(run, module_name, cfg.n_batches))):
            # batch.gates: [B,T,C]
            save_gate_chunk(out_dir / f"chunk_{i:05d}.pt", batch, threshold=cfg.threshold)

if __name__ == "__main__":
    ...
```

You will probably need to adapt this to Goodfire's actual object model. The point of the mini-project codebase is to isolate that repo-specific logic in `io.py`, so the analysis code is clean.

---

## 8. Stage 2: same-position kernels

### 8.1 Sufficient statistics

For each module/layer:

\[
S = G^\top G,
\qquad
\mu = \sum_n G_n,
\qquad
q = \sum_n G_n\odot G_n.
\]

Then cosine kernel:

\[
K_{c,c'}=\frac{S_{c,c'}}{\sqrt{q_cq_{c'}}+\epsilon}.
\]

Centered covariance:

\[
\Sigma=\frac{1}{N}G^\top G - \bar G^\top\bar G.
\]

Correlation:

\[
R_{c,c'}=\frac{\Sigma_{c,c'}}{\sigma_c\sigma_{c'}+\epsilon}.
\]

### 8.2 Implementation sketch

```python
# src/vpd_gate_geometry/kernels.py
import torch

@torch.no_grad()
def accumulate_gram(chunks, device="cuda", dtype=torch.float32):
    gram = None
    sums = None
    sq_sums = None
    n_total = 0

    for G in chunks:  # [B,T,C] or [N,C]
        G = G.reshape(-1, G.shape[-1]).to(device=device, dtype=dtype)
        n_total += G.shape[0]
        gram_i = G.T @ G
        sums_i = G.sum(dim=0)
        sq_i = (G * G).sum(dim=0)
        gram = gram_i if gram is None else gram + gram_i
        sums = sums_i if sums is None else sums + sums_i
        sq_sums = sq_i if sq_sums is None else sq_sums + sq_i

    return {"gram": gram.cpu(), "sums": sums.cpu(), "sq_sums": sq_sums.cpu(), "n": n_total}


def cosine_from_stats(stats, eps=1e-8):
    q = stats["sq_sums"]
    denom = torch.sqrt(torch.outer(q, q)).clamp_min(eps)
    return stats["gram"] / denom


def corr_from_stats(stats, eps=1e-8):
    n = stats["n"]
    gram = stats["gram"] / n
    mean = stats["sums"] / n
    cov = gram - torch.outer(mean, mean)
    var = cov.diag().clamp_min(eps)
    return cov / torch.sqrt(torch.outer(var, var))
```

### 8.3 Outputs

For each module/layer:

```text
processed/kernels/layer_2_mlp_down_cosine.pt
processed/kernels/layer_2_mlp_down_corr.pt
processed/kernels/layer_2_mlp_down_stats.pt
figures/final/layer_2_mlp_down_spectrum.pdf
figures/final/layer_2_mlp_down_clustered_heatmap.pdf
```

---

## 9. Stage 3: token residualization

### 9.1 Data needed

- input token IDs \([B,T]\);
- gate matrix \([B,T,C]\);
- vocabulary size or observed token set;
- minimum token count threshold.

### 9.2 Token baseline

Compute token-conditioned means with shrinkage:

```python
# src/vpd_gate_geometry/residualization.py
import torch

@torch.no_grad()
def token_baseline(chunks, token_chunks, vocab_size, C, shrinkage=100.0, device="cuda"):
    global_sum = torch.zeros(C, device=device)
    global_count = 0
    token_sum = torch.zeros(vocab_size, C, device=device)
    token_count = torch.zeros(vocab_size, device=device)

    for G, toks in zip(chunks, token_chunks):
        G = G.reshape(-1, C).to(device)
        z = toks.reshape(-1).to(device)
        global_sum += G.sum(dim=0)
        global_count += G.shape[0]
        token_sum.index_add_(0, z, G)
        token_count.index_add_(0, z, torch.ones_like(z, dtype=torch.float32))

    mu = global_sum / global_count
    raw_token_mean = token_sum / token_count.clamp_min(1)[:, None]
    shrink = token_count / (token_count + shrinkage)
    token_offset = shrink[:, None] * (raw_token_mean - mu[None, :])
    return mu.cpu(), token_offset.cpu(), token_count.cpu()


def residualize_chunk(G, toks, mu, token_offset):
    return G - mu.to(G.device)[None, None, :] - token_offset.to(G.device)[toks]
```

### 9.3 Explained variance

```python
@torch.no_grad()
def explained_variance(chunks, token_chunks, mu, token_offset):
    sse = 0.0
    sst = 0.0
    for G, toks in zip(chunks, token_chunks):
        pred = mu.to(G.device)[None, None, :] + token_offset.to(G.device)[toks]
        resid = G - pred
        centered = G - mu.to(G.device)[None, None, :]
        sse += (resid ** 2).sum(dim=(0,1)).cpu()
        sst += (centered ** 2).sum(dim=(0,1)).cpu()
    return 1 - sse / sst.clamp_min(1e-8)
```

### 9.4 Outputs

- histogram of \(R^2_c\);
- scatter: mean causal importance vs token-explained \(R^2\);
- raw vs residualized spectra;
- raw vs residualized top-edge overlap;
- raw vs residualized cluster heatmap.

---

## 10. Stage 4: lagged co-importance kernels

### 10.1 Core computation

For modules A and B and lag \(\tau\):

\[
S^{A,B}(\tau)=\sum_{b,t} G^A_{b,t,:}\otimes G^B_{b,t+\tau,:}.
\]

Implementation:

```python
# src/vpd_gate_geometry/lagged.py
import torch

@torch.no_grad()
def lagged_gram(GA, GB, tau: int):
    # GA: [B,T,Ca], GB: [B,T,Cb]
    if tau >= 0:
        A = GA[:, :GA.shape[1]-tau, :]
        B = GB[:, tau:, :]
    else:
        A = GA[:, -tau:, :]
        B = GB[:, :GB.shape[1]+tau, :]
    A = A.reshape(-1, A.shape[-1])
    B = B.reshape(-1, B.shape[-1])
    return A.T @ B
```

For large modules, compute top edges without materializing all module pairs. For example:

- restrict to alive components;
- restrict to components with support above threshold;
- analyze per module pair;
- use fp16/bf16 accumulation with fp32 final correction;
- use blockwise multiplication.

### 10.2 Lags to test

Start with:

```python
lags = list(range(-8, 9))
```

If expensive:

```python
lags = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
```

For attention, negative/positive convention must be stated clearly. For a destination token \(t\), source tokens usually live at earlier positions \(t-\delta\). Decide and document:

- \(K^{A\to B}(\tau)=\operatorname{corr}(A_t,B_{t+\tau})\);
- if \(\tau<0\), B is earlier than A.

### 10.3 Module pairs worth prioritizing

Priority 1:

- `attn.q_proj` vs `attn.k_proj` within the same layer;
- `attn.o_proj` vs `attn.v_proj` within the same layer.

Priority 2:

- layer \(\ell\) MLP down vs layer \(\ell+1\) attention Q/K;
- layer \(\ell\) attention output vs layer \(\ell+1\) MLP c_fc/down.

Priority 3:

- all module-pair summary over alive components only.

### 10.4 Outputs

- lag profile plot for top 20 cross-position pairs;
- heatmap: module-pair vs best nonzero lag score;
- distribution: max \(|K(\tau)|\) at \(\tau=0\) vs \(\tau\neq0\);
- case study: one pair with strong nonzero lag and interpretable token examples.

---

## 11. Stage 5: clustering

### 11.1 Methods

Start with three clusterers:

1. threshold graph connected components;
2. agglomerative clustering on \(1-K\);
3. spectral clustering on top eigenvectors.

Keep only one in the final post unless comparing is interesting.

### 11.2 Graph clustering sketch

```python
import networkx as nx
import numpy as np


def threshold_components(K, threshold=0.25, min_size=3):
    K = K.copy()
    np.fill_diagonal(K, 0)
    rows, cols = np.where(K > threshold)
    G = nx.Graph()
    G.add_nodes_from(range(K.shape[0]))
    G.add_edges_from((int(i), int(j)) for i, j in zip(rows, cols) if i < j)
    comps = [sorted(c) for c in nx.connected_components(G) if len(c) >= min_size]
    return sorted(comps, key=len, reverse=True)
```

### 11.3 Cluster report format

For each cluster:

```markdown
### Cluster 7: layer 2 MLP down, 14 subcomponents

- mean gate mass: ...
- top tokens by PMI: ...
- top contexts: ...
- top subcomponents: ...
- raw cluster survives token residualization? yes/no
- strongest lagged partners: ...
- tentative interpretation: ...
```

This can be generated automatically into `reports/cluster_cards.md`.

---

## 12. Stage 6: plots

### 12.1 Goodfire-style visual aesthetic

Goodfire's posts tend to use:

- clean off-white or white backgrounds;
- high whitespace;
- thin axes;
- muted but distinct accent colors;
- small multiples;
- annotated examples;
- interactive or semi-interactive visualizations where useful;
- text labels directly on plots rather than dense legends;
- simple color semantics: orange/purple/blue/green accents, not rainbow unless heatmap.

Your static plot style:

```python
# src/vpd_gate_geometry/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns

GOODFIRE_RC = {
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "axes.facecolor": "#fbfaf7",
    "figure.facecolor": "#fbfaf7",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.frameon": False,
}

def set_style():
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update(GOODFIRE_RC)
```

Optional palette:

```python
PALETTE = {
    "raw": "#6C5CE7",          # muted purple
    "residual": "#00A6A6",     # teal
    "token": "#E76F51",        # orange/red
    "lagged": "#2A9D8F",       # green/teal
    "baseline": "#8D99AE",     # gray-blue
}
```

If you want a closer Goodfire visual feel, use less saturated colors and avoid default matplotlib blue/orange.

### 12.2 Required figures

Minimum final post figures:

1. **Schematic:** VPD gives rank-one atoms plus gate field. You analyze the gate field.
2. **Spectrum:** eigenvalue spectrum of raw co-importance kernel for 2--4 modules/layers.
3. **Raw vs residualized:** token residualization changes the kernel spectrum and/or heatmap.
4. **Lagged co-importance:** heatmap or line plot showing strongest nonzero-lag pairs.
5. **Cluster card:** qualitative examples for one cluster or pair.

### 12.3 Nice-to-have figures

- component support histogram;
- mean gate activity by layer/module;
- token-explained variance histogram;
- graph of top lagged component pairs;
- prefix/importance curves;
- module-pair summary matrix.

---

## 13. Experiment menu

### Experiment 0: sanity checks

Purpose: ensure the extracted gates match paper-scale statistics.

Compute:

- alive component counts using mean \(g>10^{-6}\);
- mean \(L_0\) per token;
- per-layer totals;
- histogram of \(g\) values;
- distribution of component support.

Expected approximate target for full Pile run:

- ~9,972 alive;
- ~205 active per token;
- layerwise L0 roughly 44.6 / 18.9 / 49.5 / 92.0.

If not close, debug extraction or thresholding.

### Experiment 1: gate spectra

For each layer and module:

- compute cosine/correlation kernel;
- eigenvalue spectrum;
- effective rank:

\[
r_{\mathrm{eff}}=\exp\left(-\sum_i p_i\log p_i\right),
\quad p_i=\lambda_i/\sum_j\lambda_j;
\]

- participation ratio:

\[
\mathrm{PR}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
\]

Plots:

- spectrum by module;
- effective rank by layer/module;
- heatmap for one clean module.

### Experiment 2: token identity baseline

For each module/layer:

- fit token-only baseline;
- compute per-component \(R^2\);
- compute residualized kernel;
- compare raw/residualized spectra;
- compare top edges.

Plots:

- \(R^2\) histogram;
- mean activity vs \(R^2\) scatter;
- raw/residualized spectrum overlay;
- raw/residualized clustered heatmaps.

### Experiment 3: lagged co-importance

For selected module pairs:

- compute lagged correlations for \(\tau\in[-8,8]\);
- identify top pairs by max nonzero-lag score;
- compare \(\tau=0\) and \(\tau\neq0\);
- extract token examples where both components are active at the corresponding lag.

Plots:

- top lag profiles;
- module-pair max-lag matrix;
- top pair context examples.

### Experiment 4: ranking / variable-resolution proxy

Rank components by:

- mean gate mass;
- residualized gate energy;
- spectral centrality;
- cross-lag centrality.

Plot prefix curves:

- gate mass captured;
- residualized energy captured;
- top-edge coverage;
- optional reconstruction KL if implementable.

This is optional but good for connecting to BAE's analytic importance ordering.

### Experiment 5: qualitative case study

Pick the best cluster/pair from Experiments 1--3.

For each, show:

- component labels if autointerp labels exist;
- top contexts where all/most cluster members fire;
- top tokens by PMI;
- whether raw cluster survives residualization;
- lag relation if relevant;
- tentative mechanism story.

This is what makes the post feel like interpretability rather than pure linear algebra.

---

## 14. Concrete day-by-day plan

### Day 1: setup + data access

- Clone Goodfire repo.
- Verify install.
- Authenticate W&B.
- Load canonical run or SimpleStories run.
- Run harvest/postprocess or inspect existing artifacts.
- Extract one small module's gates or correlations.
- Reproduce basic paper stats for that module.

Deliverable: `notebooks/00_sanity_check.ipynb` and one printed table.

### Day 2: kernels + spectra

- Implement `kernels.py`.
- Compute same-position kernels for 2--4 modules.
- Plot spectra and heatmaps.
- Choose the most visually/technically interesting module.

Deliverable: 2 draft figures.

### Day 3: token residualization

- Implement token baseline.
- Compute \(R^2\) histograms.
- Compare raw/residualized spectra and clusters.
- Identify clusters that survive residualization.

Deliverable: 2 draft figures and one paragraph result.

### Day 4: lagged kernels

- Implement lagged kernels for priority module pairs.
- Compute top nonzero-lag pairs.
- Generate lag profile plots.
- Extract qualitative examples.

Deliverable: main novel plot.

### Day 5: write + polish

- Draft LessWrong post.
- Add caveats.
- Export final figures.
- Add citation section.
- Post draft to trusted reviewers or keep as local draft for comments.

Deliverable: `posts/lesswrong_draft.md`.

---

## 15. Fallback plans

### Fallback A: no raw gates, only harvested correlations

Write:

> I analyze the coactivation statistics already produced by Goodfire's harvest pipeline and identify what kinds of structure would be needed for lagged analysis.

Experiments possible:

- same-position spectra;
- token stats;
- cluster cards;
- comparison to VPD's clustering limitation.

Missing:

- true lagged kernels;
- full residualized kernels unless token-conditioned component data is rich enough.

### Fallback B: full Pile run too hard

Use SimpleStories.

The post becomes:

> I built a gate-geometry diagnostic on the smaller public SimpleStories VPD run, with a path to reproducing on the 67M Pile run.

This is still useful if the analysis is clean.

### Fallback C: Goodfire artifacts are difficult

Train a tiny VPD decomposition using `nano_param_decomp` or a small registered experiment. Then make the post explicitly about the diagnostic, not the model.

### Fallback D: all code is hard

Run only on Goodfire's existing `component_correlations.pt` and write a narrower post:

> What should VPD clustering optimize? A first look at same-position gate coactivation.

Not ideal, but better than no post.

---

## 16. Expected final claims by result type

### If raw and residualized clusters are clean

Claim:

> VPD causal-importance fields contain low-dimensional structure beyond token identity. This suggests post-hoc clustering can be improved by treating gates as a geometric object.

### If raw clusters vanish after residualization

Claim:

> A large part of VPD gate geometry is lexical. Future clustering should separate token-support structure from contextual mechanism structure.

### If lagged structure is strong

Claim:

> Same-position clustering misses a meaningful part of parameter-subcomponent organization. Lagged co-importance is a simple diagnostic for cross-position circuits.

### If lagged structure is weak

Claim:

> Cross-position circuits may not be visible in marginal gate correlations, so attribution graphs or conditional/causal statistics are necessary.

### If ranking curves are strong

Claim:

> Simple global gate-derived rankings already give variable-resolution descriptions of the decomposition. This is a lightweight precursor to learned continuous cut-off scales.

---

## 17. Reproducibility checklist

Before posting, include:

- repo commit hash for Goodfire `param-decomp`;
- W&B run path;
- model/dataset config;
- number of batches/sequences analyzed;
- threshold for alive/active gates;
- kernel type;
- residualization shrinkage parameter;
- lags analyzed;
- modules included/excluded;
- random seeds;
- plot generation scripts;
- known deviations from the VPD paper setup.

---

## 18. README skeleton

```markdown
# VPD Gate Geometry

This repo contains analysis code for the LessWrong post
"The Gate Geometry of Parameter Decompositions".

The project analyzes the causal-importance field produced by Goodfire's
adVersarial Parameter Decomposition (VPD):

    g[layer, batch, token, component]

We ask whether this gate field contains mechanism-level structure by computing:

1. same-position component co-importance kernels;
2. token-identity residualized gate geometry;
3. lagged co-importance kernels for cross-position circuits;
4. simple component importance rankings.

## Setup

```bash
git clone https://github.com/goodfire-ai/param-decomp.git
cd param-decomp
make install-dev
wandb login
```

Then install this repo:

```bash
git clone <this repo>
cd vpd-gate-geometry
pip install -e .
```

## Data

By default, this code expects a Goodfire W&B run path such as:

    goodfire/spd/runs/s-55ea3f9b

See `configs/pile_canonical.yaml`.

## Reproduce figures

```bash
python scripts/01_harvest_or_load.py --config configs/pile_canonical.yaml
python scripts/02_extract_gate_matrix.py --config configs/pile_canonical.yaml
python scripts/03_compute_kernels.py --config configs/pile_canonical.yaml
python scripts/04_token_residualize.py --config configs/pile_canonical.yaml
python scripts/05_lagged_kernels.py --config configs/pile_canonical.yaml
python scripts/07_make_figures.py --config configs/pile_canonical.yaml
```
```

---

## 19. Practical recommendation

For the actual 4--5 day sprint:

1. Do **not** start by retraining the 67M VPD run.
2. First load/harvest the canonical run or the SimpleStories run.
3. Make the same-position kernel and token-residualization plots.
4. Only then attempt lagged kernels.
5. Include full-run H200 training as an appendix / reproducibility extension, not as the dependency of the post.

The mini-project should be judged by whether it produces a clear diagnostic and one surprising result, not by whether it reruns Goodfire's whole paper.

