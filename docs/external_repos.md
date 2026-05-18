# External repos and references

Status of every external dependency mentioned in the planning docs.
Probed on 2026-05-18 from this RunPod environment.

| Repo / page | URL | Status | Notes |
| --- | --- | --- | --- |
| `goodfire-ai/param-decomp` | https://github.com/goodfire-ai/param-decomp | shallow-cloned to `external/param-decomp` (125 MB on disk, full size ~954 MB) | commit `74146b5` on `main`. Authoritative source for VPD code, harvest, postprocess, and the canonical paper run pointer. |
| Canonical VPD W&B run | `wandb:goodfire/spd/runs/s-55ea3f9b` | not yet accessed | requires `WANDB_API_KEY` to load. Stored in `.env`. |
| Pile config | `param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml` | inspected in cloned repo | 4-layer Llama-MLP-only, 24 decomposed matrices, 38,912 atoms. |
| SimpleStories config | `param_decomp/experiments/lm/ss_llama_simple_mlp-2L.yaml` | inspected in cloned repo | 2-layer Llama-MLP-only fallback. |
| Postprocess umbrella config | `param_decomp/postprocess/pile.yaml` | inspected in cloned repo | runs harvest → autointerp → attributions → graph-interp → clustering. |
| `goodfire-ai/spd` | https://github.com/goodfire-ai/spd | HTTP 301 → merged into `param-decomp` `spd-paper` branch | README of `param-decomp` confirms this. Not separately cloned. |
| `bartbussmann/nn_decompositions` (vpd_paper branch) | https://github.com/bartbussmann/nn_decompositions/tree/vpd_paper | exists (`vpd_paper` head present), ~65 MB | CLT/PLT comparison. Not cloned in this commit — pull only if Phase 6 wants a baseline. |
| Bilinear MLPs paper | https://arxiv.org/abs/2410.08417 | theoretical reference | no implementation dependency for the spectral baseline. |
| Bilinear Autoencoders (BAE) paper | https://arxiv.org/abs/2510.16820 | theoretical reference | — |
| BAE project page | https://tdooms.github.io/research/bae/ | not fetched in this commit | reachable via WebFetch if needed. |
| `tdooms/bae` (code, as listed in `docs/code.md`) | https://github.com/tdooms/bae | **404 / not found** as of 2026-05-18 | the URL in the planning doc is stale. If we need BAE code for a follow-up, search the BAE project page for the current canonical link, or use the arXiv supplementary materials. |
| Lee Sharkey MATS stream | https://www.matsprogram.org/stream/sharkey | reference only | — |

## What we depend on for the first commit

- `external/param-decomp/` at the shallow-cloned `74146b5` commit.
  We do not import it yet; the `repo` backend in `extract_gates.py`
  imports `param_decomp.models.component_model.ComponentModel` only
  if the user runs `--backend repo`.

## What we deliberately do not pull yet

- `nn_decompositions` (CLT/PLT comparison) — not needed for the
  same-position / residualized / lagged gate-geometry baseline.
- Any BAE code — out of scope for the first post.
