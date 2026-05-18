# vpd_gate_geometry

Spectral / tensor geometry of the VPD causal-importance gate field

    g[layer, batch, token, component] ∈ [0, 1]

produced by Goodfire's `param-decomp` (VPD paper, April 2026).

## What this is

A small analysis pipeline:

```
extract_gates  →  gate_matrix  →  spectral / residualize / lagged  →  plots + summary.json
```

It is **not** a reimplementation of VPD. The point is to treat the
causal-importance gate field as a first-class object and run a few
linear baselines on it: same-position kernel + spectrum, token-identity
residualization, lagged cross-position kernels.

## Backends

`vpd_gate_geometry.extract_gates.iter_gate_batches(cfg)` yields
`GateBatch` records from one of three sources:

- `mock`     — synthetic gates with planted clusters / lagged pair.
- `repo`     — live `ComponentModel.calc_causal_importances`. Requires
               `param_decomp` to be importable; see the readiness
               report for install steps.
- `artifact` — load precomputed harvest artifacts (stub for now).

## Quick start (mock)

```bash
source ../scripts/runpod_activate.sh
uv sync
uv run python -m vpd_gate_geometry.run_analysis \
    --backend mock \
    --output-dir outputs/gate_geometry/mock_smoke \
    --max-components 128 --max-lag 4
```

This populates `outputs/gate_geometry/mock_smoke/` with `summary.json`
and a `plots/` directory containing 8 PNGs.

## File map

| File | Role |
| --- | --- |
| `config.py` | `AnalysisConfig` dataclass + CLI parsing |
| `extract_gates.py` | `GateBatch` + mock/repo/artifact backends |
| `gate_matrix.py` | `ComponentKey` + `GateMatrix` + flattening |
| `spectral.py` | gate stats, alive filter, kernels, eigvals, clustering order |
| `residualize.py` | token-identity baseline + per-component R² |
| `lagged.py` | lagged co-importance kernels + top-pair surfacing |
| `plotting.py` | Goodfire-style rcParams + 8 plot helpers |
| `run_analysis.py` | end-to-end CLI |
