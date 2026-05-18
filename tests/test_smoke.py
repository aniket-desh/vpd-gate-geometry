"""End-to-end mock smoke test for the vpd_gate_geometry pipeline.

Run with:

    uv run pytest tests/

These tests do **not** touch upstream `param_decomp` or W&B; they use the
`mock` backend so they're cheap and side-effect-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vpd_gate_geometry import gate_matrix as gm_mod
from vpd_gate_geometry import lagged, nulls, plotting, residualize, spectral
from vpd_gate_geometry.config import AnalysisConfig
from vpd_gate_geometry.extract_gates import iter_gate_batches
from vpd_gate_geometry.run_analysis import run


def _mock_cfg(tmp_path: Path, **overrides) -> AnalysisConfig:
    cfg = AnalysisConfig(
        backend="mock",
        output_dir=tmp_path / "out",
        max_components=64,
        max_lag=3,
        mock_n_layers=2,
        mock_components_per_module=16,
        mock_batch_size=4,
        mock_seq_len=64,
        mock_vocab_size=128,
        mock_n_batches=4,
        lagged_top_k=32,
        residualize_max_vocab_tokens=64,
        residualize_min_count=4,
        n_null_runs=2,
        null_kind="circular",
        device="cpu",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_pipeline_end_to_end_mock(tmp_path: Path) -> None:
    cfg = _mock_cfg(tmp_path)
    summary = run(cfg)

    # Summary structure
    for key in ("backend", "provenance", "modules", "module_a", "module_b",
                "gate_matrix", "per_module_stats",
                "spectrum_raw_cosine", "spectrum_pearson",
                "spectrum_shuffled_null_cosine", "spectrum_residual_cosine",
                "token_residualization", "lagged", "config"):
        assert key in summary, f"missing {key!r}"

    # Spectrum sanity
    top_eigs = summary["spectrum_raw_cosine"]["top10"]
    assert all(torch.isfinite(torch.tensor(top_eigs)).tolist()), "non-finite eigvals"
    assert top_eigs[0] > 0, "top eigenvalue should be positive"

    # Null path was exercised
    lg = summary["lagged"]
    assert lg["n_null_runs"] == cfg.n_null_runs
    assert lg["null_kind"] == cfg.null_kind
    # real_stats keys are str-converted JSON keys after the round trip…
    # but here `summary` is the in-memory dict, keys are still ints.
    for tau in lg["lags"]:
        s = lg["real_stats"][tau]
        for stat_key in ("max", "p99_9", "mean_top_100", "n_above_0_5"):
            assert stat_key in s
        ns = lg["null_summary"][tau]
        for stat_key in ("max_mean", "max_p95", "mean_top_100_mean"):
            assert stat_key in ns

    # JSON serializability of the summary file
    out_json = json.loads((cfg.output_dir / "summary.json").read_text())
    assert out_json["modules"], "no modules in serialized summary"

    # Plots exist + nonempty
    plot_dir = cfg.output_dir / "plots"
    expected = {
        "01_gate_activity.png",
        "02_spectrum_raw.png",
        "03_kernel_clustered_heatmap.png",
        "04_spectral_embedding.png",
        "05_token_r2.png",
        "06_spectrum_raw_vs_residual.png",
        "07_lag_profile.png",
        "08_top_lagged_pairs.png",
        "09_kernel_variants.png",
    }
    for name in expected:
        p = plot_dir / name
        assert p.is_file(), f"missing plot: {p}"
        assert p.stat().st_size > 1024, f"plot too small: {p}"


def test_provenance_round_trip_through_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.pt"
    cfg1 = _mock_cfg(
        tmp_path / "run1",
        cache_gate_matrix=str(cache_path),
    )
    run(cfg1)
    assert cache_path.exists()
    cached = torch.load(cache_path, weights_only=False)
    assert "provenance" in cached
    assert cached["provenance"]["backend"] == "mock"
    assert cached["provenance"]["n_batches"] == cfg1.mock_n_batches

    cfg2 = _mock_cfg(
        tmp_path / "run2",
        load_gate_matrix=str(cache_path),
    )
    summary2 = run(cfg2)
    # Provenance survives through the cache load.
    assert summary2["provenance"]["backend"] == "mock"
    assert summary2["provenance"]["n_batches"] == cfg1.mock_n_batches


def test_null_preserves_marginals(tmp_path: Path) -> None:
    """Circular shift and column permute must keep each module's marginal
    histogram (per-row totals are preserved up to floating-point noise)."""
    cfg = _mock_cfg(tmp_path)
    batches = list(iter_gate_batches(cfg))
    gm = gm_mod.build_gate_matrix(batches, modules=list(batches[0].gates.keys()))
    target = list(batches[0].gates.keys())[0]

    cols = [i for i, k in enumerate(gm.keys) if k.module_name == target]
    cols_t = torch.tensor(cols, dtype=torch.long)
    real_marginal = gm.G[:, cols_t].sum(dim=1)

    for kind_fn in (nulls.circular_shift_per_sequence_b, nulls.permute_components_b):
        gm_null = kind_fn(gm, target, seed=0)
        null_marginal = gm_null.G[:, cols_t].sum(dim=1).sort().values
        real_sorted = real_marginal.sort().values
        # Either column permute (exact preservation) or circular shift
        # (preserves per-sequence totals, so global sorted totals match exactly).
        assert torch.allclose(null_marginal, real_sorted, atol=1e-5), (
            f"{kind_fn.__name__} should preserve per-row sums"
        )


def test_lagged_real_beats_null(tmp_path: Path) -> None:
    """Mock data plants a τ=+1 cross-module pair. With enough sample, the
    real lagged max |r| at τ=+1 should beat both the τ=+3 noise floor and
    the τ=+1 circular-shift null."""
    cfg = _mock_cfg(
        tmp_path,
        mock_n_batches=24, mock_seq_len=128, mock_vocab_size=32,
        mock_components_per_module=24, n_null_runs=3,
    )
    batches = list(iter_gate_batches(cfg))
    gm = gm_mod.build_gate_matrix(batches, modules=list(batches[0].gates.keys()))
    modules = list(batches[0].gates.keys())
    result = lagged.lagged_kernel_with_nulls(
        gm,
        module_a=modules[0],
        module_b=modules[1],
        lags=list(range(-3, 4)),
        top_k=24,
        normalize=True,
        device="cpu",
        n_null_runs=cfg.n_null_runs,
        null_kind="circular",
    )
    real_p1 = result["real_stats"][1]["max"]
    real_p3 = result["real_stats"][3]["max"]
    null_p1 = result["null_summary"][1]["max_p95"]
    # Planted τ=+1 signal should beat far-lag noise.
    assert real_p1 > real_p3, (
        f"planted τ=+1 signal should beat τ=+3 noise; real_p1={real_p1:.3f} "
        f"real_p3={real_p3:.3f}"
    )
    # And it should beat the circular-shift null at τ=+1.
    assert real_p1 > null_p1, (
        f"planted τ=+1 should beat null; real_max={real_p1:.3f} "
        f"null_max_p95={null_p1:.3f}"
    )


def test_spectral_decompose_returns_finite(tmp_path: Path) -> None:
    G = torch.randn(200, 32)
    K = spectral.cosine_kernel(G)
    eigvals, eigvecs = spectral.spectral_decompose(K)
    assert torch.isfinite(eigvals).all()
    assert torch.isfinite(eigvecs).all()
    assert eigvals.shape == (32,)
    assert eigvecs.shape == (32, 32)
    # Monotone non-increasing.
    diffs = eigvals[:-1] - eigvals[1:]
    assert (diffs >= -1e-5).all()
