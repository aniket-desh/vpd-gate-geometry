"""Analysis configuration + CLI parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Backend = Literal["mock", "repo", "artifact"]


@dataclass
class AnalysisConfig:
    backend: Backend = "mock"
    output_dir: Path = Path("outputs/gate_geometry/mock_smoke")

    # Common knobs
    max_components: int = 512
    max_lag: int = 8
    alive_threshold: float = 1e-3
    seed: int = 0

    # Mock backend
    mock_n_layers: int = 2
    mock_modules: tuple[str, ...] = ("attn.q_proj", "attn.k_proj", "mlp.c_fc", "mlp.down_proj")
    mock_components_per_module: int = 32
    mock_batch_size: int = 8
    mock_seq_len: int = 128
    mock_vocab_size: int = 512
    mock_n_batches: int = 8

    # Repo backend
    run_path: str = ""               # e.g. "wandb:goodfire/spd/runs/s-55ea3f9b"
    dataset_name: str = "danbraunai/pile-uncopyrighted-tok-shuffled"
    dataset_split: str = "train"
    dataset_column: str = "input_ids"
    n_batches: int = 32
    batch_size: int = 8
    seq_len: int = 256
    device: str = "cuda"
    sampling: str = ""               # overrides run_info.config.sampling if non-empty

    # Artifact backend
    harvest_dir: str = ""            # path to harvest sub-run

    # Residualization
    residualize_max_vocab_tokens: int = 4096
    residualize_min_count: int = 16
    residualize_shrinkage: float = 100.0

    # Lagged kernel restriction
    lagged_top_k: int = 256

    # Plot toggles
    skip_plots: bool = False

    # Resolved at runtime; not from CLI
    extras: dict[str, object] = field(default_factory=dict)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vpd_gate_geometry.run_analysis")
    p.add_argument("--backend", choices=["mock", "repo", "artifact"], default="mock")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/gate_geometry/mock_smoke"))
    p.add_argument("--max-components", type=int, default=512)
    p.add_argument("--max-lag", type=int, default=8)
    p.add_argument("--alive-threshold", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--mock-n-layers", type=int, default=2)
    p.add_argument("--mock-components-per-module", type=int, default=32)
    p.add_argument("--mock-batch-size", type=int, default=8)
    p.add_argument("--mock-seq-len", type=int, default=128)
    p.add_argument("--mock-vocab-size", type=int, default=512)
    p.add_argument("--mock-n-batches", type=int, default=8)

    p.add_argument("--run-path", type=str, default="")
    p.add_argument("--dataset-name", type=str, default="danbraunai/pile-uncopyrighted-tok-shuffled")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--dataset-column", type=str, default="input_ids")
    p.add_argument("--n-batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sampling", type=str, default="")

    p.add_argument("--harvest-dir", type=str, default="")

    p.add_argument("--residualize-max-vocab-tokens", type=int, default=4096)
    p.add_argument("--residualize-min-count", type=int, default=16)
    p.add_argument("--residualize-shrinkage", type=float, default=100.0)

    p.add_argument("--lagged-top-k", type=int, default=256)
    p.add_argument("--skip-plots", action="store_true")
    return p


def config_from_argv(argv: list[str] | None = None) -> AnalysisConfig:
    args = build_arg_parser().parse_args(argv)
    return AnalysisConfig(
        backend=args.backend,
        output_dir=args.output_dir,
        max_components=args.max_components,
        max_lag=args.max_lag,
        alive_threshold=args.alive_threshold,
        seed=args.seed,
        mock_n_layers=args.mock_n_layers,
        mock_components_per_module=args.mock_components_per_module,
        mock_batch_size=args.mock_batch_size,
        mock_seq_len=args.mock_seq_len,
        mock_vocab_size=args.mock_vocab_size,
        mock_n_batches=args.mock_n_batches,
        run_path=args.run_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_column=args.dataset_column,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        sampling=args.sampling,
        harvest_dir=args.harvest_dir,
        residualize_max_vocab_tokens=args.residualize_max_vocab_tokens,
        residualize_min_count=args.residualize_min_count,
        residualize_shrinkage=args.residualize_shrinkage,
        lagged_top_k=args.lagged_top_k,
        skip_plots=args.skip_plots,
    )
