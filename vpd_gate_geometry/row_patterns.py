"""Row-pattern vocabulary of the gate field.

Each token position $n$ has an *active set*
$S_n = \\{c : g_{n,c} > \\tau\\} \\subseteq [C]$. With $\\tau = 0.5$ the
typical $|S_n|$ is around 150 components out of 38,912. This module
asks: how many distinct $S_n$ patterns does the model actually use,
and how is the row-pattern vocabulary distributed?

We report:

- the number of *unique* active sets across the 65,536 token positions;
- the frequency distribution of patterns (the most-shared support, the
  rate of singleton patterns, the cumulative coverage);
- whether neighbouring tokens share more support than random pairs
  (mean Jaccard at lag 1 vs at lag = uniform over the sequence).

The patterns are hashed at the byte level on a CPU-packed bitmask, so
there's no quadratic memory blowup. Computing the row-row Jaccard
matrix on 65k × 65k would be 17 GB; we don't materialize it.

CLI:

    python -m vpd_gate_geometry.row_patterns \\
        --cache outputs/gate_geometry/cache/pile4L_16x8x512.pt \\
        --output-dir outputs/gate_geometry/row_patterns
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import cache_io, plotting
from .lagged import _sequence_views


def _hash_row_supports(B: np.ndarray) -> list[bytes]:
    """Hash each row of a boolean array to a fixed-size digest.

    `B`: bool numpy array of shape [N, C]. We hash the raw byte
    representation of each row so that two rows with identical support
    sets produce the same digest. SHA-1 chosen for speed + small
    output; we do not need cryptographic strength here.
    """
    # Pack the bool array to bytes once. Numpy already gives one byte per bool.
    rows = B.view(np.uint8)
    return [hashlib.sha1(rows[i].tobytes()).digest() for i in range(rows.shape[0])]


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard between two boolean rows. Cheap."""
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vpd_gate_geometry.row_patterns")
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binary 'on' threshold for g; default 0.5.")
    p.add_argument("--alive-only", action="store_true",
                   help="Hash patterns only over alive components (mean g > alive-threshold).")
    p.add_argument("--alive-threshold", type=float, default=1e-4)
    p.add_argument("--n-random-pairs", type=int, default=20_000,
                   help="Random row-pairs to estimate baseline Jaccard.")
    p.add_argument("--n-adjacent-pairs", type=int, default=20_000,
                   help="Within-sequence adjacent pairs to estimate t-vs-(t+1) Jaccard.")
    args = p.parse_args(argv)

    print(f"[row_patterns] loading {args.cache} ...", flush=True)
    gm, _meta = cache_io.load_gate_matrix_cache(args.cache)
    G = gm.G
    print(f"[row_patterns] G={tuple(G.shape)}, threshold={args.threshold}", flush=True)

    # Optionally restrict to alive columns to keep hashing cheaper and the
    # patterns interpretable (dead components are always 0 anyway).
    if args.alive_only:
        col_keep = (G.mean(dim=0) > args.alive_threshold)
        G_for_hash = G[:, col_keep]
        n_alive = int(col_keep.sum().item())
        print(f"[row_patterns] hashing over {n_alive:,} alive columns "
              f"(of {G.shape[1]:,})", flush=True)
    else:
        G_for_hash = G

    B = (G_for_hash > args.threshold).cpu().numpy().astype(bool)
    N, Ceff = B.shape
    avg_L0 = float(B.sum(axis=1).mean())
    print(f"[row_patterns] N={N:,}  Ceff={Ceff:,}  avg |S_n|={avg_L0:.1f}", flush=True)

    print(f"[row_patterns] hashing {N:,} rows ...", flush=True)
    digests = _hash_row_supports(B)
    counts = Counter(digests)
    n_unique = len(counts)
    n_singletons = sum(1 for v in counts.values() if v == 1)
    most_common = counts.most_common(8)
    n_patterns_covering_50pct = next(
        i + 1 for i, _ in enumerate(
            np.cumsum(sorted(counts.values(), reverse=True))
        ) if np.cumsum(sorted(counts.values(), reverse=True))[i] >= 0.5 * N
    )

    # Within-sequence adjacent Jaccard vs random-row Jaccard.
    views = _sequence_views(gm)
    rng = np.random.default_rng(0)
    adj_pairs: list[float] = []
    for _, s, e in views:
        for t in range(s, e - 1):
            adj_pairs.append((s, t, t + 1))
    rng.shuffle(adj_pairs)
    adj_pairs = adj_pairs[: args.n_adjacent_pairs]
    adj_jaccards = [_jaccard(B[i], B[j]) for (_, i, j) in adj_pairs]

    rand_jaccards = []
    for _ in range(args.n_random_pairs):
        i = int(rng.integers(0, N))
        j = int(rng.integers(0, N))
        if i == j:
            continue
        rand_jaccards.append(_jaccard(B[i], B[j]))

    summary = {
        "config": {
            "threshold": args.threshold,
            "alive_only": args.alive_only,
            "alive_threshold": args.alive_threshold if args.alive_only else None,
            "C_eff": int(Ceff),
            "N": int(N),
            "avg_L0_per_row": avg_L0,
            "n_random_pairs": args.n_random_pairs,
            "n_adjacent_pairs": args.n_adjacent_pairs,
        },
        "patterns": {
            "n_unique_patterns": n_unique,
            "frac_unique": n_unique / N,
            "n_singletons": n_singletons,
            "frac_singletons": n_singletons / N,
            "max_pattern_freq": int(most_common[0][1]),
            "top8_pattern_freqs": [c for (_, c) in most_common],
            "n_patterns_covering_50pct_of_rows": int(n_patterns_covering_50pct),
        },
        "jaccard": {
            "adjacent_mean": float(np.mean(adj_jaccards)),
            "adjacent_median": float(np.median(adj_jaccards)),
            "adjacent_p95": float(np.percentile(adj_jaccards, 95)),
            "random_mean": float(np.mean(rand_jaccards)),
            "random_median": float(np.median(rand_jaccards)),
            "random_p95": float(np.percentile(rand_jaccards, 95)),
            "adjacent_over_random_mean_ratio":
                float(np.mean(adj_jaccards) / np.mean(rand_jaccards))
                if np.mean(rand_jaccards) > 0 else float("inf"),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    plotting.set_style()
    import matplotlib.pyplot as plt

    # Plot 1: pattern frequency distribution (sorted, log-log).
    sorted_counts = np.array(sorted(counts.values(), reverse=True))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.loglog(np.arange(1, len(sorted_counts) + 1), sorted_counts,
              color=plotting.PALETTE["raw"], lw=1.8)
    ax.set_xlabel("pattern rank")
    ax.set_ylabel("# rows sharing this pattern")
    ax.margins(x=0)
    plotting.save_close(fig, args.output_dir / "01_pattern_freq_distribution.png")

    # Plot 2: Jaccard histogram, adjacent vs random.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, 1, 60)
    ax.hist(rand_jaccards, bins=bins, color=plotting.PALETTE["baseline"],
            alpha=0.65, label="random row pairs",
            edgecolor=plotting.PALETTE["bg"], linewidth=0.4)
    ax.hist(adj_jaccards, bins=bins, color=plotting.PALETTE["raw"],
            alpha=0.85, label="within-sequence (t, t+1)",
            edgecolor=plotting.PALETTE["bg"], linewidth=0.4)
    ax.axvline(np.mean(rand_jaccards), color=plotting.PALETTE["baseline"],
               lw=1.1, ls="--", zorder=4)
    ax.axvline(np.mean(adj_jaccards), color=plotting.PALETTE["raw"],
               lw=1.1, ls="--", zorder=4)
    ax.legend(loc="upper right", borderpad=0.4)
    ax.set_xlabel("Jaccard similarity of active sets")
    ax.set_ylabel("# pairs")
    ax.margins(x=0)
    plotting.save_close(fig, args.output_dir / "02_jaccard_adjacent_vs_random.png")

    print(f"[row_patterns] wrote summary + plots -> {args.output_dir}", flush=True)
    print(
        f"[row_patterns] {n_unique:,} unique active-set patterns out of {N:,} rows "
        f"({n_unique/N*100:.1f}% unique); top pattern shared by "
        f"{most_common[0][1]:,} rows; "
        f"adjacent Jaccard mean {summary['jaccard']['adjacent_mean']:.3f} vs "
        f"random {summary['jaccard']['random_mean']:.3f} "
        f"({summary['jaccard']['adjacent_over_random_mean_ratio']:.1f}× lift).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
