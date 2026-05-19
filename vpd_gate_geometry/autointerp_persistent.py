"""Snippet extraction for autointerp on persistent VPD gate components.

The persistent population is small (32 of 9,014 alive components have
$P(\\text{on at } t{+}1 \\mid \\text{on at } t) > 0.8$ at threshold
$g > 0.5$). For each persistent component we collect:

- top positive snippets: rows where the gate is high, with surrounding
  context. The active token is marked `<<TOK>>`.
- held-out positives: a separate pool of high-gate rows used for
  later validation by the LLM.
- matched negatives: rows where the gate is near 0; we try to match
  on token id first, and fall back to random low-gate rows.

This module is deliberately API-free. It writes `examples.jsonl` for
downstream LLM-driven hypothesis generation. Decoded text uses the
canonical `EleutherAI/gpt-neox-20b` tokenizer (the same one VPD's
target model was trained against).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from . import cache_io
from .lagged import _sequence_views


def _compute_persistence(
    G: torch.Tensor, starts: list[int], ends: list[int], threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (persistence, baseline) per column on `G.device`."""
    C = G.shape[1]
    on_count = torch.zeros(C, device=G.device, dtype=torch.float64)
    cont_count = torch.zeros(C, device=G.device, dtype=torch.float64)
    prev_count = torch.zeros(C, device=G.device, dtype=torch.float64)
    for s, e in zip(starts, ends, strict=True):
        chunk = G[s:e]
        if chunk.shape[0] <= 1:
            on_count += (chunk > threshold).to(torch.float64).sum(dim=0)
            continue
        on = chunk > threshold
        on_count += on.to(torch.float64).sum(dim=0)
        prev_on = on[:-1]
        next_on = on[1:]
        cont_count += (prev_on & next_on).to(torch.float64).sum(dim=0)
        prev_count += prev_on.to(torch.float64).sum(dim=0)
    persistence = (cont_count / prev_count.clamp_min(1)).cpu()
    baseline = (on_count / G.shape[0]).cpu()
    return persistence, baseline


def _build_snippets(
    token_ids: torch.Tensor,
    starts: list[int],
    ends: list[int],
    row_ids: list[int],
    tokenizer,
    left_ctx: int,
    right_ctx: int,
    g_values: list[float],
) -> list[dict[str, object]]:
    """Build context snippets for the given row indices."""
    out: list[dict[str, object]] = []
    seq_start_of: dict[int, int] = {}
    seq_end_of: dict[int, int] = {}
    for s, e in zip(starts, ends, strict=True):
        for r in range(s, e):
            seq_start_of[r] = s
            seq_end_of[r] = e
    for row, g_val in zip(row_ids, g_values, strict=True):
        s = seq_start_of.get(row)
        e = seq_end_of.get(row)
        if s is None:
            continue
        local = row - s
        lo = max(s, row - left_ctx)
        hi = min(e, row + right_ctx + 1)
        # Pre / focus / post strings, decoded separately so we can mark the
        # focus token without surprising the tokenizer's space rules.
        pre_ids = token_ids[lo:row].tolist()
        focus_id = int(token_ids[row].item())
        post_ids = token_ids[row + 1:hi].tolist()
        pre_str = tokenizer.decode(pre_ids, clean_up_tokenization_spaces=False)
        focus_str = tokenizer.decode([focus_id], clean_up_tokenization_spaces=False)
        post_str = tokenizer.decode(post_ids, clean_up_tokenization_spaces=False)
        snippet = f"{pre_str}<<{focus_str}>>{post_str}"
        out.append({
            "row": int(row),
            "seq_start": int(s),
            "local_pos": int(local),
            "focus_token_id": focus_id,
            "focus_token_str": focus_str,
            "gate_value": float(g_val),
            "context_left_ids": pre_ids[-left_ctx:],
            "context_right_ids": post_ids[:right_ctx],
            "snippet": snippet,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vpd_gate_geometry.autointerp_persistent")
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    p.add_argument("--persistence-threshold", type=float, default=0.8)
    p.add_argument("--alive-threshold", type=float, default=1e-4)
    p.add_argument("--gate-threshold", type=float, default=0.5,
                   help="Binary 'on' threshold for snippet selection.")
    p.add_argument("--n-train-pos", type=int, default=40)
    p.add_argument("--n-heldout-pos", type=int, default=30)
    p.add_argument("--n-heldout-neg", type=int, default=30)
    p.add_argument("--context-left", type=int, default=48)
    p.add_argument("--context-right", type=int, default=24)
    p.add_argument("--neg-low-gate", type=float, default=0.01,
                   help="Negatives must have gate < this value for the component.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-components", type=int, default=0,
                   help="Cap components processed (0 = all persistent).")
    p.add_argument("--include-matched-control", action="store_true",
                   help="Also extract snippets for a matched set of non-persistent "
                        "components (same mean activity, persistence < 0.3). "
                        "Useful as a baseline for autointerp difficulty.")
    args = p.parse_args(argv)

    rng = torch.Generator().manual_seed(args.seed)

    print(f"[autointerp] loading {args.cache} ...", flush=True)
    gm, meta = cache_io.load_gate_matrix_cache(args.cache)
    G = gm.G
    if args.device == "cuda" and torch.cuda.is_available():
        G = G.to(args.device)
    print(f"[autointerp] G={tuple(G.shape)} on {G.device}", flush=True)

    views = _sequence_views(gm)
    starts = [v[1] for v in views]
    ends = [v[2] for v in views]

    print("[autointerp] computing persistence + baseline ...", flush=True)
    persistence, baseline = _compute_persistence(G, starts, ends, args.gate_threshold)
    col_mean = G.mean(dim=0).cpu()
    alive_mask = (col_mean > args.alive_threshold)
    persistent_mask = alive_mask & (persistence > args.persistence_threshold)
    n_pers = int(persistent_mask.sum().item())
    print(f"[autointerp] {n_pers} persistent components (alive & persistence > "
          f"{args.persistence_threshold})", flush=True)

    selected_idx = torch.nonzero(persistent_mask, as_tuple=False).squeeze(-1).tolist()
    selected_idx.sort(key=lambda i: float(-persistence[i].item()))
    if args.max_components > 0:
        selected_idx = selected_idx[: args.max_components]

    if args.include_matched_control:
        # Pick the same number of non-persistent alive components with
        # similar mean activity; useful as an autointerp difficulty baseline.
        target_mean = col_mean[selected_idx].mean().item()
        candidate_idx = torch.nonzero(
            alive_mask & (persistence < 0.3), as_tuple=False
        ).squeeze(-1)
        if candidate_idx.numel() > 0:
            cand_means = col_mean[candidate_idx]
            score = (cand_means - target_mean).abs()
            top = torch.topk(score, k=min(len(selected_idx), candidate_idx.numel()),
                             largest=False).indices
            control_idx = candidate_idx[top].tolist()
        else:
            control_idx = []
        print(f"[autointerp] matched-control: {len(control_idx)} non-persistent "
              f"components (persistence < 0.3, mean ≈ {target_mean:.4f}).", flush=True)
    else:
        control_idx = []

    print(f"[autointerp] loading tokenizer {args.tokenizer} ...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer,
        token=os.environ.get("HF_TOKEN") or None,
    )

    token_ids_cpu = gm.token_ids.cpu()
    G_cpu = G.cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = args.output_dir / "examples.jsonl"
    metadata_path = args.output_dir / "extraction_summary.json"

    n_total = args.n_train_pos + args.n_heldout_pos + args.n_heldout_neg

    def _extract_for_component(c: int, label: str) -> dict[str, object]:
        col = G_cpu[:, c]
        on_rows = torch.nonzero(col > args.gate_threshold, as_tuple=False).squeeze(-1)
        on_g = col[on_rows]
        order = torch.argsort(on_g, descending=True)
        on_rows = on_rows[order]
        on_g = on_g[order]

        # Train: top-N by gate value.
        n_train = min(args.n_train_pos, on_rows.numel())
        train_rows = on_rows[:n_train].tolist()
        train_g = on_g[:n_train].tolist()
        # Held-out positives: next-N by gate, no overlap with train.
        remaining = on_rows[n_train:]
        if remaining.numel() == 0:
            heldout_pos_rows = []
            heldout_pos_g = []
        else:
            n_pick = min(args.n_heldout_pos, remaining.numel())
            # Choose uniformly from the remaining high-gate pool (not literal
            # next-N, so the held-out distribution is less biased toward the
            # very top of the distribution).
            perm = torch.randperm(remaining.numel(), generator=rng)[:n_pick]
            heldout_pos_rows = remaining[perm].tolist()
            heldout_pos_g = col[heldout_pos_rows].tolist()

        # Negatives: prefer same-token-id with low gate; fall back to random
        # low-gate rows.
        pos_token_ids = token_ids_cpu[train_rows + heldout_pos_rows].tolist()
        same_tok_negs: list[int] = []
        seen = set()
        for tid in pos_token_ids:
            if tid in seen: continue
            seen.add(tid)
            mask = (token_ids_cpu == tid) & (col < args.neg_low_gate)
            cand = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if cand.numel() == 0:
                continue
            # Take a handful per matched token; cap to keep it diverse.
            perm = torch.randperm(cand.numel(), generator=rng)[: max(1, args.n_heldout_neg // 10)]
            same_tok_negs.extend(cand[perm].tolist())
            if len(same_tok_negs) >= args.n_heldout_neg:
                break
        same_tok_negs = same_tok_negs[: args.n_heldout_neg]
        if len(same_tok_negs) < args.n_heldout_neg:
            # Fill with random low-gate rows.
            mask = col < args.neg_low_gate
            cand = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            need = args.n_heldout_neg - len(same_tok_negs)
            if cand.numel() >= need:
                perm = torch.randperm(cand.numel(), generator=rng)[:need]
                same_tok_negs.extend(cand[perm].tolist())
        heldout_neg_rows = same_tok_negs
        heldout_neg_g = col[heldout_neg_rows].tolist() if heldout_neg_rows else []

        # Build text snippets.
        train_snippets = _build_snippets(
            token_ids_cpu, starts, ends, train_rows, tok,
            args.context_left, args.context_right, train_g,
        )
        heldout_pos_snippets = _build_snippets(
            token_ids_cpu, starts, ends, heldout_pos_rows, tok,
            args.context_left, args.context_right, heldout_pos_g,
        )
        heldout_neg_snippets = _build_snippets(
            token_ids_cpu, starts, ends, heldout_neg_rows, tok,
            args.context_left, args.context_right, heldout_neg_g,
        )

        # Token-identity baseline: are positives concentrated on one token?
        pos_tok_ids = [s["focus_token_id"] for s in train_snippets + heldout_pos_snippets]
        tok_counter: dict[int, int] = defaultdict(int)
        for t in pos_tok_ids:
            tok_counter[t] += 1
        if tok_counter:
            top_tid, top_cnt = max(tok_counter.items(), key=lambda kv: kv[1])
            top_frac = top_cnt / len(pos_tok_ids)
            top_str = tok.decode([top_tid], clean_up_tokenization_spaces=False)
        else:
            top_tid, top_cnt, top_frac, top_str = None, 0, 0.0, ""

        key = gm.keys[c]
        return {
            "component_idx": int(c),
            "module": key.module_name,
            "local_idx": key.local_idx,
            "category": label,
            "mean_gate": float(col_mean[c].item()),
            "persistence": float(persistence[c].item()),
            "baseline": float(baseline[c].item()),
            "n_on_rows": int(on_rows.numel()),
            "token_identity_baseline": {
                "top_token_id": top_tid,
                "top_token_str": top_str,
                "top_token_count": int(top_cnt),
                "top_token_fraction_of_positives": float(top_frac),
                "n_positives_used": len(pos_tok_ids),
            },
            "train_positives": train_snippets,
            "heldout_positives": heldout_pos_snippets,
            "heldout_negatives": heldout_neg_snippets,
        }

    records: list[dict[str, object]] = []
    print(f"[autointerp] extracting snippets for {len(selected_idx)} persistent "
          f"+ {len(control_idx)} control components ...", flush=True)
    for c in selected_idx:
        rec = _extract_for_component(c, label="persistent")
        records.append(rec)
    for c in control_idx:
        rec = _extract_for_component(c, label="control_non_persistent")
        records.append(rec)

    with examples_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"[autointerp] wrote {examples_path} ({len(records)} components)", flush=True)

    summary = {
        "n_persistent": n_pers,
        "n_selected": len(selected_idx),
        "n_control": len(control_idx),
        "n_total_components": len(records),
        "n_train_pos": args.n_train_pos,
        "n_heldout_pos": args.n_heldout_pos,
        "n_heldout_neg": args.n_heldout_neg,
        "persistence_threshold": args.persistence_threshold,
        "gate_threshold": args.gate_threshold,
        "alive_threshold": args.alive_threshold,
        "context_left": args.context_left,
        "context_right": args.context_right,
        "tokenizer": args.tokenizer,
        "cache_path": str(args.cache),
    }
    with metadata_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[autointerp] wrote {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
