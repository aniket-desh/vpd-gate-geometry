"""LLM hypothesis + held-out classification for persistent VPD gates.

Two stages per component:

1. **Hypothesis generation.** Given a set of "positive" snippets where
   the gate is highly active, ask the model for a short label + one
   sentence describing when the gate activates. The model must say so
   when no coherent pattern is visible.

2. **Held-out classification.** Given the hypothesis and a shuffled
   mix of held-out positives and held-out negatives (no labels), ask
   the model to predict whether each snippet would activate the gate.
   We score balanced accuracy / precision / recall against the ground
   truth held-out labels.

Both stages cache their responses by (component_idx, stage,
prompt_hash) so the pipeline is resumable; pre-existing entries are
read from disk and reused.

The OpenAI Chat Completions API is the default backend (the `openai`
Python package is already installed via the upstream venv's
`openrouter>=0.1.1` dependency, which pulls `openai>=1.x`). With
`--dry-run`, prompts are written to disk but no API call is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


HYPOTHESIS_SYSTEM = """You are an interpretability analyst. You will see a small set of
text snippets where some hidden "gate" inside a language model
fires strongly. Each snippet has its focus token marked with
<<...>>. Your job is to propose one short label and one
sentence describing what the snippets have in common; if they
have no coherent pattern, say so.

Constraints:
- Do not overfit to a single repeated literal token. Look for
  syntactic, contextual, or document-structure cues.
- The gate is a VPD causal-importance score, not an SAE feature.
- Do not claim causality. Do not claim this is "a feature for X".
  Phrase the hypothesis as a prediction about when the gate fires.
- If the snippets look incoherent or noise-like, your label
  should be "unclear" and you should say so."""

HYPOTHESIS_USER_TMPL = """Component {component_id} ({module}#{local_idx}).
Mean gate over the corpus: {mean_gate:.4f}.
Persistence  P(on_{{t+1}} | on_t) = {persistence:.3f}.

Below are {n_positives} snippets where this gate fires strongly.
The focus token is marked with <<...>>.

{snippets}

Return JSON with this exact schema:

{{
  "label": "short slug, <= 5 words",
  "hypothesis": "one sentence describing when the gate fires",
  "positive_cues": ["..."],
  "negative_cues": ["..."],
  "confidence": 0.0,
  "notes": "..."
}}

Set confidence to 0 if the snippets look incoherent. Set label
to "unclear" in that case.
"""

CLASSIFY_SYSTEM = """You are an interpretability analyst. You will be given a
hypothesis describing when a hidden gate in a language model
fires. Then you will see a shuffled list of unlabeled text
snippets. Your job is to predict for each snippet whether the
gate would fire on the marked token.

Constraints:
- Use only the hypothesis to decide. Do not invent new rules
  midway through.
- If a snippet is ambiguous, return predicted_active=false with
  low confidence; do not flip a coin.
- Confidences must reflect how strongly the snippet matches the
  hypothesis."""

CLASSIFY_USER_TMPL = """Component {component_id} ({module}#{local_idx}).

Hypothesis:
{hypothesis}

Below are {n_total} unlabeled snippets. For each one, predict
whether the gate would fire on the marked token.

{snippets_with_ids}

Return JSON of the form:

{{
  "predictions": [
    {{"example_id": "ex_0", "predicted_active": true, "confidence": 0.0, "rationale": "..."}}
  ]
}}

Order the predictions by example_id ascending. Make sure every
example_id is covered exactly once.
"""


def _prompt_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _format_snippets(snippets: list[dict[str, object]], max_chars: int = 300) -> str:
    out = []
    for i, s in enumerate(snippets):
        snippet = str(s["snippet"]).replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 1] + "…"
        out.append(f"[{i+1}] g={float(s['gate_value']):.3f}\n    {snippet}")
    return "\n".join(out)


def _format_snippets_with_ids(snippets: list[tuple[str, dict[str, object]]],
                              max_chars: int = 300) -> str:
    out = []
    for example_id, s in snippets:
        snippet = str(s["snippet"]).replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 1] + "…"
        out.append(f"{example_id}: {snippet}")
    return "\n".join(out)


def _call_openai(model: str, messages: list[dict[str, str]],
                 temperature: float = 0.2, max_tokens: int = 800) -> tuple[str, dict[str, Any]]:
    """Call OpenAI Chat Completions. Returns (content, raw_response_meta).

    Reads OPENAI_API_KEY from env. Raises if missing.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    meta = {
        "model": resp.model,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
            "total_tokens": resp.usage.total_tokens if resp.usage else None,
        },
        "finish_reason": resp.choices[0].finish_reason,
    }
    return content, meta


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        # Try to find a fenced JSON block.
        if "```" in text:
            chunks = text.split("```")
            for chunk in chunks:
                chunk = chunk.lstrip()
                if chunk.startswith("json"):
                    chunk = chunk[4:]
                try:
                    return json.loads(chunk)
                except Exception:
                    continue
        return None


def _score_predictions(
    preds: list[dict[str, Any]],
    ground_truth: dict[str, bool],
) -> dict[str, float]:
    """Balanced accuracy + precision + recall + counts.

    Robust to LLMs that drop zero-padding from the example_id (ex_0 vs
    ex_000). We normalize on both sides via integer parsing.
    """
    import re as _re

    def _norm(eid: str | None) -> str | None:
        if not isinstance(eid, str):
            return None
        m = _re.match(r"ex[_-]?(\d+)", eid.strip())
        if not m:
            return None
        return f"ex_{int(m.group(1)):03d}"

    gt_norm = {_norm(k): v for k, v in ground_truth.items() if _norm(k) is not None}
    tp = fp = tn = fn = 0
    for p in preds:
        eid = _norm(p.get("example_id"))
        if eid is None or eid not in gt_norm:
            continue
        gt = gt_norm[eid]
        pa = bool(p.get("predicted_active", False))
        if pa and gt:        tp += 1
        elif pa and not gt:  fp += 1
        elif not pa and gt:  fn += 1
        else:                tn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (sens + spec)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "n": int(n),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(sens),
        "specificity": float(spec),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vpd_gate_geometry.openai_autointerp")
    p.add_argument("--examples", type=Path, required=True,
                   help="examples.jsonl from autointerp_persistent")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str,
                   default=os.environ.get("OPENAI_AUTOINTERP_MODEL", "gpt-4o-mini"))
    p.add_argument("--max-components", type=int, default=0,
                   help="Cap components processed (0 = all from input).")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--seed", type=int, default=0,
                   help="Shuffle seed for held-out mix.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write prompts to disk but skip the API call.")
    p.add_argument("--only-category", type=str, default="",
                   choices=["", "persistent", "control_non_persistent"])
    p.add_argument("--min-excess-persistence", type=float, default=0.0,
                   help="Drop components whose (persistence - baseline) is below this. "
                        "Set to ~0.5 to focus on bursty/persistent components rather "
                        "than always-on ones.")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    print(f"[llm] reading {args.examples} ...", flush=True)
    records = [json.loads(l) for l in open(args.examples)]
    if args.only_category:
        records = [r for r in records if r["category"] == args.only_category]
    if args.min_excess_persistence > 0:
        before = len(records)
        records = [r for r in records
                   if (r["persistence"] - r["baseline"]) >= args.min_excess_persistence]
        print(f"[llm] excess-persistence filter: kept {len(records)}/{before}", flush=True)
    if args.max_components > 0:
        records = records[: args.max_components]
    print(f"[llm] {len(records)} components to process; model={args.model}; "
          f"dry_run={args.dry_run}", flush=True)

    cache_dir = args.output_dir / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = args.output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    hypotheses: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    api_used_cents = 0.0

    for ridx, rec in enumerate(records):
        c = rec["component_idx"]
        category = rec["category"]
        train_pos = rec["train_positives"]
        heldout_pos = rec["heldout_positives"]
        heldout_neg = rec["heldout_negatives"]
        tag = f"c{c:05d}_{category}"
        print(f"[llm] [{ridx+1}/{len(records)}] {tag}  module={rec['module']}#{rec['local_idx']}  "
              f"pos={len(train_pos)} held_pos={len(heldout_pos)} held_neg={len(heldout_neg)}",
              flush=True)

        # ----- Stage 1: hypothesis generation -----
        hypo_user = HYPOTHESIS_USER_TMPL.format(
            component_id=c,
            module=rec["module"],
            local_idx=rec["local_idx"],
            mean_gate=rec["mean_gate"],
            persistence=rec["persistence"],
            n_positives=len(train_pos),
            snippets=_format_snippets(train_pos),
        )
        hypo_hash = _prompt_hash(HYPOTHESIS_SYSTEM, hypo_user, args.model)
        hypo_cache = cache_dir / f"{tag}_hypothesis_{hypo_hash}.json"
        (prompts_dir / f"{tag}_hypothesis.txt").write_text(
            f"SYSTEM:\n{HYPOTHESIS_SYSTEM}\n\nUSER:\n{hypo_user}\n"
        )

        if hypo_cache.exists():
            hypo_payload = json.loads(hypo_cache.read_text())
            print(f"  cached hypothesis hit", flush=True)
        elif args.dry_run:
            hypo_payload = {"dry_run": True, "label": "(dry-run)",
                            "hypothesis": "(dry-run)", "confidence": 0.0}
        else:
            t0 = time.time()
            content, meta = _call_openai(
                args.model,
                [{"role": "system", "content": HYPOTHESIS_SYSTEM},
                 {"role": "user", "content": hypo_user}],
                temperature=args.temperature, max_tokens=args.max_tokens,
            )
            parsed = _try_parse_json(content) or {"raw": content, "parse_error": True}
            hypo_payload = {**parsed, "_meta": meta, "_elapsed_s": time.time() - t0}
            hypo_cache.write_text(json.dumps(hypo_payload, indent=2))
            print(f"  hypothesis call: {hypo_payload.get('label', '?')} "
                  f"(conf={hypo_payload.get('confidence', '?')}, "
                  f"{hypo_payload.get('_elapsed_s', 0):.1f}s)", flush=True)

        hypotheses.append({
            "component_idx": c,
            "module": rec["module"],
            "local_idx": rec["local_idx"],
            "category": category,
            "mean_gate": rec["mean_gate"],
            "persistence": rec["persistence"],
            "token_identity_baseline": rec["token_identity_baseline"],
            "label": hypo_payload.get("label", ""),
            "hypothesis": hypo_payload.get("hypothesis", ""),
            "positive_cues": hypo_payload.get("positive_cues", []),
            "negative_cues": hypo_payload.get("negative_cues", []),
            "confidence": hypo_payload.get("confidence", 0.0),
            "notes": hypo_payload.get("notes", ""),
            "n_train_positives": len(train_pos),
        })

        # ----- Stage 2: held-out classification -----
        if not heldout_pos or not heldout_neg:
            print(f"  skipping classification (no held-out pairs)", flush=True)
            validation_rows.append({
                "component_idx": c, "category": category,
                "skipped": True, "reason": "no held-out data",
            })
            continue

        all_heldout = [(s, True) for s in heldout_pos] + [(s, False) for s in heldout_neg]
        rng.shuffle(all_heldout)
        labeled = [(f"ex_{i:03d}", s, gt) for i, (s, gt) in enumerate(all_heldout)]
        ground_truth = {eid: gt for (eid, _s, gt) in labeled}
        snippets_for_prompt = [(eid, s) for (eid, s, _gt) in labeled]

        cls_user = CLASSIFY_USER_TMPL.format(
            component_id=c, module=rec["module"], local_idx=rec["local_idx"],
            hypothesis=hypo_payload.get("hypothesis", "(missing)"),
            n_total=len(snippets_for_prompt),
            snippets_with_ids=_format_snippets_with_ids(snippets_for_prompt),
        )
        cls_hash = _prompt_hash(CLASSIFY_SYSTEM, cls_user, args.model)
        cls_cache = cache_dir / f"{tag}_classify_{cls_hash}.json"
        (prompts_dir / f"{tag}_classify.txt").write_text(
            f"SYSTEM:\n{CLASSIFY_SYSTEM}\n\nUSER:\n{cls_user}\n"
        )

        if cls_cache.exists():
            cls_payload = json.loads(cls_cache.read_text())
            print(f"  cached classify hit", flush=True)
        elif args.dry_run:
            cls_payload = {"dry_run": True, "predictions": []}
        else:
            t0 = time.time()
            content, meta = _call_openai(
                args.model,
                [{"role": "system", "content": CLASSIFY_SYSTEM},
                 {"role": "user", "content": cls_user}],
                temperature=args.temperature,
                max_tokens=max(args.max_tokens, 60 * len(snippets_for_prompt)),
            )
            parsed = _try_parse_json(content) or {"raw": content, "parse_error": True}
            cls_payload = {**parsed, "_meta": meta, "_elapsed_s": time.time() - t0}
            cls_cache.write_text(json.dumps(cls_payload, indent=2))
            print(f"  classify call: ({cls_payload.get('_elapsed_s', 0):.1f}s)", flush=True)

        preds = cls_payload.get("predictions", []) or []
        scores = _score_predictions(preds, ground_truth)
        # Random-label control: shuffle gt, score again.
        gt_keys = list(ground_truth.keys())
        gt_values = list(ground_truth.values())
        np_rng.shuffle(gt_values)
        shuffled_gt = dict(zip(gt_keys, gt_values, strict=True))
        rand_scores = _score_predictions(preds, shuffled_gt)

        validation_rows.append({
            "component_idx": c, "category": category,
            "module": rec["module"], "local_idx": rec["local_idx"],
            "mean_gate": rec["mean_gate"], "persistence": rec["persistence"],
            "label": hypo_payload.get("label", ""),
            "hypothesis": hypo_payload.get("hypothesis", ""),
            "scores": scores,
            "random_label_scores": rand_scores,
            "token_identity_baseline": rec["token_identity_baseline"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "hypotheses.json").open("w") as fh:
        json.dump(hypotheses, fh, indent=2)
    with (args.output_dir / "validation_scores.json").open("w") as fh:
        json.dump(validation_rows, fh, indent=2)
    summary = {
        "model": args.model,
        "n_components": len(records),
        "dry_run": args.dry_run,
        "seed": args.seed,
    }
    with (args.output_dir / "run_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[llm] wrote hypotheses + scores -> {args.output_dir}", flush=True)
    if not args.dry_run:
        ok_rows = [v for v in validation_rows if not v.get("skipped")]
        if ok_rows:
            mean_acc = np.mean([v["scores"]["balanced_accuracy"] for v in ok_rows])
            rand_acc = np.mean([v["random_label_scores"]["balanced_accuracy"] for v in ok_rows])
            print(f"[llm] mean balanced-accuracy: {mean_acc:.3f}  vs random label: {rand_acc:.3f}",
                  flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
