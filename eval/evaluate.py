#!/usr/bin/env python3
"""
Evaluation script for all three task heads of the PT Debugger.

Metrics:
  Summarization  — ROUGE-1, ROUGE-2, ROUGE-L  (via `evaluate` library)
  Q&A            — Exact Match (EM), Token-level F1
  Chatbot        — Perplexity on held-out assistant turns

Reads:  data/datasets/val.jsonl  (or a separate eval JSONL)
        outputs/llama-pt-debugger  (fine-tuned model + adapter)
Writes: eval/results/eval_report.json

Usage:
  python eval/evaluate.py
  python eval/evaluate.py --val_file data/datasets/val.jsonl \\
      --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \\
      --adapter_path outputs/llama-pt-debugger \\
      --device auto --max_samples 200
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

BOS = "<|begin_of_text|>"
SOH = "<|start_header_id|>"
EOH = "<|end_header_id|>"
EOT = "<|eot_id|>"

# ---------------------------------------------------------------------------
# Text normalization for QA metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower-case, strip punctuation and extra whitespace (SQuAD-style)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


# ---------------------------------------------------------------------------
# Sample parsing (val.jsonl has {"text": "..."} entries)
# ---------------------------------------------------------------------------

def detect_task(text: str) -> Optional[str]:
    if "[TASK: SUMMARIZE]" in text:
        return "summarize"
    if "[TASK: QA]" in text:
        return "qa"
    if BOS in text or SOH in text:
        return "chatbot"
    return None


def split_prompt_completion_text(text: str, task: str) -> tuple[str, str]:
    """
    Split a text sample into (prompt, expected_completion).
    For text tasks: everything up to the gold answer section.
    For chatbot: all but the last assistant turn.
    """
    if task in ("summarize", "qa"):
        # The completion follows the last "### Summary\n" or "### Answer\n" header
        marker = "### Summary\n" if task == "summarize" else "### Answer\n"
        idx = text.rfind(marker)
        if idx == -1:
            return text, ""
        split_at = idx + len(marker)
        return text[:split_at], text[split_at:].strip()

    # chatbot: messages are separated by LLaMA-3 tokens
    # Split at the last <|start_header_id|>assistant<|end_header_id|> block
    pattern = f"{SOH}assistant{EOH}\n\n"
    idx = text.rfind(pattern)
    if idx == -1:
        return text, ""
    split_at = idx + len(pattern)
    prompt = text[:split_at]
    # Gold completion = everything up to the next EOT
    rest = text[split_at:]
    eot_idx = rest.find(EOT)
    gold = rest[:eot_idx].strip() if eot_idx != -1 else rest.strip()
    return prompt, gold


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_summarize(samples: list[tuple[str, str]], model) -> dict:
    try:
        import evaluate as hf_evaluate
        rouge = hf_evaluate.load("rouge")
    except ImportError:
        raise ImportError("pip install evaluate rouge_score")

    predictions, references = [], []
    for prompt, gold in samples:
        if not gold:
            continue
        pred = model.generate(prompt, max_new_tokens=256)
        predictions.append(pred)
        references.append(gold)

    if not predictions:
        return {"error": "No valid summarize samples"}

    scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return {k: round(v, 4) for k, v in scores.items()}


def evaluate_qa(samples: list[tuple[str, str]], model) -> dict:
    em_scores, f1_scores = [], []
    for prompt, gold in samples:
        if not gold:
            continue
        pred = model.generate(prompt, max_new_tokens=256)
        em_scores.append(_exact_match(pred, gold))
        f1_scores.append(_token_f1(pred, gold))

    if not em_scores:
        return {"error": "No valid QA samples"}

    return {
        "exact_match": round(sum(em_scores) / len(em_scores), 4),
        "token_f1":    round(sum(f1_scores) / len(f1_scores), 4),
        "num_samples": len(em_scores),
    }


def evaluate_chatbot_perplexity(samples: list[tuple[str, str]], model) -> dict:
    """
    Compute average perplexity on held-out assistant turns.
    Uses the model to score the gold completion.
    """
    import torch

    nll_sum = 0.0
    tok_count = 0
    skipped = 0

    for prompt, gold in samples:
        if not gold:
            skipped += 1
            continue
        full_text = prompt + gold

        try:
            inputs = model.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            prompt_len = len(model.tokenizer(prompt, return_tensors="pt")["input_ids"][0])

            device_obj = "hpu" if model.device == "hpu" else next(model.model.parameters()).device
            input_ids = inputs["input_ids"].to(device_obj)

            with torch.no_grad():
                outputs = model.model(input_ids, labels=input_ids)

            # Only score the completion tokens (after prompt)
            shift_logits = outputs.logits[0, prompt_len - 1: -1, :]
            shift_labels = input_ids[0, prompt_len:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="sum"
            )
            nll_sum  += loss.item()
            tok_count += shift_labels.shape[0]
        except Exception as exc:
            logger.debug("Perplexity computation skipped for sample: %s", exc)
            skipped += 1

    if tok_count == 0:
        return {"error": "No tokens scored"}

    ppl = math.exp(nll_sum / tok_count)
    return {
        "perplexity":   round(ppl, 2),
        "tokens_scored": tok_count,
        "samples_skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the PT Debugger fine-tuned model")
    ap.add_argument(
        "--val_file", type=Path,
        default=Path("data/datasets/val.jsonl"),
    )
    ap.add_argument(
        "--model_path", type=str,
        default=os.getenv("MODEL_PATH", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    )
    ap.add_argument(
        "--adapter_path", type=str,
        default=os.getenv("ADAPTER_PATH", "outputs/llama-pt-debugger"),
    )
    ap.add_argument(
        "--device", type=str,
        default=os.getenv("DEVICE", "auto"),
    )
    ap.add_argument(
        "--max_samples", type=int, default=0,
        help="Max samples per task (0 = all)",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("eval/results/eval_report.json"),
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from inference.infer import PTDebuggerModel
    model = PTDebuggerModel(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # Load and classify val samples
    # ------------------------------------------------------------------
    task_samples: dict[str, list[tuple[str, str]]] = {
        "summarize": [],
        "qa":        [],
        "chatbot":   [],
    }

    with args.val_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec  = json.loads(line)
            text = rec.get("text", "")
            task = detect_task(text)
            if task is None:
                continue
            prompt, gold = split_prompt_completion_text(text, task)
            if prompt and gold:
                task_samples[task].append((prompt, gold))

    for task, samples in task_samples.items():
        logger.info("Loaded %d %s validation samples", len(samples), task)

    # Optionally cap sample counts
    if args.max_samples > 0:
        import random
        random.seed(42)
        for task in task_samples:
            if len(task_samples[task]) > args.max_samples:
                task_samples[task] = random.sample(task_samples[task], args.max_samples)

    # ------------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------------
    results: dict = {}

    if task_samples["summarize"]:
        logger.info("Evaluating summarization (ROUGE)...")
        results["summarize"] = evaluate_summarize(task_samples["summarize"], model)
        logger.info("Summarize results: %s", results["summarize"])

    if task_samples["qa"]:
        logger.info("Evaluating Q&A (EM + Token F1)...")
        results["qa"] = evaluate_qa(task_samples["qa"], model)
        logger.info("QA results: %s", results["qa"])

    if task_samples["chatbot"]:
        logger.info("Evaluating chatbot (perplexity)...")
        results["chatbot"] = evaluate_chatbot_perplexity(task_samples["chatbot"], model)
        logger.info("Chatbot results: %s", results["chatbot"])

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_path":   args.model_path,
        "adapter_path": args.adapter_path,
        "device":       args.device,
        "sample_counts": {t: len(s) for t, s in task_samples.items()},
        "results":      results,
    }
    with args.output.open("w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report written to %s", args.output)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for task, metrics in results.items():
        print(f"\n[{task.upper()}]")
        for k, v in metrics.items():
            print(f"  {k:30s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
