#!/usr/bin/env python3
"""
Validate, deduplicate, tokenize-filter, and split augmented datasets.

Reads:  data/datasets/augmented_*.jsonl   (falls back to raw_*.jsonl if absent)
Writes: data/datasets/train.jsonl
        data/datasets/val.jsonl

Each output line: {"text": "<full formatted sample>"}

The "text" field uses:
  - Text tasks (summarize / qa): prompt + completion concatenated
  - Chat task: LLaMA-3 ChatML template

Usage:
  python scripts/validate_datasets.py
  python scripts/validate_datasets.py --max_tokens 2048 --val_ratio 0.1
"""

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# LLaMA-3 special tokens for ChatML formatting
BOS = "<|begin_of_text|>"
SOH = "<|start_header_id|>"
EOH = "<|end_header_id|>"
EOT = "<|eot_id|>"


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def format_text_sample(rec: dict) -> Optional[str]:
    """Format a summarize or qa record as a single training string."""
    prompt     = rec.get("prompt", "").strip()
    completion = rec.get("completion", "").strip()
    if not prompt or not completion or completion.startswith("__"):
        return None
    return prompt + completion


def format_chat_sample(rec: dict) -> Optional[str]:
    """Format a chatbot record using the LLaMA-3 ChatML template."""
    messages = rec.get("messages", [])
    if not messages:
        return None

    parts = [BOS]
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content or content == "__AUGMENT__" or content == "__LLM_FAILED__":
            return None  # Drop incomplete conversations
        parts.append(f"{SOH}{role}{EOH}\n\n{content}{EOT}")

    # Signal the model to generate from assistant perspective
    parts.append(f"{SOH}assistant{EOH}\n\n")
    return "".join(parts)


def format_record(rec: dict) -> Optional[str]:
    task = rec.get("task", "")
    if task in ("summarize", "qa"):
        return format_text_sample(rec)
    elif task == "chatbot":
        return format_chat_sample(rec)
    logger.debug("Unknown task type '%s'; skipping record", task)
    return None


# ---------------------------------------------------------------------------
# Token-length estimator (character-based; avoids loading a tokenizer)
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 3.5  # conservative estimate for LLaMA tokenizer


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------

TASK_FILES = {
    "summarize": ("augmented_summarize.jsonl", "raw_summarize.jsonl"),
    "qa":        ("augmented_qa.jsonl",        "raw_qa.jsonl"),
    "chatbot":   ("augmented_chatbot.jsonl",   "raw_chatbot.jsonl"),
}


def load_task(datasets_dir: Path, task: str) -> list[dict]:
    primary, fallback = TASK_FILES[task]
    for fname in (primary, fallback):
        path = datasets_dir / fname
        if path.exists():
            records: list[dict] = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            logger.info("[%s] loaded %d records from %s", task, len(records), fname)
            return records
    logger.warning("[%s] no input file found; skipping", task)
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate and split augmented datasets into train/val"
    )
    ap.add_argument(
        "--datasets_dir", type=Path,
        default=Path("data/datasets"),
    )
    ap.add_argument(
        "--max_tokens", type=int, default=2048,
        help="Discard samples exceeding this estimated token count",
    )
    ap.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Fraction of data to use for validation (default: 0.10)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load and format all tasks
    # ------------------------------------------------------------------
    all_texts: list[str] = []
    stats: dict[str, dict] = {}

    for task in ("summarize", "qa", "chatbot"):
        records = load_task(args.datasets_dir, task)
        task_stats = {"loaded": len(records), "formatted": 0, "too_long": 0, "empty": 0}

        for rec in records:
            text = format_record(rec)
            if text is None:
                task_stats["empty"] += 1
                continue
            tokens = estimate_tokens(text)
            if tokens > args.max_tokens:
                task_stats["too_long"] += 1
                continue
            all_texts.append(text)
            task_stats["formatted"] += 1

        stats[task] = task_stats
        logger.info(
            "[%s] formatted=%d  too_long=%d  empty/invalid=%d",
            task,
            task_stats["formatted"],
            task_stats["too_long"],
            task_stats["empty"],
        )

    # ------------------------------------------------------------------
    # Deduplicate
    # ------------------------------------------------------------------
    seen: set[str] = set()
    unique: list[str] = []
    for text in all_texts:
        fp = fingerprint(text)
        if fp not in seen:
            seen.add(fp)
            unique.append(text)

    duplicates = len(all_texts) - len(unique)
    logger.info(
        "Total: %d samples  |  Duplicates removed: %d  |  Unique: %d",
        len(all_texts), duplicates, len(unique),
    )

    if not unique:
        raise RuntimeError(
            "No valid samples after filtering. "
            "Check that augmented_*.jsonl files exist and contain completed records."
        )

    # ------------------------------------------------------------------
    # Shuffle and split
    # ------------------------------------------------------------------
    random.shuffle(unique)
    n_val   = max(1, int(len(unique) * args.val_ratio))
    n_train = len(unique) - n_val
    train_texts = unique[:n_train]
    val_texts   = unique[n_train:]

    logger.info("Split: train=%d  val=%d", n_train, n_val)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    args.datasets_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.datasets_dir / "train.jsonl"
    val_path   = args.datasets_dir / "val.jsonl"

    for path, texts in ((train_path, train_texts), (val_path, val_texts)):
        with path.open("w", encoding="utf-8") as f:
            for text in texts:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        logger.info("Wrote %d records → %s", len(texts), path)

    # ------------------------------------------------------------------
    # Token length distribution report
    # ------------------------------------------------------------------
    all_lens = [estimate_tokens(t) for t in unique]
    avg_len  = sum(all_lens) / len(all_lens) if all_lens else 0
    max_len  = max(all_lens) if all_lens else 0
    p95_len  = sorted(all_lens)[int(len(all_lens) * 0.95)] if all_lens else 0

    logger.info(
        "Token length — avg: %.0f  p95: %d  max: %d  (estimated, chars/%.1f)",
        avg_len, p95_len, max_len, CHARS_PER_TOKEN,
    )

    # Write a brief stats JSON for reference
    stats_path = args.datasets_dir / "dataset_stats.json"
    with stats_path.open("w") as f:
        json.dump({
            "per_task": stats,
            "total_unique": len(unique),
            "duplicates_removed": duplicates,
            "train_size": n_train,
            "val_size": n_val,
            "token_length": {
                "avg": round(avg_len, 1),
                "p95": p95_len,
                "max": max_len,
            },
        }, f, indent=2)
    logger.info("Dataset stats → %s", stats_path)


if __name__ == "__main__":
    main()
