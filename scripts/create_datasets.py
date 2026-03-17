#!/usr/bin/env python3
"""
Build three instruction-tuning datasets from parsed PT failure records.

Reads:  data/processed/parsed_failures.jsonl
Writes:
  data/datasets/raw_summarize.jsonl   — summarization task
  data/datasets/raw_qa.jsonl          — question-answer task
  data/datasets/raw_chatbot.jsonl     — multi-turn chatbot task

All three tasks share a single task-prefix prompt convention so one unified
model can handle all three at inference time.

Usage:
  python scripts/create_datasets.py
  python scripts/create_datasets.py --input data/processed/parsed_failures.jsonl \\
      --out_dir data/datasets --failures_only
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SUMMARIZE_INSTRUCTION = (
    "You are an expert PyTorch training framework debugger. "
    "Summarize the test failure below. Identify: (1) root cause, "
    "(2) affected component, (3) recommended action."
)

SUMMARIZE_PROMPT = """\
[TASK: SUMMARIZE]
{instruction}

### Test Failure
Suite:        {suite_name}
Test:         {test_name}
Class:        {class_name}
Status:       {status}
Error Type:   {error_type}
Error Msg:    {error_message}
Stack Trace:
{stack_trace}

### Summary
"""

QA_INSTRUCTION = (
    "You are an expert PyTorch/Gaudi training framework debugger. "
    "Answer the question using only the provided test failure context."
)

QA_PROMPT = """\
[TASK: QA]
{instruction}

### Test Failure Context
Suite:        {suite_name}
Test:         {test_name}
Class:        {class_name}
Status:       {status}
Error Type:   {error_type}
Error Msg:    {error_message}
Stack Trace:
{stack_trace}

### Question
{question}

### Answer
"""

CHAT_SYSTEM = (
    "You are an expert PyTorch and Gaudi2 training framework debugger. "
    "Help the user understand and fix test failures. "
    "Be concise, specific, and actionable. "
    "Always reference specific error types, modules, or stack frames where possible."
)

# Seed Q&A questions — augment_data.py will generate more per failure
QA_SEED_QUESTIONS: list[str] = [
    "What is the root cause of this test failure?",
    "Which software component is most likely responsible for this error?",
    "What steps would you take to reproduce this failure?",
    "What is the recommended fix for this failure?",
    "Is this failure related to distributed training? Explain why or why not.",
    "What additional environment information would help debug this further?",
    "Could this be a flaky test or a genuine regression? How would you determine that?",
    "What other tests in the same suite might be affected by the same root cause?",
    "How would you differentiate between a framework bug and a user code bug here?",
    "What monitoring or logging would you add to catch this failure earlier?",
]

# Chat turn templates — placeholders filled by augmentation step
CHAT_TURN_TEMPLATES: list[list[tuple[str, str]]] = [
    [
        ("user",      "Can you help me understand this test failure?\n\n"
                      "Suite: {suite_name}\nTest: {test_name}\n"
                      "Error: {error_type}: {error_message}\n\nStack:\n{stack_trace_short}"),
        ("assistant", "__AUGMENT__"),
        ("user",      "What is the most likely root cause?"),
        ("assistant", "__AUGMENT__"),
        ("user",      "How should I fix it?"),
        ("assistant", "__AUGMENT__"),
    ],
    [
        ("user",      "I'm seeing a {status} in {test_name}. Here is the error:\n"
                      "{error_type}: {error_message}\n\nStack:\n{stack_trace_short}"),
        ("assistant", "__AUGMENT__"),
        ("user",      "Is this a distributed training issue?"),
        ("assistant", "__AUGMENT__"),
        ("user",      "What should I check first?"),
        ("assistant", "__AUGMENT__"),
    ],
    [
        ("user",      "I'm running tests and {suite_name} is failing.\n"
                      "Error: {error_message}\nWhat component should I look at?"),
        ("assistant", "__AUGMENT__"),
        ("user",      "Can you suggest a workaround while I investigate the root cause?"),
        ("assistant", "__AUGMENT__"),
    ],
]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

MAX_STACK_LINES = 40


def build_ctx(rec: dict) -> dict:
    stack = rec.get("stack_trace", "")
    lines = stack.splitlines()
    if len(lines) > MAX_STACK_LINES:
        stack = "\n".join(lines[:MAX_STACK_LINES]) + "\n... (truncated)"
    stack_short = "\n".join(lines[:10]) + ("\n..." if len(lines) > 10 else "")
    return {
        "suite_name":        rec.get("suite_name", ""),
        "class_name":        rec.get("class_name", ""),
        "test_name":         rec.get("test_name", ""),
        "status":            rec.get("status", ""),
        "error_type":        rec.get("error_type", ""),
        "error_message":     rec.get("error_message", ""),
        "stack_trace":       stack,
        "stack_trace_short": stack_short,
    }


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def build_summarize_record(rec: dict) -> dict:
    ctx = build_ctx(rec)
    prompt = SUMMARIZE_PROMPT.format(instruction=SUMMARIZE_INSTRUCTION, **ctx)
    # Placeholder completion — augment_data.py fills in the real LLM-generated summary
    placeholder = (
        f"[Placeholder] {ctx['status']} in {ctx['test_name']}: "
        f"{ctx['error_type']} — {ctx['error_message'][:150]}"
    )
    return {
        "task":       "summarize",
        "prompt":     prompt,
        "completion": placeholder,
        "meta":       ctx,
    }


def build_qa_records(rec: dict) -> list[dict]:
    ctx = build_ctx(rec)
    records = []
    for q in QA_SEED_QUESTIONS:
        prompt = QA_PROMPT.format(instruction=QA_INSTRUCTION, **ctx, question=q)
        records.append({
            "task":       "qa",
            "prompt":     prompt,
            "completion": "__AUGMENT__",
            "question":   q,
            "meta":       ctx,
        })
    return records


def build_chatbot_records(rec: dict) -> list[dict]:
    ctx = build_ctx(rec)
    records = []
    for template in CHAT_TURN_TEMPLATES:
        messages = [{"role": "system", "content": CHAT_SYSTEM}]
        for role, content in template:
            try:
                filled = content.format(**ctx) if "{" in content else content
            except KeyError:
                filled = content
            messages.append({"role": role, "content": filled})
        records.append({
            "task":     "chatbot",
            "messages": messages,
            "meta":     ctx,
        })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build instruction-tuning datasets from parsed PT failures"
    )
    ap.add_argument(
        "--input", type=Path,
        default=Path("data/processed/parsed_failures.jsonl"),
    )
    ap.add_argument(
        "--out_dir", type=Path,
        default=Path("data/datasets"),
    )
    ap.add_argument(
        "--failures_only", action="store_true", default=True,
        help="Include only FAILED / ERROR records (default: True)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if args.failures_only and rec.get("status") not in ("FAILED", "ERROR"):
                continue
            records.append(rec)

    logger.info("Loaded %d records (failures_only=%s)", len(records), args.failures_only)

    sum_path  = args.out_dir / "raw_summarize.jsonl"
    qa_path   = args.out_dir / "raw_qa.jsonl"
    chat_path = args.out_dir / "raw_chatbot.jsonl"

    sum_n = qa_n = chat_n = 0
    with (
        sum_path.open("w", encoding="utf-8")  as fs,
        qa_path.open("w", encoding="utf-8")   as fq,
        chat_path.open("w", encoding="utf-8") as fc,
    ):
        for rec in records:
            # Summarization
            sr = build_summarize_record(rec)
            fs.write(json.dumps(sr, ensure_ascii=False) + "\n")
            sum_n += 1

            # Q&A
            for qr in build_qa_records(rec):
                fq.write(json.dumps(qr, ensure_ascii=False) + "\n")
                qa_n += 1

            # Chatbot
            for cr in build_chatbot_records(rec):
                fc.write(json.dumps(cr, ensure_ascii=False) + "\n")
                chat_n += 1

    logger.info("Wrote %6d summarize records → %s", sum_n,  sum_path)
    logger.info("Wrote %6d QA records       → %s", qa_n,   qa_path)
    logger.info("Wrote %6d chatbot records  → %s", chat_n, chat_path)


if __name__ == "__main__":
    main()
