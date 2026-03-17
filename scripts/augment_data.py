#!/usr/bin/env python3
"""
Synthetic data augmentation via an OpenAI-compatible LLM API.

Reads:  data/datasets/raw_summarize.jsonl
        data/datasets/raw_qa.jsonl
        data/datasets/raw_chatbot.jsonl
Writes: data/datasets/augmented_summarize.jsonl
        data/datasets/augmented_qa.jsonl
        data/datasets/augmented_chatbot.jsonl

Supports any OpenAI-compatible endpoint (OpenAI, local vLLM, TGI).
Configure via environment variables:
  LLM_BASE_URL  — API base URL (default: https://api.openai.com/v1)
  LLM_API_KEY   — API key       (required; use 'EMPTY' for local endpoints)
  LLM_MODEL     — model name    (default: gpt-4o)

Usage:
  export LLM_API_KEY=sk-...
  python scripts/augment_data.py

  # Local vLLM / TGI:
  export LLM_BASE_URL=http://localhost:8000/v1
  export LLM_API_KEY=EMPTY
  export LLM_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
  python scripts/augment_data.py --tasks summarize qa chatbot
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (self-contained; mirrored from create_datasets.py)
# ---------------------------------------------------------------------------

_QA_INSTRUCTION = (
    "You are an expert PyTorch/Gaudi training framework debugger. "
    "Answer the question using only the provided test failure context."
)

_QA_PROMPT = """\
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

_EXTRA_Q_PROMPT = """\
Given this PyTorch/Gaudi test failure, generate exactly 5 additional diagnostic questions \
that a developer would ask to debug it. Return a JSON array of strings and nothing else.

Error Type:    {error_type}
Error Message: {error_message}
Stack Trace (first 20 lines):
{stack_trace_short}

JSON array:"""

_SUMMARIZE_SYSTEM = (
    "You are an expert PyTorch and Gaudi2 training framework engineer. "
    "Write concise, accurate summaries of test failures. "
    "Always cover: (1) root cause, (2) affected component, (3) recommended action. "
    "Be factual and specific; do not speculate beyond what the stack trace shows."
)

_QA_SYSTEM = (
    "You are an expert PyTorch/Gaudi2 training framework debugger. "
    "Answer questions about test failures accurately, concisely, and actionably."
)

_CHAT_FILL_SYSTEM = (
    "You are an expert PyTorch and Gaudi2 training framework debugger helping a user debug failures. "
    "Keep responses concise (2-4 sentences), specific, and actionable."
)


# ---------------------------------------------------------------------------
# LLM client (thin wrapper over openai SDK — works with any compatible endpoint)
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not found. Install with: pip install openai>=1.0"
            )
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.retries     = retries
        self.retry_delay = retry_delay

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Single-turn completion."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._call(messages)

    def chat_complete(self, messages: list[dict]) -> str:
        """Multi-turn completion from a list of message dicts."""
        return self._call(messages)

    def _call(self, messages: list[dict]) -> str:
        for attempt in range(self.retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1, self.retries, exc,
                )
                if attempt < self.retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return "__LLM_FAILED__"


# ---------------------------------------------------------------------------
# Per-task augmentation functions
# ---------------------------------------------------------------------------

def augment_summarize(records: list[dict], client: LLMClient) -> list[dict]:
    """Generate a high-quality summary completion for each summarization record."""
    out: list[dict] = []
    for i, rec in enumerate(records):
        completion = client.complete(rec["prompt"], system=_SUMMARIZE_SYSTEM)
        out.append({**rec, "completion": completion})
        if (i + 1) % 50 == 0:
            logger.info("  [summarize] %d / %d done", i + 1, len(records))
    return out


def augment_qa(records: list[dict], client: LLMClient) -> list[dict]:
    """
    Fill in answers for seed Q&A records and generate 5 extra questions per failure.
    Groups records by failure identity to avoid calling the extra-question prompt
    once per question (it runs once per unique failure).
    """
    out: list[dict] = []

    # Group by failure key
    by_failure: dict[str, list[dict]] = {}
    for rec in records:
        key = f"{rec['meta']['suite_name']}::{rec['meta']['test_name']}"
        by_failure.setdefault(key, []).append(rec)

    for i, (_, recs) in enumerate(by_failure.items()):
        # 1. Answer existing seed questions
        for rec in recs:
            answer = client.complete(rec["prompt"], system=_QA_SYSTEM)
            out.append({**rec, "completion": answer})

        # 2. Generate 5 extra questions for this failure
        meta = recs[0]["meta"]
        stack_short = "\n".join(
            meta["stack_trace"].splitlines()[:20]
        )
        eq_prompt = _EXTRA_Q_PROMPT.format(
            **meta, stack_trace_short=stack_short
        )
        eq_raw = client.complete(eq_prompt, system=_QA_SYSTEM)
        try:
            extra_questions: list[str] = json.loads(eq_raw)
            if not isinstance(extra_questions, list):
                raise ValueError("Expected JSON list")
        except (json.JSONDecodeError, ValueError):
            logger.debug("Could not parse extra questions for %s: %s", meta["test_name"], eq_raw[:80])
            extra_questions = []

        for q in extra_questions[:5]:
            if not isinstance(q, str) or not q.strip():
                continue
            prompt = _QA_PROMPT.format(instruction=_QA_INSTRUCTION, **meta, question=q.strip())
            answer = client.complete(prompt, system=_QA_SYSTEM)
            out.append({
                "task":       "qa",
                "prompt":     prompt,
                "completion": answer,
                "question":   q.strip(),
                "meta":       meta,
            })

        if (i + 1) % 20 == 0:
            logger.info("  [qa] %d / %d failures done", i + 1, len(by_failure))

    return out


def augment_chatbot(records: list[dict], client: LLMClient) -> list[dict]:
    """
    Fill __AUGMENT__ placeholder turns in chat conversations.
    Replays the conversation up to each placeholder so the LLM
    has the full context before generating the next assistant turn.
    """
    out: list[dict] = []
    for i, rec in enumerate(records):
        messages = list(rec["messages"])
        filled: list[dict] = []
        history: list[dict] = []  # Growing conversation to feed the LLM

        for msg in messages:
            if msg["role"] == "assistant" and msg["content"] == "__AUGMENT__":
                # Include history so far (system + all preceding turns)
                response = client.chat_complete(history)
                filled_msg = {"role": "assistant", "content": response}
            else:
                filled_msg = dict(msg)
            filled.append(filled_msg)
            history.append(filled_msg)

        out.append({**rec, "messages": filled})
        if (i + 1) % 50 == 0:
            logger.info("  [chatbot] %d / %d done", i + 1, len(records))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Augment raw datasets with LLM-generated completions"
    )
    ap.add_argument(
        "--datasets_dir", type=Path,
        default=Path("data/datasets"),
    )
    ap.add_argument(
        "--llm_base_url", type=str,
        default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL (env: LLM_BASE_URL)",
    )
    ap.add_argument(
        "--llm_api_key", type=str,
        default=os.getenv("LLM_API_KEY", ""),
        help="API key — use 'EMPTY' for local endpoints (env: LLM_API_KEY)",
    )
    ap.add_argument(
        "--llm_model", type=str,
        default=os.getenv("LLM_MODEL", "gpt-4o"),
        help="Model name (env: LLM_MODEL)",
    )
    ap.add_argument("--max_tokens",  type=int,   default=512)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument(
        "--tasks", nargs="+",
        choices=["summarize", "qa", "chatbot"],
        default=["summarize", "qa", "chatbot"],
    )
    args = ap.parse_args()

    if not args.llm_api_key:
        ap.error(
            "LLM API key is required. "
            "Set the LLM_API_KEY environment variable or pass --llm_api_key. "
            "For local vLLM / TGI endpoints use: export LLM_API_KEY=EMPTY"
        )

    client = LLMClient(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    task_registry = {
        "summarize": ("raw_summarize.jsonl", "augmented_summarize.jsonl", augment_summarize),
        "qa":        ("raw_qa.jsonl",        "augmented_qa.jsonl",        augment_qa),
        "chatbot":   ("raw_chatbot.jsonl",   "augmented_chatbot.jsonl",   augment_chatbot),
    }

    for task in args.tasks:
        in_name, out_name, augment_fn = task_registry[task]
        in_path  = args.datasets_dir / in_name
        out_path = args.datasets_dir / out_name

        if not in_path.exists():
            logger.warning("Input not found for task '%s': %s — skipping", task, in_path)
            continue

        raw: list[dict] = []
        with in_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

        logger.info("Augmenting [%s]: %d records ...", task, len(raw))
        augmented = augment_fn(raw, client)

        with out_path.open("w", encoding="utf-8") as f:
            for rec in augmented:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info("[%s] wrote %d augmented records → %s", task, len(augmented), out_path)


if __name__ == "__main__":
    main()
