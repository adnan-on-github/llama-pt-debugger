"""
Summarization task module.

Accepts raw failure text, XML snippet, or a structured dict and
returns a human-readable failure summary from the fine-tuned model.

Usage:
  from inference.summarize import Summarizer
  s = Summarizer(model)
  print(s.run("Suite: ... Test: ... Error: ..."))
"""

import xml.etree.ElementTree as ET
from typing import Union

from inference.infer import PTDebuggerModel

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


class Summarizer:
    def __init__(self, model: PTDebuggerModel) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        failure_input: Union[str, dict],
        max_new_tokens: int = 256,
        stream: bool = False,
    ) -> str:
        """
        Summarize a test failure.

        Args:
            failure_input: Either:
                - A raw text description of the failure
                - A dict with keys: suite_name, test_name, class_name,
                  status, error_type, error_message, stack_trace
                - A JUnit XML string (auto-parsed)
            max_new_tokens: Maximum tokens for the generated summary.
            stream:         If True, return a token streamer instead of a string.
        """
        ctx = self._normalize(failure_input)
        prompt = SUMMARIZE_PROMPT.format(instruction=SUMMARIZE_INSTRUCTION, **ctx)
        return self.model.generate(prompt, max_new_tokens=max_new_tokens, stream=stream)

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    def _normalize(self, inp: Union[str, dict]) -> dict:
        if isinstance(inp, dict):
            return self._fill_defaults(inp)

        # Try to parse as JUnit XML
        stripped = inp.strip()
        if stripped.startswith("<"):
            try:
                ctx = self._parse_xml_snippet(stripped)
                if ctx:
                    return ctx
            except ET.ParseError:
                pass

        # Treat as free-form text failure description
        return {
            "suite_name":    "Unknown Suite",
            "class_name":    "",
            "test_name":     "Unknown Test",
            "status":        "FAILED",
            "error_type":    "",
            "error_message": "",
            "stack_trace":   inp.strip(),
        }

    def _fill_defaults(self, d: dict) -> dict:
        defaults = {
            "suite_name":    "",
            "class_name":    "",
            "test_name":     "",
            "status":        "FAILED",
            "error_type":    "",
            "error_message": "",
            "stack_trace":   "",
        }
        return {**defaults, **d}

    def _parse_xml_snippet(self, xml_str: str) -> dict:
        """Extract failure info from a single <testcase> or <testsuite> XML fragment."""
        root = ET.fromstring(xml_str)

        # Normalize — find the first testcase
        if root.tag == "testcase":
            tc = root
        elif root.tag in ("testsuite", "testsuites"):
            tc = root.find(".//testcase")
        else:
            return {}

        if tc is None:
            return {}

        def attr(e, k, d=""):
            return (e.get(k) or d).strip()

        def child_text(e, tag):
            c = e.find(tag)
            return (c.text or "").strip() if c is not None else ""

        failure = tc.find("failure")
        error   = tc.find("error")
        node    = failure if failure is not None else error

        suite = tc.getparent() if hasattr(tc, "getparent") else root

        return {
            "suite_name":    attr(suite, "name", "Unknown Suite"),
            "class_name":    attr(tc, "classname"),
            "test_name":     attr(tc, "name", "Unknown Test"),
            "status":        "FAILED" if failure is not None else "ERROR",
            "error_type":    attr(node, "type") if node is not None else "",
            "error_message": attr(node, "message") if node is not None else "",
            "stack_trace":   (node.text or "").strip() if node is not None else "",
        }
