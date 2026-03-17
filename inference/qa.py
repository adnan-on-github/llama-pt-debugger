"""
Question-Answer task module.

Accepts a failure context dict or text plus a question string and
returns an answer from the fine-tuned model.

Usage:
  from inference.qa import QAModule
  qa = QAModule(model)
  answer = qa.run(context=failure_dict, question="What is the root cause?")
"""

from typing import Union

from inference.infer import PTDebuggerModel

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


class QAModule:
    def __init__(self, model: PTDebuggerModel) -> None:
        self.model = model

    def run(
        self,
        context: Union[str, dict],
        question: str,
        max_new_tokens: int = 256,
        stream: bool = False,
    ) -> str:
        """
        Answer a question about a test failure.

        Args:
            context:        Either a dict with failure fields or a raw text block.
            question:       The specific question to answer.
            max_new_tokens: Maximum tokens for the generated answer.
            stream:         If True, returns a TextIteratorStreamer.
        """
        ctx = self._normalize(context)
        prompt = QA_PROMPT.format(instruction=QA_INSTRUCTION, question=question, **ctx)
        return self.model.generate(prompt, max_new_tokens=max_new_tokens, stream=stream)

    def run_batch(
        self,
        context: Union[str, dict],
        questions: list[str],
        max_new_tokens: int = 256,
    ) -> list[dict]:
        """
        Answer multiple questions about the same failure context.
        Returns a list of {"question": ..., "answer": ...} dicts.
        """
        results = []
        for q in questions:
            answer = self.run(context, q, max_new_tokens=max_new_tokens)
            results.append({"question": q, "answer": answer})
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, inp: Union[str, dict]) -> dict:
        defaults = {
            "suite_name":    "",
            "class_name":    "",
            "test_name":     "",
            "status":        "FAILED",
            "error_type":    "",
            "error_message": "",
            "stack_trace":   "",
        }
        if isinstance(inp, dict):
            return {**defaults, **inp}
        # Free-form text
        return {**defaults, "stack_trace": inp.strip()}
