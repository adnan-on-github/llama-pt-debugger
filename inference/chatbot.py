"""
Stateful multi-turn chatbot module for interactive PT failure debugging.

Maintains a conversation history per session and generates contextual
responses using the fine-tuned LLaMA-3.1-8B model.

Usage:
  from inference.chatbot import Chatbot
  bot = Chatbot(model)
  bot.set_failure_context(failure_dict)          # optional — inject failure upfront
  response = bot.chat("What is the root cause?")
  response = bot.chat("How should I fix it?")
  bot.reset()                                    # start a new session
"""

from typing import Optional, Union

from inference.infer import PTDebuggerModel

SYSTEM_PROMPT = (
    "You are an expert PyTorch and Gaudi2 training framework debugger. "
    "Help the user understand and fix test failures. "
    "Be concise (2-4 sentences), specific, and actionable. "
    "Reference specific error types, Python modules, or stack frames where relevant. "
    "If the user provides a failure context, use it to ground your answers."
)

FAILURE_CONTEXT_TEMPLATE = """\
The user is debugging the following test failure:

Suite:      {suite_name}
Test:       {test_name}
Class:      {class_name}
Status:     {status}
Error Type: {error_type}
Error Msg:  {error_message}
Stack Trace:
{stack_trace}
"""


class Chatbot:
    def __init__(
        self,
        model: PTDebuggerModel,
        system_prompt: str = SYSTEM_PROMPT,
        max_history_turns: int = 20,
        max_new_tokens: int = 256,
    ) -> None:
        self.model              = model
        self.max_history_turns  = max_history_turns
        self.max_new_tokens     = max_new_tokens
        self._system_prompt     = system_prompt
        self._history: list[dict] = []  # {"role": ..., "content": ...}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history and start a fresh session."""
        self._history = []

    def set_failure_context(self, context: Union[str, dict]) -> None:
        """
        Inject a failure context as the first user message so the model
        has grounding information before any follow-up questions.
        """
        ctx_text = self._format_context(context)
        # Replace any existing context messages to avoid duplication
        self._history = [
            m for m in self._history
            if not m.get("_is_context", False)
        ]
        self._history.insert(0, {
            "role":        "user",
            "content":     f"I need help debugging this failure:\n\n{ctx_text}",
            "_is_context": True,
        })
        # Add a placeholder assistant acknowledgement so the model knows what role to play
        self._history.insert(1, {
            "role":        "assistant",
            "content":     (
                "I can see the failure. I'll help you debug it. "
                "What would you like to know first?"
            ),
            "_is_context": True,
        })

    def get_history(self) -> list[dict]:
        """Return conversation history (role + content only, no internal flags)."""
        return [{"role": m["role"], "content": m["content"]} for m in self._history]

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        max_new_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Send a user message and return the assistant's response.

        Args:
            user_message:   The user's question or statement.
            max_new_tokens: Override for this turn.
            stream:         If True, returns a TextIteratorStreamer (non-HPU only).
        """
        self._history.append({"role": "user", "content": user_message})

        # Build the messages list for the model
        messages = [{"role": "system", "content": self._system_prompt}]
        # Trim history to avoid exceeding context window
        trimmed = self._trim_history()
        messages.extend(trimmed)

        response = self.model.chat_generate(
            messages,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            stream=stream,
        )

        if not stream:
            self._history.append({"role": "assistant", "content": response})

        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_history(self) -> list[dict]:
        """Keep only role+content, and trim to max_history_turns pairs."""
        clean = [{"role": m["role"], "content": m["content"]} for m in self._history]
        max_messages = self.max_history_turns * 2  # user + assistant per turn
        if len(clean) > max_messages:
            # Always keep the context block (first 2 messages) if present
            context_block = []
            remaining = clean
            if self._history and self._history[0].get("_is_context"):
                context_block = clean[:2]
                remaining = clean[2:]
            tail = remaining[-(max_messages - len(context_block)):]
            return context_block + tail
        return clean

    def _format_context(self, context: Union[str, dict]) -> str:
        if isinstance(context, str):
            return context.strip()
        defaults = {
            "suite_name":    "",
            "class_name":    "",
            "test_name":     "",
            "status":        "FAILED",
            "error_type":    "",
            "error_message": "",
            "stack_trace":   "",
        }
        ctx = {**defaults, **context}
        # Truncate stack trace to keep prompt manageable
        stack_lines = ctx["stack_trace"].splitlines()
        if len(stack_lines) > 30:
            ctx["stack_trace"] = "\n".join(stack_lines[:30]) + "\n... (truncated)"
        return FAILURE_CONTEXT_TEMPLATE.format(**ctx).strip()
