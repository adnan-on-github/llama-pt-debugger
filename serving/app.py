"""
Gradio three-tab web UI for the LLaMA-3.1-8B PT Debugger.

Tabs:
  1. Summarize  — paste XML / text → get a structured failure summary
  2. Q&A        — paste failure context + ask a question → get an answer
  3. Chatbot    — interactive streaming multi-turn debug assistant

Launch:
  python serving/app.py
  python serving/app.py --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \\
                        --adapter_path outputs/llama-pt-debugger \\
                        --device auto --port 7860
"""

import argparse
import logging
import os
import sys

# Add project root to path so `inference` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parse arguments early so --help works before imports
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="PT Debugger Gradio UI")
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
        choices=["auto", "hpu", "cuda", "cpu"],
    )
    ap.add_argument("--max_new_tokens", type=int, default=int(os.getenv("MAX_NEW_TOKENS", "400")))
    ap.add_argument("--port",   type=int,  default=int(os.getenv("GRADIO_PORT", "7860")))
    ap.add_argument("--share",  action="store_true", default=False)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Lazy model loading (singleton)
# ---------------------------------------------------------------------------

_model = None
_summarizer = None
_qa_module = None
_chatbot = None


def get_model(model_path, adapter_path, device, max_new_tokens):
    global _model, _summarizer, _qa_module, _chatbot
    if _model is None:
        from inference.chatbot import Chatbot
        from inference.infer import PTDebuggerModel
        from inference.qa import QAModule
        from inference.summarize import Summarizer

        logger.info("Loading model (first request)...")
        _model      = PTDebuggerModel(
            model_path=model_path,
            adapter_path=adapter_path,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        _summarizer = Summarizer(_model)
        _qa_module  = QAModule(_model)
        _chatbot    = Chatbot(_model, max_new_tokens=max_new_tokens)
        logger.info("Model ready.")
    return _summarizer, _qa_module, _chatbot


# ---------------------------------------------------------------------------
# Tab 1 — Summarize
# ---------------------------------------------------------------------------

def summarize_fn(failure_text, model_path, adapter_path, device, max_new_tokens):
    if not failure_text.strip():
        return "⚠️ Please paste a failure log, XML snippet, or error description above."
    summarizer, _, _ = get_model(model_path, adapter_path, device, max_new_tokens)
    try:
        return summarizer.run(failure_text, max_new_tokens=max_new_tokens)
    except Exception as exc:
        logger.exception("Summarize error")
        return f"❌ Error: {exc}"


# ---------------------------------------------------------------------------
# Tab 2 — Q&A
# ---------------------------------------------------------------------------

def qa_fn(context_text, question, model_path, adapter_path, device, max_new_tokens):
    if not context_text.strip():
        return "⚠️ Please provide a failure context."
    if not question.strip():
        return "⚠️ Please enter a question."
    _, qa_module, _ = get_model(model_path, adapter_path, device, max_new_tokens)
    try:
        return qa_module.run(context_text, question, max_new_tokens=max_new_tokens)
    except Exception as exc:
        logger.exception("Q&A error")
        return f"❌ Error: {exc}"


# ---------------------------------------------------------------------------
# Tab 3 — Chatbot
# ---------------------------------------------------------------------------

def chat_fn(user_message, history, context_text, model_path, adapter_path, device, max_new_tokens):
    """Gradio ChatInterface compatible function with streaming."""
    _, _, chatbot = get_model(model_path, adapter_path, device, max_new_tokens)

    # Inject failure context once per conversation
    if not history and context_text.strip():
        chatbot.reset()
        chatbot.set_failure_context(context_text)
    elif not history:
        chatbot.reset()

    try:
        streamer = chatbot.chat(user_message, stream=True)
        response = ""
        if hasattr(streamer, "__iter__"):
            for token in streamer:
                response += token
                yield response
        else:
            yield str(streamer)
    except Exception as exc:
        logger.exception("Chatbot error")
        yield f"❌ Error: {exc}"


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_app(args):
    import gradio as gr

    # Shared state passed as hidden inputs
    model_state = gr.State(args.model_path)
    adapter_state = gr.State(args.adapter_path)
    device_state = gr.State(args.device)
    max_tok_state = gr.State(args.max_new_tokens)

    with gr.Blocks(
        title="PT Test Failure Debugger",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 900px; margin: auto; }",
    ) as demo:
        gr.Markdown(
            "# 🔍 PyTorch Training Test Failure Debugger\n"
            "Powered by `LLaMA-3.1-8B-Instruct` fine-tuned on PT test failure reports.\n"
            "Choose a tab to **summarize** a failure, ask a **Q&A** question, or start a **chat** session."
        )

        # ------------------------------------------------------------------
        # Tab 1: Summarize
        # ------------------------------------------------------------------
        with gr.Tab("📋 Summarize"):
            gr.Markdown(
                "Paste a JUnit XML snippet, raw log output, or a plain-text "
                "description of a test failure. The model will produce a structured summary "
                "identifying root cause, affected component, and recommended action."
            )
            with gr.Row():
                sum_input = gr.Textbox(
                    label="Failure input (XML, log, or text)",
                    placeholder='Paste JUnit XML or error log here...',
                    lines=14,
                    max_lines=30,
                )
            sum_output = gr.Textbox(label="Summary", lines=8, interactive=False)
            with gr.Row():
                sum_btn   = gr.Button("Summarize", variant="primary")
                sum_clear = gr.ClearButton([sum_input, sum_output], value="Clear")

            sum_btn.click(
                fn=summarize_fn,
                inputs=[sum_input, model_state, adapter_state, device_state, max_tok_state],
                outputs=sum_output,
            )

        # ------------------------------------------------------------------
        # Tab 2: Q&A
        # ------------------------------------------------------------------
        with gr.Tab("❓ Q&A"):
            gr.Markdown(
                "Provide the failure context (log, XML, or plain text) and then ask "
                "a specific diagnostic question. The model answers using only the provided context."
            )
            qa_context = gr.Textbox(
                label="Failure context",
                placeholder="Paste the failure log or error description...",
                lines=10,
            )
            qa_question = gr.Textbox(
                label="Question",
                placeholder="e.g. What is the root cause? How do I fix this?",
                lines=2,
            )
            qa_output = gr.Textbox(label="Answer", lines=8, interactive=False)
            with gr.Row():
                qa_btn   = gr.Button("Ask", variant="primary")
                qa_clear = gr.ClearButton([qa_context, qa_question, qa_output], value="Clear")

            qa_btn.click(
                fn=qa_fn,
                inputs=[qa_context, qa_question, model_state, adapter_state, device_state, max_tok_state],
                outputs=qa_output,
            )

            # Preset common questions
            gr.Markdown("**Quick questions:**")
            with gr.Row():
                for q in [
                    "What is the root cause?",
                    "Which component is responsible?",
                    "How do I fix this?",
                    "Is this a distributed issue?",
                ]:
                    gr.Button(q, size="sm").click(
                        fn=lambda ctx, _q=q, mp=args.model_path, ap=args.adapter_path, d=args.device, mt=args.max_new_tokens: qa_fn(ctx, _q, mp, ap, d, mt),
                        inputs=[qa_context],
                        outputs=qa_output,
                    )

        # ------------------------------------------------------------------
        # Tab 3: Chatbot
        # ------------------------------------------------------------------
        with gr.Tab("💬 Chatbot"):
            gr.Markdown(
                "Interactive multi-turn debug assistant. Optionally paste a failure context "
                "below to give the assistant full background before you start chatting."
            )
            chat_context = gr.Textbox(
                label="(Optional) Failure context — paste before starting chat",
                placeholder="Paste failure log here to give the assistant context...",
                lines=6,
            )
            chatbot_ui = gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[chat_context, model_state, adapter_state, device_state, max_tok_state],
                title="",
                retry_btn=None,
                undo_btn="↩ Undo",
                clear_btn="🗑 Clear",
            )

        # ------------------------------------------------------------------
        # Footer
        # ------------------------------------------------------------------
        gr.Markdown(
            f"---\n"
            f"Model: `{args.model_path}` | Adapter: `{args.adapter_path}` | "
            f"Device: `{args.device}` | Max tokens: `{args.max_new_tokens}`"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    app  = build_app(args)
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
