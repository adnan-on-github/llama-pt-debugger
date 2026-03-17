"""
Hardware-aware inference core for the LLaMA-3.1-8B PT Debugger.

Loads the base model + LoRA adapter once and provides a unified
`generate()` entry-point consumed by the three task modules.

Hardware detection:
  1. Checks for Gaudi2 HPU (habana_frameworks)
  2. Falls back to CUDA
  3. Falls back to CPU (for testing only)

Usage:
  from inference.infer import PTDebuggerModel

  model = PTDebuggerModel(
      model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
      adapter_path="outputs/llama-pt-debugger",
  )
  text = model.generate(prompt="[TASK: SUMMARIZE]\n...")
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    try:
        import habana_frameworks.torch as htorch  # noqa: F401
        return "hpu"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    logger.warning("No GPU/HPU found; falling back to CPU (inference will be very slow)")
    return "cpu"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class PTDebuggerModel:
    def __init__(
        self,
        model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.max_new_tokens     = max_new_tokens
        self.temperature        = temperature
        self.top_p              = top_p
        self.repetition_penalty = repetition_penalty

        self.device = device if device != "auto" else _detect_device()
        logger.info("Loading model on device: %s", self.device)

        if self.device == "hpu":
            self._load_hpu(model_path, adapter_path)
        else:
            self._load_cuda_cpu(model_path, adapter_path, torch)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_cuda_cpu(self, model_path: str, adapter_path: Optional[str], torch) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=False,
        )

        if adapter_path and os.path.isdir(adapter_path):
            logger.info("Loading LoRA adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()  # fuse adapter for faster inference
        elif adapter_path:
            logger.warning("Adapter path '%s' not found; running base model only", adapter_path)

        self.model = model.eval()

    def _load_hpu(self, model_path: str, adapter_path: Optional[str]) -> None:
        import habana_frameworks.torch as htorch
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        )

        if adapter_path and os.path.isdir(adapter_path):
            logger.info("Loading LoRA adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
        elif adapter_path:
            logger.warning("Adapter path '%s' not found; running base model only", adapter_path)

        self.model = model.to("hpu").eval()

        # Warm-up HPU graphs for consistent latency
        try:
            import habana_frameworks.torch.hpu.graphs as htgraphs
            self._htgraphs = True
        except ImportError:
            self._htgraphs = False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt:         Full formatted prompt string.
            max_new_tokens: Override instance default.
            temperature:    Override instance default.
            stream:         If True, returns a generator of token strings.
                            (Streaming is only supported on CUDA/CPU; HPU returns full string.)
        """
        import torch

        max_tok = max_new_tokens or self.max_new_tokens
        temp    = temperature    or self.temperature

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_tok,
        )

        # Move inputs to the correct device
        if self.device == "hpu":
            device_obj = "hpu"
        elif self.device == "cuda":
            device_obj = next(self.model.parameters()).device
        else:
            device_obj = "cpu"

        inputs = {k: v.to(device_obj) for k, v in inputs.items()}

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tok,
            temperature=temp,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=temp > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if stream and self.device != "hpu":
            from transformers import TextIteratorStreamer
            import threading

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs["streamer"] = streamer
            thread = threading.Thread(
                target=self.model.generate, kwargs=gen_kwargs
            )
            thread.start()
            return streamer  # type: ignore[return-value]

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        new_ids    = output_ids[0, prompt_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Convenience: chat completion (multi-turn)
    # ------------------------------------------------------------------

    def chat_generate(
        self,
        messages: list[dict],
        max_new_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Format a list of {role, content} messages using the LLaMA-3 template
        and generate the next assistant turn.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens, stream=stream)
