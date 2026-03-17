#!/usr/bin/env python3
"""
Main fine-tuning script for LLaMA-3.1-8B + LoRA + DeepSpeed ZeRO-3.

Supports:
  NVIDIA GPU  — transformers + trl.SFTTrainer + peft + deepspeed + accelerate
  Gaudi2 HPU  — optimum-habana GaudiSFTTrainer + peft + deepspeed

Hardware is auto-detected at runtime unless overridden via --device.

Usage (single-node):
  # NVIDIA
  accelerate launch --config_file training/configs/accelerate_nvidia.yaml training/train.py

  # Gaudi2
  python training/launch/run_gaudi2.sh  (wraps gaudi_spawn.py)

  # Multi-node — use the launch scripts:
  bash training/launch/run_nvidia.sh
  bash training/launch/run_gaudi2.sh
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Return 'hpu', 'cuda', or 'cpu'."""
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
    return "cpu"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_jsonl_dataset(path: str):
    from datasets import load_dataset  # noqa: PLC0415
    return load_dataset("json", data_files=path, split="train")


# ---------------------------------------------------------------------------
# NVIDIA training path
# ---------------------------------------------------------------------------

def train_nvidia(cfg: dict, ds_config: str) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    tc  = cfg["training"]
    lc  = cfg["lora"]

    logger.info("=== NVIDIA training path ===")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tc["model_name_or_path"],
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = tc.get("truncation_side", "left")

    # Model — load in bf16 directly (no quantization needed for ZeRO-3)
    model = AutoModelForCausalLM.from_pretrained(
        tc["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",  # speeds up training on A100/H100
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        bias=lc["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lc["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Datasets
    train_ds = load_jsonl_dataset(tc["train_file"])
    val_ds   = load_jsonl_dataset(tc["val_file"])

    # Training arguments
    out_dir = tc["output_dir"]
    sft_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc["per_device_eval_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        bf16=tc["bf16"],
        fp16=tc["fp16"],
        gradient_checkpointing=tc["gradient_checkpointing"],
        eval_strategy=tc["eval_strategy"],
        eval_steps=tc["eval_steps"],
        save_strategy=tc["save_strategy"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        load_best_model_at_end=tc["load_best_model_at_end"],
        metric_for_best_model=tc["metric_for_best_model"],
        logging_steps=tc["logging_steps"],
        report_to=tc.get("report_to", ["tensorboard"]),
        run_name=tc.get("run_name", "llama-pt-debugger"),
        deepspeed=ds_config,
        # SFT-specific
        dataset_text_field=tc["dataset_text_field"],
        max_seq_length=tc["max_seq_length"],
        packing=tc["packing"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Model saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Gaudi2 training path
# ---------------------------------------------------------------------------

def train_gaudi2(cfg: dict, ds_config: str) -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from optimum.habana.trl import GaudiSFTConfig, GaudiSFTTrainer
    except ImportError:
        raise ImportError(
            "optimum-habana is required for Gaudi2 training. "
            "Install from: https://github.com/huggingface/optimum-habana"
        )

    import habana_frameworks.torch as htorch
    import torch
    from datasets import load_dataset

    tc = cfg["training"]
    lc = cfg["lora"]

    logger.info("=== Gaudi2 training path ===")
    logger.info("Gaudi device count: %d", htorch.hpu.device_count())

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tc["model_name_or_path"],
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = tc.get("truncation_side", "left")

    # Model — bf16 on HPU
    model = AutoModelForCausalLM.from_pretrained(
        tc["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        bias=lc["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lc["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Datasets
    train_ds = load_dataset("json", data_files=tc["train_file"], split="train")
    val_ds   = load_dataset("json", data_files=tc["val_file"],   split="train")

    out_dir = tc["output_dir"]

    # GaudiSFTConfig extends TrainingArguments from optimum-habana
    from optimum.habana import GaudiConfig, GaudiTrainingArguments

    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam  = True
    gaudi_config.use_fused_clip_norm = True

    sft_args = GaudiSFTConfig(
        output_dir=out_dir,
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc["per_device_eval_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        bf16=True,
        gradient_checkpointing=tc["gradient_checkpointing"],
        eval_strategy=tc["eval_strategy"],
        eval_steps=tc["eval_steps"],
        save_strategy=tc["save_strategy"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        load_best_model_at_end=tc["load_best_model_at_end"],
        metric_for_best_model=tc["metric_for_best_model"],
        logging_steps=tc["logging_steps"],
        report_to=tc.get("report_to", ["tensorboard"]),
        run_name=tc.get("run_name", "llama-pt-debugger"),
        deepspeed=ds_config,
        # HPU-specific
        use_habana=True,
        use_lazy_mode=True,       # Lazy mode significantly improves HPU throughput
        use_hpu_graphs_for_training=False,  # Disable hpu_graph during training
        gaudi_config_name=None,   # We pass explicit GaudiConfig object
        # SFT-specific
        dataset_text_field=tc["dataset_text_field"],
        max_seq_length=tc["max_seq_length"],
        packing=tc["packing"],
    )

    trainer = GaudiSFTTrainer(
        model=model,
        args=sft_args,
        gaudi_config=gaudi_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Model saved to %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune LLaMA-3.1-8B with LoRA + DeepSpeed ZeRO-3"
    )
    ap.add_argument(
        "--config", type=Path,
        default=Path("training/configs/lora_config.yaml"),
        help="Path to lora_config.yaml",
    )
    ap.add_argument(
        "--device", choices=["hpu", "cuda", "cpu", "auto"],
        default="auto",
        help="Target device (default: auto-detect)",
    )
    ap.add_argument(
        "--deepspeed_config", type=str, default=None,
        help="Path to DeepSpeed JSON config (auto-selected if not provided)",
    )
    ap.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training (set automatically by launch tools)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    device = args.device if args.device != "auto" else detect_device()
    logger.info("Target device: %s", device)

    # Select appropriate DeepSpeed config
    if args.deepspeed_config:
        ds_config = args.deepspeed_config
    elif device == "hpu":
        ds_config = str(Path(__file__).parent / "configs" / "ds_zero3_gaudi2.json")
    else:
        ds_config = str(Path(__file__).parent / "configs" / "ds_zero3_nvidia.json")

    logger.info("DeepSpeed config: %s", ds_config)

    if device == "hpu":
        train_gaudi2(cfg, ds_config)
    else:
        train_nvidia(cfg, ds_config)


if __name__ == "__main__":
    main()
