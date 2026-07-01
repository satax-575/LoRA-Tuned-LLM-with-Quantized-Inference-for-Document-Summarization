"""
QLoRA Configuration for Llama 3.1 8B
4-bit NF4 quantization + LoRA adapter (rank-16, alpha=32)
"""

import os
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType
import yaml


def get_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    """
    Build BitsAndBytes 4-bit NF4 quantization config.
    
    Key decisions:
    - bnb_4bit_quant_type='nf4': Normal Float 4-bit outperforms FP4 on NLP tasks
    - bnb_4bit_compute_dtype=bfloat16: Numerically stable; best on Ampere/newer GPUs
    - bnb_4bit_use_double_quant=True: Nested quantization saves ~0.4 bits/param extra
    """
    quant_cfg = cfg["quantization"]
    compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],   # "nf4"
        bnb_4bit_compute_dtype=compute_dtype,                    # torch.bfloat16
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )


def get_lora_config(cfg: dict) -> LoraConfig:
    """
    Build PEFT LoRA adapter config.
    
    rank=16, alpha=32 → scaling factor = alpha/rank = 2.0
    target_modules: all attention projections + MLP gate/up/down
    Trainable params: ~8M / 8B ≈ 0.1% of base model
    """
    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=lora_cfg["r"],                          # LoRA rank
        lora_alpha=lora_cfg["lora_alpha"],         # Scaling: alpha/r = 2.0
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
        # Inference mode disabled for training
        inference_mode=False,
    )


def compute_trainable_params(model) -> dict:
    """
    Compute and display trainable vs total parameter counts.
    Returns dict with counts and percentage.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    pct = 100 * trainable_params / total_params

    result = {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(pct, 4),
        "trainable_M": round(trainable_params / 1e6, 2),
        "total_B": round(total_params / 1e9, 2),
    }
    return result


def print_trainable_params(model):
    """Print a formatted trainable parameter summary."""
    stats = compute_trainable_params(model)
    print("\n" + "═" * 55)
    print("  QLoRA Parameter Summary — Llama 3.1 8B")
    print("═" * 55)
    print(f"  Trainable params : {stats['trainable_params']:>15,}  ({stats['trainable_M']:.2f}M)")
    print(f"  Total params     : {stats['total_params']:>15,}  ({stats['total_B']:.2f}B)")
    print(f"  Trainable %      : {stats['trainable_pct']:>15.4f}%")
    print("═" * 55 + "\n")


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    """Load training configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
