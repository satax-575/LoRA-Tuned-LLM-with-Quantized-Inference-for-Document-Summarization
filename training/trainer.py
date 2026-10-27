"""
training/trainer.py

Sets up the QLoRA fine-tuning run for Llama 3.1 8B on CNN/DailyMail.

A few decisions worth calling out:
  - We use DataCollatorForLanguageModeling (not DataCollatorForSeq2Seq) because
    SFTTrainer expects a causal LM setup, not encoder-decoder.
  - Flash Attention 2 is auto-detected; we fall back to SDPA if it's not available
    (e.g., on older GPUs or CPU).
  - dataset_kwargs={"skip_prepare_dataset": True} is critical: without it, SFTTrainer
    would re-tokenise our already-processed dataset and wipe our -100 label masks.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
from trl import SFTTrainer
from dotenv import load_dotenv
from rich.console import Console

from training.qlora_config import (
    get_bnb_config,
    get_lora_config,
    print_trainable_params,
    load_config,
)
from training.callbacks import (
    PerplexityEarlyStoppingCallback,
    TrainingProgressCallback,
    ModelCheckpointCallback,
)

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)


def _detect_attn_implementation() -> str:
    """
    Pick the best available attention implementation.

    We try Flash Attention 2 first (fastest, needs Ampere+ GPU and flash-attn
    installed), then fall back to SDPA (built into PyTorch 2.0+), and finally
    to the plain eager implementation if nothing else works.
    """
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            # Check for Ampere (sm_80+) or newer compute capability
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 8:
                return "flash_attention_2"
    except ImportError:
        pass

    # PyTorch 2.0+ SDPA is always a safe fallback
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"

    return "eager"


class QLoRATrainer:
    """
    End-to-end QLoRA training pipeline for Llama 3.1 8B on CNN/DailyMail.

    Stack:
      - Base model: Llama 3.1 8B loaded in 4-bit NF4 via bitsandbytes
      - Adapter: LoRA rank-16, alpha=32 (scaling=2.0), applied via PEFT
      - Trainer: SFTTrainer from TRL with perplexity early stopping
      - Optimizer: paged_adamw_8bit (much lower memory than adamw)
      - Attention: Flash Attention 2 if available, otherwise SDPA
    """

    def __init__(self, config_path: str = "configs/training_config.yaml"):
        self.cfg = load_config(config_path)
        self.hf_token = os.environ.get("HF_TOKEN")
        self.model_name = self.cfg["model"]["name"]
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model_and_tokenizer(self):
        """Download (or load from cache) Llama 3.1 8B and quantize it on the fly."""
        attn_impl = _detect_attn_implementation()
        console.print(
            f"\n[bold cyan]Loading {self.model_name} in 4-bit NF4 "
            f"(attn: {attn_impl})...[/bold cyan]"
        )

        bnb_config = get_bnb_config(self.cfg)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",              # Spread across all available GPUs, then CPU
            trust_remote_code=True,
            token=self.hf_token,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )

        # Prepares the frozen quantized layers for training and enables gradient
        # checkpointing to trade compute for memory during the backward pass.
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True,
        )
        self.model.config.use_cache = False   # Must be off when using grad checkpointing

        lora_config = get_lora_config(self.cfg)
        self.model = get_peft_model(self.model, lora_config)

        print_trainable_params(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"  # Right-pad for causal LM training

        console.print("[green]\u2713 Model and tokenizer loaded[/green]")

    def load_dataset(self, dataset_path: str = "./data/processed"):
        """Load pre-tokenized + label-masked dataset from disk."""
        console.print(f"[cyan]Loading dataset from {dataset_path}...[/cyan]")
        dataset = load_from_disk(dataset_path)
        console.print(
            f"[green]✓ Dataset: train={len(dataset['train'])}, "
            f"val={len(dataset['validation'])}[/green]"
        )
        return dataset

    def build_training_args(self) -> TrainingArguments:
        """Build HuggingFace TrainingArguments from config."""
        t_cfg = self.cfg["training"]
        Path(t_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

        return TrainingArguments(
            output_dir=t_cfg["output_dir"],
            num_train_epochs=t_cfg["num_train_epochs"],
            per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
            learning_rate=t_cfg["learning_rate"],
            weight_decay=t_cfg["weight_decay"],
            warmup_ratio=t_cfg["warmup_ratio"],
            lr_scheduler_type=t_cfg["lr_scheduler_type"],
            optim=t_cfg["optim"],
            logging_steps=t_cfg["logging_steps"],
            eval_strategy="steps",
            eval_steps=t_cfg["eval_steps"],
            save_strategy="steps",
            save_steps=t_cfg["save_steps"],
            save_total_limit=t_cfg["save_total_limit"],
            load_best_model_at_end=t_cfg["load_best_model_at_end"],
            metric_for_best_model=t_cfg["metric_for_best_model"],
            greater_is_better=t_cfg["greater_is_better"],
            bf16=t_cfg["bf16"],
            fp16=t_cfg["fp16"],
            max_grad_norm=t_cfg["max_grad_norm"],
            group_by_length=t_cfg["group_by_length"],
            dataloader_num_workers=t_cfg["dataloader_num_workers"],
            report_to=t_cfg["report_to"],
            run_name=t_cfg["run_name"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Required for PEFT
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )

    def build_trainer(self, dataset) -> SFTTrainer:
        """
        Wire up SFTTrainer with our pre-tokenized dataset.

        Two things to know:
          1. `dataset_kwargs={"skip_prepare_dataset": True}` — our dataset is already
             tokenized and label-masked (labels=-100 on prompt tokens). Without this flag,
             SFTTrainer re-tokenises the data and overwrites our masks.
          2. DataCollatorForLanguageModeling with mlm=False — pads input_ids and
             correctly propagates the -100 labels. DataCollatorForSeq2Seq is for
             encoder-decoder models and doesn't work here.
        """
        training_args = self.build_training_args()

        # ── Callbacks ──
        es_cfg = self.cfg["early_stopping"]
        callbacks = [
            PerplexityEarlyStoppingCallback(
                patience=es_cfg["patience"],
                threshold=es_cfg["threshold"],
            ),
            TrainingProgressCallback(log_every=training_args.logging_steps),
            ModelCheckpointCallback(),
        ]

        # DataCollatorForLanguageModeling in causal (mlm=False) mode pads input_ids
        # and copies labels, preserving the -100 masks we set during preprocessing.
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # Slight efficiency gain with tensor cores
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=collator,
            callbacks=callbacks,
            max_seq_length=self.cfg["data"]["max_source_length"] + self.cfg["data"]["max_target_length"] + 128,
            dataset_kwargs={"skip_prepare_dataset": True},  # Dataset already processed
        )
        return self.trainer

    def train(self, dataset_path: str = "./data/processed") -> dict:
        """Run full QLoRA training pipeline."""
        console.print("\n[bold magenta]═══ Starting QLoRA Training ═══[/bold magenta]")

        if self.model is None:
            self.load_model_and_tokenizer()

        dataset = self.load_dataset(dataset_path)
        self.build_trainer(dataset)

        console.print("\n[bold green]🚀 Training started...[/bold green]\n")
        result = self.trainer.train()

        # ── Save final adapter ──
        output_dir = self.cfg["training"]["output_dir"]
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        console.print(f"\n[bold green]✓ Training complete![/bold green]")
        console.print(f"[green]  Adapter saved to: {output_dir}[/green]")
        console.print(f"[green]  Training loss: {result.training_loss:.4f}[/green]")

        return {
            "training_loss": result.training_loss,
            "global_step": result.global_step,
            "output_dir": output_dir,
        }

    def merge_and_save(self, output_dir: str = "./outputs/merged_model"):
        """Merge LoRA adapter into base model and save full weights."""
        console.print(f"\n[cyan]Merging LoRA adapter into base model...[/cyan]")

        # Ensure we're operating on the PEFT model
        merged = self.model.merge_and_unload()

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir)

        console.print(f"[green]✓ Merged model saved to: {output_dir}[/green]")
