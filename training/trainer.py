"""
QLoRA Trainer Setup
Configures Llama 3.1 8B with QLoRA + SFTTrainer for instruction fine-tuning
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
    DataCollatorForSeq2Seq,
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


class QLoRATrainer:
    """
    Full QLoRA training pipeline for Llama 3.1 8B on CNN/DailyMail.
    
    Architecture:
    - Base: Llama 3.1 8B loaded in 4-bit NF4 via bitsandbytes
    - Adapter: LoRA rank-16, alpha=32 via PEFT
    - Trainer: SFTTrainer (TRL) with perplexity early stopping
    - Optimizer: paged_adamw_8bit (memory-efficient)
    """

    def __init__(self, config_path: str = "configs/training_config.yaml"):
        self.cfg = load_config(config_path)
        self.hf_token = os.environ.get("HF_TOKEN")
        self.model_name = self.cfg["model"]["name"]
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model_and_tokenizer(self):
        """Load Llama 3.1 8B in 4-bit NF4 from HuggingFace Hub."""
        console.print(
            f"\n[bold cyan]Loading {self.model_name} in 4-bit NF4...[/bold cyan]"
        )

        # ── 4-bit NF4 Quantization Config ──
        bnb_config = get_bnb_config(self.cfg)

        # ── Load Base Model (quantized) ──
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",              # Spread across available GPUs/CPU
            trust_remote_code=True,
            token=self.hf_token,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # Flash Attention 2 on T4
        )

        # ── Prepare for k-bit training (gradient checkpointing + cast) ──
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True,
        )
        self.model.config.use_cache = False   # Required for gradient checkpointing

        # ── Apply LoRA Adapter ──
        lora_config = get_lora_config(self.cfg)
        self.model = get_peft_model(self.model, lora_config)

        print_trainable_params(self.model)

        # ── Tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"  # Important for causal LM

        console.print("[green]✓ Model and tokenizer loaded[/green]")

    def load_dataset(self, dataset_path: str = "./data/processed"):
        """Load pre-tokenized dataset from disk."""
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
            remove_unused_columns=False,
        )

    def build_trainer(self, dataset) -> SFTTrainer:
        """Construct SFTTrainer with all callbacks."""
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

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                label_pad_token_id=-100,      # Ignore padding in loss
                pad_to_multiple_of=8,          # Efficient tensor ops
            ),
            callbacks=callbacks,
            max_seq_length=self.cfg["data"]["max_source_length"] + self.cfg["data"]["max_target_length"],
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
        from peft import PeftModel

        console.print(f"\n[cyan]Merging LoRA adapter into base model...[/cyan]")
        self.model = self.model.merge_and_unload()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        console.print(f"[green]✓ Merged model saved to: {output_dir}[/green]")
