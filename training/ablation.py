"""
Ablation Study Runner
Grid search over LoRA rank, learning rate, and tokenization strategies
Systematically identifies best hyperparameters to prevent overfitting
"""

import os
import json
import time
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import torch
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
from datasets import load_from_disk
from trl import SFTTrainer
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

from training.qlora_config import get_bnb_config
from training.callbacks import PerplexityEarlyStoppingCallback

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid")


@dataclass
class AblationResult:
    """Stores results of a single ablation run."""
    config_name: str
    lora_rank: int
    learning_rate: float
    tokenization: str
    final_train_loss: float
    final_eval_loss: float
    final_perplexity: float
    best_perplexity: float
    stopped_at_step: int
    train_time_seconds: float
    trainable_params_M: float

    def to_dict(self) -> dict:
        return asdict(self)


class AblationStudy:
    """
    Systematic ablation over:
    1. LoRA rank: [4, 8, 16, 32]
    2. Learning rate: [1e-4, 2e-4, 5e-4]
    3. Tokenization: BPE full-length vs truncated-512
    
    Each run uses 5K training / 1K validation samples for speed.
    Results are saved to JSON + visualized.
    """

    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        dataset_path: str = "./data/processed",
        output_dir: str = "./outputs/ablation",
        quick_mode: bool = True,  # Use small subsets for speed
    ):
        with open(config_path) as f:
            self.base_cfg = yaml.safe_load(f)

        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_mode = quick_mode
        self.hf_token = os.environ.get("HF_TOKEN")
        self.model_name = self.base_cfg["model"]["name"]
        self.results: List[AblationResult] = []

    def _load_model(self, lora_rank: int, lora_alpha: int = None) -> tuple:
        """Load model with specified LoRA rank."""
        if lora_alpha is None:
            lora_alpha = lora_rank * 2  # Keep alpha/rank = 2.0

        bnb_config = get_bnb_config(self.base_cfg)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.config.use_cache = False

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.base_cfg["lora"]["target_modules"],
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=self.hf_token, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model, tokenizer, round(trainable / 1e6, 2)

    def _load_dataset_subset(self, max_length: Optional[int] = None) -> tuple:
        """Load a small dataset subset for quick ablation runs."""
        dataset = load_from_disk(self.dataset_path)
        train_size = 5000 if self.quick_mode else 20000
        val_size = 1000 if self.quick_mode else 2000

        train = dataset["train"].select(range(min(train_size, len(dataset["train"]))))
        val = dataset["validation"].select(range(min(val_size, len(dataset["validation"]))))

        if max_length:
            # Truncate to max_length tokens
            def truncate(examples):
                examples["input_ids"] = [ids[:max_length] for ids in examples["input_ids"]]
                examples["attention_mask"] = [m[:max_length] for m in examples["attention_mask"]]
                examples["labels"] = [l[:max_length] for l in examples["labels"]]
                return examples
            train = train.map(truncate, batched=True)
            val = val.map(truncate, batched=True)

        return train, val

    def run_single(
        self,
        config_name: str,
        lora_rank: int,
        learning_rate: float,
        tokenization: str = "llama31_bpe",
        max_token_length: Optional[int] = None,
    ) -> AblationResult:
        """Run a single ablation configuration."""
        console.print(f"\n[bold cyan]  Running: {config_name}[/bold cyan]")
        console.print(f"  rank={lora_rank}, lr={learning_rate}, tok={tokenization}")

        run_dir = self.output_dir / config_name
        run_dir.mkdir(exist_ok=True)

        model, tokenizer, trainable_M = self._load_model(lora_rank)
        train_ds, val_ds = self._load_dataset_subset(max_token_length)

        training_args = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=100,
            bf16=True,
            max_grad_norm=0.3,
            report_to="none",  # No W&B for ablations
            gradient_checkpointing=True,
            remove_unused_columns=False,
        )

        es_callback = PerplexityEarlyStoppingCallback(patience=2, threshold=0.005)

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            callbacks=[es_callback],
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        start_time = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start_time

        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", float("inf"))
        perplexity = math.exp(min(eval_loss, 20))

        result = AblationResult(
            config_name=config_name,
            lora_rank=lora_rank,
            learning_rate=learning_rate,
            tokenization=tokenization,
            final_train_loss=train_result.training_loss,
            final_eval_loss=eval_loss,
            final_perplexity=perplexity,
            best_perplexity=es_callback.best_perplexity or perplexity,
            stopped_at_step=train_result.global_step,
            train_time_seconds=round(elapsed, 1),
            trainable_params_M=trainable_M,
        )

        # Cleanup GPU memory
        del model
        torch.cuda.empty_cache()

        return result

    def run_rank_ablation(self):
        """Ablate over LoRA ranks [4, 8, 16, 32]."""
        console.print("\n[bold magenta]── LoRA Rank Ablation ──[/bold magenta]")
        ranks = self.base_cfg["ablation"]["lora_ranks"]
        base_lr = self.base_cfg["training"]["learning_rate"]

        for rank in ranks:
            result = self.run_single(
                config_name=f"rank_{rank}",
                lora_rank=rank,
                learning_rate=base_lr,
                tokenization="llama31_bpe",
            )
            self.results.append(result)
            console.print(f"  [green]rank={rank} → perplexity={result.final_perplexity:.4f}[/green]")

    def run_lr_ablation(self):
        """Ablate over learning rates [1e-4, 2e-4, 5e-4]."""
        console.print("\n[bold magenta]── Learning Rate Ablation ──[/bold magenta]")
        lrs = self.base_cfg["ablation"]["learning_rates"]
        best_rank = 16  # From rank ablation

        for lr in lrs:
            result = self.run_single(
                config_name=f"lr_{lr:.0e}",
                lora_rank=best_rank,
                learning_rate=lr,
                tokenization="llama31_bpe",
            )
            self.results.append(result)
            console.print(f"  [green]lr={lr} → perplexity={result.final_perplexity:.4f}[/green]")

    def run_tokenization_ablation(self):
        """Ablate over tokenization strategies."""
        console.print("\n[bold magenta]── Tokenization Strategy Ablation ──[/bold magenta]")

        configs = [
            ("llama31_bpe_full", None),          # Full context ~1024 tokens
            ("llama31_bpe_512", 512),             # Truncated to 512
        ]

        for name, max_len in configs:
            result = self.run_single(
                config_name=f"tok_{name}",
                lora_rank=16,
                learning_rate=2e-4,
                tokenization=name,
                max_token_length=max_len,
            )
            self.results.append(result)
            console.print(f"  [green]{name} → perplexity={result.final_perplexity:.4f}[/green]")

    def print_results_table(self):
        """Print formatted results table."""
        table = Table(title="Ablation Study Results", style="bold magenta")
        for col in ["Config", "Rank", "LR", "Tokenization", "Train Loss", "Eval PPL", "Steps", "Time(s)"]:
            table.add_column(col, style="cyan" if col == "Config" else "white")

        sorted_results = sorted(self.results, key=lambda r: r.final_perplexity)
        for r in sorted_results:
            table.add_row(
                r.config_name,
                str(r.lora_rank),
                f"{r.learning_rate:.1e}",
                r.tokenization,
                f"{r.final_train_loss:.4f}",
                f"{r.final_perplexity:.4f}",
                str(r.stopped_at_step),
                str(r.train_time_seconds),
            )
        console.print(table)

    def plot_results(self):
        """Plot ablation results."""
        if not self.results:
            return

        df = pd.DataFrame([r.to_dict() for r in self.results])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Ablation Study — QLoRA Llama 3.1 8B", fontsize=14, fontweight="bold")

        # Rank ablation
        rank_df = df[df["config_name"].str.startswith("rank_")]
        if not rank_df.empty:
            axes[0].bar(rank_df["lora_rank"].astype(str), rank_df["final_perplexity"], color="#7c3aed")
            axes[0].set_xlabel("LoRA Rank")
            axes[0].set_ylabel("Eval Perplexity")
            axes[0].set_title("LoRA Rank Ablation")

        # LR ablation
        lr_df = df[df["config_name"].str.startswith("lr_")]
        if not lr_df.empty:
            axes[1].bar(lr_df["learning_rate"].astype(str), lr_df["final_perplexity"], color="#0ea5e9")
            axes[1].set_xlabel("Learning Rate")
            axes[1].set_ylabel("Eval Perplexity")
            axes[1].set_title("Learning Rate Ablation")

        # Tokenization ablation
        tok_df = df[df["config_name"].str.startswith("tok_")]
        if not tok_df.empty:
            axes[2].bar(tok_df["tokenization"], tok_df["final_perplexity"], color="#10b981")
            axes[2].set_xlabel("Tokenization")
            axes[2].set_ylabel("Eval Perplexity")
            axes[2].set_title("Tokenization Ablation")
            plt.setp(axes[2].get_xticklabels(), rotation=15)

        plt.tight_layout()
        out = self.output_dir / "ablation_results.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        console.print(f"[green]✓ Ablation plot saved: {out}[/green]")
        plt.close()

    def save_results(self):
        """Save all results as JSON + CSV."""
        records = [r.to_dict() for r in self.results]
        json_path = self.output_dir / "ablation_results.json"
        csv_path = self.output_dir / "ablation_results.csv"

        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)

        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)

        console.print(f"[green]✓ Results saved: {json_path}, {csv_path}[/green]")

    def run_all(self):
        """Run complete ablation study."""
        console.print("\n[bold magenta]═══ Starting Ablation Study ═══[/bold magenta]")
        console.print(f"Quick mode: {self.quick_mode}")

        self.run_rank_ablation()
        self.run_lr_ablation()
        self.run_tokenization_ablation()

        self.print_results_table()
        self.plot_results()
        self.save_results()

        best = min(self.results, key=lambda r: r.final_perplexity)
        console.print(f"\n[bold green]✓ Best config: {best.config_name} "
                      f"(perplexity={best.best_perplexity:.4f})[/bold green]")

        return self.results


if __name__ == "__main__":
    study = AblationStudy(quick_mode=True)
    study.run_all()
