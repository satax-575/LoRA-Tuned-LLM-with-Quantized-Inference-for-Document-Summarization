"""
Data Pipeline — CNN/DailyMail 50K Corpus Builder
Curates and preprocesses the dataset with Llama 3.1 BPE tokenizer
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

import yaml
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompt Formatting (Llama 3.1 Chat Template)
# ─────────────────────────────────────────────

INSTRUCTION_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an expert document summarizer. Provide concise, accurate summaries "
    "that capture the key information.<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Summarize the following article:\n\n{article}<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "{summary}<|eot_id|>"
)


def format_prompt(article: str, summary: str) -> str:
    """Format article + summary into Llama 3.1 instruction format."""
    return INSTRUCTION_TEMPLATE.format(
        article=article.strip(),
        summary=summary.strip(),
    )


def format_inference_prompt(article: str) -> str:
    """Format article for inference (no summary — model generates it)."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are an expert document summarizer. Provide concise, accurate summaries "
        "that capture the key information.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Summarize the following article:\n\n{article.strip()}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# ─────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────

class CNNDailyMailBuilder:
    """
    Builds a 50K-document corpus from CNN/DailyMail.
    
    - Downloads via HuggingFace Datasets
    - Filters and curates samples
    - Applies Llama 3.1 BPE tokenization
    - Saves processed splits to disk
    """

    def __init__(self, config_path: str = "configs/training_config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.data_cfg = self.cfg["data"]
        self.model_name = self.cfg["model"]["name"]
        self.max_source = self.data_cfg["max_source_length"]
        self.max_target = self.data_cfg["max_target_length"]
        self.seed = self.data_cfg["seed"]
        self.hf_token = os.environ.get("HF_TOKEN")

        # Llama 3.1 BPE Tokenizer
        console.print(f"[cyan]Loading tokenizer: {self.model_name}[/cyan]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def load_raw_dataset(self) -> DatasetDict:
        """Load CNN/DailyMail from HuggingFace Datasets Hub."""
        console.print("[cyan]Loading CNN/DailyMail dataset...[/cyan]")
        dataset = load_dataset(
            self.data_cfg["dataset_name"],
            self.data_cfg["dataset_version"],
            trust_remote_code=True,
        )
        console.print(
            f"[green]✓ Loaded: train={len(dataset['train'])}, "
            f"val={len(dataset['validation'])}, "
            f"test={len(dataset['test'])}[/green]"
        )
        return dataset

    def curate_corpus(self, dataset: DatasetDict) -> DatasetDict:
        """
        Curate 50K training, 5K val, 5K test samples.
        Filters out:
        - Articles shorter than 100 words
        - Summaries shorter than 20 words
        - Articles longer than 1500 words (extreme outliers)
        """
        console.print("[cyan]Curating 50K-document corpus...[/cyan]")

        def quality_filter(example):
            article_words = len(example["article"].split())
            summary_words = len(example["highlights"].split())
            return (
                100 <= article_words <= 1500
                and 20 <= summary_words <= 200
            )

        filtered = dataset.filter(
            quality_filter,
            num_proc=self.data_cfg.get("num_proc", 4),
            desc="Quality filtering",
        )

        train_size = min(self.data_cfg["train_samples"], len(filtered["train"]))
        val_size = min(self.data_cfg["val_samples"], len(filtered["validation"]))
        test_size = min(self.data_cfg["test_samples"], len(filtered["test"]))

        curated = DatasetDict({
            "train": filtered["train"].shuffle(seed=self.seed).select(range(train_size)),
            "validation": filtered["validation"].shuffle(seed=self.seed).select(range(val_size)),
            "test": filtered["test"].shuffle(seed=self.seed).select(range(test_size)),
        })

        console.print(
            f"[green]✓ Curated: train={len(curated['train'])}, "
            f"val={len(curated['validation'])}, test={len(curated['test'])}[/green]"
        )
        return curated

    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Apply Llama 3.1 BPE tokenization.
        Formats each example into instruction-following template.
        """
        console.print("[cyan]Applying Llama 3.1 BPE tokenization...[/cyan]")

        def tokenize_fn(examples):
            prompts = [
                format_prompt(article, summary)
                for article, summary in zip(
                    examples["article"], examples["highlights"]
                )
            ]
            tokenized = self.tokenizer(
                prompts,
                max_length=self.max_source + self.max_target + 64,  # +64 for template tokens
                truncation=True,
                padding=False,  # Dynamic padding in DataCollator
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=self.data_cfg.get("num_proc", 4),
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing",
        )
        console.print("[green]✓ Tokenization complete[/green]")
        return tokenized

    def save_dataset(self, dataset: DatasetDict, output_dir: str = "./data/processed"):
        """Save processed dataset to disk."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_dir)
        console.print(f"[green]✓ Saved processed dataset to {output_dir}[/green]")

        # Save tokenizer alongside data
        self.tokenizer.save_pretrained(f"{output_dir}/tokenizer")
        console.print(f"[green]✓ Saved tokenizer to {output_dir}/tokenizer[/green]")

    def build(self, output_dir: str = "./data/processed") -> DatasetDict:
        """Full pipeline: load → curate → tokenize → save."""
        console.print("\n[bold magenta]═══ Building 50K Document Corpus ═══[/bold magenta]\n")
        raw = self.load_raw_dataset()
        curated = self.curate_corpus(raw)
        tokenized = self.tokenize_dataset(curated)
        self.save_dataset(tokenized, output_dir)
        console.print("\n[bold green]✓ Dataset pipeline complete![/bold green]")
        return tokenized


if __name__ == "__main__":
    builder = CNNDailyMailBuilder()
    dataset = builder.build()
    print(f"\nDataset info:\n{dataset}")
