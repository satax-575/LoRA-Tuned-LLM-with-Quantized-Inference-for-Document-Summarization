"""
Data Pipeline — CNN/DailyMail 50K Corpus Builder
Curates and preprocesses the dataset with Llama 3.1 BPE tokenizer.

Key design decisions:
  - Proper causal label masking: only summary tokens contribute to loss
  - Text cleaning via preprocessing.py before tokenization
  - Uses `token=` (not deprecated `use_auth_token=`)
  - Dynamic padding deferred to DataCollator — no padding here
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

import yaml
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

from data.preprocessing import clean_article, clean_summary

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompt Templates (kept here for training data construction)
# Inference prompt formatting is in inference/prompt_utils.py
# ─────────────────────────────────────────────

_BOS = "<|begin_of_text|>"
_START_HEADER = "<|start_header_id|>"
_END_HEADER = "<|end_header_id|>"
_EOT = "<|eot_id|>"

_SYSTEM_PROMPT = (
    "You are an expert document summarizer. "
    "Provide concise, accurate summaries that capture the key information."
)


def format_prompt(article: str, summary: str) -> str:
    """
    Format article + summary into a full Llama 3.1 instruction-tuning example.
    Used during dataset construction and AWQ calibration.
    """
    return (
        f"{_BOS}{_START_HEADER}system{_END_HEADER}\n"
        f"{_SYSTEM_PROMPT}{_EOT}\n"
        f"{_START_HEADER}user{_END_HEADER}\n"
        f"Summarize the following article:\n\n{article.strip()}{_EOT}\n"
        f"{_START_HEADER}assistant{_END_HEADER}\n"
        f"{summary.strip()}{_EOT}"
    )


def format_inference_prompt(article: str) -> str:
    """
    Format an article for inference (no summary — model generates it).

    NOTE: For the inference server, prefer importing from inference.prompt_utils
    to avoid loading the entire training pipeline. This function is kept here
    for backwards compatibility and for use during AWQ calibration.
    """
    return (
        f"{_BOS}{_START_HEADER}system{_END_HEADER}\n"
        f"{_SYSTEM_PROMPT}{_EOT}\n"
        f"{_START_HEADER}user{_END_HEADER}\n"
        f"Summarize the following article:\n\n{article.strip()}{_EOT}\n"
        f"{_START_HEADER}assistant{_END_HEADER}\n"
    )


# ─────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────

class CNNDailyMailBuilder:
    """
    Builds a 50K-document corpus from CNN/DailyMail.

    Pipeline:
      1. Download via HuggingFace Datasets
      2. Quality filter (word count bounds)
      3. Text cleaning (bylines, boilerplate, unicode)
      4. Apply Llama 3.1 BPE tokenization
      5. Causal label masking — loss computed ONLY on summary tokens
      6. Save processed splits to disk
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

        # ── Llama 3.1 BPE Tokenizer ──
        # Uses `token=` (not deprecated `use_auth_token=`)
        console.print(f"[cyan]Loading tokenizer: {self.model_name}[/cyan]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,          # ← Fixed: was use_auth_token= (removed in 4.34)
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
        Apply Llama 3.1 BPE tokenization with proper causal label masking.

        Masking Strategy:
          - Full prompt (system + user + article + assistant header) is tokenized
          - Labels are set to -100 for all PROMPT tokens (article + instructions)
          - Only SUMMARY tokens contribute to the cross-entropy loss
          - This is the correct instruction-following training approach

        Why this matters:
          Training on the full sequence (no masking) causes the model to also try
          to predict the article content, diluting the gradient signal and typically
          reducing ROUGE-L by 1-3 points compared to masked training.
        """
        console.print("[cyan]Applying Llama 3.1 BPE tokenization with causal label masking...[/cyan]")

        max_length = self.max_source + self.max_target + 128  # +128 for template tokens

        def tokenize_fn(examples):
            all_input_ids = []
            all_attention_masks = []
            all_labels = []

            for article_raw, highlights_raw in zip(
                examples["article"], examples["highlights"]
            ):
                # ── 1. Clean text ──
                article = clean_article(article_raw)
                summary = clean_summary(highlights_raw)

                # ── 2. Build full prompt (prompt + response) ──
                full_prompt = format_prompt(article, summary)

                # ── 3. Build prompt-only prefix (to find boundary for masking) ──
                prompt_only = format_inference_prompt(article)

                # ── 4. Tokenize both ──
                full_tokenized = self.tokenizer(
                    full_prompt,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors=None,
                )
                prompt_tokenized = self.tokenizer(
                    prompt_only,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                )

                input_ids = full_tokenized["input_ids"]
                attention_mask = full_tokenized["attention_mask"]

                # ── 5. Causal Label Masking ──
                # Mask all tokens up to and including the assistant header
                # Only the summary portion contributes to loss
                prompt_len = len(prompt_tokenized["input_ids"])
                labels = [-100] * prompt_len + input_ids[prompt_len:]

                # Truncate labels to match input_ids if prompt_len > max_length
                labels = labels[:len(input_ids)]

                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_labels.append(labels)

            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_masks,
                "labels": all_labels,
            }

        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=self.data_cfg.get("num_proc", 4),
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing with causal label masking",
        )
        console.print("[green]✓ Tokenization complete (labels masked for causal LM)[/green]")
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
        """Full pipeline: load → curate → tokenize (with masking) → save."""
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
