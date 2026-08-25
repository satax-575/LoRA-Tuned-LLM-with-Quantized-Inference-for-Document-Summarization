"""
ROUGE Evaluation — QLoRA Llama 3.1 8B vs BART-Large vs T5-Base
Computes ROUGE-1, ROUGE-2, ROUGE-L on CNN/DailyMail test set
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
import numpy as np
from rouge_score import rouge_scorer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from datasets import load_from_disk, load_dataset
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)


class ROUGEEvaluator:
    """Computes ROUGE-1, ROUGE-2, ROUGE-L for a list of predictions vs references."""

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def score_batch(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute aggregate ROUGE scores over a batch."""
        assert len(predictions) == len(references)

        r1_list, r2_list, rL_list = [], [], []
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref.strip(), pred.strip())
            r1_list.append(scores["rouge1"].fmeasure)
            r2_list.append(scores["rouge2"].fmeasure)
            rL_list.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": round(float(np.mean(r1_list)), 4),
            "rouge2": round(float(np.mean(r2_list)), 4),
            "rougeL": round(float(np.mean(rL_list)), 4),
            "rouge1_std": round(float(np.std(r1_list)), 4),
            "rouge2_std": round(float(np.std(r2_list)), 4),
            "rougeL_std": round(float(np.std(rL_list)), 4),
        }

    def score_single(self, prediction: str, reference: str) -> Dict[str, float]:
        scores = self.scorer.score(reference.strip(), prediction.strip())
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }


class LlamaEvaluator:
    """Generates summaries with the QLoRA-tuned Llama 3.1 8B model."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
        use_4bit: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        console.print(f"[cyan]Loading Llama model from {model_path}...[/cyan]")

        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        console.print("[green]✓ Llama model loaded[/green]")

    def summarize_batch(self, articles: List[str], batch_size: int = 4) -> List[str]:
        """Generate summaries for a batch of articles."""
        from inference.prompt_utils import format_inference_prompt

        summaries = []
        for i in tqdm(range(0, len(articles), batch_size), desc="Llama inference"):
            batch = articles[i: i + batch_size]
            prompts = [format_inference_prompt(a) for a in batch]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for j, output in enumerate(outputs):
                # Decode only generated tokens (not the prompt)
                input_len = inputs["input_ids"][j].shape[0]
                generated = output[input_len:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                # Clean up any trailing special tokens
                text = text.replace("<|eot_id|>", "").strip()
                summaries.append(text)

        return summaries


class BARTLargeEvaluator:
    """Generates summaries with BART-Large baseline."""

    def __init__(self, device: str = "cuda"):
        console.print("[cyan]Loading BART-Large...[/cyan]")
        self.pipe = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            torch_dtype=torch.float16,
        )
        console.print("[green]✓ BART-Large loaded[/green]")

    def summarize_batch(self, articles: List[str], batch_size: int = 8) -> List[str]:
        summaries = []
        for i in tqdm(range(0, len(articles), batch_size), desc="BART-Large inference"):
            batch = articles[i: i + batch_size]
            results = self.pipe(
                batch,
                max_length=256,
                min_length=30,
                truncation=True,
                batch_size=batch_size,
            )
            summaries.extend([r["summary_text"] for r in results])
        return summaries


class T5BaseEvaluator:
    """Generates summaries with T5-Base baseline."""

    def __init__(self, device: str = "cuda"):
        console.print("[cyan]Loading T5-Base...[/cyan]")
        self.pipe = pipeline(
            "summarization",
            model="t5-base",
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            torch_dtype=torch.float32,
        )
        console.print("[green]✓ T5-Base loaded[/green]")

    def summarize_batch(self, articles: List[str], batch_size: int = 8) -> List[str]:
        summaries = []
        for i in tqdm(range(0, len(articles), batch_size), desc="T5-Base inference"):
            batch = [f"summarize: {a[:1024]}" for a in articles[i: i + batch_size]]
            results = self.pipe(
                batch,
                max_length=256,
                min_length=30,
                truncation=True,
                batch_size=batch_size,
            )
            summaries.extend([r["summary_text"] for r in results])
        return summaries
