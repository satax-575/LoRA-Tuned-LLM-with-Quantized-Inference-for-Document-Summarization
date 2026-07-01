"""
Perplexity Computation Utilities
Computes model perplexity on evaluation sets
"""

import math
import logging
from typing import List, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 4,
    max_length: int = 1024,
    device: str = "cuda",
    stride: int = 512,
) -> dict:
    """
    Compute perplexity using sliding window approach for long texts.
    
    Uses stride to handle sequences longer than max_length:
    - Avoids truncating long documents
    - Uses overlapping windows for accurate log-likelihood estimation
    
    Args:
        model: HuggingFace causal LM
        tokenizer: Corresponding tokenizer
        texts: List of texts to evaluate
        batch_size: Texts per batch
        max_length: Maximum sequence length
        device: "cuda" or "cpu"
        stride: Sliding window stride
    
    Returns:
        dict with mean/median perplexity and per-text values
    """
    model.eval()
    all_ppls = []
    all_losses = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
            batch = texts[i: i + batch_size]

            for text in batch:
                encodings = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=False,
                )
                input_ids = encodings.input_ids.to(device)
                seq_len = input_ids.size(1)

                if seq_len <= max_length:
                    # Short sequence — direct computation
                    with torch.no_grad():
                        outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss.item()
                else:
                    # Long sequence — sliding window
                    nlls = []
                    prev_end = 0
                    for begin in range(0, seq_len, stride):
                        end = min(begin + max_length, seq_len)
                        target_len = end - prev_end
                        window_input = input_ids[:, begin:end]
                        target_ids = window_input.clone()
                        target_ids[:, :-target_len] = -100  # Mask overlapping tokens

                        with torch.no_grad():
                            outputs = model(window_input, labels=target_ids)
                        nlls.append(outputs.loss.item() * target_len)
                        prev_end = end
                        if end == seq_len:
                            break

                    loss = sum(nlls) / seq_len

                ppl = math.exp(min(loss, 20))  # Cap to prevent overflow
                all_ppls.append(ppl)
                all_losses.append(loss)

    arr = np.array(all_ppls)
    return {
        "mean_perplexity": round(float(arr.mean()), 4),
        "median_perplexity": round(float(np.median(arr)), 4),
        "std_perplexity": round(float(arr.std()), 4),
        "p90_perplexity": round(float(np.percentile(arr, 90)), 4),
        "min_perplexity": round(float(arr.min()), 4),
        "max_perplexity": round(float(arr.max()), 4),
        "mean_loss": round(float(np.mean(all_losses)), 4),
        "per_text_perplexities": [round(float(p), 4) for p in all_ppls],
    }


def perplexity_from_loss(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def load_and_evaluate(
    model_path: str,
    texts: List[str],
    hf_token: Optional[str] = None,
    use_4bit: bool = True,
    batch_size: int = 4,
) -> dict:
    """
    Convenience function: load model and compute perplexity.
    """
    console.print(f"[cyan]Loading model: {model_path}[/cyan]")

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = compute_perplexity(model, tokenizer, texts, batch_size=batch_size, device=device)

    console.print(f"[green]Mean perplexity: {results['mean_perplexity']:.4f}[/green]")
    return results
