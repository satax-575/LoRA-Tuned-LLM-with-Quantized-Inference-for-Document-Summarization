"""
AWQ 4-bit Quantization Pipeline
Converts merged Llama 3.1 8B to AWQ format for production inference
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)


class AWQQuantizer:
    """
    Quantizes the merged LoRA+Llama 3.1 8B model to 4-bit AWQ format.
    
    AWQ (Activation-aware Weight Quantization):
    - Identifies salient weights via activation magnitudes
    - Preserves critical weights at higher precision
    - Outperforms GPTQ on text quality metrics
    - Fully compatible with vLLM for high-throughput serving
    
    Workflow:
    1. Load merged FP16 model
    2. Run AWQ calibration (forward pass on calibration data)
    3. Quantize to 4-bit W4A16 format
    4. Save AWQ-quantized model
    """

    def __init__(
        self,
        model_path: str = "./outputs/merged_model",
        output_path: str = "./outputs/awq_model",
        calib_samples: int = 128,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.calib_samples = calib_samples
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point

    def quantize(self):
        """Run the full AWQ quantization pipeline."""
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            console.print("[red]Error: autoawq not installed. Run: pip install autoawq[/red]")
            sys.exit(1)

        console.print("\n[bold magenta]═══ AWQ Quantization Pipeline ═══[/bold magenta]")
        console.print(f"  Source : {self.model_path}")
        console.print(f"  Output : {self.output_path}")
        console.print(f"  Bits   : {self.bits}-bit")
        console.print(f"  Groups : {self.group_size}")

        # ── Load Model ──
        console.print("\n[cyan]Loading merged model...[/cyan]")
        model = AutoAWQForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # ── AWQ Config ──
        quant_config = {
            "zero_point": self.zero_point,
            "q_group_size": self.group_size,
            "w_bit": self.bits,
            "version": "GEMM",   # GEMM kernel: best for T4 GPU
        }
        console.print(f"[cyan]AWQ config: {quant_config}[/cyan]")

        # ── Load Calibration Data ──
        console.print(f"\n[cyan]Loading calibration data ({self.calib_samples} samples)...[/cyan]")
        calib_data = self._get_calibration_data(tokenizer)

        # ── Quantize ──
        console.print("\n[cyan]Running AWQ quantization (this takes ~20-40 min on T4)...[/cyan]")
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_data,
        )

        # ── Save ──
        console.print(f"\n[cyan]Saving AWQ model to {self.output_path}...[/cyan]")
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        model.save_quantized(self.output_path)
        tokenizer.save_pretrained(self.output_path)

        # ── Save metadata ──
        metadata = {
            "base_model": "meta-llama/Llama-3.1-8B",
            "fine_tuned_with": "QLoRA (NF4, rank-16, alpha=32)",
            "quantization": "AWQ",
            "bits": self.bits,
            "group_size": self.group_size,
            "zero_point": self.zero_point,
            "calib_samples": self.calib_samples,
            "compatible_backends": ["vllm", "transformers", "autoawq"],
        }
        with open(f"{self.output_path}/quantization_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        console.print(f"\n[bold green]✓ AWQ quantization complete![/bold green]")
        console.print(f"[green]  Saved to: {self.output_path}[/green]")
        self._print_size_comparison()

    def _get_calibration_data(self, tokenizer) -> list:
        """
        Load CNN/DailyMail articles as calibration data.
        AWQ uses these to identify activation-salient weights.
        """
        from datasets import load_dataset
        from data.dataset_builder import format_inference_prompt

        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)
        dataset = dataset.shuffle(seed=42).select(range(self.calib_samples))

        calib_texts = []
        for example in dataset:
            prompt = format_inference_prompt(example["article"][:800])
            tokens = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
            calib_texts.append(tokens["input_ids"])

        return calib_texts

    def _print_size_comparison(self):
        """Print model size before/after quantization."""
        def dir_size_gb(path: str) -> float:
            total = sum(
                f.stat().st_size for f in Path(path).rglob("*") if f.is_file()
            )
            return round(total / 1e9, 2)

        orig_size = dir_size_gb(self.model_path)
        quant_size = dir_size_gb(self.output_path)
        reduction = round((1 - quant_size / orig_size) * 100, 1) if orig_size > 0 else 0

        console.print("\n  Size Comparison:")
        console.print(f"  Original (FP16) : {orig_size:.2f} GB")
        console.print(f"  AWQ 4-bit       : {quant_size:.2f} GB")
        console.print(f"  Reduction       : {reduction}%")


def verify_awq_model(model_path: str, test_text: str = "Summarize: The president held a press conference."):
    """Quick sanity check on the quantized AWQ model."""
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        console.print("[yellow]Skipping AWQ verification (autoawq not installed)[/yellow]")
        return False

    console.print(f"\n[cyan]Verifying AWQ model at {model_path}...[/cyan]")
    model = AutoAWQForCausalLM.from_quantized(
        model_path,
        fuse_layers=True,
        trust_remote_code=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    console.print(f"[green]✓ Verification passed[/green]")
    console.print(f"  Input : {test_text}")
    console.print(f"  Output: {generated[:200]}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="AWQ 4-bit quantization for Llama 3.1 8B")
    parser.add_argument("--model-path", type=str, default="./outputs/merged_model")
    parser.add_argument("--output-path", type=str, default="./outputs/awq_model")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main():
    """CLI entry point — called by qlora-quantize console script."""
    args = parse_args()
    quantizer = AWQQuantizer(
        model_path=args.model_path,
        output_path=args.output_path,
        calib_samples=args.calib_samples,
        bits=args.bits,
        group_size=args.group_size,
    )
    quantizer.quantize()

    if args.verify:
        verify_awq_model(args.output_path)


if __name__ == "__main__":
    main()
