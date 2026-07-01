"""
Main Training Entry Point
Run: python training/train.py --config configs/training_config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import wandb
from dotenv import load_dotenv
from rich.console import Console

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_builder import CNNDailyMailBuilder
from training.trainer import QLoRATrainer

load_dotenv()
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of Llama 3.1 8B on CNN/DailyMail"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/processed",
        help="Path to preprocessed dataset directory",
    )
    parser.add_argument(
        "--skip-data-build",
        action="store_true",
        help="Skip dataset building (use existing data-dir)",
    )
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Merge LoRA adapter into base model after training",
    )
    parser.add_argument(
        "--merged-output",
        type=str,
        default="./outputs/merged_model",
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.print("\n" + "═" * 60)
    console.print(
        "  [bold magenta]QLoRA-Tuned Llama 3.1 8B[/bold magenta]\n"
        "  Document Summarization on CNN/DailyMail"
    )
    console.print("═" * 60 + "\n")

    # ── Check HF Token ──
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        console.print(
            "[bold red]⚠ HF_TOKEN not found in environment.[/bold red]\n"
            "  Set it in .env file: HF_TOKEN=hf_your_token_here\n"
            "  Get your token at: https://huggingface.co/settings/tokens\n"
            "  Accept Llama 3.1 license at: https://huggingface.co/meta-llama/Llama-3.1-8B"
        )
        sys.exit(1)

    # ── W&B Setup ──
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
            console.print("[green]✓ Weights & Biases connected[/green]")

    # ── Step 1: Build Dataset ──
    if not args.skip_data_build:
        console.print("\n[bold]Step 1: Building 50K CNN/DailyMail corpus...[/bold]")
        builder = CNNDailyMailBuilder(config_path=args.config)
        builder.build(output_dir=args.data_dir)
    else:
        console.print(f"\n[yellow]Skipping data build. Using: {args.data_dir}[/yellow]")
        if not Path(args.data_dir).exists():
            console.print(f"[red]Error: Dataset not found at {args.data_dir}[/red]")
            sys.exit(1)

    # ── Step 2: QLoRA Training ──
    console.print("\n[bold]Step 2: QLoRA Fine-tuning...[/bold]")
    trainer = QLoRATrainer(config_path=args.config)
    result = trainer.train(dataset_path=args.data_dir)

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Final loss  : {result['training_loss']:.4f}")
    console.print(f"  Steps       : {result['global_step']}")
    console.print(f"  Saved to    : {result['output_dir']}")

    # ── Step 3: Merge Adapter (optional) ──
    if args.merge_adapter:
        console.print("\n[bold]Step 3: Merging LoRA adapter into base model...[/bold]")
        trainer.merge_and_save(output_dir=args.merged_output)

    console.print("\n[bold magenta]═══ Training Pipeline Complete ═══[/bold magenta]")
    console.print("\nNext steps:")
    console.print("  1. [cyan]python quantization/awq_quantize.py[/cyan]  — Quantize with AWQ")
    console.print("  2. [cyan]python evaluation/benchmark.py[/cyan]       — Run ROUGE benchmarks")
    console.print("  3. [cyan]uvicorn inference.main:app --host 0.0.0.0 --port 8000[/cyan]  — Start API")


if __name__ == "__main__":
    main()
