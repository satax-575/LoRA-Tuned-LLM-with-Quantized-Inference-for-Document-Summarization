"""
Corpus Statistics & Exploratory Data Analysis
Generates token length distributions, vocabulary stats, and plots
"""

import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table

console = Console()

# Plotting style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.facecolor"] = "#1a1a2e"
plt.rcParams["axes.facecolor"] = "#16213e"
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"


class DatasetStatsAnalyzer:
    """
    Computes and visualizes dataset statistics for the 50K CNN/DailyMail corpus.
    """

    def __init__(
        self,
        dataset_path: str = "./data/processed",
        tokenizer_path: Optional[str] = None,
        output_dir: str = "./data/stats",
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Loading dataset from {dataset_path}...[/cyan]")
        self.dataset = load_from_disk(dataset_path)

        tok_path = tokenizer_path or f"{dataset_path}/tokenizer"
        console.print(f"[cyan]Loading tokenizer from {tok_path}...[/cyan]")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

    def compute_split_stats(self, split: str = "train") -> dict:
        """Compute per-split token length statistics."""
        data = self.dataset[split]
        lengths = [len(ids) for ids in data["input_ids"]]
        arr = np.array(lengths)
        return {
            "split": split,
            "count": len(arr),
            "mean_tokens": round(float(arr.mean()), 1),
            "median_tokens": round(float(np.median(arr)), 1),
            "std_tokens": round(float(arr.std()), 1),
            "p50_tokens": round(float(np.percentile(arr, 50)), 1),
            "p90_tokens": round(float(np.percentile(arr, 90)), 1),
            "p95_tokens": round(float(np.percentile(arr, 95)), 1),
            "p99_tokens": round(float(np.percentile(arr, 99)), 1),
            "max_tokens": int(arr.max()),
            "min_tokens": int(arr.min()),
            "pct_under_512": round(float((arr <= 512).mean() * 100), 2),
            "pct_under_1024": round(float((arr <= 1024).mean() * 100), 2),
            "pct_under_1280": round(float((arr <= 1280).mean() * 100), 2),
        }

    def print_stats_table(self):
        """Print formatted stats table to console."""
        table = Table(title="Dataset Token Length Statistics", style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="green")
        table.add_column("Validation", style="yellow")
        table.add_column("Test", style="blue")

        splits = ["train", "validation", "test"]
        stats = [self.compute_split_stats(s) for s in splits]

        metrics = [
            "count", "mean_tokens", "median_tokens", "std_tokens",
            "p90_tokens", "p95_tokens", "p99_tokens",
            "min_tokens", "max_tokens",
            "pct_under_512", "pct_under_1024",
        ]

        for metric in metrics:
            table.add_row(
                metric,
                str(stats[0][metric]),
                str(stats[1][metric]),
                str(stats[2][metric]),
            )
        console.print(table)

    def plot_token_distribution(self, split: str = "train", sample_n: int = 5000):
        """Plot token length distribution histogram."""
        data = self.dataset[split]
        lengths = [len(ids) for ids in data["input_ids"]][:sample_n]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Llama 3.1 BPE Token Length Distribution — {split.title()} Split",
            color="white",
            fontsize=14,
            fontweight="bold",
        )

        # Histogram
        axes[0].hist(lengths, bins=60, color="#7c3aed", alpha=0.85, edgecolor="#a78bfa")
        axes[0].axvline(np.percentile(lengths, 95), color="#f43f5e", linestyle="--", label="P95")
        axes[0].axvline(np.percentile(lengths, 50), color="#10b981", linestyle="--", label="P50")
        axes[0].set_xlabel("Token Length", color="white")
        axes[0].set_ylabel("Count", color="white")
        axes[0].set_title("Distribution", color="white")
        axes[0].legend()

        # Box plot
        axes[1].boxplot(lengths, patch_artist=True,
                        boxprops=dict(facecolor="#7c3aed", color="#a78bfa"),
                        medianprops=dict(color="#10b981", linewidth=2),
                        whiskerprops=dict(color="white"),
                        capprops=dict(color="white"),
                        flierprops=dict(marker="o", color="#f43f5e", alpha=0.3))
        axes[1].set_ylabel("Token Length", color="white")
        axes[1].set_title("Box Plot", color="white")

        plt.tight_layout()
        out_path = self.output_dir / f"token_distribution_{split}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        console.print(f"[green]✓ Saved plot: {out_path}[/green]")
        plt.close()

    def save_stats_json(self):
        """Save all stats as JSON for reproducibility."""
        all_stats = {
            split: self.compute_split_stats(split)
            for split in ["train", "validation", "test"]
        }
        out_path = self.output_dir / "corpus_stats.json"
        with open(out_path, "w") as f:
            json.dump(all_stats, f, indent=2)
        console.print(f"[green]✓ Saved stats: {out_path}[/green]")

    def run_full_analysis(self):
        """Run all EDA steps."""
        console.print("\n[bold magenta]═══ Dataset Statistics & EDA ═══[/bold magenta]\n")
        self.print_stats_table()
        for split in ["train", "validation"]:
            self.plot_token_distribution(split)
        self.save_stats_json()
        console.print("\n[bold green]✓ EDA complete![/bold green]")


if __name__ == "__main__":
    analyzer = DatasetStatsAnalyzer()
    analyzer.run_full_analysis()
