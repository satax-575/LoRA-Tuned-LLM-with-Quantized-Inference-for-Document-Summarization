"""
training/callbacks.py

Custom HuggingFace Trainer callbacks for this QLoRA training run.

We have three of them:
  - PerplexityEarlyStoppingCallback: monitors validation perplexity and stops
    training when it plateaus — prevents overfitting on the 50K corpus
  - TrainingProgressCallback: nice Rich-formatted step logging with loss tracking
  - ModelCheckpointCallback: logs checkpoint saves so you can find them later
"""

import math
import logging
from typing import Optional

import numpy as np
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


class PerplexityEarlyStoppingCallback(TrainerCallback):
    """
    Perplexity-guided early stopping.

    We track eval perplexity (exp(eval_loss)) rather than raw loss because:
    - Perplexity is more interpretable for LMs ("confused by N words on average")
    - The exponential scale makes small loss improvements more meaningful
    - It catches plateau behaviour that's subtle in log-loss space

    We stop when perplexity doesn't improve by more than `threshold` (relative)
    for `patience` consecutive evaluations.
    """

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.001,
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.threshold = threshold
        self.min_delta = min_delta

        self.best_perplexity: Optional[float] = None
        self.wait_count = 0
        self.stopped_epoch = 0
        self.history = []

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return control

        # Convert loss → perplexity
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        self.history.append({
            "step": state.global_step,
            "eval_loss": eval_loss,
            "perplexity": perplexity,
        })

        console.print(
            f"\n[bold cyan]  [EarlyStopping] Step {state.global_step} | "
            f"Loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}[/bold cyan]"
        )

        if self.best_perplexity is None:
            self.best_perplexity = perplexity
            console.print(f"  [green]Initial best perplexity: {perplexity:.4f}[/green]")
            return control

        improvement = self.best_perplexity - perplexity
        relative_improvement = improvement / self.best_perplexity

        if relative_improvement > self.threshold:
            self.best_perplexity = perplexity
            self.wait_count = 0
            console.print(
                f"  [green]✓ Improved by {relative_improvement*100:.3f}% "
                f"→ New best: {perplexity:.4f}[/green]"
            )
        else:
            self.wait_count += 1
            console.print(
                f"  [yellow]No improvement ({self.wait_count}/{self.patience}). "
                f"Best: {self.best_perplexity:.4f}[/yellow]"
            )

            if self.wait_count >= self.patience:
                control.should_training_stop = True
                self.stopped_epoch = state.epoch
                console.print(
                    f"\n  [bold red]⚠ Early stopping triggered at epoch "
                    f"{state.epoch:.1f}, step {state.global_step}[/bold red]"
                )
                console.print(
                    f"  [red]Best perplexity: {self.best_perplexity:.4f}[/red]"
                )

        return control


class TrainingProgressCallback(TrainerCallback):
    """
    Rich-formatted training progress logging.

    Prints a compact step summary every `log_every` steps and records loss
    history so it can be inspected after training (e.g., for plotting).
    """

    def __init__(self, log_every: int = 50):
        self.log_every = log_every
        self.train_losses = []
        self.eval_losses = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict,
        **kwargs,
    ):
        if "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0)
            epoch = logs.get("epoch", 0)

            self.train_losses.append({"step": step, "loss": loss})

            if step % self.log_every == 0:
                console.print(
                    f"  [dim]Step {step:>6} | Epoch {epoch:.2f} | "
                    f"Loss: {loss:.4f} | LR: {lr:.2e}[/dim]"
                )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        eval_loss = metrics.get("eval_loss")
        # BUG FIX: `if eval_loss:` is falsy when eval_loss == 0.0.
        # Use `is not None` so we never silently skip a perfect-zero loss.
        if eval_loss is not None:
            self.eval_losses.append({
                "step": state.global_step,
                "loss": eval_loss,
                # Cap at 20 before exp() to avoid overflow on bad early losses
                "perplexity": math.exp(min(eval_loss, 20)),
            })

    def get_training_history(self) -> dict:
        return {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
        }


class ModelCheckpointCallback(TrainerCallback):
    """
    Logs a human-readable message whenever a checkpoint is written.

    Purely cosmetic — HuggingFace handles the actual saving. This just makes
    it easier to spot checkpoint events in the training log.
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        console.print(
            f"  [magenta]💾 Checkpoint saved at step {state.global_step} "
            f"→ {args.output_dir}[/magenta]"
        )
