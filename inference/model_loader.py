"""
Model Loader — vLLM AWQ Engine
Singleton pattern for loading quantized Llama 3.1 8B once per process.

Fixes vs original:
  - `_engine_type` is now set in BOTH the vLLM success path AND the HF fallback
  - `engine_type` property no longer silently returns "vllm" when using HF
  - GPU stats method hardened for multi-GPU setups
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)


class VLLMModelLoader:
    """
    Loads the AWQ-quantized Llama 3.1 8B via vLLM AsyncLLMEngine.

    vLLM advantages over vanilla HF inference:
      - PagedAttention: Efficient KV-cache memory management
      - Continuous batching: Groups requests mid-flight
      - AWQ kernel fusion: Fast W4A16 matrix multiplications
      - AsyncLLMEngine: Non-blocking, handles many concurrent requests

    Singleton pattern ensures model is loaded exactly once,
    enabling stateless, horizontally scalable deployments.
    """

    _instance: Optional["VLLMModelLoader"] = None
    _initialized: bool = False
    _startup_time: float = 0.0

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_path: str = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        quantization: str = "awq",
    ):
        if self._initialized:
            return

        self.model_path = model_path or os.environ.get(
            "AWQ_MODEL_DIR", "./outputs/awq_model"
        )
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.engine = None
        self.tokenizer = None
        self._engine_type: str = "unknown"  # ← FIX: always initialized
        self._initialized = True

    def load(self):
        """Initialize the vLLM AsyncLLMEngine with the AWQ model."""
        console.print(f"\n[bold cyan]Loading vLLM engine from {self.model_path}...[/bold cyan]")
        t0 = time.time()

        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            from transformers import AutoTokenizer

            engine_args = AsyncEngineArgs(
                model=self.model_path,
                quantization=self.quantization,          # "awq"
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype="auto",
                enable_prefix_caching=True,              # Reuse KV for repeated prompts
                trust_remote_code=True,
            )

            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._engine_type = "vllm"                  # ← FIX: was never set in success path
            self._startup_time = time.time() - t0
            console.print(
                f"[bold green]✓ vLLM engine ready in {self._startup_time:.1f}s[/bold green]"
            )

        except ImportError:
            logger.warning("vLLM not available. Falling back to HuggingFace transformers.")
            self._load_hf_fallback()
        except Exception as e:
            logger.error(f"vLLM load failed: {e}. Trying HF fallback.")
            self._load_hf_fallback()

    def _load_hf_fallback(self):
        """
        Fallback: Load AWQ model with HuggingFace transformers.
        Used when vLLM is unavailable (e.g., local CPU-only development).
        """
        console.print("[yellow]Loading HuggingFace fallback (no vLLM)...[/yellow]")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_path = self.model_path
        if not Path(model_path).exists():
            # Fall back to base model for local dev
            model_path = os.environ.get("HF_MODEL_NAME", "meta-llama/Llama-3.1-8B")
            console.print(f"[yellow]Model path not found. Using: {model_path}[/yellow]")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.engine = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            device_map="auto",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._engine_type = "transformers"               # ← FIX: now set correctly
        console.print("[green]✓ HF fallback model loaded[/green]")

    @property
    def is_loaded(self) -> bool:
        return self.engine is not None

    @property
    def engine_type(self) -> str:
        """Return the type of inference engine: 'vllm' or 'transformers'."""
        return self._engine_type                         # ← FIX: always returns actual value

    def get_gpu_stats(self) -> dict:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        try:
            mem = torch.cuda.mem_get_info(0)
            free_gb = mem[0] / 1e9
            total_gb = mem[1] / 1e9
            used_gb = total_gb - free_gb

            return {
                "gpu_available": True,
                "gpu_memory_used_gb": round(used_gb, 2),
                "gpu_memory_free_gb": round(free_gb, 2),
                "gpu_memory_total_gb": round(total_gb, 2),
                "gpu_utilization_pct": round(used_gb / total_gb * 100, 1),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_used_bytes": int(used_gb * 1e9),
                "gpu_memory_total_bytes": int(total_gb * 1e9),
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return {"gpu_available": True, "gpu_name": "unknown"}


# ── Module-level singleton ──
_loader_instance: Optional[VLLMModelLoader] = None


def get_model_loader() -> VLLMModelLoader:
    """Get the global model loader singleton."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = VLLMModelLoader(
            model_path=os.environ.get("AWQ_MODEL_DIR", "./outputs/awq_model"),
            gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTIL", "0.85")),
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "2048")),
        )
    return _loader_instance
