"""
inference/model_loader.py

vLLM AsyncLLMEngine loader for the AWQ-quantized Llama 3.1 8B model.
Singleton pattern ensures the model is loaded exactly once per process.

vLLM is a required dependency — the server will not start without it.
Run on Linux with a CUDA 12.1+ GPU (T4 16GB or better).
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from dotenv import load_dotenv

from vllm import AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)


class VLLMModelLoader:
    """
    Loads the AWQ-quantized Llama 3.1 8B via vLLM AsyncLLMEngine.

    vLLM provides:
      - PagedAttention for efficient KV-cache memory management
      - Continuous batching — groups requests mid-flight
      - AWQ kernel fusion — fast W4A16 matrix multiplications
      - AsyncLLMEngine — non-blocking, handles many concurrent requests

    Singleton pattern ensures the model is loaded exactly once.
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
        self.engine: Optional[AsyncLLMEngine] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._engine_type: str = "vllm"
        self._initialized = True

    def load(self):
        """Initialize the vLLM AsyncLLMEngine with the AWQ model."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"AWQ model not found at '{self.model_path}'. "
                "Run quantization/awq_quantize.py first, or set AWQ_MODEL_DIR in .env."
            )

        console.print(f"\n[bold cyan]Loading vLLM engine from {self.model_path}...[/bold cyan]")
        t0 = time.time()

        engine_args = AsyncEngineArgs(
            model=self.model_path,
            quantization=self.quantization,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype="auto",
            enable_prefix_caching=True,
            trust_remote_code=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._startup_time = time.time() - t0
        console.print(
            f"[bold green]✓ vLLM engine ready in {self._startup_time:.1f}s[/bold green]"
        )

    @property
    def is_loaded(self) -> bool:
        return self.engine is not None

    @property
    def engine_type(self) -> str:
        return self._engine_type

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
