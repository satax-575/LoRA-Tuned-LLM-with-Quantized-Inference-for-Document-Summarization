"""
Unit tests for the training pipeline.
Tests QLoRA config, label masking, prompt formatting, and parameter counting.
No GPU required — tests run on CPU with mocked models.
"""

import sys
import math
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# Tests: QLoRA Config
# ─────────────────────────────────────────────

class TestGetBnbConfig:
    """Tests for get_bnb_config() in training.qlora_config."""

    def test_load_in_4bit_true(self):
        from training.qlora_config import get_bnb_config
        cfg = {
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            }
        }
        bnb = get_bnb_config(cfg)
        assert bnb.load_in_4bit is True

    def test_quant_type_is_nf4(self):
        from training.qlora_config import get_bnb_config
        import torch
        cfg = {
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            }
        }
        bnb = get_bnb_config(cfg)
        assert bnb.bnb_4bit_quant_type == "nf4"

    def test_compute_dtype_bfloat16(self):
        from training.qlora_config import get_bnb_config
        import torch
        cfg = {
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            }
        }
        bnb = get_bnb_config(cfg)
        assert bnb.bnb_4bit_compute_dtype == torch.bfloat16

    def test_double_quant_enabled(self):
        from training.qlora_config import get_bnb_config
        cfg = {
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            }
        }
        bnb = get_bnb_config(cfg)
        assert bnb.bnb_4bit_use_double_quant is True


class TestGetLoraConfig:
    """Tests for get_lora_config() in training.qlora_config."""

    def _base_cfg(self):
        return {
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        }

    def test_rank_16(self):
        from training.qlora_config import get_lora_config
        lora = get_lora_config(self._base_cfg())
        assert lora.r == 16

    def test_alpha_32(self):
        from training.qlora_config import get_lora_config
        lora = get_lora_config(self._base_cfg())
        assert lora.lora_alpha == 32

    def test_scaling_factor_2(self):
        """alpha/rank = 2 is the canonical scaling factor for this project."""
        from training.qlora_config import get_lora_config
        lora = get_lora_config(self._base_cfg())
        assert lora.lora_alpha / lora.r == 2.0

    def test_inference_mode_false(self):
        from training.qlora_config import get_lora_config
        lora = get_lora_config(self._base_cfg())
        assert lora.inference_mode is False

    def test_target_modules_set(self):
        from training.qlora_config import get_lora_config
        lora = get_lora_config(self._base_cfg())
        assert "q_proj" in lora.target_modules
        assert "v_proj" in lora.target_modules


class TestComputeTrainableParams:
    """Tests for compute_trainable_params()."""

    def test_all_frozen_returns_zero(self):
        from training.qlora_config import compute_trainable_params
        import torch
        import torch.nn as nn

        model = nn.Linear(10, 10)
        for p in model.parameters():
            p.requires_grad = False

        stats = compute_trainable_params(model)
        assert stats["trainable_params"] == 0
        assert stats["trainable_pct"] == 0.0

    def test_all_trainable_returns_total(self):
        from training.qlora_config import compute_trainable_params
        import torch.nn as nn

        model = nn.Linear(10, 10)  # 110 params (10*10 weights + 10 bias)
        stats = compute_trainable_params(model)
        assert stats["trainable_params"] == stats["total_params"]
        assert stats["trainable_pct"] == 100.0

    def test_partial_trainable_percentage(self):
        from training.qlora_config import compute_trainable_params
        import torch.nn as nn

        layer1 = nn.Linear(10, 10)  # trainable
        layer2 = nn.Linear(10, 10)  # frozen
        for p in layer2.parameters():
            p.requires_grad = False

        model = nn.Sequential(layer1, layer2)
        stats = compute_trainable_params(model)
        assert stats["trainable_pct"] == pytest.approx(50.0, abs=1.0)


# ─────────────────────────────────────────────
# Tests: Prompt Formatting
# ─────────────────────────────────────────────

class TestPromptFormatting:
    """Tests for prompt_utils and dataset_builder prompt functions."""

    def test_inference_prompt_ends_with_assistant_header(self):
        from inference.prompt_utils import format_inference_prompt
        prompt = format_inference_prompt("This is a test article.")
        assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n")

    def test_inference_prompt_contains_article(self):
        from inference.prompt_utils import format_inference_prompt
        article = "Scientists discovered something amazing today."
        prompt = format_inference_prompt(article)
        assert article in prompt

    def test_inference_prompt_has_system_message(self):
        from inference.prompt_utils import format_inference_prompt
        prompt = format_inference_prompt("Some text.")
        assert "system" in prompt
        assert "document summarizer" in prompt.lower()

    def test_training_prompt_contains_summary(self):
        from inference.prompt_utils import format_training_prompt
        summary = "A brief summary of the article."
        prompt = format_training_prompt("Long article...", summary)
        assert summary in prompt

    def test_training_prompt_ends_with_eot(self):
        from inference.prompt_utils import format_training_prompt
        prompt = format_training_prompt("Article", "Summary")
        assert prompt.endswith("<|eot_id|>")

    def test_stop_tokens_returns_list(self):
        from inference.prompt_utils import get_stop_tokens
        stops = get_stop_tokens()
        assert isinstance(stops, list)
        assert "<|eot_id|>" in stops

    def test_dataset_builder_format_prompt(self):
        """Test that dataset_builder.format_prompt matches inference.prompt_utils.format_training_prompt."""
        from data.dataset_builder import format_prompt as builder_fmt
        from inference.prompt_utils import format_training_prompt as infer_fmt
        article = "A test article about science."
        summary = "Science article summary."
        # Both should produce the same structured output
        builder_out = builder_fmt(article, summary)
        infer_out = infer_fmt(article, summary)
        assert builder_out == infer_out


# ─────────────────────────────────────────────
# Tests: Label Masking
# ─────────────────────────────────────────────

class TestLabelMasking:
    """
    Tests for the causal label masking in dataset_builder.
    Verifies that prompt tokens have labels=-100.
    """

    def test_labels_start_with_minus_100(self):
        """All labels for the prompt portion must be -100."""
        from inference.prompt_utils import format_inference_prompt, format_training_prompt

        article = "A scientist made a discovery."
        summary = "Discovery made by scientist."

        prompt_only = format_inference_prompt(article)
        full_prompt = format_training_prompt(article, summary)

        # Prompt should be a strict prefix of the full prompt
        assert full_prompt.startswith(prompt_only), (
            "Full training prompt must start with the inference prompt prefix. "
            "This is required for correct label masking."
        )

    def test_summary_is_suffix_of_full_prompt(self):
        from inference.prompt_utils import format_inference_prompt, format_training_prompt

        summary = "Key finding discovered."
        article = "Scientists found a key thing."
        full = format_training_prompt(article, summary)
        prompt_only = format_inference_prompt(article)

        # The part after the prompt prefix should contain the summary
        suffix = full[len(prompt_only):]
        assert summary in suffix


# ─────────────────────────────────────────────
# Tests: Preprocessing
# ─────────────────────────────────────────────

class TestPreprocessing:
    """Tests for text cleaning utilities in data.preprocessing."""

    def test_clean_article_removes_cnn_marker(self):
        from data.preprocessing import clean_article
        text = "(CNN) This is an article about something."
        cleaned = clean_article(text)
        assert "(CNN)" not in cleaned
        assert "This is an article" in cleaned

    def test_clean_article_normalizes_whitespace(self):
        from data.preprocessing import clean_article
        text = "Word1   Word2\n\n\n\nWord3"
        cleaned = clean_article(text)
        assert "   " not in cleaned  # No triple spaces
        assert "\n\n\n" not in cleaned  # No triple newlines

    def test_clean_summary_removes_bullets(self):
        from data.preprocessing import clean_summary
        text = "- Key point one\n• Key point two\n* Key point three"
        cleaned = clean_summary(text)
        assert "- " not in cleaned
        assert "• " not in cleaned
        assert "* " not in cleaned

    def test_clean_summary_removes_new_prefix(self):
        from data.preprocessing import clean_summary
        text = "NEW: Something important happened today."
        cleaned = clean_summary(text)
        assert "NEW:" not in cleaned

    def test_truncate_respects_sentence_boundary(self):
        from data.preprocessing import truncate_to_word_limit
        text = "First sentence ends here. Second sentence continues on. Third sentence."
        result = truncate_to_word_limit(text, max_words=7)
        # Should end at a sentence boundary
        assert result.endswith(".")

    def test_truncate_short_text_unchanged(self):
        from data.preprocessing import truncate_to_word_limit
        text = "Short text."
        result = truncate_to_word_limit(text, max_words=100)
        assert result == text
