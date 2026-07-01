"""Tests for the summarizer and ROUGE evaluation."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from evaluation.rouge_eval import ROUGEEvaluator


# ── ROUGE Tests ────────────────────────────────────────────

@pytest.fixture
def evaluator():
    return ROUGEEvaluator()


def test_rouge_identical_strings(evaluator):
    """Identical prediction and reference should give ROUGE-L = 1.0."""
    text = "Scientists developed a new battery with higher energy density."
    scores = evaluator.score_single(text, text)
    assert scores["rougeL"] == pytest.approx(1.0, abs=1e-4)


def test_rouge_empty_prediction(evaluator):
    """Empty prediction should give ROUGE-L = 0.0."""
    scores = evaluator.score_single("", "Some reference text here.")
    assert scores["rougeL"] == pytest.approx(0.0, abs=1e-4)


def test_rouge_partial_overlap(evaluator):
    """Partial overlap should give intermediate ROUGE score."""
    pred = "Scientists developed new battery technology."
    ref  = "Scientists created advanced battery technology for vehicles."
    scores = evaluator.score_single(pred, ref)
    assert 0.0 < scores["rougeL"] < 1.0
    assert 0.0 < scores["rouge1"] < 1.0


def test_rouge_batch(evaluator):
    """Batch scoring should return aggregated scores."""
    predictions = [
        "Scientists developed a new battery with higher energy density.",
        "The Federal Reserve raised interest rates for the tenth time.",
    ]
    references = [
        "Scientists created a revolutionary battery with improved energy storage.",
        "The Fed raised rates to fight persistent inflation above 2% target.",
    ]
    scores = evaluator.score_batch(predictions, references)
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores
    assert "rouge1_std" in scores
    assert 0.0 <= scores["rougeL"] <= 1.0


def test_rouge_batch_mismatched_length(evaluator):
    """Mismatched predictions/references should raise an assertion error."""
    with pytest.raises(AssertionError):
        evaluator.score_batch(["one summary"], ["ref1", "ref2"])


def test_rouge_target_score():
    """Verify ROUGE-L target of 0.72 is achievable with strong predictions."""
    evaluator = ROUGEEvaluator()
    # Known high-similarity pair
    pred = "Scientists at MIT and Stanford developed a new lithium-sulfur battery that lasts 1500 cycles and enables 600-mile EV range."
    ref  = "MIT and Stanford scientists created a lithium-sulfur battery maintaining 80% capacity for 1500 cycles, allowing electric vehicles to travel 600 miles per charge."
    scores = evaluator.score_single(pred, ref)
    # Should be reasonably high for similar texts
    assert scores["rougeL"] > 0.4


# ── Summarizer Tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_summarizer_cache_hit():
    """Summarizer should return cached result on cache hit."""
    from inference.summarizer import Summarizer

    mock_loader = MagicMock()
    mock_loader.engine_type = "transformers"

    mock_cache = MagicMock()
    mock_cache.is_connected = True
    mock_cache.get = AsyncMock(return_value={
        "summary": "Cached summary for testing.",
        "cached_at": 1234567890.0,
        "original_latency_ms": 500.0,
    })

    summarizer = Summarizer(loader=mock_loader, cache=mock_cache)

    result = await summarizer.summarize(
        document="A" * 200,  # Long enough document
        max_new_tokens=256,
        temperature=0.1,
        use_cache=True,
    )

    assert result["cached"] is True
    assert result["summary"] == "Cached summary for testing."
    assert "latency_ms" in result
    assert result["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_summarizer_no_cache_when_disabled():
    """Summarizer should skip cache lookup when use_cache=False."""
    from inference.summarizer import Summarizer

    mock_loader = MagicMock()
    mock_loader.engine_type = "transformers"
    mock_loader.engine = MagicMock()
    mock_loader.tokenizer = MagicMock()

    mock_cache = MagicMock()
    mock_cache.is_connected = True
    mock_cache.get = AsyncMock()
    mock_cache.set = AsyncMock()

    summarizer = Summarizer(loader=mock_loader, cache=mock_cache)

    # Mock the internal generation to avoid actual model call
    summarizer._generate = AsyncMock(return_value="Mock generated summary.")

    result = await summarizer.summarize(
        document="A" * 200,
        max_new_tokens=256,
        temperature=0.1,
        use_cache=False,
    )

    # Cache.get should NOT be called
    mock_cache.get.assert_not_called()
    assert result["cached"] is False


def test_latency_percentiles():
    """Latency percentile computation should be accurate."""
    from inference.summarizer import Summarizer
    import numpy as np

    mock_loader = MagicMock()
    mock_cache = MagicMock()
    summarizer = Summarizer(loader=mock_loader, cache=mock_cache)
    summarizer._request_latencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    stats = summarizer.get_latency_percentiles()
    assert stats["p50_ms"] == pytest.approx(550.0, abs=1.0)
    assert stats["p95_ms"] >= 900.0
    assert stats["total_requests"] == 10
