"""
Inference Prompt Utilities — Decoupled from training data pipeline.

Provides Llama 3.1 chat-template prompt formatting for inference only.
Intentionally lives in inference/ so the serving stack has no dependency
on the training data pipeline (data/, preprocessing, etc.).
"""

# ── Llama 3.1 Chat Template Tokens ──────────────────────────────
BOS = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"

SYSTEM_PROMPT = (
    "You are an expert document summarizer. "
    "Provide concise, accurate summaries that capture the key information."
)


def format_inference_prompt(article: str) -> str:
    """
    Format an article into a Llama 3.1 instruction-following prompt for inference.

    The prompt ends right after the assistant header so the model generates
    the summary as a continuation — no reference summary is included.

    Args:
        article: The full article text to summarize.

    Returns:
        Formatted prompt string ready for tokenization.
    """
    article = article.strip()
    return (
        f"{BOS}{START_HEADER}system{END_HEADER}\n"
        f"{SYSTEM_PROMPT}{EOT}\n"
        f"{START_HEADER}user{END_HEADER}\n"
        f"Summarize the following article:\n\n{article}{EOT}\n"
        f"{START_HEADER}assistant{END_HEADER}\n"
    )


def format_training_prompt(article: str, summary: str) -> str:
    """
    Format article + summary into a full Llama 3.1 instruction-tuning example.

    Used during dataset construction and AWQ calibration.

    Args:
        article:  Full article text.
        summary:  Ground-truth highlight/summary.

    Returns:
        Complete prompt+response string for language model training.
    """
    article = article.strip()
    summary = summary.strip()
    return (
        f"{BOS}{START_HEADER}system{END_HEADER}\n"
        f"{SYSTEM_PROMPT}{EOT}\n"
        f"{START_HEADER}user{END_HEADER}\n"
        f"Summarize the following article:\n\n{article}{EOT}\n"
        f"{START_HEADER}assistant{END_HEADER}\n"
        f"{summary}{EOT}"
    )


def get_stop_tokens() -> list:
    """Return the Llama 3.1 stop token strings for vLLM SamplingParams."""
    return [EOT, "<|end_of_text|>"]
