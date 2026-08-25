"""
Server-Sent Events (SSE) Streaming Utilities
Enables token-by-token streaming output from vLLM or HF inference.

The /summarize/stream endpoint uses this module to push tokens to the client
as they are generated, rather than buffering the entire summary.

SSE format:
  data: {"token": "Scientists", "done": false}
  data: {"token": " have", "done": false}
  ...
  data: {"token": "", "done": true, "summary": "...", "latency_ms": 423.1}

Usage with JavaScript:
  const es = new EventSource('/summarize/stream?...');
  es.onmessage = (e) => { const d = JSON.parse(e.data); console.log(d.token); };
"""

import json
import time
import uuid
import logging
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


async def stream_vllm(
    engine,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from vLLM AsyncLLMEngine as SSE-formatted events.

    Yields SSE data strings, each representing one incremental token.
    The final event contains the complete summary and latency metadata.

    Args:
        engine: vLLM AsyncLLMEngine instance
        prompt: Formatted prompt string
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability

    Yields:
        SSE data strings: "data: {...}\n\n"
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.1,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()
    full_text = ""
    previous_text = ""

    try:
        async for request_output in engine.generate(prompt, sampling_params, request_id):
            if request_output.outputs:
                current_text = request_output.outputs[0].text
                # Compute the incremental new token(s)
                new_token = current_text[len(previous_text):]
                previous_text = current_text
                full_text = current_text

                if new_token:
                    payload = json.dumps({"token": new_token, "done": False})
                    yield f"data: {payload}\n\n"

        # Final event with complete summary
        latency_ms = (time.perf_counter() - t_start) * 1000
        summary = full_text.replace("<|eot_id|>", "").strip()
        final_payload = json.dumps({
            "token": "",
            "done": True,
            "summary": summary,
            "latency_ms": round(latency_ms, 2),
        })
        yield f"data: {final_payload}\n\n"

    except Exception as e:
        logger.error(f"vLLM streaming error: {e}", exc_info=True)
        error_payload = json.dumps({"error": str(e), "done": True})
        yield f"data: {error_payload}\n\n"


async def stream_hf(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from HuggingFace model using TextIteratorStreamer.

    HF streaming is less efficient than vLLM but works on any device.
    Uses asyncio to run the blocking generation in a thread pool.

    Yields:
        SSE data strings: "data: {...}\n\n"
    """
    import asyncio
    from transformers import TextIteratorStreamer
    from threading import Thread

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": 1.1,
        "do_sample": temperature > 0,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    t_start = time.perf_counter()
    full_text = ""

    # Run model.generate in a thread (it's blocking)
    loop = asyncio.get_running_loop()
    thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()

    try:
        # Yield tokens as they arrive from the streamer
        for token_text in streamer:
            clean = token_text.replace("<|eot_id|>", "")
            if clean:
                full_text += clean
                payload = json.dumps({"token": clean, "done": False})
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0)  # Yield control to event loop

        # Final event
        latency_ms = (time.perf_counter() - t_start) * 1000
        final_payload = json.dumps({
            "token": "",
            "done": True,
            "summary": full_text.strip(),
            "latency_ms": round(latency_ms, 2),
        })
        yield f"data: {final_payload}\n\n"

    except Exception as e:
        logger.error(f"HF streaming error: {e}", exc_info=True)
        error_payload = json.dumps({"error": str(e), "done": True})
        yield f"data: {error_payload}\n\n"
    finally:
        thread.join(timeout=5)
