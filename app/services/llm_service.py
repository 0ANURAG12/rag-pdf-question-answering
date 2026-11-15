"""
Stage 7: Generation.

Sends the constructed prompt to a locally running Ollama server and
returns the generated answer. No API key is required since Ollama
runs on the local machine.
"""

import asyncio
import httpx

from app.config import settings
from app.exceptions import OllamaServiceError
from app.logging_config import get_logger

logger = get_logger(__name__)

_client = httpx.AsyncClient(timeout=settings.ollama_timeout_seconds)

_MAX_RETRIES = 2
_BASE_BACKOFF_SECONDS = 2.0


async def generate_answer(prompt: str) -> str:
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await _client.post(url, json=payload)
            response.raise_for_status()
            break
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                backoff = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    "llm.generate_answer.retrying",
                    extra={
                        "attempt": attempt + 1,
                        "max_attempts": _MAX_RETRIES + 1,
                        "backoff_seconds": backoff,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(backoff)
            else:
                break

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise OllamaServiceError(
            f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
        )

    data = response.json()
    answer = data.get("response", "").strip()

    if not answer:
        raise OllamaServiceError("Ollama returned an empty response.")

    logger.info("llm.generate_answer", extra={"model": settings.ollama_model, "answer_chars": len(answer)})
    return answer


async def list_models() -> list[dict]:
    """Call Ollama /api/tags to list available models (used by /health/deep)."""
    url = f"{settings.ollama_base_url}/api/tags"
    response = await _client.get(url)
    response.raise_for_status()
    data = response.json()
    return data.get("models", [])


async def close_client() -> None:
    """Close the shared AsyncClient — call on app shutdown."""
    await _client.aclose()