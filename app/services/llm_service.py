"""
Stage 7: Generation.

Sends the constructed prompt to a locally running Ollama server and
returns the generated answer. No API key is required since Ollama
runs on the local machine.
"""

import httpx

from app.config import settings
from app.exceptions import OllamaServiceError
from app.logging_config import get_logger

logger = get_logger(__name__)

_client = httpx.AsyncClient(timeout=settings.ollama_timeout_seconds)


async def generate_answer(prompt: str) -> str:
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = await _client.post(url, json=payload)
        response.raise_for_status()
    except httpx.RequestError as exc:
        raise OllamaServiceError(
            f"Could not connect to Ollama at {settings.ollama_base_url}. "
            f"Make sure Ollama is running and the model '{settings.ollama_model}' is pulled. "
            f"Details: {exc}"
        )
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