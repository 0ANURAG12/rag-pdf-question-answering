"""
Stage 7: Generation.

Sends the constructed prompt to a locally running Ollama server and
returns the generated answer. No API key is required since Ollama
runs on the local machine.
"""

import requests

from app.config import settings
from app.exceptions import OllamaServiceError


def generate_answer(prompt: str) -> str:
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=settings.ollama_timeout_seconds)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise OllamaServiceError(
            f"Could not get a response from Ollama at {settings.ollama_base_url}. "
            f"Make sure Ollama is running and the model '{settings.ollama_model}' is pulled. "
            f"Details: {exc}"
        )

    data = response.json()
    answer = data.get("response", "").strip()

    if not answer:
        raise OllamaServiceError("Ollama returned an empty response.")

    return answer
