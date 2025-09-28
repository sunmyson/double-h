"""Remote LLM client using Groq's free API tier.

Set the ``GROQ_API_KEY`` environment variable with your personal token before
calling :func:`prompt_llm`.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

DEFAULT_MODEL = "mixtral-8x7b-32768"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
SYSTEM_PROMPT = "You are a fast, precise assistant. Keep replies short and clear."


class LLMError(RuntimeError):
    """Raised when the remote LLM client cannot complete a request."""


def prompt_llm(prompt: str, *, model: str = DEFAULT_MODEL, api_key: Optional[str] = None) -> str:
    """Return a completion from Groq's hosted models.

    Args:
        prompt: User content to send to the model.
        model: Groq model identifier. Defaults to ``mixtral-8x7b-32768`` (free tier).
        api_key: Optional explicit API key. Falls back to ``GROQ_API_KEY`` env var.
    """
    token = api_key or os.getenv("GROQ_API_KEY")
    if not token:
        raise LLMError(
            "Missing GROQ_API_KEY environment variable. Create a free API token at "
            "https://console.groq.com/, then export it before running the app."
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=45.0) as client:
            response = client.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        raise LLMError(f"Groq API returned HTTP {exc.response.status_code}: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"Groq API request failed: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise LLMError("Groq API returned no completion choices.")

    message = choices[0].get("message", {}).get("content")
    if not message:
        raise LLMError("Groq API completion had empty content.")

    return message.strip()


if __name__ == "__main__":  # pragma: no cover - manual check
    print(prompt_llm("What is the capital of Greece?"))
