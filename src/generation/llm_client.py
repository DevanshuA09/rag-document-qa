"""
llm_client.py — Thin wrapper around the OpenAI Chat Completions API.

Handles auth, error translation, token counting, and cost estimation.
All callers should use the module-level singleton `get_client()` or
instantiate `LLMClient()` directly.

Primary interface:
    client = LLMClient()
    text   = client.generate(system_prompt, user_prompt)
    result = client.generate_with_usage(system_prompt, user_prompt)
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# GPT-4o pricing as of mid-2024 (USD per token).
_COST_PER_INPUT_TOKEN  = 5.00  / 1_000_000   # $5 / 1M input tokens
_COST_PER_OUTPUT_TOKEN = 15.00 / 1_000_000   # $15 / 1M output tokens

_TIMEOUT_SECONDS = 60


class LLMClient:
    """OpenAI Chat Completions client with error handling and cost tracking.

    Args:
        model: Override the model name.  Reads ``OPENAI_MODEL`` from env;
            defaults to ``"gpt-4o"``.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self.model: str = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self._client = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return the response text.

        Args:
            system_prompt: System instruction message.
            user_prompt: User turn message.

        Returns:
            The assistant's reply as a plain string.

        Raises:
            ValueError: Invalid or missing API key.
            RuntimeError: Rate limit, timeout, or other API error.
        """
        return self.generate_with_usage(system_prompt, user_prompt)["response"]

    def generate_with_usage(self, system_prompt: str, user_prompt: str) -> dict:
        """Call the LLM and return the response together with token/cost metadata.

        Args:
            system_prompt: System instruction message.
            user_prompt: User turn message.

        Returns:
            Dict with keys::

                {
                    "response":           str,
                    "model":              str,
                    "prompt_tokens":      int,
                    "completion_tokens":  int,
                    "total_tokens":       int,
                    "cost_usd":           float,
                }

        Raises:
            ValueError: Invalid or missing API key.
            RuntimeError: Rate limit, timeout, or other API error.
        """
        client = self._get_client()

        try:
            completion = client.chat.completions.create(
                model=self.model,
                temperature=0,
                timeout=_TIMEOUT_SECONDS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
        except Exception as exc:
            self._translate_error(exc)

        response_text    = completion.choices[0].message.content or ""
        prompt_tokens    = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        total_tokens     = completion.usage.total_tokens
        cost_usd         = (
            prompt_tokens    * _COST_PER_INPUT_TOKEN
            + completion_tokens * _COST_PER_OUTPUT_TOKEN
        )

        logger.info(
            "LLM call: model=%s  tokens=%d (in=%d out=%d)  cost=$%.5f",
            self.model,
            total_tokens,
            prompt_tokens,
            completion_tokens,
            cost_usd,
        )

        return {
            "response":           response_text,
            "model":              self.model,
            "prompt_tokens":      prompt_tokens,
            "completion_tokens":  completion_tokens,
            "total_tokens":       total_tokens,
            "cost_usd":           cost_usd,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return a shared OpenAI client, initialising it on first call.

        Raises:
            ValueError: If ``OPENAI_API_KEY`` is missing or blank.
        """
        if self._client is not None:
            return self._client

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key: "
                "OPENAI_API_KEY=sk-..."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=api_key)
        logger.debug("OpenAI client initialised (model=%s).", self.model)
        return self._client

    @staticmethod
    def _translate_error(exc: Exception) -> None:
        """Re-raise an OpenAI SDK exception as a more descriptive error.

        Args:
            exc: The raw exception from the OpenAI SDK.

        Raises:
            ValueError: For auth errors.
            RuntimeError: For all other API errors.
        """
        name = type(exc).__name__
        msg  = str(exc)

        if "AuthenticationError" in name or "401" in msg:
            raise ValueError(
                "OpenAI authentication failed. "
                "Check that OPENAI_API_KEY in your .env is valid."
            ) from exc

        if "RateLimitError" in name or "429" in msg:
            raise RuntimeError(
                "OpenAI rate limit hit. Wait a moment and retry, "
                "or check your usage quota at platform.openai.com."
            ) from exc

        if "Timeout" in name or "timeout" in msg.lower():
            raise RuntimeError(
                f"OpenAI request timed out after {_TIMEOUT_SECONDS}s. "
                "The prompt may be too long, or the API is slow. Try again."
            ) from exc

        raise RuntimeError(f"OpenAI API error ({name}): {msg}") from exc
