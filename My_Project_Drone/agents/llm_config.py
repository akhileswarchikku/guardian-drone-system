"""
Phase 2 — OpenRouter LLM configuration.
All LLM agents share the same ChatOpenAI instance with tenacity retry.
Rule-based fallback is injected at the call site — this module only handles
connectivity and retry logic.
"""
from __future__ import annotations

import os
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL       = "google/gemini-2.5-flash"


def get_llm(
    model: str       = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int  = 512,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=0,       # tenacity handles retries externally
        default_headers={
            "HTTP-Referer": "https://github.com/guardian-drone",
            "X-Title":      "Guardian Drone System",
        },
    )


def with_llm_retry(max_attempts: int = 3):
    """Retry decorator for LLM calls: 3 attempts, exponential backoff 2–10 s."""
    def decorator(func):
        @retry(
            stop   = stop_after_attempt(max_attempts),
            wait   = wait_exponential(multiplier=1, min=2, max=10),
            retry  = retry_if_exception_type(Exception),
            reraise = True,
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
