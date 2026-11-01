"""Shared Anthropic client.

The zero-argument constructor resolves credentials itself (ANTHROPIC_API_KEY, or an
`ant auth login` profile), so no key ever passes through application config.
"""

from __future__ import annotations

from functools import lru_cache

import anthropic


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=3)


@lru_cache(maxsize=1)
def get_async_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(max_retries=3)


NO_CREDENTIALS_MESSAGE = (
    "No Anthropic credentials are configured. Set ANTHROPIC_API_KEY (for example in "
    "backend/.env) or run `ant auth login`."
)


def is_missing_credentials(exc: BaseException) -> bool:
    """The SDK raises a plain TypeError at request time when it finds no credentials."""
    return isinstance(exc, TypeError) and "authentication method" in str(exc)


# Prices per million tokens, used only for the cost estimates printed before paid runs.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, *, batch: bool = False
) -> float:
    in_price, out_price = PRICE_PER_MTOK.get(model, (4.00, 20.00))
    cost = (input_tokens * in_price + output_tokens * out_price) / 1_000_000
    return cost * (0.5 if batch else 1.0)
