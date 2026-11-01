"""Polite, cached fetching. Raw bytes are cached under data/raw/ and checksummed, so
re-ingest is reproducible offline and an unchanged source is skipped entirely."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class Fetched:
    path: Path
    content: bytes
    sha256: str
    from_cache: bool


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_cached(
    url: str,
    dest: Path,
    *,
    user_agent: str,
    refresh: bool = False,
    timeout: float = 60.0,
    retries: int = 3,
) -> Fetched:
    if dest.exists() and not refresh:
        data = dest.read_bytes()
        return Fetched(dest, data, sha256_bytes(data), from_cache=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = httpx.get(
                url, headers={"User-Agent": user_agent}, timeout=timeout, follow_redirects=True
            )
            response.raise_for_status()
            data = response.content
            dest.write_bytes(data)
            return Fetched(dest, data, sha256_bytes(data), from_cache=False)
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error
