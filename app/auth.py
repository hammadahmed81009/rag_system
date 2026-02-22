import time
from collections import defaultdict

from fastapi import Header, HTTPException, Request

from app.config import settings

# In-memory rate limit: key -> list of request timestamps (sliding window)
_rate_limit_cache: dict[str, list[float]] = defaultdict(list)
_rate_limit_window = 60.0  # seconds


def require_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    if not settings.api_key:
        return
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _rate_limit_key(request: Request, x_api_key: str | None) -> str:
    return x_api_key or request.client.host if request.client else "unknown"


def check_rate_limit(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    if settings.rate_limit_per_minute <= 0:
        return
    key = _rate_limit_key(request, x_api_key)
    now = time.monotonic()
    window_start = now - _rate_limit_window
    cache = _rate_limit_cache[key]
    cache[:] = [t for t in cache if t > window_start]
    if len(cache) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    cache.append(now)
