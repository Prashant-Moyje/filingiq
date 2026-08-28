"""Thread-safe token-bucket rate limiter.

The SEC will block you if you exceed ~10 requests/second. This is not a
suggestion -- they enforce it at the CDN. Every EDGAR call goes through here.
"""
from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        self._min_interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last_call = now

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        return None
