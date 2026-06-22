"""Centralized API gatekeeper (guidelines §5).

Every external (third-party) API call — e.g. live provider pricing checks in the cost
analysis — must go through this gatekeeper. It enforces a per-minute rate limit from
config/rate_limits.json (never hardcoded), retries transient failures with backoff, queues
on overflow rather than dropping, and logs every call.
"""
import logging
import time
from collections import deque

logger = logging.getLogger("orch5.gatekeeper")


class ApiGatekeeper:
    """Rate-limited, retrying, logged entry point for all external API calls."""

    def __init__(self, config: dict):
        svc = config.get("services", {}).get("default", {})
        self.rpm = int(svc.get("requests_per_minute", 30))
        self.max_retries = int(svc.get("max_retries", 3))
        self.retry_after = float(svc.get("retry_after_seconds", 30))
        self._calls: deque[float] = deque()      # timestamps within the last 60 s
        self._queued = 0

    def _wait_for_slot(self) -> None:
        """Block until under the per-minute limit (queue, don't drop)."""
        while True:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= 60.0:
                self._calls.popleft()
            if len(self._calls) < self.rpm:
                self._calls.append(now)
                return
            self._queued += 1
            sleep_s = 60.0 - (now - self._calls[0])
            logger.info("rate limit reached; queueing for %.1fs", sleep_s)
            time.sleep(max(sleep_s, 0.05))
            self._queued -= 1

    def execute(self, api_call, *args, **kwargs):
        """Run ``api_call(*args, **kwargs)`` through the gatekeeper with retries."""
        self._wait_for_slot()
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = api_call(*args, **kwargs)
                logger.info("api call ok (attempt %d)", attempt)
                return result
            except Exception as exc:  # noqa: BLE001 — retry transient failures
                last_exc = exc
                logger.warning("api call failed (attempt %d/%d): %s",
                               attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_after)
        raise RuntimeError(f"API call failed after {self.max_retries} attempts") from last_exc

    def queue_status(self) -> dict:
        return {"in_window": len(self._calls), "rpm_limit": self.rpm, "queued": self._queued}
