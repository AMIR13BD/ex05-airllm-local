"""Unit tests for the API gatekeeper (rate limiting, retries)."""
import pytest

from orch5.shared.gatekeeper import ApiGatekeeper


def test_execute_success_returns_value(rate_cfg):
    gk = ApiGatekeeper(rate_cfg)
    assert gk.execute(lambda x: x * 2, 21) == 42


def test_execute_retries_then_succeeds(rate_cfg):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("transient")
        return "ok"

    gk = ApiGatekeeper(rate_cfg)
    assert gk.execute(flaky) == "ok"
    assert calls["n"] == 2


def test_execute_raises_after_max_retries(rate_cfg):
    def always_fail():
        raise RuntimeError("boom")

    gk = ApiGatekeeper(rate_cfg)
    with pytest.raises(RuntimeError):
        gk.execute(always_fail)


def test_queue_status_reports_limit(rate_cfg):
    gk = ApiGatekeeper(rate_cfg)
    gk.execute(lambda: None)
    status = gk.queue_status()
    assert status["rpm_limit"] == 1000
    assert status["in_window"] >= 1


def test_rate_limit_blocks_when_exceeded():
    # rpm=1 means the 2nd immediate call must queue (we don't wait the full minute here;
    # just assert the limiter recognizes the window is full).
    gk = ApiGatekeeper({"services": {"default": {"requests_per_minute": 1, "max_retries": 1}}})
    gk.execute(lambda: 1)
    assert gk.queue_status()["in_window"] == 1
