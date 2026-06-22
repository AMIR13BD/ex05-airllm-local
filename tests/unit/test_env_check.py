"""Unit test for the environment-check service (structure only)."""
from orch5.services import env_check


def test_collect_returns_expected_keys():
    info = env_check.collect()
    assert "python" in info and "platform" in info and "packages" in info
    # the pinned core packages must be reported (value may be a version or 'MISSING (...)')
    for pkg in ["torch", "transformers", "airllm", "numpy"]:
        assert pkg in info["packages"]


def test_report_prints_and_returns(capsys):
    info = env_check.report()
    out = capsys.readouterr().out
    assert "Python" in out
    assert info["packages"]  # same structured snapshot returned
