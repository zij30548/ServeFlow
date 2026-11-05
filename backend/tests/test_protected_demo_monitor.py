from enum import Enum

import pytest

from core.protected_demo import monitor_snapshot


@pytest.mark.parametrize("value", [None, True, -1, float("nan"), float("inf"), 10 ** 1000, "private text"])
def test_unknown_or_invalid_numbers_are_not_converted_to_zero(value):
    raw = {"agent_stats": {"billing_0": {"total": value, "success_rate": value, "avg_ms": value, "recovery_state": "healthy"}}}
    row = monitor_snapshot(raw)["agent_stats"]["billing_0"]
    assert row["total"] is row["success_rate"] is row["avg_ms"] is None


def test_numeric_zero_and_enum_alert_are_preserved_without_free_text():
    class Severity(Enum):
        WARNING = "warning"
    raw = {"agent_stats": {"billing_0": {"total": 0, "success_rate": 0, "avg_ms": 0, "recovery_state": "healthy"}},
        "active_alerts": [{"severity": Severity.WARNING, "metric": "agent_avg_ms:billing_0", "value": 2, "threshold": 1, "message": "private"}]}
    result = monitor_snapshot(raw)
    assert result["agent_stats"]["billing_0"]["total"] == 0
    assert result["active_alerts"] == [{"severity": "warning", "metric": "agent_avg_ms:billing_0", "value": 2, "threshold": 1, "resolved": False}]


def test_malformed_structures_unknown_states_and_metrics_are_removed():
    raw = {"agent_stats": {"general_0": {"total": 1, "success_rate": .5, "avg_ms": 2, "recovery_state": "private"}},
        "tool_stats": [], "active_alerts": [{"severity": "private", "metric": "agent_avg_ms:general_0"}],
        "suggestions": [{"action": "private"}]}
    result = monitor_snapshot(raw)
    assert result["agent_stats"]["general_0"]["recovery_state"] == "unknown"
    assert result["tool_stats"] == {} and result["active_alerts"] == [] and result["suggestions"] == []
    assert monitor_snapshot(None) == {"agent_stats": {}, "tool_stats": {}, "active_alerts": [], "suggestions": []}
