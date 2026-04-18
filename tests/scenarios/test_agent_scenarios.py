# tests/scenarios/test_agent_scenarios.py
"""
Scenario tests for the Replenishment Agent.

Tests cover agent behaviour without a live server —
tools are mocked to test decision logic in isolation.

Scenarios:
  1. Healthy system — complete plan produced
  2. Serving layer down — graceful failure
  3. Monitoring unavailable — degraded plan
  4. Retraining recommended — warning in plan
  5. Guardrail enforcement — replenishment clipped
  6. Audit log written correctly
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.replenishment_agent import (
    MAX_REPLENISHMENT_UNITS,
    MIN_REPLENISHMENT_UNITS,
    REPLENISHMENT_BUFFER_MULTIPLIER,
    ReplenishmentAgent,
    ReplenishmentPlan,
    _compute_replenishment,
)
from src.agents.tools import ToolResult

# ── Fixtures ────────────────────────────────────────────────────────────────


def make_tool_result(
    success: bool,
    tool_name: str,
    outputs: dict,
    error: str | None = None,
) -> ToolResult:
    return ToolResult(
        success=success,
        tool_name=tool_name,
        inputs={},
        outputs=outputs,
        error=error,
    )


HEALTHY_FORECAST_OUTPUT = {
    "store_id": 1,
    "start_date": "2015-07-01",
    "horizon_days": 7,
    "forecasts": [
        {
            "forecast_date": f"2015-07-0{i+1}",
            "point_forecast": 2000.0,
            "lower_90": 1000.0,
            "upper_90": 3000.0,
        }
        for i in range(7)
    ],
    "model_version": "test",
    "generated_at": "2026-01-01T00:00:00",
}

HEALTHY_TRIGGER_OUTPUT = {
    "should_retrain": False,
    "recommended_action": "No action required.",
    "drift_summary": {"drift_detected": False},
}

RETRAIN_TRIGGER_OUTPUT = {
    "should_retrain": True,
    "recommended_action": "RETRAIN REQUIRED. Critical threshold breached.",
    "drift_summary": {"drift_detected": True},
}

NARRATIVE_OUTPUT = {
    "narrative": "Store 1 forecast: 2000 units/day. System healthy.",
    "template_used": "combined",
    "prompt_version": "v1",
    "generated_at": "2026-01-01T00:00:00",
    "model": "fallback",
    "grounded": True,
    "error": None,
}


# ── Guardrail unit tests ────────────────────────────────────────────────────


class TestGuardrails:
    def test_normal_replenishment_applies_buffer(self):
        result = _compute_replenishment(10_000.0)
        assert result == 10_000.0 * REPLENISHMENT_BUFFER_MULTIPLIER

    def test_zero_forecast_returns_zero(self):
        result = _compute_replenishment(0.0)
        assert result == 0.0

    def test_excessive_forecast_clipped_to_max(self):
        result = _compute_replenishment(MAX_REPLENISHMENT_UNITS * 10)
        assert result == MAX_REPLENISHMENT_UNITS

    def test_negative_forecast_clipped_to_min(self):
        result = _compute_replenishment(-1000.0)
        assert result == MIN_REPLENISHMENT_UNITS

    def test_replenishment_never_exceeds_max(self):
        for units in [0, 1000, 10_000, 100_000, 1_000_000]:
            result = _compute_replenishment(float(units))
            assert result <= MAX_REPLENISHMENT_UNITS

    def test_replenishment_never_below_min(self):
        for units in [-1000, -100, 0, 100]:
            result = _compute_replenishment(float(units))
            assert result >= MIN_REPLENISHMENT_UNITS


# ── Agent scenario tests ────────────────────────────────────────────────────


class TestAgentScenarios:
    def _make_agent(self, tmp_path: Path) -> ReplenishmentAgent:
        return ReplenishmentAgent(
            base_url="http://localhost:8000",
            audit_log_path=tmp_path / "audit.jsonl",
        )

    @patch("src.agents.replenishment_agent.AgentTools")
    def test_scenario_healthy_system(
        self, mock_tools_class: MagicMock, tmp_path: Path
    ) -> None:
        """Scenario 1: All tools succeed — complete plan produced."""
        mock_tools = MagicMock()
        mock_tools_class.return_value.__enter__.return_value = mock_tools

        mock_tools.check_health.return_value = make_tool_result(
            True, "ready", {"status": "ready", "model_loaded": True}
        )
        mock_tools.get_forecast.return_value = make_tool_result(
            True, "forecast", HEALTHY_FORECAST_OUTPUT
        )
        mock_tools.get_monitoring_trigger.return_value = make_tool_result(
            True, "monitoring_trigger", HEALTHY_TRIGGER_OUTPUT
        )
        mock_tools.generate_narrative.return_value = make_tool_result(
            True, "genai_narrative", NARRATIVE_OUTPUT
        )

        agent = self._make_agent(tmp_path)
        plan = agent.run(
            store_id=1,
            start_date=date(2015, 7, 1),
            horizon_days=7,
        )

        assert plan.status == "complete"
        assert plan.error is None
        assert plan.total_forecast_units == 14_000.0
        assert plan.recommended_replenishment_units == pytest.approx(
            14_000.0 * REPLENISHMENT_BUFFER_MULTIPLIER
        )
        assert len(plan.reasoning) > 0
        assert len(plan.tool_calls) == 4

    @patch("src.agents.replenishment_agent.AgentTools")
    def test_scenario_serving_layer_down(
        self, mock_tools_class: MagicMock, tmp_path: Path
    ) -> None:
        """Scenario 2: Serving layer down — graceful failure."""
        mock_tools = MagicMock()
        mock_tools_class.return_value.__enter__.return_value = mock_tools

        mock_tools.check_health.return_value = make_tool_result(
            False, "ready", {}, error="Connection refused"
        )

        agent = self._make_agent(tmp_path)
        plan = agent.run(
            store_id=1,
            start_date=date(2015, 7, 1),
        )

        assert plan.status == "failed"
        assert plan.error is not None
        assert plan.total_forecast_units == 0.0
        assert plan.recommended_replenishment_units == 0.0

    @patch("src.agents.replenishment_agent.AgentTools")
    def test_scenario_monitoring_unavailable(
        self, mock_tools_class: MagicMock, tmp_path: Path
    ) -> None:
        """Scenario 3: Monitoring unavailable — degraded plan."""
        mock_tools = MagicMock()
        mock_tools_class.return_value.__enter__.return_value = mock_tools

        mock_tools.check_health.return_value = make_tool_result(
            True, "ready", {"status": "ready", "model_loaded": True}
        )
        mock_tools.get_forecast.return_value = make_tool_result(
            True, "forecast", HEALTHY_FORECAST_OUTPUT
        )
        mock_tools.get_monitoring_trigger.return_value = make_tool_result(
            False, "monitoring_trigger", {}, error="Monitoring service unavailable"
        )
        mock_tools.generate_narrative.return_value = make_tool_result(
            True, "genai_narrative", NARRATIVE_OUTPUT
        )

        agent = self._make_agent(tmp_path)
        plan = agent.run(
            store_id=1,
            start_date=date(2015, 7, 1),
        )

        assert plan.status == "degraded"
        assert plan.monitoring_available is False
        assert plan.total_forecast_units > 0
        assert len(plan.warnings) > 0

    @patch("src.agents.replenishment_agent.AgentTools")
    def test_scenario_retraining_recommended(
        self, mock_tools_class: MagicMock, tmp_path: Path
    ) -> None:
        """Scenario 4: Retraining recommended — warning in plan."""
        mock_tools = MagicMock()
        mock_tools_class.return_value.__enter__.return_value = mock_tools

        mock_tools.check_health.return_value = make_tool_result(
            True, "ready", {"status": "ready", "model_loaded": True}
        )
        mock_tools.get_forecast.return_value = make_tool_result(
            True, "forecast", HEALTHY_FORECAST_OUTPUT
        )
        mock_tools.get_monitoring_trigger.return_value = make_tool_result(
            True, "monitoring_trigger", RETRAIN_TRIGGER_OUTPUT
        )
        mock_tools.generate_narrative.return_value = make_tool_result(
            True, "genai_narrative", NARRATIVE_OUTPUT
        )

        agent = self._make_agent(tmp_path)
        plan = agent.run(
            store_id=1,
            start_date=date(2015, 7, 1),
        )

        assert plan.should_retrain is True
        assert plan.drift_detected is True
        assert any("Retraining" in w for w in plan.warnings)
        assert plan.status == "complete"

    @patch("src.agents.replenishment_agent.AgentTools")
    def test_scenario_audit_log_written(
        self, mock_tools_class: MagicMock, tmp_path: Path
    ) -> None:
        """Scenario 5: Audit log written after every run."""
        mock_tools = MagicMock()
        mock_tools_class.return_value.__enter__.return_value = mock_tools

        mock_tools.check_health.return_value = make_tool_result(
            True, "ready", {"status": "ready", "model_loaded": True}
        )
        mock_tools.get_forecast.return_value = make_tool_result(
            True, "forecast", HEALTHY_FORECAST_OUTPUT
        )
        mock_tools.get_monitoring_trigger.return_value = make_tool_result(
            True, "monitoring_trigger", HEALTHY_TRIGGER_OUTPUT
        )
        mock_tools.generate_narrative.return_value = make_tool_result(
            True, "genai_narrative", NARRATIVE_OUTPUT
        )

        audit_path = tmp_path / "audit.jsonl"
        agent = ReplenishmentAgent(
            base_url="http://localhost:8000",
            audit_log_path=audit_path,
        )

        agent.run(store_id=1, start_date=date(2015, 7, 1))
        time.sleep(1)  # Ensure different timestamp-based plan_id
        agent.run(store_id=2, start_date=date(2015, 7, 1))

        assert audit_path.exists()
        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) == 2

        plan1 = json.loads(lines[0])
        plan2 = json.loads(lines[1])
        assert plan1["store_id"] == 1
        assert plan2["store_id"] == 2
        assert plan1["plan_id"] != plan2["plan_id"]

    @patch("src.agents.replenishment_agent.AgentTools")
    def test_scenario_plan_always_returned(
        self, mock_tools_class: MagicMock, tmp_path: Path
    ) -> None:
        """Scenario 6: Agent never raises — always returns a plan."""
        mock_tools = MagicMock()
        mock_tools_class.return_value.__enter__.return_value = mock_tools

        # All tools fail
        mock_tools.check_health.return_value = make_tool_result(
            False, "ready", {}, error="Everything is broken"
        )

        agent = self._make_agent(tmp_path)

        # Should never raise
        plan = agent.run(store_id=999, start_date=date(2015, 1, 1))
        assert isinstance(plan, ReplenishmentPlan)
        assert plan.status == "failed"
