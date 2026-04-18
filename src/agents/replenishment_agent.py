# src/agents/replenishment_agent.py
"""
Replenishment Agent for demand-forecast-ops.

Orchestrates the full replenishment workflow:
  1. Check serving layer health
  2. Get demand forecast for requested store
  3. Check monitoring trigger for retraining signals
  4. Generate stakeholder narrative
  5. Produce a structured ReplenishmentPlan
  6. Write full audit log

Design rules (non-negotiable):
  - Never imports model weights directly
  - All model access via AgentTools (HTTP only)
  - Constrained action space — agent cannot do anything
    not explicitly defined in AgentTools
  - Every decision is logged with reasoning
  - Deterministic guardrails — no hallucinated authority
  - Failures produce degraded but valid outputs,
    never silent errors
  - Audit log is append-only
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from src.agents.tools import AgentTools
from src.config.logging_config import get_logger

logger = get_logger(__name__)

# ── Audit log path ──────────────────────────────────────────────────────────
DEFAULT_AUDIT_LOG_PATH = Path("logs/agent_audit.jsonl")

# ── Replenishment thresholds ────────────────────────────────────────────────
# Agent can only recommend within these bounds — hard guardrails
MIN_REPLENISHMENT_UNITS = 0
MAX_REPLENISHMENT_UNITS = 50_000
REPLENISHMENT_BUFFER_MULTIPLIER = 1.1  # 10% safety buffer over forecast


@dataclass
class ReplenishmentPlan:
    """
    Output of the Replenishment Agent.

    Contains the recommended replenishment quantity,
    full reasoning, monitoring status, and narrative.

    The agent never places orders directly — it produces
    a plan for human review and downstream systems.
    """

    plan_id: str
    generated_at: str
    store_id: int
    forecast_horizon_days: int
    forecast_start_date: str

    # Forecast outputs
    total_forecast_units: float
    mean_daily_forecast: float
    recommended_replenishment_units: float

    # Monitoring status
    monitoring_available: bool
    should_retrain: bool
    retraining_action: str
    drift_detected: bool

    # Narrative
    narrative: str
    narrative_model: str

    # Agent reasoning
    reasoning: list[str]
    warnings: list[str]
    tool_calls: list[dict]

    # Status
    status: str  # "complete", "degraded", "failed"
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "generated_at": self.generated_at,
            "store_id": self.store_id,
            "forecast_horizon_days": self.forecast_horizon_days,
            "forecast_start_date": self.forecast_start_date,
            "forecast": {
                "total_units": round(self.total_forecast_units, 2),
                "mean_daily_units": round(self.mean_daily_forecast, 2),
                "recommended_replenishment_units": round(
                    self.recommended_replenishment_units, 2
                ),
            },
            "monitoring": {
                "available": self.monitoring_available,
                "should_retrain": self.should_retrain,
                "retraining_action": self.retraining_action,
                "drift_detected": self.drift_detected,
            },
            "narrative": self.narrative,
            "narrative_model": self.narrative_model,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
            "tool_calls": self.tool_calls,
            "status": self.status,
            "error": self.error,
        }


def _generate_plan_id() -> str:
    """Generate a unique plan ID based on timestamp."""
    return f"plan_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"


def _compute_replenishment(
    total_forecast: float,
    buffer_multiplier: float = REPLENISHMENT_BUFFER_MULTIPLIER,
) -> float:
    """
    Compute recommended replenishment units.

    Applies a safety buffer over total forecast.
    Clips to MIN/MAX guardrails — agent cannot recommend
    quantities outside these bounds.
    """
    raw = total_forecast * buffer_multiplier
    clipped = max(MIN_REPLENISHMENT_UNITS, min(raw, MAX_REPLENISHMENT_UNITS))

    if clipped != raw:
        logger.warning(
            f"Replenishment clipped: raw={raw:.0f} → clipped={clipped:.0f} "
            f"(guardrail: [{MIN_REPLENISHMENT_UNITS}, {MAX_REPLENISHMENT_UNITS}])"
        )

    return clipped


class ReplenishmentAgent:
    """
    Replenishment Agent — orchestrates forecast and monitoring tools.

    Constrained action space:
      - get_forecast: retrieve demand forecast
      - get_monitoring_trigger: check retraining signals
      - generate_narrative: produce stakeholder summary
      - run_drift_check: trigger drift computation

    The agent cannot:
      - Access model weights directly
      - Place orders (produces plan only)
      - Take actions outside AgentTools
      - Override guardrail thresholds
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH,
    ) -> None:
        self.base_url = base_url
        self.audit_log_path = audit_log_path
        logger.info(
            f"ReplenishmentAgent initialised: "
            f"base_url={base_url}, "
            f"audit_log={audit_log_path}"
        )

    def run(
        self,
        store_id: int,
        start_date: date,
        horizon_days: int = 7,
        promo: int = 0,
    ) -> ReplenishmentPlan:
        """
        Run the full replenishment workflow for a store.

        Workflow:
          1. Check serving layer health
          2. Get demand forecast
          3. Check monitoring trigger
          4. Generate narrative
          5. Compute replenishment plan
          6. Write audit log

        Args:
            store_id:      Store to plan replenishment for.
            start_date:    First date of the forecast horizon.
            horizon_days:  Number of days to forecast.
            promo:         Promotion flag for forecast period.

        Returns:
            ReplenishmentPlan — always returns, never raises.
            status="failed" if critical tools unavailable.
            status="degraded" if monitoring unavailable.
            status="complete" if all tools succeeded.
        """
        plan_id = _generate_plan_id()
        generated_at = datetime.now(UTC).isoformat()
        reasoning: list[str] = []
        warnings: list[str] = []
        tool_calls: list[dict] = []

        logger.info(
            f"Agent run started: plan_id={plan_id}, "
            f"store={store_id}, horizon={horizon_days}"
        )

        with AgentTools(base_url=self.base_url) as tools:
            # ── Step 1: Health check ────────────────────────────────────────
            health_result = tools.check_health()
            tool_calls.append(health_result.to_dict())

            if not health_result.success:
                msg = "Serving layer unavailable — cannot proceed"
                logger.error(msg)
                plan = self._failed_plan(
                    plan_id,
                    generated_at,
                    store_id,
                    horizon_days,
                    str(start_date),
                    msg,
                )
                self._write_audit_log(plan)
                return plan

            reasoning.append("Serving layer healthy — proceeding with workflow.")

            # ── Step 2: Get forecast ────────────────────────────────────────
            forecast_result = tools.get_forecast(
                store_id=store_id,
                start_date=start_date,
                horizon_days=horizon_days,
                promo=promo,
            )
            tool_calls.append(forecast_result.to_dict())

            if not forecast_result.success:
                msg = f"Forecast tool failed: {forecast_result.error}"
                logger.error(msg)
                plan = self._failed_plan(
                    plan_id,
                    generated_at,
                    store_id,
                    horizon_days,
                    str(start_date),
                    msg,
                )
                self._write_audit_log(plan)
                return plan

            # Extract forecast values
            forecasts = forecast_result.outputs.get("forecasts", [])
            if not forecasts:
                msg = "Forecast returned empty results"
                plan = self._failed_plan(
                    plan_id,
                    generated_at,
                    store_id,
                    horizon_days,
                    str(start_date),
                    msg,
                )
                self._write_audit_log(plan)
                return plan

            point_forecasts = [f["point_forecast"] for f in forecasts]
            total_forecast = sum(point_forecasts)
            mean_daily = total_forecast / len(point_forecasts)

            reasoning.append(
                f"Forecast retrieved: {len(forecasts)} days, "
                f"total={total_forecast:.0f} units, "
                f"mean_daily={mean_daily:.0f} units."
            )

            # ── Step 3: Check monitoring trigger ───────────────────────────
            trigger_result = tools.get_monitoring_trigger()
            tool_calls.append(trigger_result.to_dict())

            should_retrain = False
            retraining_action = "Monitoring data unavailable."
            drift_detected = False
            monitoring_available = False

            if trigger_result.success:
                monitoring_available = True
                outputs = trigger_result.outputs

                if outputs.get("status") == "no_data":
                    reasoning.append(
                        "Monitoring trigger has no data yet — "
                        "forecast requests needed to populate."
                    )
                else:
                    should_retrain = outputs.get("should_retrain", False)
                    retraining_action = outputs.get(
                        "recommended_action",
                        "No action required.",
                    )
                    drift_summary = outputs.get("drift_summary", {})
                    drift_detected = drift_summary.get("drift_detected", False)

                    reasoning.append(
                        f"Monitoring trigger: should_retrain={should_retrain}, "
                        f"drift_detected={drift_detected}."
                    )

                    if should_retrain:
                        warnings.append(f"Retraining recommended: {retraining_action}")
            else:
                warnings.append(
                    "Monitoring trigger unavailable — " "proceeding with forecast only."
                )
                reasoning.append(
                    "Monitoring unavailable — plan status will be 'degraded'."
                )

            # ── Step 4: Compute replenishment ───────────────────────────────
            replenishment_units = _compute_replenishment(total_forecast)

            reasoning.append(
                f"Replenishment computed: "
                f"{total_forecast:.0f} units × "
                f"{REPLENISHMENT_BUFFER_MULTIPLIER} buffer = "
                f"{replenishment_units:.0f} units "
                f"(guardrails: [{MIN_REPLENISHMENT_UNITS}, "
                f"{MAX_REPLENISHMENT_UNITS}])."
            )

            # ── Step 5: Generate narrative ──────────────────────────────────
            narrative_metadata = {
                "store_id": store_id,
                "horizon_days": horizon_days,
                "mean_forecast": round(mean_daily, 2),
                "total_forecast": round(total_forecast, 2),
                "recommended_replenishment_units": round(replenishment_units, 2),
                "drift_detected": drift_detected,
                "should_retrain": should_retrain,
                "recommended_action": retraining_action,
            }

            narrative_result = tools.generate_narrative(
                metadata=narrative_metadata,
                template="combined",
            )
            tool_calls.append(narrative_result.to_dict())

            if narrative_result.success:
                narrative = narrative_result.outputs.get(
                    "narrative",
                    "Narrative unavailable.",
                )
                narrative_model = narrative_result.outputs.get("model", "unknown")
                reasoning.append("Stakeholder narrative generated successfully.")
            else:
                narrative = (
                    f"Store {store_id} forecast: {mean_daily:.0f} units/day "
                    f"over {horizon_days} days. "
                    f"Recommended replenishment: {replenishment_units:.0f} units."
                )
                narrative_model = "fallback"
                warnings.append("Narrative generation failed — using fallback.")

            # ── Step 6: Determine status ────────────────────────────────────
            if not monitoring_available:
                status = "degraded"
            else:
                status = "complete"

            reasoning.append(f"Plan complete with status='{status}'.")

            # ── Build final plan ────────────────────────────────────────────
            plan = ReplenishmentPlan(
                plan_id=plan_id,
                generated_at=generated_at,
                store_id=store_id,
                forecast_horizon_days=horizon_days,
                forecast_start_date=str(start_date),
                total_forecast_units=total_forecast,
                mean_daily_forecast=mean_daily,
                recommended_replenishment_units=replenishment_units,
                monitoring_available=monitoring_available,
                should_retrain=should_retrain,
                retraining_action=retraining_action,
                drift_detected=drift_detected,
                narrative=narrative,
                narrative_model=narrative_model,
                reasoning=reasoning,
                warnings=warnings,
                tool_calls=tool_calls,
                status=status,
                error=None,
            )

        self._write_audit_log(plan)
        logger.info(
            f"Agent run complete: plan_id={plan_id}, "
            f"status={status}, "
            f"replenishment={replenishment_units:.0f} units"
        )

        return plan

    def _failed_plan(
        self,
        plan_id: str,
        generated_at: str,
        store_id: int,
        horizon_days: int,
        start_date: str,
        error: str,
    ) -> ReplenishmentPlan:
        """Produce a failed plan with error information."""
        return ReplenishmentPlan(
            plan_id=plan_id,
            generated_at=generated_at,
            store_id=store_id,
            forecast_horizon_days=horizon_days,
            forecast_start_date=start_date,
            total_forecast_units=0.0,
            mean_daily_forecast=0.0,
            recommended_replenishment_units=0.0,
            monitoring_available=False,
            should_retrain=False,
            retraining_action="Unknown — agent failed.",
            drift_detected=False,
            narrative="Plan generation failed. See error field.",
            narrative_model="none",
            reasoning=[f"Agent failed: {error}"],
            warnings=[],
            tool_calls=[],
            status="failed",
            error=error,
        )

    def _write_audit_log(self, plan: ReplenishmentPlan) -> None:
        """
        Append plan to the audit log.

        Append-only — historical plans are never overwritten.
        This is the authoritative record of all agent decisions.
        """
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(plan.to_dict()) + "\n")

        logger.info(
            f"Audit log updated: plan_id={plan.plan_id}, " f"path={self.audit_log_path}"
        )
