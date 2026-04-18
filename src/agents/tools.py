# src/agents/tools.py
"""
HTTP tool wrappers for the Replenishment Agent.

Design rules (non-negotiable):
  - Agent never imports model weights directly
  - Agent calls src/serving/ exclusively via HTTP
  - Every tool call is logged with inputs and outputs
  - Tool failures return structured error responses — never raise
  - Tools are stateless — no shared state between calls

Available tools:
  - get_forecast: Call /forecast endpoint
  - get_monitoring_trigger: Call /monitoring/trigger endpoint
  - get_forecast_health: Call /monitoring/forecast endpoint
  - generate_narrative: Call /genai/narrative endpoint
  - run_drift_check: Call /monitoring/run-drift-check endpoint
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

from src.config.logging_config import get_logger

logger = get_logger(__name__)

# ── Tool configuration ──────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8000"
TOOL_TIMEOUT_SECONDS = 15


@dataclass
class ToolResult:
    """
    Structured result from any tool call.

    success:   True if the tool call succeeded.
    tool_name: Name of the tool called — for audit log.
    inputs:    Inputs passed to the tool — for audit log.
    outputs:   Response data from the tool.
    error:     None if successful, error message if failed.
    """

    success: bool
    tool_name: str
    inputs: dict
    outputs: dict
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
        }


class AgentTools:
    """
    HTTP tool wrappers for the Replenishment Agent.

    All tools call the serving layer via HTTP.
    No direct model imports — ever.

    Args:
        base_url: Base URL of the serving layer.
                  Defaults to localhost:8000 for local development.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=TOOL_TIMEOUT_SECONDS)
        logger.info(f"AgentTools initialised with base_url={self.base_url}")

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> AgentTools:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _post(self, endpoint: str, payload: dict) -> ToolResult:
        """
        Make a POST request to the serving layer.

        Returns ToolResult — never raises on failure.
        """
        tool_name = endpoint.lstrip("/").replace("/", "_")
        url = f"{self.base_url}{endpoint}"

        logger.info(f"Tool call: POST {endpoint} — inputs={list(payload.keys())}")

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            logger.info(f"Tool success: POST {endpoint}")
            return ToolResult(
                success=True,
                tool_name=tool_name,
                inputs=payload,
                outputs=data,
                error=None,
            )

        except httpx.TimeoutException:
            msg = f"Timeout calling {endpoint} after {TOOL_TIMEOUT_SECONDS}s"
            logger.error(msg)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                inputs=payload,
                outputs={},
                error=msg,
            )

        except httpx.HTTPStatusError as e:
            msg = f"HTTP {e.response.status_code} from {endpoint}"
            logger.error(msg)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                inputs=payload,
                outputs={},
                error=msg,
            )

        except Exception as e:
            msg = f"Tool call failed: {str(e)}"
            logger.error(msg)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                inputs=payload,
                outputs={},
                error=msg,
            )

    def _get(self, endpoint: str) -> ToolResult:
        """
        Make a GET request to the serving layer.

        Returns ToolResult — never raises on failure.
        """
        tool_name = endpoint.lstrip("/").replace("/", "_")
        url = f"{self.base_url}{endpoint}"

        logger.info(f"Tool call: GET {endpoint}")

        try:
            response = self._client.get(url)
            response.raise_for_status()
            data = response.json()

            logger.info(f"Tool success: GET {endpoint}")
            return ToolResult(
                success=True,
                tool_name=tool_name,
                inputs={},
                outputs=data,
                error=None,
            )

        except httpx.TimeoutException:
            msg = f"Timeout calling {endpoint} after {TOOL_TIMEOUT_SECONDS}s"
            logger.error(msg)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                inputs={},
                outputs={},
                error=msg,
            )

        except httpx.HTTPStatusError as e:
            msg = f"HTTP {e.response.status_code} from {endpoint}"
            logger.error(msg)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                inputs={},
                outputs={},
                error=msg,
            )

        except Exception as e:
            msg = f"Tool call failed: {str(e)}"
            logger.error(msg)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                inputs={},
                outputs={},
                error=msg,
            )

    # ── Public tool interface ───────────────────────────────────────────────

    def get_forecast(
        self,
        store_id: int,
        start_date: date,
        horizon_days: int,
        promo: int = 0,
    ) -> ToolResult:
        """
        Get demand forecast for a store.

        Calls POST /forecast on the serving layer.
        Agent never accesses model weights directly.
        """
        return self._post(
            "/forecast",
            {
                "store_id": store_id,
                "start_date": str(start_date),
                "horizon_days": horizon_days,
                "promo": promo,
            },
        )

    def get_monitoring_trigger(self) -> ToolResult:
        """
        Get the latest retraining trigger decision.

        Calls GET /monitoring/trigger on the serving layer.
        """
        return self._get("/monitoring/trigger")

    def get_forecast_health(self) -> ToolResult:
        """
        Get the latest forecast health report.

        Calls GET /monitoring/forecast on the serving layer.
        """
        return self._get("/monitoring/forecast")

    def generate_narrative(
        self,
        metadata: dict,
        template: str = "combined",
    ) -> ToolResult:
        """
        Generate a plain-language narrative from metadata.

        Calls POST /genai/narrative on the serving layer.
        """
        payload = {"template": template, **metadata}
        return self._post("/genai/narrative", payload)

    def run_drift_check(self) -> ToolResult:
        """
        Trigger a drift computation.

        Calls POST /monitoring/run-drift-check on the serving layer.
        """
        return self._post("/monitoring/run-drift-check", {})

    def check_health(self) -> ToolResult:
        """
        Check if the serving layer is healthy and model is loaded.

        Calls GET /ready on the serving layer.
        """
        return self._get("/ready")
