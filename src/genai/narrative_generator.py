# src/genai/narrative_generator.py
"""
Narrative generator for demand-forecast-ops.

Receives structured metadata dicts from the serving and monitoring
layers. Returns plain-language narrative strings for operations teams.

Design rules (non-negotiable):
  - Never invents data points not present in the input dict
  - Prompt templates are versioned — version logged with every output
  - LLM call failures never crash the serving layer
  - GenAI augments — never replaces — structured prediction logic
  - Grounding is enforced at the prompt level, not post-hoc
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from src.config.logging_config import get_logger
from src.genai.prompt_templates import (
    CURRENT_PROMPT_VERSION,
    PromptTemplate,
    get_prompt,
)

logger = get_logger(__name__)

# ── Model configuration ─────────────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 300
TIMEOUT_SECONDS = 10


@dataclass
class NarrativeResult:
    """
    Output of the narrative generator.

    narrative:      The generated plain-language text.
    template_used:  Prompt template name — for audit trail.
    prompt_version: Version of the prompt template used.
    generated_at:   UTC timestamp of generation.
    model:          Model used for generation.
    grounded:       Always True — enforced by prompt design.
    error:          None if successful, error message if failed.
    """

    narrative: str
    template_used: str
    prompt_version: str
    generated_at: str
    model: str
    grounded: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "narrative": self.narrative,
            "template_used": self.template_used,
            "prompt_version": self.prompt_version,
            "generated_at": self.generated_at,
            "model": self.model,
            "grounded": self.grounded,
            "error": self.error,
        }


def _format_metadata_for_prompt(metadata: dict) -> str:
    """
    Serialise the input metadata dict to a clean JSON string.

    The model receives only what is in the metadata dict —
    this is the grounding boundary.
    """
    return json.dumps(metadata, indent=2, default=str)


def _build_user_message(template: PromptTemplate, metadata: dict) -> str:
    """Build the full user message from template prefix + formatted data."""
    return template.user_prefix + _format_metadata_for_prompt(metadata)


def _call_anthropic(
    system_prompt: str,
    user_message: str,
    api_key: str,
) -> str:
    """
    Call the Anthropic Messages API synchronously.

    Args:
        system_prompt: System prompt from the template.
        user_message:  User message containing the structured data.
        api_key:       Anthropic API key from environment.

    Returns:
        Generated text string.

    Raises:
        httpx.HTTPError on network or API failure.
        ValueError on unexpected response structure.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    response = httpx.post(
        ANTHROPIC_API_URL,
        headers=headers,
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )

    response.raise_for_status()
    data = response.json()

    content = data.get("content", [])
    if not content or content[0].get("type") != "text":
        raise ValueError(f"Unexpected response structure: {data}")

    return str(content[0]["text"])


def _fallback_narrative(metadata: dict) -> str:
    """
    Generate a safe, data-grounded fallback narrative without LLM.

    Used when the API is unavailable or misconfigured.
    Only uses values present in the metadata dict.
    """
    parts = []

    store_id = metadata.get("store_id")
    horizon = metadata.get("horizon_days")
    mean_forecast = metadata.get("mean_forecast")

    if store_id and horizon and mean_forecast:
        parts.append(
            f"Store {store_id} forecast over {horizon} days: "
            f"mean daily demand of {mean_forecast:,.0f} units."
        )

    drift_detected = metadata.get("drift_detected")
    should_retrain = metadata.get("should_retrain")

    if drift_detected is not None:
        status = (
            "Drift detected in input features."
            if drift_detected
            else "No significant input drift detected."
        )
        parts.append(status)

    if should_retrain is not None:
        action = (
            "Retraining is recommended."
            if should_retrain
            else "No retraining action required at this time."
        )
        parts.append(action)

    if not parts:
        return (
            "Forecast and monitoring data available. "
            "Narrative generation unavailable."
        )

    return " ".join(parts)


def generate_narrative(
    metadata: dict,
    template_name: str = "combined",
    api_key: str | None = None,
) -> NarrativeResult:
    """
    Generate a plain-language narrative from structured metadata.

    The metadata dict is the grounding boundary — the model is
    instructed to use only what is in this dict. Nothing is
    invented or inferred beyond the provided data.

    Args:
        metadata:      Structured data dict. Can contain forecast
                       results, monitoring reports, or both.
        template_name: Prompt template to use. One of:
                       "forecast", "monitoring", "combined"
        api_key:       Anthropic API key. Defaults to
                       ANTHROPIC_API_KEY environment variable.

    Returns:
        NarrativeResult with generated text and audit metadata.
        On failure, returns NarrativeResult with error field set
        and a safe fallback narrative — never raises.
    """
    generated_at = datetime.now(UTC).isoformat()

    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not resolved_key:
        logger.warning("ANTHROPIC_API_KEY not set — returning fallback narrative")
        return NarrativeResult(
            narrative=_fallback_narrative(metadata),
            template_used=template_name,
            prompt_version=CURRENT_PROMPT_VERSION,
            generated_at=generated_at,
            model="fallback",
            grounded=True,
            error="ANTHROPIC_API_KEY not configured",
        )

    try:
        template = get_prompt(template_name)
    except ValueError as e:
        logger.error(f"Invalid template name: {e}")
        return NarrativeResult(
            narrative=_fallback_narrative(metadata),
            template_used=template_name,
            prompt_version=CURRENT_PROMPT_VERSION,
            generated_at=generated_at,
            model="fallback",
            grounded=True,
            error=str(e),
        )

    user_message = _build_user_message(template, metadata)

    logger.info(
        f"Generating narrative: template={template_name}, "
        f"version={template.version}, "
        f"metadata_keys={list(metadata.keys())}"
    )

    try:
        narrative = _call_anthropic(
            system_prompt=template.system,
            user_message=user_message,
            api_key=resolved_key,
        )

        logger.info(
            f"Narrative generated: {len(narrative)} chars, " f"template={template_name}"
        )

        return NarrativeResult(
            narrative=narrative,
            template_used=template_name,
            prompt_version=template.version,
            generated_at=generated_at,
            model=ANTHROPIC_MODEL,
            grounded=True,
            error=None,
        )

    except httpx.TimeoutException:
        msg = f"Anthropic API timeout after {TIMEOUT_SECONDS}s"
        logger.error(msg)
        return NarrativeResult(
            narrative=_fallback_narrative(metadata),
            template_used=template_name,
            prompt_version=template.version,
            generated_at=generated_at,
            model="fallback",
            grounded=True,
            error=msg,
        )

    except httpx.HTTPStatusError as e:
        msg = f"Anthropic API HTTP error: {e.response.status_code}"
        logger.error(msg)
        return NarrativeResult(
            narrative=_fallback_narrative(metadata),
            template_used=template_name,
            prompt_version=template.version,
            generated_at=generated_at,
            model="fallback",
            grounded=True,
            error=msg,
        )

    except Exception as e:
        msg = f"Narrative generation failed: {str(e)}"
        logger.error(msg)
        return NarrativeResult(
            narrative=_fallback_narrative(metadata),
            template_used=template_name,
            prompt_version=template.version,
            generated_at=generated_at,
            model="fallback",
            grounded=True,
            error=msg,
        )
