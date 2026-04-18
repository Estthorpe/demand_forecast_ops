"""
Versioned prompt templates for the GenAI narrative generator.

Design rules (non-negotiable):
  - Prompts are versioned — changes are tracked explicitly
  - All prompts are grounded — the model is instructed to use
    only the data provided in the input, never invent figures
  - Safety constraints are embedded in every prompt
  - Prompt versions are logged with every generated narrative

Current version: v1
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Prompt version registry ─────────────────────────────────────────────────
CURRENT_PROMPT_VERSION = "v1"


@dataclass
class PromptTemplate:
    """
    A versioned prompt template.

    version:      Explicit version string — logged with every output.
    system:       System prompt — sets model role and safety constraints.
    user_prefix:  Prefix prepended to every user message.
    """

    version: str
    system: str
    user_prefix: str


# ── Safety rules embedded in every prompt ──────────────────────────────────
_SAFETY_RULES = """
STRICT RULES — NEVER VIOLATE:
1. Only use numbers and facts explicitly provided in the input data.
2. Never invent, estimate, or extrapolate figures not present in the input.
3. If a metric is missing from the input, say "not available" — do not guess.
4. Do not make investment, purchasing, or business recommendations beyond
   what the data directly supports.
5. Never claim certainty about future events — forecasts are probabilistic.
6. Keep the narrative factual, concise, and free of filler phrases.
"""

# ── Prompt v1 — Forecast narrative ─────────────────────────────────────────
FORECAST_NARRATIVE_V1 = PromptTemplate(
    version="v1",
    system=f"""You are a supply chain analytics assistant for a retail
forecasting system. Your role is to translate structured forecast
data into clear, concise narrative summaries for operations teams.

{_SAFETY_RULES}

OUTPUT FORMAT:
- Write in plain English, 3-5 sentences maximum
- Lead with the key forecast figure
- Include confidence interval context
- Mention monitoring status if relevant
- End with one operational implication if clearly supported by the data
- Never use bullet points — prose only
""",
    user_prefix="Generate a narrative summary for the following forecast data:\n\n",
)


# ── Prompt v1 — Monitoring status narrative ────────────────────────────────
MONITORING_NARRATIVE_V1 = PromptTemplate(
    version="v1",
    system=f"""You are a machine learning operations analyst for a retail
demand forecasting system. Your role is to translate monitoring
reports into clear operational summaries for engineering teams.

{_SAFETY_RULES}

OUTPUT FORMAT:
- Write in plain English, 3-5 sentences maximum
- Lead with the overall system health status
- State drift status using the PSI values provided
- State retraining recommendation if present
- Never speculate about root causes not evidenced in the data
- Never use bullet points — prose only
""",
    user_prefix="Generate a monitoring status narrative for the following report:\n\n",
)


# ── Prompt v1 — Combined forecast + monitoring narrative ───────────────────
COMBINED_NARRATIVE_V1 = PromptTemplate(
    version="v1",
    system=f"""You are a supply chain analytics assistant for a retail
demand forecasting system. Your role is to produce integrated
summaries combining forecast outputs and system health status
for operations and engineering stakeholders.

{_SAFETY_RULES}

OUTPUT FORMAT:
- Write in plain English, 4-6 sentences maximum
- Paragraph 1: Forecast outlook (use only provided figures)
- Paragraph 2: System health and any monitoring alerts
- Never blend forecast uncertainty with monitoring uncertainty
- Never use bullet points — prose only
""",
    user_prefix=(
        "Generate an integrated forecast and monitoring narrative "
        "for the following data:\n\n"
    ),
)


# ── Template registry ───────────────────────────────────────────────────────
PROMPT_REGISTRY: dict[str, PromptTemplate] = {
    "forecast": FORECAST_NARRATIVE_V1,
    "monitoring": MONITORING_NARRATIVE_V1,
    "combined": COMBINED_NARRATIVE_V1,
}


def get_prompt(template_name: str) -> PromptTemplate:
    """
    Retrieve a prompt template by name.

    Args:
        template_name: One of "forecast", "monitoring", "combined"

    Returns:
        PromptTemplate for the requested narrative type.

    Raises:
        ValueError if template name is not registered.
    """
    if template_name not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt template: '{template_name}'. "
            f"Available: {list(PROMPT_REGISTRY.keys())}"
        )
    return PROMPT_REGISTRY[template_name]
