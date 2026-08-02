"""
recommendations.py

Generates actionable business recommendations grounded in the dataset
context, categorized as HIGH PRIORITY / OPPORTUNITY / RISK / COST / GROWTH.
"""

from __future__ import annotations

from typing import Optional

from modules.ai_service import AIService, AIResponse, get_ai_service


def generate_recommendations(context: str, ai_service: Optional[AIService] = None) -> AIResponse:
    service = ai_service or get_ai_service()

    prompt = f"""Here is the structured summary of a business dataset:

{context}

Task: Act as an experienced business consultant. Based ONLY on the facts above,
produce a short set of actionable recommendations (aim for 3-6 total), each
labeled with exactly one category tag from this list: HIGH PRIORITY, OPPORTUNITY,
RISK, COST, GROWTH.

Format each recommendation like this:
[CATEGORY] Short recommendation title
Recommended action: one or two sentences describing what to do next.

Rules:
- Only reference facts that appear in the context above.
- Do not claim causal certainty when the data only shows a correlation or a
  single-period change — phrase those as "may be worth investigating" rather
  than a confirmed cause.
- If the context does not support a confident recommendation in a category,
  skip that category rather than inventing one."""

    return service.generate(prompt)
