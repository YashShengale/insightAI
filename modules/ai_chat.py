"""
ai_chat.py

Implements the "Ask Your Data" natural-language Q&A feature. Answers are
generated strictly from the structured dataset context (see ai_insights.py)
plus recent chat history for follow-up questions.
"""

from __future__ import annotations

from typing import List, Tuple

from modules.ai_service import AIService, AIResponse, get_ai_service

MAX_HISTORY_TURNS = 5


def format_chat_history(history: List[Tuple[str, str]]) -> str:
    """Format the last few (question, answer) turns for context continuity."""
    if not history:
        return "(no previous questions in this session)"

    recent = history[-MAX_HISTORY_TURNS:]
    lines = []
    for question, answer in recent:
        lines.append(f"User asked: {question}")
        lines.append(f"You answered: {answer}")
    return "\n".join(lines)


def answer_question(
    question: str,
    dataset_context: str,
    history: List[Tuple[str, str]],
    ai_service: AIService = None,
) -> AIResponse:
    """Answer a natural-language question about the dataset."""
    service = ai_service or get_ai_service()

    history_text = format_chat_history(history)

    prompt = f"""Dataset context (the only source of truth you may use):

{dataset_context}

Previous conversation in this session (for continuity only, do not treat as new facts):
{history_text}

The user's new question:
"{question}"

Answer the question using ONLY the dataset context above. If the dataset context
does not contain the information needed to answer, respond exactly with:
"The uploaded dataset does not contain enough information to answer that."
Keep the answer concise and business-appropriate (a few sentences, or a short
list if comparing multiple items). Do not invent numbers."""

    return service.generate(prompt)
