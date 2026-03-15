"""
query_classifier.py — Lightweight query type classifier for adaptive RAG routing.

Classifies a user query into one of four semantic types and maps each type to
the most appropriate RAG chain.  Based on the adaptive routing pattern described
in NirDiamant's RAG Techniques repository (2024).

Query types and their routing rationale
---------------------------------------
Factual      → stuff chain
    Simple look-up questions with a single, direct answer.
    Example: "What year was the company founded?"

Numerical    → stuff chain
    Questions requiring extraction of numbers or statistics.
    Example: "What was the net revenue in Q3 2023?"

Analytical   → reciprocal chain
    Questions requiring synthesis across multiple evidence points.
    Example: "How did the company's strategy evolve over the past three years?"

Comparative  → reciprocal chain
    Questions contrasting two or more entities / time periods.
    Example: "How do the margins in 2022 compare to 2021?"

The reciprocal chain generates 4 targeted sub-questions, retrieves evidence for
each, and synthesises a richer answer — more expensive but significantly better
for multi-faceted queries.

Primary interface:
    label = classify_query(query)         # returns QueryType string
    chain = route_query(query)            # returns "stuff" | "reciprocal"
"""

import logging
import os
import re
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

QueryType = Literal["Factual", "Analytical", "Numerical", "Comparative"]

# Maps query type → RAG chain name
_ROUTING: dict[str, str] = {
    "Factual":     "stuff",
    "Numerical":   "stuff",
    "Analytical":  "reciprocal",
    "Comparative": "reciprocal",
}

_SYSTEM_PROMPT = (
    "You are a query classification assistant. "
    "Classify the user's question into EXACTLY ONE of the following types:\n\n"
    "  Factual     — a single specific fact answerable from one passage\n"
    "                e.g. 'Who is the CEO?' / 'When was the company founded?'\n"
    "  Numerical   — a specific number or figure answerable from one table or paragraph\n"
    "                e.g. 'What was net revenue in 2023?' / 'How many employees?'\n"
    "  Analytical  — requires combining evidence from MULTIPLE SEPARATE sections\n"
    "                e.g. 'Why did margins decline?' / 'How did strategy evolve over time?'\n"
    "  Comparative — explicitly contrasts two or more entities or time periods\n"
    "                e.g. 'How do 2022 margins compare to 2021?'\n\n"
    "KEY RULE: If the answer can be found in a single table or paragraph, "
    "choose Factual or Numerical. "
    "Only choose Analytical if the question genuinely requires reasoning across "
    "multiple separate sections of the document.\n\n"
    "Respond with ONLY the label, nothing else."
)

# Lazy singleton to avoid import-time model loading
_classifier_llm = None


def _get_classifier_llm():
    """Return a lightweight LLMClient (gpt-4o-mini) for classification."""
    global _classifier_llm
    if _classifier_llm is None:
        from src.generation.llm_client import LLMClient
        _classifier_llm = LLMClient(
            model=os.getenv("CLASSIFIER_MODEL", "gpt-4o-mini"),
        )
    return _classifier_llm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_query(query: str) -> QueryType:
    """Classify a query into a semantic type using GPT-4o-mini.

    Falls back to heuristic classification if the LLM call fails (avoids
    breaking the pipeline due to a classification error).

    Args:
        query: User's natural-language question.

    Returns:
        One of ``"Factual"``, ``"Analytical"``, ``"Numerical"``, ``"Comparative"``.
    """
    if not query.strip():
        return "Factual"

    try:
        llm = _get_classifier_llm()
        raw = llm.generate(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Query: {query}",
        ).strip()

        for label in ("Factual", "Analytical", "Numerical", "Comparative"):
            if label.lower() in raw.lower():
                logger.info("Classified query as %r: %r", label, query[:60])
                return label  # type: ignore[return-value]

        # Unexpected output — fall through to heuristic
        logger.warning("Unexpected classifier output %r; using heuristic.", raw)

    except Exception as exc:
        logger.warning("Classifier LLM call failed (%s); using heuristic.", exc)

    return _heuristic_classify(query)


def route_query(query: str) -> str:
    """Return the RAG chain name appropriate for the given query.

    Args:
        query: User's natural-language question.

    Returns:
        ``"stuff"`` or ``"reciprocal"``.
    """
    label = classify_query(query)
    chain = _ROUTING[label]
    logger.info("Routing query to %r chain (type=%r).", chain, label)
    return chain


# ---------------------------------------------------------------------------
# Heuristic fallback (no LLM call)
# ---------------------------------------------------------------------------

_COMPARE_WORDS = re.compile(
    r"\b(compar|contrast|differ|versus|vs\.?|than|relative to|"
    r"better|worse|higher|lower|more|less|between)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = re.compile(
    r"\b(how much|how many|revenue|profit|loss|margin|percent|"
    r"number of|count|total|sum|average|growth|rate|ratio)\b",
    re.IGNORECASE,
)
_ANALYTICAL_WORDS = re.compile(
    r"\b(why|explain|analyse|analyze|impact|effect|implication|"
    r"strategy|trend|evolve|evolution|reason|cause|consequence)\b",
    re.IGNORECASE,
)


def _heuristic_classify(query: str) -> QueryType:
    """Simple keyword-based fallback classifier."""
    q = query.lower()
    if _COMPARE_WORDS.search(q):
        return "Comparative"
    if _ANALYTICAL_WORDS.search(q):
        return "Analytical"
    if _NUMBER_WORDS.search(q):
        return "Numerical"
    return "Factual"
