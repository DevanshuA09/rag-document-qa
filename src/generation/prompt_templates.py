"""
prompt_templates.py — Prompt construction for Stuff and Reciprocal RAG modes.

All prompts follow these principles:
  - System prompt locks the model to context-only answers with page citations.
  - Abstention instruction prevents hallucination on unanswerable questions.
  - Context is formatted with explicit [CHUNK N — Page X] headers so the
    model can reliably produce [Page X] citations in its output.

Public API:
    build_stuff_prompt(query, chunks)            → (system, user)
    build_reciprocal_system_prompt()             → system str
    build_reciprocal_subquestion_prompt(query)   → user str
    build_reciprocal_final_prompt(query, chunks) → (system, user)
    format_chunk_context(chunks)                 → str
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared system prompt (identical for Stuff and Reciprocal final answer)
# ---------------------------------------------------------------------------

_GROUNDED_SYSTEM_PROMPT = """You are a precise document analysis assistant.

Answer questions ONLY using the context passages provided. Do NOT use any outside knowledge, training data, or assumptions beyond what is explicitly stated in the context.

CITATION RULES:
- After every factual claim write [Page X] where X is the page number from the context.
- If multiple pages support a claim, cite each: [Page 3, Page 7].
- For numbers from tables: name the row AND column you are reading (e.g. "Operating expenses row, fiscal 2023 column"). Copy the figure verbatim — do not round or paraphrase.

TABLE RULES:
- Context passages labelled [TABLE] contain structured data. Read each column header carefully to identify the time period or category before citing a number.
- If a table has multiple year columns, confirm which column corresponds to the year in the question before extracting a figure.

ABSTENTION RULE:
- Only respond with "I cannot find sufficient information in the document to answer this question." if the context contains NO relevant information whatsoever.
- If the context is partially relevant, provide a partial answer and note what is missing.
- Do NOT use the abstention phrase simply because the answer requires synthesising across multiple passages.

FORMATTING:
- Be direct and concise. No bullet points unless the question explicitly asks for a list."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_stuff_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Build the (system, user) prompt pair for Stuff RAG mode.

    All retrieved chunks are concatenated into a single context block and
    sent in one LLM call.

    Args:
        query: The user's question.
        chunks: Retrieved and reranked chunk dicts.  Each must have
            ``text`` and ``page_number``.

    Returns:
        ``(system_prompt, user_prompt)`` tuple ready for :class:`LLMClient`.
    """
    context = format_chunk_context(chunks)
    user_prompt = (
        f"Context from document:\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite page numbers for every claim):"
    )
    return _GROUNDED_SYSTEM_PROMPT, user_prompt


def build_reciprocal_system_prompt() -> str:
    """Return the system prompt for sub-question generation.

    This is a *different* system prompt — lighter, instructional — used only
    for the sub-question generation step, not for the final answer.

    Returns:
        System prompt string.
    """
    return (
        "You are a precise query decomposition assistant. "
        "Your only job is to rewrite a user question into specific sub-questions. "
        "Output ONLY the numbered list — no explanations, no preamble."
    )


def build_reciprocal_subquestion_prompt(query: str) -> str:
    """Build the user prompt that asks the LLM to generate 4 sub-questions.

    Args:
        query: The original user question.

    Returns:
        User prompt string (system prompt is :func:`build_reciprocal_system_prompt`).
    """
    return (
        f"Original question: {query}\n\n"
        "Generate exactly 4 alternative sub-questions that are more specific and "
        "less ambiguous. These sub-questions should together cover the full intent "
        "of the original question.\n\n"
        "Output ONLY the 4 questions as a numbered list, nothing else. Example format:\n"
        "1. <sub-question>\n"
        "2. <sub-question>\n"
        "3. <sub-question>\n"
        "4. <sub-question>"
    )


def build_reciprocal_final_prompt(
    query: str, chunks: list[dict]
) -> tuple[str, str]:
    """Build the (system, user) prompt for the Reciprocal RAG final answer.

    The context was gathered from multiple sub-question retrievals, so the
    user prompt notes that broader coverage was used.

    Args:
        query: The *original* user question (not the sub-questions).
        chunks: Deduplicated, score-filtered chunks from all sub-retrievals.

    Returns:
        ``(system_prompt, user_prompt)`` tuple.
    """
    context = format_chunk_context(chunks)
    user_prompt = (
        "The following context was retrieved using multiple search queries to "
        "ensure comprehensive coverage:\n\n"
        f"{context}\n\n"
        f"Original question: {query}\n\n"
        "Answer (cite page numbers for every claim):"
    )
    return _GROUNDED_SYSTEM_PROMPT, user_prompt


def format_chunk_context(chunks: list[dict]) -> str:
    """Render a list of chunks as a labelled context block for the prompt.

    Each chunk gets a header ``[CHUNK N — Page X]`` so the model can
    reliably produce ``[Page X]`` citations.

    Args:
        chunks: List of chunk dicts with ``text`` and ``page_number``.

    Returns:
        Multi-line string ready to embed in a prompt.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        if chunk.get("chunk_type") == "table":
            header = f"[TABLE {i} — Page {chunk['page_number']}]"
        else:
            header = f"[CHUNK {i} — Page {chunk['page_number']}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)
