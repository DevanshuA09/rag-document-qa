"""
chunker.py — Split page-level text into overlapping chunks using LangChain.

Implements **Contextual Headers** (NirDiamant RAG Techniques, 2024):
each chunk's text is prefixed with a structured header that encodes the
document name, page number, and detected section heading.  This gives the
dense embedder and the cross-encoder richer context about *where* the text
comes from, improving retrieval accuracy by ~27.9 % on open-QA benchmarks.

Primary interface:
    chunk_pages(pages: list[dict]) -> list[dict]

Secondary interface:
    get_chunk_stats(chunks: list[dict]) -> dict
"""

import logging
import os
import re
import uuid

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

# Minimum characters a chunk must contain to be kept (noise filter).
_MIN_CHUNK_LENGTH = 50

# Regex that matches a likely section heading: short line (≤ 80 chars),
# possibly starting with a numbering scheme like "1.", "2.3", "A.", etc.
_HEADING_RE = re.compile(
    r"^(?:[\d]+(?:\.\d+)*\.?\s+)?[A-Z][^a-z\n]{0,60}$",
    re.MULTILINE,
)


def _build_table_summary(markdown: str) -> str:
    """Build a searchable text summary of a Markdown table for the contextual header.

    Conventional tables (product comparisons, data grids) have meaningful
    labels in the header row — columns like "Product", "Revenue", "Growth".
    Financial statement tables (income statements, balance sheets) have
    meaningful labels in the row label column — rows like "Total net sales",
    "Operating expenses", "Net income" — while the column headers are just
    fiscal years or currency symbols.

    Reading only the first row therefore fails silently for half of all tables.

    This function reads both dimensions:
    - All cells from the first (header) row that are not purely numeric or symbolic
    - The first cell of every subsequent data row (the row label column)

    It deduplicates, filters out currency symbols, numbers, percentages, and
    separator rows, and returns up to 12 unique text labels joined by commas.

    For the Apple income statement this produces:
        "Products, Services, Total net sales, Cost of sales, Total cost of sales,
         Gross margin, Operating expenses, Research and development,
         Selling general and administrative, Total operating expenses,
         Operating income, Net income"

    Args:
        markdown: Raw GitHub-flavoured Markdown table string.

    Returns:
        Comma-joined label string, or empty string if no labels found.
    """
    # Pattern for cells that carry no semantic meaning: pure numbers, currency,
    # percentage signs, parenthesised negatives, empty, or Markdown separators.
    _NUMERIC_CELL = re.compile(r"^[\d$,%.\s()/\-–—]+$")
    _SEPARATOR = re.compile(r"^:?-+:?$")

    rows: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|")]
        cells = [c for c in cells if c]  # drop empty boundary cells
        if not cells:
            continue
        # Skip separator rows (all cells are --- or :---:)
        if all(_SEPARATOR.match(c) for c in cells):
            continue
        rows.append(cells)

    if not rows:
        return ""

    seen: set[str] = set()
    labels: list[str] = []

    def _keep(cell: str) -> bool:
        """Return True if the cell is a meaningful text label worth indexing."""
        c = cell.strip()
        return (
            len(c) >= 3
            and c not in seen
            and not _NUMERIC_CELL.match(c)
        )

    # Column headers: all cells from the first row
    for cell in rows[0]:
        if _keep(cell):
            labels.append(cell)
            seen.add(cell)

    # Row labels: first cell of every subsequent row
    for row in rows[1:]:
        if row and _keep(row[0]):
            labels.append(row[0])
            seen.add(row[0])

    return ", ".join(labels[:12])


def _detect_heading(text: str) -> str | None:
    """Return the first heading-like line in *text*, or None.

    A heading is a short (≤ 80 chars), predominantly-uppercase line that may
    optionally start with a section number (e.g. "3.1 Methodology").
    """
    for line in text.splitlines():
        stripped = line.strip()
        if 4 <= len(stripped) <= 80 and _HEADING_RE.match(stripped):
            return stripped
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split a list of parsed pages into smaller overlapping text chunks.

    Each input page dict must have keys ``text``, ``page_number``, and
    ``source_filename`` (as produced by :func:`pdf_parser.parse_pdf`).

    Each output chunk dict has the shape::

        {
            "text":            str,  # contextual-header + chunk text (used for embedding/reranking)
            "raw_text":        str,  # original chunk text without header (used for display)
            "page_number":     int,  # page this chunk came from
            "source_filename": str,  # carried over from the page
            "chunk_id":        str,  # UUID4 unique identifier
            "chunk_index":     int,  # global sequential index (0-based)
        }

    Args:
        pages: List of page dicts from :func:`pdf_parser.parse_pdf`.

    Returns:
        List of chunk dicts, filtered for minimum length, globally indexed.

    Raises:
        ValueError: If ``pages`` is empty.
    """
    if not pages:
        raise ValueError("chunk_pages received an empty page list.")

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Separators are tried in order; the splitter falls through to the next
    # one only when the current separator produces a piece that is still
    # larger than chunk_size.  Ordering: paragraph → sentence → clause →
    # word → character.  This biases splits toward semantic boundaries so
    # we never cut inside a word and rarely cut inside a sentence.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
        separators=[
            "\n\n",   # paragraph break (highest priority)
            "\n",     # line break
            ". ",     # sentence end
            "? ",     # question end
            "! ",     # exclamation end
            "; ",     # clause boundary
            ", ",     # phrase boundary
            " ",      # word boundary
            "",       # character fallback (last resort)
        ],
    )

    all_chunks: list[dict] = []

    for page in pages:
        page_text: str = page["text"]
        page_number: int = page["page_number"]
        source_filename: str = page["source_filename"]

        # ── Prose chunks ──────────────────────────────────────────────────
        # Split this page's text independently to keep page_number accurate.
        splits = splitter.split_text(page_text)

        for raw_chunk in splits:
            cleaned = raw_chunk.strip()
            if len(cleaned) < _MIN_CHUNK_LENGTH:
                logger.debug(
                    "Dropping short chunk (%d chars) from page %d.",
                    len(cleaned),
                    page_number,
                )
                continue

            # ── Contextual Header (NirDiamant 2024) ──────────────────────
            heading = _detect_heading(cleaned)
            if heading:
                header = (
                    f"[Document: {source_filename} | Page: {page_number} "
                    f"| Section: {heading}]"
                )
            else:
                header = f"[Document: {source_filename} | Page: {page_number}]"
            contextual_text = f"{header}\n{cleaned}"

            all_chunks.append(
                {
                    "text": contextual_text,
                    "raw_text": cleaned,
                    "page_number": page_number,
                    "source_filename": source_filename,
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": len(all_chunks),
                    "chunk_type": "text",
                    "bbox": None,
                }
            )

        # ── Table chunks (atomic — never split) ───────────────────────────
        # Each table is stored as a single indivisible chunk with its exact
        # bounding-box coordinates so the UI can render a highlighted excerpt
        # directly from the PDF, and so the LLM must retrieve the full table
        # rather than recalling numbers from pretraining memory.
        for table in page.get("tables", []):
            md = table.get("markdown", "").strip()
            if not md:
                continue
            # Include column headers in the contextual header so dense and
            # sparse retrieval both see the table's subject in natural language
            # rather than just "[Section: TABLE]".
            col_headers = _build_table_summary(md)
            if col_headers:
                header = (
                    f"[Document: {source_filename} | Page: {page_number}"
                    f" | TABLE: {col_headers}]"
                )
            else:
                header = f"[Document: {source_filename} | Page: {page_number} | TABLE]"
            contextual_text = f"{header}\n{md}"
            all_chunks.append(
                {
                    "text": contextual_text,
                    "raw_text": md,
                    "page_number": page_number,
                    "source_filename": source_filename,
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": len(all_chunks),
                    "chunk_type": "table",
                    "bbox": table.get("bbox"),   # [x0, y0, x1, y1] from PyMuPDF
                }
            )

    # Re-index sequentially so chunk_index is always 0..N-1 with no gaps.
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = i

    logger.info(
        "Chunking complete: %d chunks from %d pages "
        "(chunk_size=%d, overlap=%d).",
        len(all_chunks),
        len(pages),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks


def get_chunk_stats(chunks: list[dict]) -> dict:
    """Compute descriptive statistics over a list of chunks.

    Args:
        chunks: List of chunk dicts (output of :func:`chunk_pages`).

    Returns:
        Dict with keys:
            - ``total_chunks``      (int)
            - ``avg_chunk_length``  (float)
            - ``min_chunk_length``  (int)
            - ``max_chunk_length``  (int)
            - ``pages_covered``     (int)  — distinct page numbers present
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_length": 0.0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
            "pages_covered": 0,
        }

    lengths = [len(c.get("raw_text", c["text"])) for c in chunks]
    pages_covered = len({c["page_number"] for c in chunks})

    return {
        "total_chunks": len(chunks),
        "avg_chunk_length": round(sum(lengths) / len(lengths), 1),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths),
        "pages_covered": pages_covered,
    }
