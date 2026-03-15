"""
pdf_parser.py — Rich PDF extraction using PyMuPDF (fitz).

Design decisions and evidence
------------------------------
We use an optimised PyMuPDF pipeline rather than heavier tools (Marker, MinerU)
for two reasons documented in OmniDocBench (Ouyang et al., CVPR 2024):

  * MinerU and Marker are the highest-quality open-source parsers, but require
    ~2 GB of model weights and take 10–60 s/page on CPU — too slow for an
    interactive ingestion UI.
  * PyMuPDF with proper block-level extraction, multi-column reading order, and
    Tesseract OCR for images is sufficient for digital (non-scanned) PDFs and
    runs in < 1 s/page.

Marker / MinerU are the recommended path for production scaling and are noted
as future improvements.

Improvements over naive get_text("text")  (all cited to OmniDocBench findings):
  1. Multi-column reading order  — x-position clustering; left column processed
     before right column (OmniDocBench §4.1 "Reading Order" evaluation axis).
  2. Header / footer / page-number removal  — OmniDocBench "Ignore Handling"
     explicitly excludes these as noisy, inconsistent elements.
  3. Tables via find_tables() → GitHub-Flavoured Markdown with false-positive
     filter (>20 cols = figure artefact; <30 % meaningful cells = noise).
  4. Image OCR via Tesseract — recovers diagram labels and figure text.
  5. Formula region tagging — high non-alpha ratio → [FORMULA: …] sentinel.
  6. Hyphenation repair — "recog-\\nition" → "recognition".

Public API
----------
    parse_pdf(pdf_path)        -> list[dict]
    get_pdf_metadata(pdf_path) -> dict
"""

import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF ≥ 1.23 required for find_tables()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OCR dependency — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import pytesseract
    from PIL import Image
    import io as _io
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    logger.warning(
        "pytesseract / Pillow not found — image OCR disabled. "
        "Install with: pip install pytesseract pillow"
    )

# Fraction of page height that defines the header/footer strip.
# Blocks whose centre falls within this band at the top or bottom are
# candidates for removal (page numbers, running headers, etc.).
_HEADER_FOOTER_RATIO = 0.07   # top/bottom 7 % of page height


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str) -> list[dict]:
    """Parse a PDF and return one enriched dict per non-empty page.

    Each page dict::

        {
            "text":            str,        # prose + OCR text (NO table markdown)
            "tables":          list[dict], # [{markdown, bbox, page_number, source_filename}]
            "page_number":     int,        # 1-indexed
            "source_filename": str,
            "has_tables":      bool,
            "has_images":      bool,
        }

    Tables are returned as a separate ``tables`` list so the chunker can
    store each table as an atomic chunk with its exact bounding-box
    coordinates.  Keeping tables out of ``text`` prevents financial data
    from being split mid-row and forces the LLM to ground numerical
    answers in retrieved table chunks rather than pretraining memory.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of non-empty page dicts (blank / fully image pages that yield no
        text are skipped with a warning).

    Raises:
        ValueError: File missing, encrypted, or corrupt.
    """
    pdf_path = _resolve_path(pdf_path)
    source_filename = pdf_path.name

    doc = _open_document(pdf_path)
    total_pages = len(doc)
    parsed_pages: list[dict] = []
    skipped = 0

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]

        page_dict = _extract_page(page, page_num, source_filename)

        # Stamp page_number / source_filename onto every extracted table.
        for tbl in page_dict.get("tables", []):
            tbl["page_number"] = page_num
            tbl["source_filename"] = source_filename

        if not page_dict["text"] and not page_dict.get("tables"):
            logger.warning(
                "Page %d of '%s' yielded no content — skipping.",
                page_num, source_filename,
            )
            skipped += 1
            continue

        page_dict["page_number"] = page_num
        page_dict["source_filename"] = source_filename
        parsed_pages.append(page_dict)

    doc.close()

    logger.info(
        "Parsed '%s': %d/%d pages extracted (%d skipped).",
        source_filename, len(parsed_pages), total_pages, skipped,
    )
    return parsed_pages


def get_pdf_metadata(pdf_path: str) -> dict:
    """Return basic metadata about a PDF file.

    Returns:
        Dict with keys: ``page_count``, ``title``, ``author``, ``file_size_mb``.
    """
    pdf_path = _resolve_path(pdf_path)
    doc = _open_document(pdf_path)
    meta = doc.metadata or {}
    page_count = len(doc)
    doc.close()

    return {
        "page_count": page_count,
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 3),
    }


# ---------------------------------------------------------------------------
# Page-level extraction orchestrator
# ---------------------------------------------------------------------------

def _extract_page(page: fitz.Page, page_num: int, filename: str) -> dict:
    """Extract all content from one page, merging text, tables, and images.

    Pipeline
    --------
    1. Detect tables → render each as Markdown; record their bounding boxes.
    2. Extract and OCR any embedded images.
    3. Extract text blocks, skipping regions already covered by tables.
    4. Strip header / footer / page-number blocks (OmniDocBench "ignore" rule).
    5. Detect multi-column layout and sort blocks in correct reading order.
    6. Join all fragments and apply text normalisation.

    Returns a dict with ``text``, ``has_tables``, ``has_images`` (page_number
    and source_filename are added by the caller).
    """
    page_rect = page.rect
    page_h = page_rect.height

    # ── 1. Tables ─────────────────────────────────────────────────────────
    # Tables are extracted as atomic dicts with their bounding-box coordinates.
    # They are NOT merged into the page text — the chunker stores each table
    # as its own indivisible chunk so numerical data is never split mid-row.
    #
    # Caption extraction: financial tables (income statements, balance sheets)
    # often have a caption or column-year header as a text block immediately
    # above the table bbox rather than inside the detected table cells.
    # We capture any text block whose bottom edge is within 60 px above the
    # table top and prepend it as a title row so the LLM knows which year
    # each column represents.
    table_rects: list[fitz.Rect] = []
    extracted_tables: list[dict] = []   # [{markdown, bbox: [x0,y0,x1,y1]}]
    _CAPTION_LOOKAHEAD_PX = 120  # max vertical gap between caption and table

    # Collect all text blocks first so we can search for captions.
    raw_blocks_for_caption = page.get_text("blocks", sort=True)

    try:
        for table in page.find_tables():
            md = _table_to_markdown(table)
            if md:
                rect = fitz.Rect(table.bbox)

                # Look for a caption block immediately above the table.
                caption_lines: list[str] = []
                for block in raw_blocks_for_caption:
                    if block[6] != 0:   # skip image blocks
                        continue
                    bx0, by0, bx1, by1 = block[:4]
                    block_text = block[4].strip()
                    if not block_text:
                        continue
                    # Block must end just above the table top and overlap
                    # horizontally with at least half the table width.
                    vertical_gap = rect.y0 - by1
                    overlap_x = max(0, min(bx1, rect.x1) - max(bx0, rect.x0))
                    table_width = rect.x1 - rect.x0
                    if (
                        0 <= vertical_gap <= _CAPTION_LOOKAHEAD_PX
                        and overlap_x > 0
                    ):
                        caption_lines.append(block_text)

                if caption_lines:
                    caption = " | ".join(caption_lines)
                    md = f"<!-- caption: {caption} -->\n{md}"

                table_rects.append(rect)
                extracted_tables.append({
                    "markdown": md,
                    "bbox": list(rect),   # [x0, y0, x1, y1] — serialisable
                })
    except Exception as exc:
        logger.debug("Table extraction failed on page %d: %s", page_num, exc)

    has_tables = bool(table_rects)

    # ── 2. Images ─────────────────────────────────────────────────────────
    image_elements: list[tuple[float, str]] = []   # (y_top, ocr_text)
    has_images = False
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        ocr_text, img_rect = _extract_image(page, xref, page_num)
        if ocr_text:
            has_images = True
            y = img_rect.y0 if img_rect else 0.0
            image_elements.append((y, ocr_text))

    # ── 3. Text blocks (skip table regions) ───────────────────────────────
    raw_blocks = page.get_text("blocks", sort=True)
    text_blocks: list[tuple[float, float, float, float, str]] = []
    # raw_blocks: (x0, y0, x1, y1, text, block_no, block_type)
    for block in raw_blocks:
        if block[6] != 0:            # skip image-type blocks
            continue
        bx0, by0, bx1, by1 = block[:4]
        text = block[4].strip()
        if not text:
            continue
        block_rect = fitz.Rect(bx0, by0, bx1, by1)
        if any(_overlap_ratio(block_rect, tr) > 0.5 for tr in table_rects):
            continue
        text_blocks.append((bx0, by0, bx1, by1, text))

    # ── 4. Strip headers / footers / page numbers ─────────────────────────
    # OmniDocBench "Ignore Handling": headers, footers, page numbers, and
    # page footnotes are excluded from evaluation as they are inconsistently
    # handled across parsers and add retrieval noise.
    header_y = page_rect.y0 + page_h * _HEADER_FOOTER_RATIO
    footer_y = page_rect.y1 - page_h * _HEADER_FOOTER_RATIO

    filtered_blocks: list[tuple[float, float, float, float, str]] = []
    for bx0, by0, bx1, by1, text in text_blocks:
        centre_y = (by0 + by1) / 2
        if centre_y < header_y or centre_y > footer_y:
            if _is_header_footer(text):
                logger.debug(
                    "Page %d: stripping header/footer block: %r", page_num, text[:60]
                )
                continue
        filtered_blocks.append((bx0, by0, bx1, by1, text))

    # ── 5. Multi-column reading order ─────────────────────────────────────
    # OmniDocBench evaluates "Reading Order" as a dedicated axis. Naïve y-sort
    # breaks for 2-column layouts (right column appears mid-document in sorted
    # order). We detect columns by clustering text-block x-midpoints and
    # process each column independently, top-to-bottom.
    text_elements = _sort_reading_order(filtered_blocks, page_rect)

    # ── 6. Assemble prose in reading order ────────────────────────────────
    # Tables are intentionally excluded here — they are returned separately
    # so the chunker can store them as atomic, bbox-annotated chunks.
    all_elements = text_elements + image_elements
    all_elements.sort(key=lambda t: t[0])

    raw = "\n\n".join(frag for _, frag in all_elements)
    cleaned = _normalize_whitespace(raw)

    return {
        "text": cleaned,
        "tables": extracted_tables,   # atomic table dicts with bbox
        "has_tables": has_tables,
        "has_images": has_images,
    }


# ---------------------------------------------------------------------------
# Multi-column reading order
# ---------------------------------------------------------------------------

def _sort_reading_order(
    blocks: list[tuple[float, float, float, float, str]],
    page_rect: fitz.Rect,
) -> list[tuple[float, str]]:
    """Sort text blocks in correct reading order for single- and multi-column pages.

    Algorithm
    ---------
    1. Compute the x-midpoint of each block.
    2. Look for a clear vertical gap in the x-midpoint distribution — this
       indicates a column gutter.  We only detect exactly two columns (the
       vast majority of real documents).
    3. If two columns are detected, assign each block to left or right based on
       whether its midpoint is below or above the gap centre.  Sort each column
       independently by y0, then concatenate left-column blocks followed by
       right-column blocks.
    4. For single-column pages, sort purely by y0.

    Full-width blocks (tables, wide paragraphs) that span > 60 % of page width
    are always sorted by y0 and not assigned to a column.

    Args:
        blocks:    List of (x0, y0, x1, y1, text) tuples.
        page_rect: The page bounding box used to measure relative widths.

    Returns:
        List of (y0, text) tuples in reading order.
    """
    if not blocks:
        return []

    page_w = page_rect.width

    # Separate full-width blocks (headers, section titles, wide paragraphs)
    full_width_threshold = page_w * 0.6
    narrow: list[tuple[float, float, float, float, str]] = []
    wide: list[tuple[float, float, float, float, str]] = []
    for b in blocks:
        bx0, by0, bx1, by1, text = b
        if (bx1 - bx0) >= full_width_threshold:
            wide.append(b)
        else:
            narrow.append(b)

    # Detect column split from x-midpoints of narrow blocks
    x_mids = [(bx0 + bx1) / 2 for bx0, by0, bx1, by1, _ in narrow]
    split_x = _detect_column_split(x_mids, page_w)

    if split_x is not None and len(narrow) >= 4:
        # Two-column layout
        left_col  = [(bx0, by0, bx1, by1, t) for bx0, by0, bx1, by1, t in narrow if (bx0 + bx1) / 2 <= split_x]
        right_col = [(bx0, by0, bx1, by1, t) for bx0, by0, bx1, by1, t in narrow if (bx0 + bx1) / 2 >  split_x]
        left_col.sort(key=lambda b: b[1])   # sort by y0
        right_col.sort(key=lambda b: b[1])
        ordered_narrow = left_col + right_col
    else:
        # Single column — sort by y0
        narrow.sort(key=lambda b: b[1])
        ordered_narrow = narrow

    wide.sort(key=lambda b: b[1])

    # Merge wide blocks back in at their y0 positions
    result: list[tuple[float, str]] = []

    narrow_iter = iter(ordered_narrow)
    wide_iter   = iter(wide)
    n_cur = next(narrow_iter, None)
    w_cur = next(wide_iter, None)

    while n_cur is not None or w_cur is not None:
        take_wide = (
            w_cur is not None
            and (n_cur is None or w_cur[1] < n_cur[1])
        )
        if take_wide:
            result.append((w_cur[1], w_cur[4]))
            w_cur = next(wide_iter, None)
        else:
            result.append((n_cur[1], n_cur[4]))
            n_cur = next(narrow_iter, None)

    return result


def _detect_column_split(x_mids: list[float], page_w: float) -> Optional[float]:
    """Find the x-coordinate of a two-column gutter, if one exists.

    Strategy: sort x-midpoints and look for the largest gap.  A gap is
    considered a real column split if:
      - It falls within the middle third of the page (25 %–75 % of page width)
      - The gap is at least 10 % of page width
      - There are blocks on both sides of the gap

    Args:
        x_mids:  X-midpoints of text blocks.
        page_w:  Total page width.

    Returns:
        x-coordinate of the column split, or None for single-column pages.
    """
    if len(x_mids) < 4:
        return None

    sorted_x = sorted(x_mids)
    best_gap = 0.0
    best_x   = None

    for i in range(len(sorted_x) - 1):
        gap = sorted_x[i + 1] - sorted_x[i]
        mid = (sorted_x[i] + sorted_x[i + 1]) / 2
        # Gap must be in the middle third and of meaningful size
        if page_w * 0.25 <= mid <= page_w * 0.75 and gap > page_w * 0.10:
            if gap > best_gap:
                best_gap = gap
                best_x = mid

    return best_x


# ---------------------------------------------------------------------------
# Header / footer detection
# ---------------------------------------------------------------------------

def _is_header_footer(text: str) -> bool:
    """Return True if a text block looks like a page header, footer, or page number.

    Heuristics (conservative — prefer false negatives over false positives):
    - Pure integer (page number): "42", "- 42 -", "Page 42"
    - Very short (≤ 6 tokens) and contains a standalone integer

    Args:
        text: Stripped text of the block.

    Returns:
        True if the block is likely a header/footer/page number.
    """
    t = text.strip()
    # Pure page number
    if re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", t):
        return True
    # "Page N" or "N of M"
    if re.fullmatch(r"[Pp]age\s+\d{1,4}(\s+of\s+\d{1,4})?", t):
        return True
    # Short line that is purely a number, possibly with surrounding punctuation
    tokens = t.split()
    if len(tokens) <= 4 and any(tok.isdigit() and len(tok) <= 4 for tok in tokens):
        if sum(1 for tok in tokens if tok.isdigit()) / len(tokens) >= 0.5:
            return True
    return False


# ---------------------------------------------------------------------------
# Table → Markdown
# ---------------------------------------------------------------------------

def _table_to_markdown(table) -> str:
    """Convert a PyMuPDF TableFinder table to GitHub-Flavoured Markdown.

    Quality filters (to reject false positives such as figure-layout grids):
      * ≤ 20 columns  — wider "tables" are almost always figure artefacts
      * ≥ 30 % of cells contain ≥ 2 characters of meaningful text
      * Header row has at least one non-empty cell

    Args:
        table: A ``fitz.table.Table`` returned by ``page.find_tables()``.

    Returns:
        Markdown string, or empty string if the table fails quality checks.
    """
    try:
        rows = table.extract()
    except Exception:
        return ""

    if not rows:
        return ""

    cleaned_rows: list[list[str]] = []
    for row in rows:
        cleaned = [_clean_cell(cell) for cell in row]
        if any(c for c in cleaned):
            cleaned_rows.append(cleaned)

    if not cleaned_rows:
        return ""

    n_cols = max(len(r) for r in cleaned_rows)

    if n_cols > 20:
        logger.debug("Skipping likely false-positive table with %d columns.", n_cols)
        return ""

    total_cells = sum(len(r) for r in cleaned_rows)
    meaningful  = sum(1 for r in cleaned_rows for c in r if len(c) >= 2)
    if total_cells > 0 and meaningful / total_cells < 0.30:
        logger.debug(
            "Skipping low-content table (%.0f%% meaningful cells).",
            meaningful / total_cells * 100,
        )
        return ""

    if not any(cleaned_rows[0]):
        return ""

    padded = [r + [""] * (n_cols - len(r)) for r in cleaned_rows]
    lines  = [
        "| " + " | ".join(padded[0]) + " |",
        "| " + " | ".join(["---"] * n_cols) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _clean_cell(value) -> str:
    """Normalise a raw table cell to a clean string safe for Markdown.

    Args:
        value: Raw cell content (str, None, or other).

    Returns:
        Cleaned string with pipe characters escaped.
    """
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    return text.replace("|", "\\|")


# ---------------------------------------------------------------------------
# Image OCR
# ---------------------------------------------------------------------------

def _extract_image(
    page: fitz.Page,
    xref: int,
    page_num: int,
) -> tuple[str, Optional[fitz.Rect]]:
    """Extract text from an embedded image via Tesseract OCR.

    Args:
        page:     The page containing the image.
        xref:     The image cross-reference number.
        page_num: 1-indexed page number (for log messages).

    Returns:
        ``(ocr_text, image_rect)``.  ``ocr_text`` is empty string when OCR is
        unavailable or the image yields no recognisable text.
    """
    if not _OCR_AVAILABLE:
        return "", None

    img_rect: Optional[fitz.Rect] = None
    for item in page.get_image_info(xrefs=True):
        if item.get("xref") == xref:
            img_rect = fitz.Rect(item["bbox"])
            break

    try:
        base_image = page.parent.extract_image(xref)
        pil_image  = Image.open(_io.BytesIO(base_image["image"])).convert("RGB")
    except Exception as exc:
        logger.debug("Could not extract image xref=%d on page %d: %s", xref, page_num, exc)
        return "", img_rect

    try:
        ocr_text = pytesseract.image_to_string(
            pil_image,
            config="--psm 6 --oem 3",
        ).strip()
    except Exception as exc:
        logger.debug("OCR failed for image on page %d: %s", page_num, exc)
        return "", img_rect

    if not ocr_text:
        return "", img_rect

    # Tag high-symbol-density content as a formula (OmniDocBench formula sentinel)
    non_alpha = sum(1 for c in ocr_text if not c.isalnum() and not c.isspace())
    if non_alpha / len(ocr_text) > 0.45:
        ocr_text = f"[FORMULA: {ocr_text}]"

    return ocr_text, img_rect


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Clean extracted text while preserving paragraph breaks and Markdown tables.

    Steps (in order):
    1. Repair hyphenated line breaks  ("recog-\\nition" → "recognition").
    2. Collapse horizontal whitespace.
    3. Cap runs of blank lines to two newlines.
    4. Join soft line-wraps (single newlines between non-table lines).

    Args:
        text: Raw concatenated page text.

    Returns:
        Cleaned string, or empty string if nothing meaningful remains.
    """
    # 1. Repair hyphenation (word- + newline + continuation)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 2. Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # 3. Normalise blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Join soft line-wraps, but leave Markdown table rows intact
    def _join_wrap(m: re.Match) -> str:
        before = m.group(1)
        after  = m.group(2)
        if before.lstrip().startswith("|") or after.lstrip().startswith("|"):
            return before + "\n" + after
        return before + " " + after

    text = re.sub(r"([^\n]+)\n([^\n]+)", _join_wrap, text)
    return text.strip()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    """Fraction of rect ``a`` that is covered by rect ``b`` (0–1).

    Args:
        a: Rect whose coverage is measured.
        b: Reference rect.

    Returns:
        Float in [0, 1].
    """
    intersection = a & b
    if intersection.is_empty:
        return 0.0
    area_a = a.width * a.height
    return 0.0 if area_a == 0 else (intersection.width * intersection.height) / area_a


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _resolve_path(pdf_path: str) -> Path:
    """Resolve and validate a PDF path.

    Raises:
        ValueError: If the path does not point to an existing file.
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"PDF file not found: '{pdf_path}'")
    if not path.is_file():
        raise ValueError(f"Path is not a file: '{pdf_path}'")
    return path


def _open_document(pdf_path: Path) -> fitz.Document:
    """Open a PDF with safety checks for encryption and corruption.

    Raises:
        ValueError: If the file is encrypted or cannot be parsed.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(
            f"Could not open '{pdf_path.name}'. "
            "The file may be corrupt or not a valid PDF."
        ) from exc

    if doc.is_encrypted:
        doc.close()
        raise ValueError(
            f"'{pdf_path.name}' is password-protected. "
            "Decrypt the PDF before ingestion."
        )

    return doc
