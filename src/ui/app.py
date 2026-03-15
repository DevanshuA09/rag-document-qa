"""
Streamlit UI - Document Q&A System
Run: bash run.sh  (from project root)
"""

import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import streamlit as st
from src.ingestion.pipeline import ingest_document
from src.generation.rag_chain import answer

st.set_page_config(page_title="DocChat", page_icon="💬", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 0;
    max-width: 780px;
    margin-left: auto;
    margin-right: auto;
}

/* ── Landing ────────────────────────────────── */
.landing-wrap {
    display: flex; flex-direction: column; align-items: center;
    padding: 6rem 1rem 2rem; text-align: center;
}
.landing-title {
    font-size: 2rem; font-weight: 700;
    letter-spacing: -0.02em; margin: 0 0 0.5rem;
}
.landing-sub {
    font-size: 0.95rem; color: #8e8ea0;
    max-width: 400px; margin: 0 auto 2.5rem; line-height: 1.65;
}
.prompts-row {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    justify-content: center; max-width: 600px;
}
.prompt-pill {
    background: #2a2a2a; border: 1px solid #3a3a3a;
    border-radius: 999px; padding: 0.45rem 1rem;
    font-size: 0.82rem; color: #ccc; cursor: default;
    transition: border-color 0.15s;
}

/* ── Document banner ────────────────────────── */
.doc-banner {
    display: flex; align-items: center; justify-content: space-between;
    background: #1c1c1c; border: 1px solid #2e2e2e;
    border-radius: 10px; padding: 0.65rem 1rem; margin-bottom: 1rem;
}
.doc-name { font-weight: 600; font-size: 0.88rem; color: #ececec; }
.doc-meta { font-size: 0.72rem; color: #6b6b7e; margin-top: 2px; }
.ready-pill {
    background: #10a37f22; color: #10a37f; font-size: 0.62rem;
    font-weight: 700; padding: 0.2rem 0.65rem;
    border-radius: 999px; letter-spacing: .07em;
}

/* ── Empty chat state ───────────────────────── */
.empty-state {
    text-align: center; padding: 4rem 1rem 2rem; color: #6b6b7e;
}
.empty-title { font-size: 0.95rem; font-weight: 600; color: #aaa; margin-bottom: 1.5rem; }

/* ── Message metadata ───────────────────────── */
.chips {
    display: flex; gap: 0.3rem; flex-wrap: wrap;
    margin-top: 0.65rem; padding-top: 0.5rem;
    border-top: 1px solid #2a2a2a;
}
.chip {
    background: transparent; border: 1px solid #2e2e2e;
    border-radius: 999px; padding: 0.12rem 0.55rem;
    font-size: 0.68rem; color: #6b6b7e;
}
.chip-mode {
    color: #10a37f; border-color: #10a37f33;
}

/* ── Source cards ───────────────────────────── */
.src-card {
    border-left: 2px solid #2e2e2e; background: #1c1c1c;
    border-radius: 0 8px 8px 0; padding: 0.6rem 0.85rem;
    margin-bottom: 0.35rem; font-size: 0.81rem;
    color: #b0b0b0; line-height: 1.55;
}
.src-card:hover { border-left-color: #10a37f; }
.badge {
    display: inline-block; font-size: 0.62rem; font-weight: 700;
    padding: 0.1rem 0.45rem; border-radius: 4px;
    margin-bottom: 0.3rem; margin-right: 0.3rem; letter-spacing: .05em;
}
.badge-page { background: #1e2e26; color: #10a37f; }
.badge-table { background: #1e2433; color: #6b9fff; }
.badge-text  { background: #262626; color: #888; }

/* ── Sub-question cards ─────────────────────── */
.subq {
    background: #1c1c1c; border: 1px solid #2a2a2a;
    border-radius: 8px; padding: 0.5rem 0.85rem;
    margin-bottom: 0.3rem; font-size: 0.82rem; color: #bbb;
    display: flex; gap: 0.5rem; align-items: flex-start;
}
.sq-num { color: #10a37f; font-weight: 700; min-width: 1rem; }

/* ── Sidebar ────────────────────────────────── */
.sidebar-section {
    font-size: 0.7rem; font-weight: 600; letter-spacing: .08em;
    color: #555; text-transform: uppercase; margin: 0.5rem 0 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "ingested": False, "collection_name": None,
    "source_filename": None, "page_count": 0,
    "chunk_count": 0, "chat_history": [],
}.items():
    st.session_state.setdefault(k, v)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### DocChat")
    st.divider()

    st.markdown("<div class='sidebar-section'>Document</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF", type=["pdf"], label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.source_filename:
            st.session_state.ingested = False
            st.session_state.collection_name = None
            st.session_state.chat_history = []

        size_mb = uploaded_file.size / (1024 * 1024)
        st.caption(f"`{uploaded_file.name}` · {size_mb:.1f} MB")

        if not st.session_state.ingested:
            if st.button("Ingest Document", type="primary", use_container_width=True):
                data_dir = os.path.join(ROOT, "data")
                os.makedirs(data_dir, exist_ok=True)
                pdf_path = os.path.join(data_dir, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner("Parsing and indexing..."):
                    try:
                        res = ingest_document(pdf_path)
                        st.session_state.update({
                            "ingested": True,
                            "collection_name": res["collection_name"],
                            "source_filename": res["source_filename"],
                            "page_count": res["page_count"],
                            "chunk_count": res["chunk_count"],
                        })
                        if res.get("already_existed"):
                            st.info("Already indexed.")
                        else:
                            st.success(
                                f"{res['page_count']} pages / "
                                f"{res['chunk_count']} chunks indexed."
                            )
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

    if st.session_state.ingested:
        st.success("Ready to query")

    st.divider()
    st.markdown("<div class='sidebar-section'>Mode</div>", unsafe_allow_html=True)

    mode_label = st.radio(
        "Mode", ["Auto", "Fast (Stuff)", "Accurate (Reciprocal)"],
        label_visibility="collapsed",
    )
    st.caption(
        "**Auto** classifies query type and routes automatically.  \n"
        "**Fast** uses multi-query retrieval for direct lookups.  \n"
        "**Accurate** decomposes the query into sub-questions."
    )

    if st.session_state.chat_history:
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _mode(label: str) -> str:
    if "Reciprocal" in label: return "reciprocal"
    if "Auto" in label:       return "auto"
    return "stuff"


def _truncate(text: str, n: int = 240) -> str:
    if len(text) <= n:
        return text
    cut = text[:n]
    last = cut.rfind(". ")
    return (cut[:last + 1] if last > n // 2 else cut.rstrip()) + "..."


def _render_entry(entry: dict) -> None:
    res     = entry["result"]
    sources = res.get("sources", [])
    mode    = res.get("mode", "stuff")
    chunks  = res.get("chunks_used_in_prompt") or res.get("chunks_retrieved", 0)
    mode_label = "Reciprocal" if mode == "reciprocal" else "Stuff"

    with st.chat_message("user"):
        st.markdown(entry["query"])

    with st.chat_message("assistant"):
        st.markdown(res["answer"])

        qtype = (
            f"<span class='chip'>{res['query_type']}</span>"
            if res.get("auto_routed") and res.get("query_type") else ""
        )
        st.markdown(
            f"<div class='chips'>"
            f"<span class='chip chip-mode'>{mode_label}</span>"
            f"{qtype}"
            f"<span class='chip'>{chunks} chunks</span>"
            f"<span class='chip'>{res.get('tokens_used', 0):,} tokens</span>"
            f"<span class='chip'>${res.get('cost_usd', 0):.4f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if sources:
        with st.expander(f"{len(sources)} sources", expanded=False):
            for s in sources:
                is_table = s.get("chunk_type") == "table"
                type_badge = (
                    "<span class='badge badge-table'>TABLE</span>"
                    if is_table else
                    "<span class='badge badge-text'>TEXT</span>"
                )
                st.markdown(
                    f"<div class='src-card'>"
                    f"<span class='badge badge-page'>Page {s['page_number']}</span>"
                    f"{type_badge}<br>"
                    f"{_truncate(s.get('text_excerpt', ''))}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if mode == "reciprocal" and res.get("sub_questions"):
        with st.expander("Sub-questions", expanded=False):
            html = "".join(
                f"<div class='subq'><span class='sq-num'>{i}.</span><span>{q}</span></div>"
                for i, q in enumerate(res["sub_questions"], 1)
            )
            st.markdown(html, unsafe_allow_html=True)


# ── Landing ───────────────────────────────────────────────────────────────────
if not st.session_state.ingested:
    st.markdown(
        "<div class='landing-wrap'>"
        "<div class='landing-title'>Ask anything about your document</div>"
        "<div class='landing-sub'>"
        "Upload a PDF in the sidebar and ask questions in plain English. "
        "Every answer is grounded in the document with page citations."
        "</div>"
        "<div class='prompts-row'>"
        "<span class='prompt-pill'>What were total revenues in 2023?</span>"
        "<span class='prompt-pill'>Summarise the key risk factors</span>"
        "<span class='prompt-pill'>What is the R&amp;D strategy?</span>"
        "<span class='prompt-pill'>Compare performance across years</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Chat ──────────────────────────────────────────────────────────────────────
else:
    st.markdown(
        f"<div class='doc-banner'>"
        f"<div>"
        f"<div class='doc-name'>{st.session_state.source_filename}</div>"
        f"<div class='doc-meta'>"
        f"{st.session_state.page_count} pages &nbsp;&middot;&nbsp; "
        f"{st.session_state.chunk_count} chunks"
        f"</div>"
        f"</div>"
        f"<span class='ready-pill'>READY</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:
        st.markdown(
            "<div class='empty-state'>"
            "<div class='empty-title'>Document indexed. Start asking.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for entry in st.session_state.chat_history:
            _render_entry(entry)

    query = st.chat_input("Ask a question about your document...")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        with st.spinner(""):
            try:
                result = answer(
                    query=query,
                    collection_name=st.session_state.collection_name,
                    mode=_mode(mode_label),
                )
                st.session_state.chat_history.append({"query": query, "result": result})
                st.rerun()
            except Exception as e:
                st.error(f"{e}")
