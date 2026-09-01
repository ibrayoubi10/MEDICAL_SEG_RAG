# Author: Ibrahim M. AlAyoubi

"""Streamlit interface for asking questions over the arXiv RAG corpus."""

from __future__ import annotations

from html import escape

import streamlit as st

from rag_answer import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_N_RESULTS,
    DEFAULT_OLLAMA_MODEL,
    generate_answer,
    retrieve_sources,
)


st.set_page_config(
    page_title="Medical Segmentation RAG",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.16), transparent 30rem),
            radial-gradient(circle at 85% 12%, rgba(245, 158, 11, 0.14), transparent 24rem),
            linear-gradient(180deg, #f8fafc 0%, #edf7f4 52%, #f5f3ff 100%);
        color: #111827;
    }

    .stApp p,
    .stApp li,
    .stApp label,
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stMarkdownContainer"] p {
        color: #111827;
    }

    div[data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 2.25rem;
    }

    div[data-testid="stSidebarContent"] {
        background:
            linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.96)),
            radial-gradient(circle at top, rgba(20, 184, 166, 0.22), transparent 18rem);
        color: #f8fafc;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    div[data-testid="stSidebarContent"] h1,
    div[data-testid="stSidebarContent"] h2,
    div[data-testid="stSidebarContent"] h3,
    div[data-testid="stSidebarContent"] label,
    div[data-testid="stSidebarContent"] p,
    div[data-testid="stSidebarContent"] span,
    div[data-testid="stSidebarContent"] div[data-testid="stMarkdownContainer"],
    div[data-testid="stSidebarContent"] div[data-testid="stMarkdownContainer"] p {
        color: #e2e8f0;
    }

    .hero {
        position: relative;
        padding: 2rem 0 1.7rem;
        margin-bottom: 0.25rem;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 1rem;
        align-items: end;
    }

    .hero h1 {
        color: #0f172a;
        font-size: clamp(2.25rem, 5vw, 4.6rem);
        line-height: 1.05;
        margin: 0;
    }

    .hero-kicker {
        color: #0f766e;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-line {
        width: 8rem;
        height: 0.34rem;
        margin-top: 1rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #14b8a6, #f59e0b, #7c3aed);
    }

    .hero-stats {
        display: grid;
        grid-template-columns: repeat(2, minmax(8rem, 1fr));
        gap: 0.75rem;
    }

    .hero-stat {
        min-width: 8rem;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(15, 23, 42, 0.10);
        border-radius: 0.9rem;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.07);
    }

    .hero-stat-value {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.1;
    }

    .hero-stat-label {
        color: #475569;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        color: #111827;
        background: rgba(255, 255, 255, 0.96);
        border-color: #cbd5e1;
    }

    div[data-testid="stForm"] {
        padding: 1rem 1.1rem 1.2rem;
        border: 1px solid rgba(15, 23, 42, 0.10);
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.76);
        box-shadow: 0 20px 55px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(10px);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(15, 23, 42, 0.12);
        background: rgba(255, 255, 255, 0.90);
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.07);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] li,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] {
        color: #111827;
    }

    .source-card {
        position: relative;
        padding: 1rem 1.1rem 1rem 1.35rem;
        border: 1px solid rgba(15, 23, 42, 0.10);
        border-left: 0.35rem solid #14b8a6;
        border-radius: 0.85rem;
        background: rgba(255, 255, 255, 0.92);
        margin: 0.7rem 0 0.75rem;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
    }

    .source-card h3 {
        color: #0f172a;
        font-size: 1.08rem;
        line-height: 1.28;
        margin: 0 0 0.45rem;
    }

    .source-meta {
        color: #334155;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    h2, h3 {
        color: #0f172a;
    }

    @media (max-width: 760px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }

        .hero-stats {
            grid-template-columns: 1fr 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_article_card(source) -> None:
    published_year = source.published[:4] if source.published else ""
    meta_items = [item for item in [published_year, source.categories] if item]
    authors = escape(source.authors or "Authors unavailable")
    title = escape(source.title)
    meta = escape(" | ".join(meta_items))

    st.markdown(
        f"""
        <div class="source-card">
            <h3>{title}</h3>
            <div class="source-meta">{authors}</div>
            <div class="source-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(horizontal=True):
        if source.abstract_page:
            st.link_button(
                "arXiv",
                source.abstract_page,
                icon=":material/article:",
                width="content",
            )
        if source.pdf:
            st.link_button(
                "PDF",
                source.pdf,
                icon=":material/picture_as_pdf:",
                width="content",
            )


def render_source(source) -> None:
    render_article_card(source)


st.markdown(
    """
    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="hero-kicker">Local arXiv research assistant</div>
                <h1>Medical Segmentation RAG</h1>
                <div class="hero-line"></div>
            </div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="hero-stat-value">3,527</div>
                    <div class="hero-stat-label">papers</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">Ollama</div>
                    <div class="hero-stat-label">local LLM</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    n_results = st.slider("Number of sources", 1, 12, DEFAULT_N_RESULTS)
    model = st.text_input("Ollama model", DEFAULT_OLLAMA_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_context_chars = st.number_input(
        "Max context characters",
        min_value=1000,
        max_value=50000,
        value=DEFAULT_MAX_CONTEXT_CHARS,
        step=1000,
    )

with st.form("question_form", border=False):
    question = st.text_area(
        "Question",
        placeholder="Which articles did Hicham Messaoudi publish?",
        height=130,
    )
    ask_clicked = st.form_submit_button(
        "Search",
        type="primary",
        width="stretch",
        icon=":material/search:",
    )

if ask_clicked:
    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning("Write a question before starting the search.")
        st.stop()

    with st.spinner("Searching ChromaDB..."):
        try:
            sources = retrieve_sources(cleaned_question, n_results)
        except ValueError as error:
            st.error(str(error))
            st.stop()

    with st.spinner("Generating the answer..."):
        try:
            answer = generate_answer(
                question=cleaned_question,
                sources=sources,
                provider="ollama",
                model=model,
                max_context_chars=int(max_context_chars),
                temperature=temperature,
            )
        except RuntimeError as error:
            st.error(f"Answer generation failed: {error}")
            st.stop()

    st.subheader("Answer")
    with st.container(border=True):
        st.markdown(answer)

    st.subheader("Articles")
    for source in sources:
        render_source(source)
