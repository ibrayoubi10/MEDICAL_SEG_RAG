# Author: Ibrahim M. AlAyoubi

"""Answer questions with a simple RAG loop over the medical segmentation corpus."""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from chroma_index import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    get_paper_id,
    iter_query_hits,
    load_papers,
    paper_to_document,
    paper_to_metadata,
    query_vector_database,
)

DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
DEFAULT_MODEL = DEFAULT_OPENAI_MODEL
DEFAULT_PROVIDER = os.environ.get("RAG_LLM_PROVIDER", "ollama").lower()
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_N_RESULTS = 6
DEFAULT_MAX_CONTEXT_CHARS = 14000

SYSTEM_PROMPT = """You are a careful research assistant for medical image segmentation.
Answer in English.
Use only the retrieved arXiv context.
If the retrieved context is not enough, say so clearly.
When citing sources, cite the article titles exactly.
Do not cite sources with labels such as [S1] or [S2].
Keep the answer structured and useful for a student building a first RAG project."""


@dataclass
class RetrievedSource:
    label: str
    paper_id: str
    title: str
    authors: str
    published: str
    categories: str
    abstract_page: str
    pdf: str
    distance: float
    document: str
    retrieval_method: str = "vector"
    lexical_score: float = 0.0


SEARCH_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "article",
    "articles",
    "author",
    "authors",
    "did",
    "for",
    "in",
    "of",
    "on",
    "person",
    "published",
    "the",
    "this",
    "to",
    "worked",
    "with",
}


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


def normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def search_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_search_text(text).split()
        if len(token) > 1 and token not in SEARCH_STOPWORDS
    }


def source_from_metadata(
    label: str,
    paper_id: str,
    metadata,
    document: str,
    distance: float,
    retrieval_method: str,
    lexical_score: float = 0.0,
) -> RetrievedSource:
    return RetrievedSource(
        label=label,
        paper_id=paper_id,
        title=str(metadata.get("title") or "Untitled"),
        authors=str(metadata.get("authors") or ""),
        published=str(metadata.get("published") or ""),
        categories=str(metadata.get("all_categories") or ""),
        abstract_page=str(metadata.get("abstract_page") or ""),
        pdf=str(metadata.get("pdf") or ""),
        distance=distance,
        document=document,
        retrieval_method=retrieval_method,
        lexical_score=lexical_score,
    )


def score_author_match(question_tokens: set[str], question_text: str, authors) -> float:
    if not authors:
        return 0.0

    author_names = authors if isinstance(authors, list) else str(authors).split(",")
    best_score = 0.0

    for author in author_names:
        author_text = normalize_search_text(str(author))
        author_tokens = [token for token in author_text.split() if len(token) > 1]

        if not author_tokens:
            continue

        if len(author_tokens) >= 2 and author_text in question_text:
            best_score = max(best_score, 130.0)
            continue

        if len(author_tokens) >= 2 and all(
            token in question_tokens for token in author_tokens
        ):
            best_score = max(best_score, 120.0)
            continue

        first_name = author_tokens[0]
        last_name = author_tokens[-1]

        if first_name in question_tokens and last_name in question_tokens:
            best_score = max(best_score, 110.0)
        elif last_name in question_tokens and any(
            token in question_tokens for token in author_tokens[:-1]
        ):
            best_score = max(best_score, 90.0)
        elif last_name in question_tokens and len(last_name) >= 5:
            best_score = max(best_score, 65.0)

    return best_score


def score_token_overlap(question_tokens: set[str], text: str, weight: float) -> float:
    if not question_tokens:
        return 0.0

    text_tokens = search_tokens(text)
    return len(question_tokens.intersection(text_tokens)) * weight


def score_catalog_paper(question: str, paper: dict) -> float:
    question_text = normalize_search_text(question)
    question_tokens = search_tokens(question)

    author_score = score_author_match(
        question_tokens=question_tokens,
        question_text=question_text,
        authors=paper.get("authors"),
    )
    title_score = score_token_overlap(question_tokens, str(paper.get("title") or ""), 8.0)
    category_score = score_token_overlap(
        question_tokens,
        " ".join(str(category) for category in paper.get("all_categories") or []),
        3.0,
    )
    abstract_score = min(
        score_token_overlap(question_tokens, str(paper.get("abstract") or ""), 1.0),
        12.0,
    )

    return author_score + title_score + category_score + abstract_score


def retrieve_vector_sources(
    question: str,
    n_results: int,
) -> list[RetrievedSource]:
    result = query_vector_database(
        query=question,
        chroma_path=DEFAULT_CHROMA_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
        n_results=n_results,
    )

    sources = []

    for index, hit in enumerate(iter_query_hits(result), start=1):
        metadata = hit["metadata"]
        sources.append(
            source_from_metadata(
                label=f"S{index}",
                paper_id=hit["id"],
                metadata=metadata,
                document=hit["document"],
                distance=float(hit["distance"]),
                retrieval_method="vector",
            )
        )

    return sources


def retrieve_catalog_sources(
    question: str,
    n_results: int,
) -> list[RetrievedSource]:
    papers = load_papers(DEFAULT_CATALOG_PATH)
    scored_papers = []

    for paper in papers:
        score = score_catalog_paper(question, paper)
        if score >= 20.0:
            scored_papers.append((score, paper))

    scored_papers.sort(key=lambda item: item[0], reverse=True)

    sources = []
    for index, (score, paper) in enumerate(scored_papers[:n_results], start=1):
        sources.append(
            source_from_metadata(
                label=f"S{index}",
                paper_id=get_paper_id(paper),
                metadata=paper_to_metadata(paper),
                document=paper_to_document(paper),
                distance=0.0,
                retrieval_method="catalog",
                lexical_score=score,
            )
        )

    return sources


def source_key(source: RetrievedSource) -> str:
    return source.paper_id or source.abstract_page or source.title


def relabel_sources(sources: list[RetrievedSource]) -> list[RetrievedSource]:
    for index, source in enumerate(sources, start=1):
        source.label = f"S{index}"

    return sources


def merge_sources(
    vector_sources: list[RetrievedSource],
    catalog_sources: list[RetrievedSource],
    n_results: int,
) -> list[RetrievedSource]:
    strong_catalog_sources = [
        source for source in catalog_sources if source.lexical_score >= 50.0
    ]
    if strong_catalog_sources:
        return relabel_sources(dedupe_sources(strong_catalog_sources)[:n_results])

    merged = []
    seen = set()

    for source in vector_sources + catalog_sources:
        key = source_key(source)
        if key in seen:
            continue

        merged.append(source)
        seen.add(key)

        if len(merged) >= n_results:
            break

    return relabel_sources(merged)


def dedupe_sources(sources: list[RetrievedSource]) -> list[RetrievedSource]:
    deduped = []
    seen = set()

    for source in sources:
        key = source_key(source)
        if key in seen:
            continue

        deduped.append(source)
        seen.add(key)

    return deduped


def retrieve_sources(
    question: str,
    n_results: int,
) -> list[RetrievedSource]:
    catalog_sources = retrieve_catalog_sources(question, n_results)
    try:
        vector_sources = retrieve_vector_sources(question, max(n_results, n_results * 2))
    except ValueError:
        vector_sources = []

    return merge_sources(vector_sources, catalog_sources, n_results)


def format_sources_for_prompt(
    sources: list[RetrievedSource],
    max_context_chars: int,
) -> str:
    blocks = []
    remaining_chars = max_context_chars

    for source in sources:
        header = "\n".join(
            [
                f"Source label: {source.label}",
                f"Title: {source.title}",
                f"Authors: {source.authors}",
                f"Published: {source.published}",
                f"Categories: {source.categories}",
                f"arXiv URL: {source.abstract_page}",
                f"PDF URL: {source.pdf}",
                f"Retrieval method: {source.retrieval_method}",
                f"Vector distance: {source.distance:.4f}",
                f"Lexical score: {source.lexical_score:.1f}",
                "Content:",
            ]
        )
        available_for_document = max(0, remaining_chars - len(header) - 8)

        if available_for_document <= 0:
            break

        document = truncate_text(source.document, available_for_document)
        block = f"{header}\n{document}"
        blocks.append(block)
        remaining_chars -= len(block)

    return "\n\n---\n\n".join(blocks)


def build_user_prompt(
    question: str,
    sources: list[RetrievedSource],
    max_context_chars: int,
) -> str:
    context = format_sources_for_prompt(sources, max_context_chars)

    return f"""Question:
{question}

Retrieved arXiv context:
{context}

Answer the question using only this context.
Answer in English.
When you cite a source, cite the exact article title in quotation marks.
Do not use source labels like [S1], [S2], or [S3] in the answer."""


def replace_source_labels_with_titles(
    answer: str,
    sources: list[RetrievedSource],
) -> str:
    for source in sources:
        title_citation = f'"{source.title}"'
        answer = answer.replace(f"[{source.label}]", title_citation)

    return answer


def generate_openai_answer(
    question: str,
    sources: list[RetrievedSource],
    model: str,
    max_context_chars: int,
    temperature: float,
) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is not installed. Run: "
            ".venv/bin/python -m pip install -r requirements.txt"
        ) from error

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=build_user_prompt(question, sources, max_context_chars),
        temperature=temperature,
    )

    return response.output_text


def normalize_ollama_url(base_url: str) -> str:
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"

    return base_url.rstrip("/")


def generate_ollama_answer(
    question: str,
    sources: list[RetrievedSource],
    model: str,
    max_context_chars: int,
    temperature: float,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
) -> str:
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": build_user_prompt(question, sources, max_context_chars),
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    request = Request(
        url=f"{normalize_ollama_url(base_url)}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed: {message}") from error
    except URLError as error:
        raise RuntimeError(
            "Ollama is not reachable. Start it with the Ollama app or run: "
            "ollama serve"
        ) from error

    answer = data.get("response")
    if not answer:
        raise RuntimeError(f"Ollama returned no response: {data}")

    return str(answer).strip()


def get_default_model(provider: str) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL

    if provider == "ollama":
        return DEFAULT_OLLAMA_MODEL

    raise ValueError(f"Unknown provider: {provider}")


def generate_answer(
    question: str,
    sources: list[RetrievedSource],
    provider: str,
    model: str,
    max_context_chars: int,
    temperature: float,
) -> str:
    if provider == "openai":
        answer = generate_openai_answer(
            question=question,
            sources=sources,
            model=model,
            max_context_chars=max_context_chars,
            temperature=temperature,
        )
        return replace_source_labels_with_titles(answer, sources)

    if provider == "ollama":
        answer = generate_ollama_answer(
            question=question,
            sources=sources,
            model=model,
            max_context_chars=max_context_chars,
            temperature=temperature,
        )
        return replace_source_labels_with_titles(answer, sources)

    raise RuntimeError(f"Unknown LLM provider: {provider}")


def print_retrieved_sources(sources: list[RetrievedSource]) -> None:
    print("Retrieved sources:\n")

    for index, source in enumerate(sources, start=1):
        print(f"{index}. {source.title}")
        print(f"     authors: {source.authors}")
        print(f"     arXiv: {source.abstract_page}")
        print(f"     PDF: {source.pdf}")
        print(f"     categories: {source.categories}")
        print(f"     retrieval: {source.retrieval_method}")
        print(f"     distance: {source.distance:.4f}")
        print(f"     lexical score: {source.lexical_score:.1f}\n")


def positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error

    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")

    return parsed_value


def non_negative_float(value: str) -> float:
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error

    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")

    return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions over the medical segmentation Chroma corpus."
    )
    parser.add_argument("question", help="Question to ask the RAG assistant.")
    parser.add_argument(
        "--n-results",
        type=positive_int,
        default=DEFAULT_N_RESULTS,
        help="Number of Chroma results to retrieve.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default=DEFAULT_PROVIDER if DEFAULT_PROVIDER in {"ollama", "openai"} else "ollama",
        help="LLM provider used for answer generation.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model used for answer generation. Defaults to the provider model.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=positive_int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help="Maximum number of context characters sent to the LLM.",
    )
    parser.add_argument(
        "--temperature",
        type=non_negative_float,
        default=0.2,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only print retrieved Chroma sources without calling an LLM.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the context that would be sent to the LLM.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        sources = retrieve_sources(args.question, args.n_results)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    if args.retrieval_only:
        print_retrieved_sources(sources)
        return

    if args.show_context:
        print(format_sources_for_prompt(sources, args.max_context_chars))
        print()

    model = args.model or get_default_model(args.provider)

    try:
        answer = generate_answer(
            question=args.question,
            sources=sources,
            provider=args.provider,
            model=model,
            max_context_chars=args.max_context_chars,
            temperature=args.temperature,
        )
    except RuntimeError as error:
        raise SystemExit(f"Error: answer generation failed: {error}") from error

    print(answer)
    print()
    print_retrieved_sources(sources)


if __name__ == "__main__":
    main()
