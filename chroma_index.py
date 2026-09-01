# Author: Ibrahim M. AlAyoubi

"""Build and query a local Chroma vector database for medical segmentation papers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import (
    Embeddable,
    EmbeddingFunction,
    Include,
    Metadata,
    QueryResult,
)
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


DEFAULT_CATALOG_PATH = Path("data/medical_segmentation/arxiv_papers_catalog.json")
DEFAULT_CHROMA_PATH = Path("data/medical_segmentation/chroma_db")
DEFAULT_COLLECTION_NAME = "medical_segmentation_papers"


def load_papers(catalog_path: Path = DEFAULT_CATALOG_PATH) -> list[dict[str, Any]]:
    """Load papers from the merged medical segmentation arXiv catalog."""

    with catalog_path.open("r", encoding="utf-8") as catalog_file:
        catalog = json.load(catalog_file)

    papers = catalog.get("papers", [])

    if not isinstance(papers, list):
        raise ValueError(f"No paper list found in {catalog_path}")

    return papers


def paper_to_document(paper: dict[str, Any]) -> str:
    """Create the text that will be embedded for one paper."""

    title = paper.get("title") or "Untitled"
    abstract = paper.get("abstract") or ""
    authors = join_values(paper.get("authors"))
    categories = join_values(paper.get("all_categories"))

    return "\n".join(
        [
            f"Title: {title}",
            f"Authors: {authors}",
            f"Categories: {categories}",
            f"Abstract: {abstract}",
        ]
    )


def paper_to_metadata(paper: dict[str, Any]) -> dict[str, str]:
    """Keep Chroma metadata scalar and easy to display after retrieval."""

    return {
        "arxiv_id": as_text(paper.get("arxiv_id")),
        "base_arxiv_id": as_text(paper.get("base_arxiv_id")),
        "title": as_text(paper.get("title")),
        "authors": join_values(paper.get("authors")),
        "published": as_text(paper.get("published")),
        "updated": as_text(paper.get("updated")),
        "primary_category": as_text(paper.get("primary_category")),
        "all_categories": join_values(paper.get("all_categories")),
        "abstract_page": as_text(paper.get("abstract_page")),
        "pdf": as_text(paper.get("pdf")),
    }


def as_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


def join_values(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item is not None)

    return str(value)


def get_paper_id(paper: dict[str, Any]) -> str:
    paper_id = paper.get("base_arxiv_id") or paper.get("arxiv_id")

    if not paper_id:
        raise ValueError(f"Paper is missing an arXiv ID: {paper}")

    return str(paper_id)


def batched(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class QueryHit(TypedDict):
    id: str
    distance: float
    metadata: Metadata
    document: str


def get_chroma_collection(
    chroma_path: Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    """Open the persistent Chroma collection used by this project."""

    client = chromadb.PersistentClient(path=str(chroma_path))
    default_embedding_function = DefaultEmbeddingFunction()
    embedding_function = cast(
        EmbeddingFunction[Embeddable],
        default_embedding_function,
    )

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={
            "description": (
                "Medical image segmentation arXiv paper abstracts and metadata"
            ),
            "embedding_function": default_embedding_function.name(),
            "corpus": "medical_image_segmentation",
        },
    )


def reset_collection(
    chroma_path: Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    client = chromadb.PersistentClient(path=str(chroma_path))

    try:
        client.delete_collection(collection_name)
    except NotFoundError:
        pass

    return get_chroma_collection(chroma_path, collection_name)


def build_vector_database(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 100,
    limit: Optional[int] = None,
    reset: bool = False,
) -> Collection:
    """Index medical segmentation papers into a persistent Chroma collection."""

    papers = load_papers(catalog_path)

    if limit is not None:
        papers = papers[:limit]

    collection = (
        reset_collection(chroma_path, collection_name)
        if reset
        else get_chroma_collection(chroma_path, collection_name)
    )

    for paper_batch in batched(papers, batch_size):
        collection.upsert(
            ids=[get_paper_id(paper) for paper in paper_batch],
            documents=[paper_to_document(paper) for paper in paper_batch],
            metadatas=[paper_to_metadata(paper) for paper in paper_batch],
        )

    return collection


def query_vector_database(
    query: str,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    n_results: int = 5,
) -> QueryResult:
    """Search the medical segmentation vector database."""

    collection = get_chroma_collection(chroma_path, collection_name)
    result_count = min(n_results, collection.count())

    if result_count < 1:
        raise ValueError(
            f"Collection '{collection_name}' is empty. Build the database before searching."
        )

    return collection.query(
        query_texts=[query],
        n_results=result_count,
        include=cast(Include, ["documents", "metadatas", "distances"]),
    )


def first_query_batch(
    result: QueryResult,
) -> tuple[list[str], list[Metadata], list[float], list[str]]:
    ids = result["ids"][0] if result["ids"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []
    documents = result["documents"][0] if result["documents"] else []

    return ids, metadatas, distances, documents


def iter_query_hits(result: QueryResult) -> Iterable[QueryHit]:
    ids, metadatas, distances, documents = first_query_batch(result)

    for paper_id, metadata, distance, document in zip(
        ids,
        metadatas,
        distances,
        documents,
    ):
        yield {
            "id": paper_id,
            "distance": distance,
            "metadata": metadata,
            "document": document,
        }


def print_query_results(result: QueryResult) -> None:
    for rank, hit in enumerate(iter_query_hits(result), start=1):
        metadata = hit["metadata"]
        print(f"{rank}. {metadata.get('title', 'Untitled')}")
        print(f"   arXiv: {metadata.get('arxiv_id', hit['id'])}")
        print(f"   category: {metadata.get('primary_category', '')}")
        print(f"   distance: {hit['distance']:.4f}")
        print(f"   url: {metadata.get('abstract_page', '')}")


def positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error

    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")

    return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and query a local Chroma DB for medical segmentation arXiv papers."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    build_parser.add_argument("--db", type=Path, default=DEFAULT_CHROMA_PATH)
    build_parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    build_parser.add_argument("--batch-size", type=positive_int, default=100)
    build_parser.add_argument("--limit", type=positive_int)
    build_parser.add_argument("--reset", action="store_true")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--db", type=Path, default=DEFAULT_CHROMA_PATH)
    search_parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    search_parser.add_argument("--n-results", type=positive_int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        if args.command == "build":
            collection = build_vector_database(
                catalog_path=args.catalog,
                chroma_path=args.db,
                collection_name=args.collection,
                batch_size=args.batch_size,
                limit=args.limit,
                reset=args.reset,
            )
            print(
                f"Collection '{args.collection}' contains {collection.count()} papers."
            )
            print(f"Chroma DB path: {args.db}")
            return

        result = query_vector_database(
            query=args.query,
            chroma_path=args.db,
            collection_name=args.collection,
            n_results=args.n_results,
        )
        print_query_results(result)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
