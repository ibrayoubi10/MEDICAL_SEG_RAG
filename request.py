# Author: Ibrahim M. AlAyoubi

"""Query the arXiv API and save paper metadata to a JSON file."""

import json
import os
import re
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import feedparser


BASE_URL = "https://export.arxiv.org/api/query"

START = 0
MAX_RESULTS = 50
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
OUTPUT_DIR = "data"


def build_query_url(
    search_query: str,
    start: int = 0,
    max_results: int = 5,
) -> str:
    """Build an encoded arXiv API query URL."""

    parameters = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
    }

    return f"{BASE_URL}?{urlencode(parameters)}"


def build_search_query(keyword: str) -> str:
    """Build a simple arXiv search query from user input."""

    if keyword.startswith("cat:"):
        return keyword

    if re.fullmatch(r"[a-z-]+\.[A-Z]{2}", keyword):
        return f"cat:{keyword}"

    return f"all:{keyword}"


def fetch_arxiv_feed(url: str) -> feedparser.FeedParserDict:
    """Download and parse the arXiv Atom feed."""

    for attempt in range(1, MAX_RETRIES + 1):
        request = Request(
            url,
            headers={
                "User-Agent": "arxiv-rag-learning-project/1.0",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                content = response.read()
                break

        except HTTPError as error:
            if error.code == 429 and attempt < MAX_RETRIES:
                print(
                    "arXiv limite temporairement les requetes. "
                    f"Nouvel essai dans {RETRY_DELAY_SECONDS} secondes..."
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            raise RuntimeError(
                f"arXiv returned HTTP error {error.code}: {error.reason}"
            ) from error

        except URLError as error:
            raise RuntimeError(
                f"Unable to connect to arXiv: {error.reason}"
            ) from error

        except socket.timeout as error:
            if attempt < MAX_RETRIES:
                print(
                    "arXiv met trop longtemps a repondre. "
                    f"Nouvel essai dans {RETRY_DELAY_SECONDS} secondes..."
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            raise RuntimeError("arXiv request timed out") from error

    feed = feedparser.parse(content)

    if feed.bozo:
        print(f"Warning: feed parsing issue: {feed.bozo_exception}")

    return feed


def extract_feed_metadata(feed: feedparser.FeedParserDict) -> dict:
    """Extract general metadata about the API response."""

    return {
        "feed_title": feed.feed.get("title", "Unknown"),
        "feed_last_updated": feed.feed.get("updated", "Unknown"),
        "total_results": feed.feed.get("opensearch_totalresults", "Unknown"),
        "items_per_page": feed.feed.get("opensearch_itemsperpage", "Unknown"),
        "start_index": feed.feed.get("opensearch_startindex", "Unknown"),
    }


def extract_entry(entry: feedparser.FeedParserDict) -> dict:
    """Extract metadata for one arXiv paper."""

    arxiv_id = entry.get("id", "").split("/abs/")[-1]

    authors = [
        author.get("name", "Unknown")
        for author in entry.get("authors", [])
    ]

    abstract_url = None
    pdf_url = None

    for link in entry.get("links", []):
        if link.get("rel") == "alternate":
            abstract_url = link.get("href")

        if (
            link.get("title") == "pdf"
            or link.get("type") == "application/pdf"
        ):
            pdf_url = link.get("href")

    categories = [
        tag.get("term")
        for tag in entry.get("tags", [])
        if tag.get("term")
    ]

    primary_category = entry.get("arxiv_primary_category", {})

    if isinstance(primary_category, dict):
        primary_category = primary_category.get("term")

    # Fallback to the first category when the namespaced field is absent.
    if not primary_category and categories:
        primary_category = categories[0]

    abstract = entry.get("summary", "No abstract found")
    abstract = " ".join(abstract.split())

    return {
        "arxiv_id": arxiv_id,
        "published": entry.get("published", "Unknown"),
        "updated": entry.get("updated", "Unknown"),
        "title": entry.get("title", "Unknown").strip(),
        "authors": authors,
        "affiliation": entry.get("arxiv_affiliation"),
        "abstract_page": abstract_url,
        "pdf": pdf_url,
        "journal_reference": entry.get("arxiv_journal_ref"),
        "comments": entry.get("arxiv_comment"),
        "primary_category": primary_category,
        "all_categories": categories,
        "abstract": abstract,
    }


def make_output_filename(keyword: str, article_count: int) -> str:
    """Create a simple and safe JSON filename."""

    clean_keyword = re.sub(r"[^a-zA-Z0-9_-]+", "_", keyword.strip())
    clean_keyword = clean_keyword.strip("_").lower() or "arxiv"
    return f"{clean_keyword}_{article_count}.json"


def main() -> None:
    keyword = input(
        "Quel domaine ou mot-cle veux-tu chercher sur arXiv ? "
        "(ex: TransUnet, artificial intelligence, cat:cs.AI) "
    ).strip()

    if not keyword:
        print("Aucun mot-cle donne. Le script s'arrete.")
        return

    search_query = build_search_query(keyword)
    query_url = build_query_url(
        search_query=search_query,
        start=START,
        max_results=MAX_RESULTS,
    )

    print(f"Query URL: {query_url}\n")

    try:
        feed = fetch_arxiv_feed(query_url)
    except RuntimeError as error:
        print(f"Erreur: {error}")
        print(
            "Essaie un mot-cle plus precis, attends quelques secondes, "
            "ou utilise une categorie comme cat:cs.AI."
        )
        return

    if not feed.entries:
        print("\nNo papers were found.")
        return

    papers = [extract_entry(entry) for entry in feed.entries]
    output_data = {
        "keyword": keyword,
        "search_query": search_query,
        "query_url": query_url,
        "metadata": extract_feed_metadata(feed),
        "papers": papers,
    }

    output_filename = make_output_filename(keyword, len(papers))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=2)

    print(f"{len(papers)} articles sauvegardes dans {output_path}")


if __name__ == "__main__":
    main()
