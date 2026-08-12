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
MAX_RESULTS = 500
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
OUTPUT_DIR = "data"
MERGED_OUTPUT_FILENAME = "arxiv_papers_catalog.json"
RECOMMENDED_CATEGORIES = {
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.DB",
    "cs.IR",
    "cs.LG",
    "cs.MA",
    "cs.NE",
    "cs.RO",
    "cs.SE",
    "eess.AS",
    "eess.IV",
    "stat.ML",
}


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


def validate_search_query(search_query: str) -> bool:
    """Catch common arXiv category typos before calling the API."""

    if not search_query.startswith("cat:"):
        return True

    category = search_query.removeprefix("cat:")

    if category in RECOMMENDED_CATEGORIES:
        return True

    if category.startswith("cd."):
        print(f"Category '{category}' does not exist. Did you mean 'cs.AI'?")
        return False

    print(f"Warning: '{category}' is not in the recommended category list.")
    print(
        "Useful categories:",
        ", ".join(sorted(RECOMMENDED_CATEGORIES)),
    )
    return True


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Remove the arXiv version suffix from an ID."""

    return re.sub(r"v\d+$", "", arxiv_id)


def get_paper_key(paper: dict) -> str:
    """Return the stable key used to deduplicate papers."""

    arxiv_id = paper.get("base_arxiv_id") or paper.get("arxiv_id", "")
    return normalize_arxiv_id(arxiv_id)


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
    base_arxiv_id = normalize_arxiv_id(arxiv_id)

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
        "base_arxiv_id": base_arxiv_id,
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


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers using the arXiv ID without version."""

    seen_ids = set()
    unique_papers = []

    for paper in papers:
        paper_id = get_paper_key(paper)

        if paper_id in seen_ids:
            continue

        seen_ids.add(paper_id)
        unique_papers.append(paper)

    return unique_papers


def load_existing_papers(output_dir: str) -> list[dict]:
    """Load papers from JSON files in the data folder."""

    if not os.path.isdir(output_dir):
        return []

    papers = []

    for filename in os.listdir(output_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(output_dir, filename)

        with open(path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        for paper in data.get("papers", []):
            if "base_arxiv_id" not in paper:
                paper["base_arxiv_id"] = normalize_arxiv_id(
                    paper.get("arxiv_id", "")
                )

            papers.append(paper)

    return papers


def make_output_filename(
    search_query: str,
    article_count: int,
    start: int = 0,
) -> str:
    """Create a simple and safe JSON filename."""

    if search_query.startswith("cat:"):
        filename_keyword = search_query.removeprefix("cat:")
    elif search_query.startswith("all:"):
        filename_keyword = search_query.removeprefix("all:")
    else:
        filename_keyword = search_query

    clean_keyword = re.sub(r"[^a-zA-Z0-9_-]+", "_", filename_keyword.strip())
    clean_keyword = clean_keyword.strip("_").lower() or "arxiv"

    if start > 0:
        return f"{clean_keyword}_start_{start}_{article_count}.json"

    return f"{clean_keyword}_{article_count}.json"


def ask_start_index() -> int:
    """Ask which arXiv result page should be fetched."""

    value = input(
        "Start index ? Mets 0 pour les premiers 500, "
        "500 pour les suivants, 1000 apres, etc. "
    ).strip()

    if not value:
        return START

    try:
        start = int(value)
    except ValueError:
        print("Start invalide. J'utilise 0.")
        return START

    if start < 0:
        print("Start ne peut pas etre negatif. J'utilise 0.")
        return START

    return start


def main() -> None:
    keyword = input(
        "Quel domaine ou mot-cle veux-tu chercher sur arXiv ? "
        "(ex: TransUnet, artificial intelligence, cat:cs.AI) "
    ).strip()

    if not keyword:
        print("Aucun mot-cle donne. Le script s'arrete.")
        return

    search_query = build_search_query(keyword)

    if not validate_search_query(search_query):
        return

    start_index = ask_start_index()
    query_url = build_query_url(
        search_query=search_query,
        start=start_index,
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

    papers_before_deduplication = [extract_entry(entry) for entry in feed.entries]
    papers = deduplicate_papers(papers_before_deduplication)
    duplicates_removed = len(papers_before_deduplication) - len(papers)

    output_data = {
        "keyword": keyword,
        "search_query": search_query,
        "query_url": query_url,
        "metadata": extract_feed_metadata(feed),
        "start": start_index,
        "requested_results": MAX_RESULTS,
        "received_results": len(papers_before_deduplication),
        "saved_results": len(papers),
        "duplicates_removed": duplicates_removed,
        "papers": papers,
    }

    output_filename = make_output_filename(search_query, len(papers), start_index)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    merged_output_path = os.path.join(OUTPUT_DIR, MERGED_OUTPUT_FILENAME)
    unique_count_before_save = len(deduplicate_papers(load_existing_papers(OUTPUT_DIR)))

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=2)

    all_papers = load_existing_papers(OUTPUT_DIR)
    merged_papers = deduplicate_papers(all_papers)
    global_duplicates_removed = len(all_papers) - len(merged_papers)
    new_unique_articles = len(merged_papers) - unique_count_before_save

    output_data["global_unique_articles"] = len(merged_papers)
    output_data["global_duplicates_removed"] = global_duplicates_removed
    output_data["new_unique_articles_added"] = new_unique_articles

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=2)

    merged_output_data = {
        "total_unique_articles": len(merged_papers),
        "total_source_articles": len(all_papers),
        "duplicates_removed": global_duplicates_removed,
        "papers": merged_papers,
    }

    with open(merged_output_path, "w", encoding="utf-8") as json_file:
        json.dump(merged_output_data, json_file, ensure_ascii=False, indent=2)

    print(f"{len(papers)} articles sauvegardes dans {output_path}")
    print(f"{duplicates_removed} doublons retires dans cette recherche")
    print(f"{new_unique_articles} nouveaux articles uniques ajoutes au merge")
    print(
        f"{len(merged_papers)} articles uniques disponibles dans "
        f"{merged_output_path}"
    )


if __name__ == "__main__":
    main()
