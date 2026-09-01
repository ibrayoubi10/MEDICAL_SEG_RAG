# Author: Ibrahim M. AlAyoubi

"""Collect arXiv metadata for a medical image segmentation RAG corpus."""

import argparse
import json
import os
import re
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import feedparser


BASE_URL = "https://export.arxiv.org/api/query"

START = 0
MAX_RESULTS = 500
DEFAULT_MEDICAL_TARGET_ARTICLES = 3500
DEFAULT_MEDICAL_RESULTS_PER_QUERY = 200
DEFAULT_MEDICAL_PAGES_PER_QUERY = 5
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
REQUEST_DELAY_SECONDS = 3
OUTPUT_DIR = "data"
MEDICAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "medical_segmentation")
MERGED_OUTPUT_FILENAME = "arxiv_papers_catalog.json"
MEDICAL_CATEGORY_FILTER = (
    "(cat:cs.CV OR cat:cs.LG OR cat:cs.AI OR cat:eess.IV "
    "OR cat:eess.SP OR cat:stat.ML OR cat:q-bio.QM OR cat:q-bio.NC)"
)
MEDICAL_SEGMENTATION_SEARCHES = [
    (
        "medical_image_segmentation",
        f"(all:medical AND all:image AND all:segmentation) AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "unet_medical_image_segmentation",
        f"(all:U-Net OR all:UNet OR all:unet) AND all:medical "
        f"AND all:segmentation AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "nnunet",
        f"(all:nnU-Net OR all:nnUNet OR all:nnunet) AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "cnn_vs_transformer_segmentation",
        f"(all:CNN OR all:convolutional OR all:transformer) AND all:medical "
        f"AND all:segmentation AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "transunet_unetr_swin_unetr",
        f"(all:TransUNet OR all:UNETR OR (all:Swin AND all:UNETR)) "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "medical_data_augmentation",
        f"(all:medical AND all:data AND all:augmentation) AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "diffusion_medical_augmentation",
        f"(all:diffusion AND all:medical AND all:augmentation) "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "semi_supervised_medical_segmentation",
        f"(all:semi-supervised OR (all:semi AND all:supervised)) "
        f"AND all:medical AND all:segmentation "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "weakly_supervised_medical_segmentation",
        f"(all:weakly-supervised OR (all:weakly AND all:supervised)) "
        f"AND all:medical AND all:segmentation "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "self_supervised_medical_image_analysis",
        f"(all:self-supervised OR (all:self AND all:supervised)) "
        f"AND all:medical AND all:image "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "domain_adaptation_generalization_medical_segmentation",
        f"(all:domain AND (all:adaptation OR all:generalization) "
        f"AND all:medical AND all:segmentation) "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "federated_learning_medical_segmentation",
        f"(all:federated AND all:learning AND all:medical AND all:segmentation) "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "three_dimensional_medical_segmentation",
        f"(all:3D OR all:volumetric) AND all:medical AND all:segmentation "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "mri_ct_ultrasound_xray_segmentation",
        f"(all:MRI OR all:CT OR all:ultrasound OR all:X-ray OR all:radiography) "
        f"AND all:segmentation AND all:deep AND all:learning "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "evaluation_metrics_medical_segmentation",
        f"(all:Dice OR all:Hausdorff OR all:IoU) AND all:medical "
        f"AND all:segmentation AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "uncertainty_medical_segmentation",
        f"(all:uncertainty AND all:medical AND all:segmentation) "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "explainable_ai_medical_imaging",
        f"(all:explainable OR all:interpretability OR all:XAI) "
        f"AND all:medical AND all:imaging "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
    (
        "foundation_models_sam_medsam_medical_sam2",
        f"(all:SAM OR all:MedSAM OR (all:Medical AND all:SAM) "
        f"OR (all:Segment AND all:Anything)) AND all:medical "
        f"AND {MEDICAL_CATEGORY_FILTER}",
    ),
]
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
    "eess.SP",
    "q-bio.NC",
    "q-bio.QM",
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


def load_existing_papers(
    output_dir: str,
    ignored_filenames: set[str] | None = None,
) -> list[dict]:
    """Load papers from JSON files in the data folder."""

    if not os.path.isdir(output_dir):
        return []

    ignored_filenames = ignored_filenames or {MERGED_OUTPUT_FILENAME}
    papers = []

    for filename in os.listdir(output_dir):
        if filename in ignored_filenames:
            continue

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


def fetch_papers_for_query(
    search_query: str,
    start_index: int = START,
    max_results: int = MAX_RESULTS,
) -> tuple[str, feedparser.FeedParserDict, list[dict]]:
    """Fetch and extract one arXiv result page."""

    query_url = build_query_url(
        search_query=search_query,
        start=start_index,
        max_results=max_results,
    )

    feed = fetch_arxiv_feed(query_url)
    papers = [extract_entry(entry) for entry in feed.entries]

    return query_url, feed, papers


def save_query_results(
    output_dir: str,
    output_filename: str,
    output_data: dict,
) -> str:
    """Save one query result file and return its path."""

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=2)

    return output_path


def save_merged_catalog(
    output_dir: str,
    merged_papers: list[dict],
    total_source_articles: int,
    query_summaries: list[dict] | None = None,
) -> str:
    """Save the deduplicated catalog used by the next RAG steps."""

    merged_output_path = os.path.join(output_dir, MERGED_OUTPUT_FILENAME)
    merged_output_data = {
        "total_unique_articles": len(merged_papers),
        "total_source_articles": total_source_articles,
        "duplicates_removed": total_source_articles - len(merged_papers),
        "query_summaries": query_summaries or [],
        "papers": merged_papers,
    }

    with open(merged_output_path, "w", encoding="utf-8") as json_file:
        json.dump(merged_output_data, json_file, ensure_ascii=False, indent=2)

    return merged_output_path


def collect_single_query() -> None:
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
    query_url = build_query_url(search_query, start_index, MAX_RESULTS)

    print(f"Query URL: {query_url}\n")

    try:
        query_url, feed, papers_before_deduplication = fetch_papers_for_query(
            search_query=search_query,
            start_index=start_index,
            max_results=MAX_RESULTS,
        )
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
    unique_count_before_save = len(deduplicate_papers(load_existing_papers(OUTPUT_DIR)))

    output_path = save_query_results(OUTPUT_DIR, output_filename, output_data)

    all_papers = load_existing_papers(OUTPUT_DIR)
    merged_papers = deduplicate_papers(all_papers)
    global_duplicates_removed = len(all_papers) - len(merged_papers)
    new_unique_articles = len(merged_papers) - unique_count_before_save

    output_data["global_unique_articles"] = len(merged_papers)
    output_data["global_duplicates_removed"] = global_duplicates_removed
    output_data["new_unique_articles_added"] = new_unique_articles

    save_query_results(OUTPUT_DIR, output_filename, output_data)
    merged_output_path = save_merged_catalog(
        OUTPUT_DIR,
        merged_papers,
        total_source_articles=len(all_papers),
    )

    print(f"{len(papers)} articles sauvegardes dans {output_path}")
    print(f"{duplicates_removed} doublons retires dans cette recherche")
    print(f"{new_unique_articles} nouveaux articles uniques ajoutes au merge")
    print(
        f"{len(merged_papers)} articles uniques disponibles dans "
        f"{merged_output_path}"
    )


def collect_medical_segmentation_corpus(
    target_articles: int = DEFAULT_MEDICAL_TARGET_ARTICLES,
    results_per_query: int = DEFAULT_MEDICAL_RESULTS_PER_QUERY,
    pages_per_query: int = DEFAULT_MEDICAL_PAGES_PER_QUERY,
    output_dir: str = MEDICAL_OUTPUT_DIR,
    request_delay_seconds: int = REQUEST_DELAY_SECONDS,
) -> None:
    """Build a specialized corpus for medical image segmentation RAG."""

    if target_articles < 1:
        raise ValueError("target_articles must be at least 1")

    if results_per_query < 1:
        raise ValueError("results_per_query must be at least 1")

    if pages_per_query < 1:
        raise ValueError("pages_per_query must be at least 1")

    results_per_query = min(results_per_query, MAX_RESULTS)
    os.makedirs(output_dir, exist_ok=True)

    all_papers = load_existing_papers(output_dir)
    merged_papers = deduplicate_papers(all_papers)
    query_summaries = []

    print(
        "Corpus cible: medical image segmentation, U-Net, augmentation, "
        "deep learning medical imaging."
    )
    print(f"Objectif: {target_articles} articles uniques")
    print(f"Articles demandes par theme et par page: {results_per_query}")
    print(f"Pages maximum par theme: {pages_per_query}")
    print(f"Deja presents dans {output_dir}: {len(merged_papers)} articles uniques\n")

    for page in range(pages_per_query):
        if len(merged_papers) >= target_articles:
            break

        start_index = page * results_per_query
        print(f"=== Page {page + 1}/{pages_per_query} | start={start_index} ===")

        for index, (label, search_query) in enumerate(
            MEDICAL_SEGMENTATION_SEARCHES,
            start=1,
        ):
            if len(merged_papers) >= target_articles:
                break

            print(f"[{index}/{len(MEDICAL_SEGMENTATION_SEARCHES)}] {label}")
            print(f"Search query: {search_query}")

            try:
                query_url, feed, fetched_papers = fetch_papers_for_query(
                    search_query=search_query,
                    start_index=start_index,
                    max_results=results_per_query,
                )
            except RuntimeError as error:
                print(f"Erreur pour {label}, page {page + 1}: {error}")
                continue

            if not fetched_papers:
                print("Aucun article trouve pour cette page.\n")
                continue

            query_papers = deduplicate_papers(fetched_papers)
            unique_count_before_query = len(merged_papers)
            all_papers.extend(query_papers)
            merged_papers = deduplicate_papers(all_papers)
            new_unique_articles = len(merged_papers) - unique_count_before_query
            duplicates_removed = len(fetched_papers) - len(query_papers)

            output_data = {
                "keyword": label,
                "search_query": search_query,
                "query_url": query_url,
                "metadata": extract_feed_metadata(feed),
                "start": start_index,
                "requested_results": results_per_query,
                "received_results": len(fetched_papers),
                "saved_results": len(query_papers),
                "duplicates_removed": duplicates_removed,
                "new_unique_articles_added": new_unique_articles,
                "global_unique_articles": len(merged_papers),
                "papers": query_papers,
            }

            output_filename = f"{label}_start_{start_index}_{len(query_papers)}.json"
            output_path = save_query_results(output_dir, output_filename, output_data)

            query_summaries.append(
                {
                    "label": label,
                    "search_query": search_query,
                    "start": start_index,
                    "received_results": len(fetched_papers),
                    "saved_results": len(query_papers),
                    "new_unique_articles_added": new_unique_articles,
                    "output_path": output_path,
                }
            )

            merged_output_path = save_merged_catalog(
                output_dir,
                merged_papers,
                total_source_articles=len(all_papers),
                query_summaries=query_summaries,
            )

            print(f"{len(query_papers)} articles sauvegardes dans {output_path}")
            print(f"{new_unique_articles} nouveaux articles uniques ajoutes")
            print(f"Total actuel: {len(merged_papers)} articles uniques")
            print(f"Catalogue merge: {merged_output_path}\n")

            if len(merged_papers) >= target_articles:
                break

            time.sleep(request_delay_seconds)

    merged_output_path = save_merged_catalog(
        output_dir,
        merged_papers,
        total_source_articles=len(all_papers),
        query_summaries=query_summaries,
    )

    print("Collecte terminee.")
    print(f"Articles uniques: {len(merged_papers)}")
    print(f"Catalogue final: {merged_output_path}")

    if len(merged_papers) < target_articles:
        print(
            "Objectif non atteint. Ajoute d'autres requetes specialisees "
            "ou augmente --results-per-query."
        )


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
        description="Collect arXiv papers for the RAG project."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("single", help="Run one interactive arXiv search.")

    medical_parser = subparsers.add_parser(
        "medical",
        help="Build a medical image segmentation corpus.",
    )
    medical_parser.add_argument(
        "--target",
        type=positive_int,
        default=DEFAULT_MEDICAL_TARGET_ARTICLES,
        help="Target number of unique papers.",
    )
    medical_parser.add_argument(
        "--results-per-query",
        type=positive_int,
        default=DEFAULT_MEDICAL_RESULTS_PER_QUERY,
        help=f"Number of arXiv results per themed query, max {MAX_RESULTS}.",
    )
    medical_parser.add_argument(
        "--pages-per-query",
        type=positive_int,
        default=DEFAULT_MEDICAL_PAGES_PER_QUERY,
        help="Number of arXiv result pages to try for each themed query.",
    )
    medical_parser.add_argument(
        "--output-dir",
        default=MEDICAL_OUTPUT_DIR,
        help="Directory where query files and the merged catalog are saved.",
    )
    medical_parser.add_argument(
        "--delay",
        type=positive_int,
        default=REQUEST_DELAY_SECONDS,
        help="Seconds to wait between arXiv API requests.",
    )

    return parser.parse_args()


def main() -> None:
    if len(sys.argv) == 1:
        collect_single_query()
        return

    args = parse_args()

    if args.command == "medical":
        collect_medical_segmentation_corpus(
            target_articles=args.target,
            results_per_query=args.results_per_query,
            pages_per_query=args.pages_per_query,
            output_dir=args.output_dir,
            request_delay_seconds=args.delay,
        )
        return

    collect_single_query()


if __name__ == "__main__":
    main()
