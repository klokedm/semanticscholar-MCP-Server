import semanticscholar as sch
from semanticscholar import SemanticScholar, Author, Paper
from semanticscholar.SemanticScholarException import (
    InternalServerErrorException,
    GatewayTimeoutException,
)
from typing import List, Dict, Any, Callable
import os
import logging
import random
import threading
import time

logger = logging.getLogger(__name__)

REQUEST_INTERVAL_SECONDS = 1.0  # Semantic Scholar rate limit: 1 request per second
MAX_RETRIES = int(os.getenv("S2_MAX_RETRIES", "5"))
BACKOFF_BASE_SECONDS = float(os.getenv("S2_BACKOFF_BASE_SECONDS", "1.0"))
BACKOFF_MAX_SECONDS = float(os.getenv("S2_BACKOFF_MAX_SECONDS", "32.0"))
BACKOFF_JITTER_SECONDS = float(os.getenv("S2_BACKOFF_JITTER_SECONDS", "0.25"))

_rate_limit_lock = threading.Lock()
_next_request_time = 0.0


def _wait_for_request_slot() -> None:
    """Reserve and wait for the next allowed request slot (global 1 RPS)."""
    global _next_request_time

    wait_seconds = 0.0
    with _rate_limit_lock:
        now = time.monotonic()
        reserved_time = max(now, _next_request_time)
        wait_seconds = reserved_time - now
        _next_request_time = reserved_time + REQUEST_INTERVAL_SECONDS

    if wait_seconds > 0:
        time.sleep(wait_seconds)


def _is_retriable_error(error: Exception) -> bool:
    if isinstance(error, (ConnectionRefusedError, TimeoutError)):
        return True

    if isinstance(error, (InternalServerErrorException, GatewayTimeoutException)):
        return True

    message = str(error).lower()
    retriable_tokens = (
        "429",
        "too many requests",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "502",
        "503",
        "504",
    )
    return any(token in message for token in retriable_tokens)


def _call_with_rate_limit_and_backoff(
    operation: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Execute an API operation under 1 RPS + exponential backoff retries."""
    for attempt in range(MAX_RETRIES + 1):
        _wait_for_request_slot()
        try:
            return operation(*args, **kwargs)
        except Exception as error:
            should_retry = attempt < MAX_RETRIES and _is_retriable_error(error)
            if not should_retry:
                raise

            backoff = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** attempt))
            jitter = random.uniform(0.0, BACKOFF_JITTER_SECONDS)
            delay = backoff + jitter
            logger.warning(
                "Semantic Scholar request failed with retriable error (%s). "
                "Retrying in %.2fs (%d/%d).",
                error,
                delay,
                attempt + 1,
                MAX_RETRIES,
            )
            time.sleep(delay)

def initialize_client() -> SemanticScholar:
    """Initialize the SemanticScholar client."""
    api_key = os.getenv("S2_API_KEY", "").strip()
    if api_key:
        return SemanticScholar(api_key=api_key, retry=False)
    return SemanticScholar(retry=False)

def search_papers(client: SemanticScholar, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for papers using a query string."""
    results = _call_with_rate_limit_and_backoff(client.search_paper, query, limit=limit)
    return [
        {
            "paperId": paper.paperId,
            "title": paper.title,
            "abstract": paper.abstract,
            "year": paper.year,
            "authors": [{"name": author.name, "authorId": author.authorId} for author in paper.authors],
            "url": paper.url,
            "venue": paper.venue,
            "publicationTypes": paper.publicationTypes,
            "citationCount": paper.citationCount
        } for paper in results
    ]

def get_paper_details(client: SemanticScholar, paper_id: str) -> Paper:
    """Get details of a specific paper."""
    return _call_with_rate_limit_and_backoff(client.get_paper, paper_id)

def get_author_details(client: SemanticScholar, author_id: str) -> Author:
    """Get details of a specific author."""
    return _call_with_rate_limit_and_backoff(client.get_author, author_id)

def get_citations_and_references(paper: Paper) -> Dict[str, List[Dict[str, Any]]]:
    """Get citations and references for a paper."""
    return {
        "citations": paper.citations,
        "references": paper.references
    }

def main():
    try:
        # Initialize the client
        client = initialize_client()

        # Search for papers
        search_results = search_papers(client, "machine learning")
        print(f"Search results: {search_results[:2]}")  # Print first 2 results

        # Get paper details
        if search_results:
            paper_id = search_results[0]['paperId']
            paper = get_paper_details(client, paper_id)
            print(f"Paper details: {paper}")

            # Get citations and references
            citations_refs = get_citations_and_references(paper)
            print(f"Citations: {citations_refs['citations'][:2]}")  # Print first 2 citations
            print(f"References: {citations_refs['references'][:2]}")  # Print first 2 references

        # Get author details
        author_id = "1741101"  # Example author ID
        author = get_author_details(client, author_id)
        print(f"Author details: {author}")

    except sch.SemanticScholarException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
