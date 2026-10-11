import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Create searchable terms for Chinese text and alphanumeric tokens."""
    terms: List[str] = []
    for chunk in _TOKEN_PATTERN.findall((text or "").lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            terms.append(chunk)
            terms.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))
        else:
            terms.append(chunk)
    return terms


def bm25_search(
    query: str,
    documents: Dict[int, Dict[str, Any]],
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Dict[str, Any]]:
    """Search the loaded medical corpus with a transparent local BM25 scorer."""
    query_terms = set(tokenize(query))
    if not query_terms or not documents:
        return []

    prepared = []
    document_frequency = Counter()
    total_length = 0
    for doc_id, doc in documents.items():
        title = str(doc.get("title", ""))
        content = str(doc.get("abstract", doc.get("content", "")))
        terms = tokenize(f"{title} {content}")
        counts = Counter(terms)
        prepared.append((doc_id, title, content, counts))
        total_length += len(terms)
        document_frequency.update(set(terms))

    average_length = max(total_length / max(len(prepared), 1), 1.0)
    scored = []
    for doc_id, title, content, counts in prepared:
        document_length = sum(counts.values())
        score = 0.0
        title_text = title.lower()
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1 + (len(prepared) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * document_length / average_length
            )
            score += idf * frequency * (k1 + 1) / denominator
            if term in title_text:
                score += 0.25 * idf

        if score > 0:
            scored.append(
                {
                    "doc_id": doc_id,
                    "score": round(score, 6),
                    "matched_terms": sorted(term for term in query_terms if counts.get(term)),
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def merge_lexical_hits(
    queries: Iterable[str],
    documents: Dict[int, Dict[str, Any]],
    top_k: int,
) -> Dict[int, Dict[str, Any]]:
    """Merge lexical hits while retaining ranks from every rewritten query."""
    merged: Dict[int, Dict[str, Any]] = {}
    for query in queries:
        for rank, hit in enumerate(bm25_search(query, documents, top_k), 1):
            current = merged.get(hit["doc_id"])
            candidate = {
                **hit,
                "rank": rank,
                "matched_query": query,
                "ranks": [rank],
                "matched_queries": [query],
            }
            if current is None:
                merged[hit["doc_id"]] = candidate
                continue

            current["ranks"] = sorted(set(current.get("ranks", []) + [rank]))
            current["matched_queries"] = list(
                dict.fromkeys(current.get("matched_queries", []) + [query])
            )
            if candidate["score"] > current["score"] or (
                candidate["score"] == current["score"]
                and candidate["rank"] < current["rank"]
            ):
                current.update(
                    {
                        "score": candidate["score"],
                        "matched_terms": candidate["matched_terms"],
                        "rank": candidate["rank"],
                        "matched_query": candidate["matched_query"],
                    }
                )
    return merged
