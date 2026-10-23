from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.research.store import ResearchItem

METHODS = (
    "bm25",
    "dense_text",
    "clip_visual",
    "hybrid_rrf",
    "hybrid_mmr",
)
RRF_K = 60
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class QueryEncoder(Protocol):
    def embed_text_query(self, text: str, model: str = "dense") -> list[float]: ...


@dataclass(frozen=True, slots=True)
class RankedItem:
    item: ResearchItem
    score: float


class ResearchRetriever:
    def __init__(self, index: QueryEncoder) -> None:
        self.index = index

    def search(
        self,
        items: Sequence[ResearchItem],
        query: str,
        method: str,
        top_k: int = 5,
    ) -> list[RankedItem]:
        if method not in METHODS:
            raise ValueError(f"Unknown retrieval method {method!r}; choose from {METHODS}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        text_items = [item for item in items if item.kind == "transcript" and item.text.strip()]
        image_items = [
            item for item in items if item.kind == "keyframe" and item.clip_vector is not None
        ]
        if method == "bm25":
            scores = _bm25_scores(text_items, query)
            return _rank(
                {item_id: pair for item_id, pair in scores.items() if pair[1] > 0},
                top_k,
            )
        if method == "dense_text":
            query_vector = self.index.embed_text_query(query, model="dense")
            return _rank(_cosine_scores(text_items, query_vector, "dense_vector"), top_k)
        if method == "clip_visual":
            query_vector = self.index.embed_text_query(query, model="clip")
            return _rank(_cosine_scores(image_items, query_vector, "clip_vector"), top_k)

        dense_vector = self.index.embed_text_query(query, model="dense")
        clip_vector = self.index.embed_text_query(query, model="clip")
        rankings = [
            _rank(
                {
                    item_id: result
                    for item_id, result in _bm25_scores(text_items, query).items()
                    if result[1] > 0
                },
                max(top_k * 3, 20),
            ),
            _rank(_cosine_scores(text_items, dense_vector, "dense_vector"), max(top_k * 3, 20)),
            _rank(_cosine_scores(image_items, clip_vector, "clip_vector"), max(top_k * 3, 20)),
        ]
        fused = _rrf(rankings)
        if method == "hybrid_mmr":
            return _temporal_mmr(fused, top_k)
        return fused[:top_k]


def _bm25_scores(
    items: Sequence[ResearchItem], query: str
) -> dict[str, tuple[ResearchItem, float]]:
    tokens_by_item = {str(item.id): TOKEN_PATTERN.findall(item.text.casefold()) for item in items}
    if not tokens_by_item:
        return {}
    document_frequency: Counter[str] = Counter()
    for tokens in tokens_by_item.values():
        document_frequency.update(set(tokens))
    average_length = sum(map(len, tokens_by_item.values())) / len(tokens_by_item)
    query_terms = TOKEN_PATTERN.findall(query.casefold())
    count = len(tokens_by_item)
    scores: dict[str, tuple[ResearchItem, float]] = {}
    for item in items:
        tokens = tokens_by_item[str(item.id)]
        term_counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_counts[term]
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1 + (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            length_normalization = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(tokens) / max(average_length, 1)
            )
            score += inverse_document_frequency * frequency * 2.5 / length_normalization
        scores[str(item.id)] = (item, score)
    return scores


def _cosine_scores(
    items: Sequence[ResearchItem],
    query_vector: Sequence[float],
    vector_field: str,
) -> dict[str, tuple[ResearchItem, float]]:
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    scores: dict[str, tuple[ResearchItem, float]] = {}
    for item in items:
        vector = getattr(item, vector_field)
        if vector is None:
            continue
        vector_norm = math.sqrt(sum(value * value for value in vector))
        similarity = (
            sum(left * right for left, right in zip(vector, query_vector, strict=True))
            / (vector_norm * query_norm)
            if vector_norm and query_norm and len(vector) == len(query_vector)
            else 0.0
        )
        scores[str(item.id)] = (item, similarity)
    return scores


def _rank(
    scores: dict[str, tuple[ResearchItem, float]],
    top_k: int,
) -> list[RankedItem]:
    return [
        RankedItem(item, score)
        for item, score in sorted(scores.values(), key=lambda pair: (-pair[1], str(pair[0].id)))[
            :top_k
        ]
    ]


def _rrf(rankings: Sequence[Sequence[RankedItem]]) -> list[RankedItem]:
    items_by_id: dict[str, ResearchItem] = {}
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            item_id = str(result.item.id)
            items_by_id[item_id] = result.item
            scores[item_id] += 1 / (RRF_K + rank)
    return [
        RankedItem(items_by_id[item_id], score)
        for item_id, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def _temporal_mmr(items: Sequence[RankedItem], top_k: int) -> list[RankedItem]:
    if not items:
        return []
    max_score = max(item.score for item in items) or 1.0
    remaining = list(items)
    selected: list[RankedItem] = []
    while remaining and len(selected) < top_k:
        best = max(
            remaining,
            key=lambda candidate: (
                0.7 * (candidate.score / max_score)
                - 0.3
                * max(
                    (_temporal_overlap(candidate.item, chosen.item) for chosen in selected),
                    default=0.0,
                ),
                -candidate.item.start_s,
                str(candidate.item.id),
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def _temporal_overlap(first: ResearchItem, second: ResearchItem) -> float:
    distance = max(
        0.0,
        max(first.start_s, second.start_s) - min(first.end_s, second.end_s),
    )
    return math.exp(-distance / 5.0)
