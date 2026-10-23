from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceSpan(BaseModel):
    video_id: UUID
    start_s: float = Field(ge=0, allow_inf_nan=False)
    end_s: float = Field(ge=0, allow_inf_nan=False)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_interval(self) -> EvidenceSpan:
        if self.end_s < self.start_s:
            raise ValueError("Evidence interval end_s must be greater than or equal to start_s")
        return self


class BenchmarkCase(BaseModel):
    id: str = Field(min_length=1)
    video_id: UUID
    question: str = Field(min_length=1)
    answerable: bool
    acceptable_answers: list[str] = Field(default_factory=list)
    answer_options: list[str] = Field(default_factory=list)
    gold_evidence: list[EvidenceSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_annotations(self) -> BenchmarkCase:
        if not self.question.strip():
            raise ValueError("Question cannot be empty")
        if self.answerable and not self.acceptable_answers:
            raise ValueError("Answerable cases require at least one acceptable answer")
        if self.answerable and not (self.gold_evidence or self.answer_options):
            raise ValueError("Answerable cases require gold_evidence or answer_options")
        if self.answer_options and (
            len(self.answer_options) < 2
            or len({_normalize(option) for option in self.answer_options})
            != len(self.answer_options)
        ):
            raise ValueError("Multiple-choice cases require at least two unique answer options")
        if self.answer_options and any(
            _normalize(answer) not in {_normalize(option) for option in self.answer_options}
            for answer in self.acceptable_answers
        ):
            raise ValueError("Acceptable answers must match one of the answer_options")
        if not self.answerable and (
            self.acceptable_answers or self.answer_options or self.gold_evidence
        ):
            raise ValueError(
                "Unanswerable cases must not have acceptable_answers, "
                "answer_options, or gold_evidence"
            )
        if any(span.video_id != self.video_id for span in self.gold_evidence):
            raise ValueError("Gold evidence must belong to the case video")
        return self


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    exact_match: float | None
    token_f1: float | None
    multiple_choice_accuracy: float | None
    retrieval_recall_at_1: float | None
    retrieval_recall_at_5: float | None
    retrieval_recall_at_10: float | None
    reciprocal_rank: float | None
    citation_precision: float | None
    citation_recall: float | None
    abstention_correct: float | None
    retrieved_count: int
    citation_count: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
            "multiple_choice_accuracy": self.multiple_choice_accuracy,
            "retrieval_recall_at_1": self.retrieval_recall_at_1,
            "retrieval_recall_at_5": self.retrieval_recall_at_5,
            "retrieval_recall_at_10": self.retrieval_recall_at_10,
            "reciprocal_rank": self.reciprocal_rank,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "abstention_correct": self.abstention_correct,
            "retrieved_count": self.retrieved_count,
            "citation_count": self.citation_count,
        }


def score_case(
    case: BenchmarkCase,
    answer: str,
    retrieved: Sequence[EvidenceSpan],
    citations: Sequence[EvidenceSpan],
    abstained: bool = False,
    *,
    iou_threshold: float = 0.5,
) -> CaseMetrics:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in the interval (0, 1]")
    if not case.answerable:
        return CaseMetrics(
            exact_match=None,
            token_f1=None,
            multiple_choice_accuracy=None,
            retrieval_recall_at_1=None,
            retrieval_recall_at_5=None,
            retrieval_recall_at_10=None,
            reciprocal_rank=None,
            citation_precision=None,
            citation_recall=None,
            abstention_correct=float(abstained),
            retrieved_count=len(retrieved),
            citation_count=len(citations),
        )

    accepted = [_normalize(text) for text in case.acceptable_answers]
    normalized_answer = _normalize(answer)
    exact_match = None if case.answer_options else float(normalized_answer in accepted)
    token_f1 = (
        None
        if case.answer_options
        else max(
            (_token_f1(normalized_answer, reference) for reference in accepted),
            default=0.0,
        )
    )
    multiple_choice_accuracy = float(normalized_answer in accepted) if case.answer_options else None
    if not case.gold_evidence:
        return CaseMetrics(
            exact_match=exact_match,
            token_f1=token_f1,
            multiple_choice_accuracy=multiple_choice_accuracy,
            retrieval_recall_at_1=None,
            retrieval_recall_at_5=None,
            retrieval_recall_at_10=None,
            reciprocal_rank=None,
            citation_precision=None,
            citation_recall=None,
            abstention_correct=None,
            retrieved_count=len(retrieved),
            citation_count=len(citations),
        )
    relevant_gold_by_rank = [
        _matched_gold_count(case.gold_evidence, retrieved[:cutoff], iou_threshold)
        for cutoff in (1, 5, 10)
    ]
    reciprocal_rank = next(
        (
            1 / rank
            for rank, item in enumerate(retrieved, start=1)
            if _matches_any(item, case.gold_evidence, iou_threshold)
        ),
        0.0,
    )
    relevant_citations = sum(
        _matches_any(item, case.gold_evidence, iou_threshold) for item in citations
    )
    return CaseMetrics(
        exact_match=exact_match,
        token_f1=token_f1,
        multiple_choice_accuracy=multiple_choice_accuracy,
        retrieval_recall_at_1=relevant_gold_by_rank[0] / len(case.gold_evidence),
        retrieval_recall_at_5=relevant_gold_by_rank[1] / len(case.gold_evidence),
        retrieval_recall_at_10=relevant_gold_by_rank[2] / len(case.gold_evidence),
        reciprocal_rank=reciprocal_rank,
        citation_precision=(relevant_citations / len(citations) if citations else 0.0),
        citation_recall=_matched_gold_count(
            case.gold_evidence,
            citations,
            iou_threshold,
        )
        / len(case.gold_evidence),
        abstention_correct=None,
        retrieved_count=len(retrieved),
        citation_count=len(citations),
    )


def aggregate_metrics(
    cases: Sequence[BenchmarkCase],
    scores: Sequence[CaseMetrics],
) -> dict[str, float | int | None]:
    if len(cases) != len(scores):
        raise ValueError("Each benchmark case must have exactly one metrics record")
    if not cases:
        raise ValueError("At least one benchmark case is required")

    answerable = [score for case, score in zip(cases, scores, strict=True) if case.answerable]
    unanswerable = [score for case, score in zip(cases, scores, strict=True) if not case.answerable]
    multiple_choice = [
        score for case, score in zip(cases, scores, strict=True) if case.answer_options
    ]
    free_response = [
        score for case, score in zip(cases, scores, strict=True) if not case.answer_options
    ]
    evidence_annotated = [
        score for case, score in zip(cases, scores, strict=True) if case.gold_evidence
    ]

    def mean(name: str, selected: Sequence[CaseMetrics]) -> float | None:
        values = [getattr(score, name) for score in selected]
        numeric_values = [value for value in values if isinstance(value, (int, float))]
        return sum(numeric_values) / len(numeric_values) if numeric_values else None

    return {
        "case_count": len(cases),
        "answerable_case_count": len(answerable),
        "unanswerable_case_count": len(unanswerable),
        "multiple_choice_case_count": len(multiple_choice),
        "evidence_annotated_case_count": len(evidence_annotated),
        "multiple_choice_accuracy": mean("multiple_choice_accuracy", multiple_choice),
        "answer_exact_match": mean("exact_match", free_response),
        "answer_token_f1": mean("token_f1", free_response),
        "retrieval_recall_at_1": mean("retrieval_recall_at_1", evidence_annotated),
        "retrieval_recall_at_5": mean("retrieval_recall_at_5", evidence_annotated),
        "retrieval_recall_at_10": mean("retrieval_recall_at_10", evidence_annotated),
        "retrieval_mrr": mean("reciprocal_rank", evidence_annotated),
        "citation_precision": mean("citation_precision", evidence_annotated),
        "citation_recall": mean("citation_recall", evidence_annotated),
        "unanswerable_abstention_accuracy": mean("abstention_correct", unanswerable),
    }


def _matched_gold_count(
    gold: Sequence[EvidenceSpan],
    retrieved: Sequence[EvidenceSpan],
    threshold: float,
) -> int:
    return sum(
        any(_overlaps(gold_span, item, threshold) for item in retrieved) for gold_span in gold
    )


def _matches_any(
    item: EvidenceSpan,
    gold: Sequence[EvidenceSpan],
    threshold: float,
) -> bool:
    return any(_overlaps(gold_span, item, threshold) for gold_span in gold)


def _overlaps(gold: EvidenceSpan, candidate: EvidenceSpan, threshold: float) -> bool:
    if gold.video_id != candidate.video_id:
        return False
    if gold.start_s == gold.end_s:
        return candidate.start_s <= gold.start_s <= candidate.end_s
    if candidate.start_s == candidate.end_s:
        return gold.start_s <= candidate.start_s <= gold.end_s
    intersection = max(
        0.0,
        min(gold.end_s, candidate.end_s) - max(gold.start_s, candidate.start_s),
    )
    union = max(gold.end_s, candidate.end_s) - min(gold.start_s, candidate.start_s)
    return union > 0 and intersection / union >= threshold


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _token_f1(answer: str, reference: str) -> float:
    answer_tokens = answer.split()
    reference_tokens = reference.split()
    if not answer_tokens or not reference_tokens:
        return float(answer_tokens == reference_tokens)
    common = Counter(answer_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
