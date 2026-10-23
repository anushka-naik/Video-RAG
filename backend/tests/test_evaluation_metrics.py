from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.evaluation.metrics import (
    BenchmarkCase,
    EvidenceSpan,
    aggregate_metrics,
    score_case,
)
from app.evaluation.runner import _load_dataset


def _case(
    *,
    answerable: bool = True,
    video_id: UUID | None = None,
    gold_evidence: list[EvidenceSpan] | None = None,
    acceptable_answers: list[str] | None = None,
) -> BenchmarkCase:
    video_id = video_id or uuid4()
    return BenchmarkCase(
        id="case-1",
        video_id=video_id,
        question="What happened?",
        answerable=answerable,
        acceptable_answers=acceptable_answers or (["A person entered."] if answerable else []),
        gold_evidence=gold_evidence
        if gold_evidence is not None
        else ([EvidenceSpan(video_id=video_id, start_s=10, end_s=20)] if answerable else []),
    )


def test_score_case_reports_answer_retrieval_and_citation_metrics() -> None:
    case = _case(acceptable_answers=["A person entered."])
    irrelevant = EvidenceSpan(video_id=uuid4(), start_s=10, end_s=20)
    relevant = EvidenceSpan(video_id=case.video_id, start_s=12, end_s=18)

    score = score_case(
        case,
        "A PERSON entered!",
        [irrelevant, relevant],
        [relevant],
    )

    assert score.exact_match == 1.0
    assert score.token_f1 == 1.0
    assert score.retrieval_recall_at_1 == 0.0
    assert score.retrieval_recall_at_5 == 1.0
    assert score.reciprocal_rank == 0.5
    assert score.citation_precision == 1.0
    assert score.citation_recall == 1.0


def test_keyframe_point_matches_gold_interval_and_partial_retrieval_is_counted() -> None:
    video_id = uuid4()
    case = _case(
        gold_evidence=[
            EvidenceSpan(video_id=video_id, start_s=10, end_s=20),
            EvidenceSpan(video_id=video_id, start_s=30, end_s=40),
        ],
        video_id=video_id,
    )
    keyframe = EvidenceSpan(video_id=video_id, start_s=15, end_s=15)
    score = score_case(case, "A person entered.", [keyframe], [keyframe])

    assert score.retrieval_recall_at_1 == 0.5
    assert score.citation_precision == 1.0
    assert score.citation_recall == 0.5


def test_unanswerable_case_scores_abstention_without_answer_quality_claims() -> None:
    case = _case(answerable=False)

    score = score_case(case, "I couldn't find evidence.", [], [], abstained=True)

    assert score.exact_match is None
    assert score.retrieval_recall_at_1 is None
    assert score.citation_precision is None
    assert score.abstention_correct == 1.0


def test_aggregate_metrics_separates_answerable_and_unanswerable_cases() -> None:
    answerable = _case()
    unanswerable = _case(answerable=False)
    answered = score_case(answerable, "A person entered.", answerable.gold_evidence, [])
    abstained = score_case(unanswerable, "No evidence.", [], [], abstained=True)

    metrics = aggregate_metrics([answerable, unanswerable], [answered, abstained])

    assert metrics["case_count"] == 2
    assert metrics["answer_exact_match"] == 1.0
    assert metrics["unanswerable_abstention_accuracy"] == 1.0


def test_multiple_choice_case_scores_accuracy_without_claiming_evidence_metrics() -> None:
    video_id = uuid4()
    case = BenchmarkCase(
        id="perception:mc:1",
        video_id=video_id,
        question="What is visible?",
        answerable=True,
        answer_options=["a cup", "a book", "a spoon"],
        acceptable_answers=["a book"],
    )

    score = score_case(case, "A book.", [], [])
    metrics = aggregate_metrics([case], [score])

    assert score.multiple_choice_accuracy == 1.0
    assert score.exact_match is None
    assert score.retrieval_recall_at_1 is None
    assert score.citation_precision is None
    assert metrics["multiple_choice_accuracy"] == 1.0
    assert metrics["evidence_annotated_case_count"] == 0
    assert metrics["retrieval_mrr"] is None


def test_benchmark_case_requires_explicit_gold_for_answerable_examples() -> None:
    with pytest.raises(ValidationError, match="at least one acceptable answer"):
        BenchmarkCase(
            id="invalid",
            video_id=uuid4(),
            question="What happened?",
            answerable=True,
        )


def test_multiple_choice_case_requires_correct_answer_to_be_an_option() -> None:
    with pytest.raises(ValidationError, match="match one of the answer_options"):
        BenchmarkCase(
            id="invalid",
            video_id=uuid4(),
            question="What happened?",
            answerable=True,
            acceptable_answers=["something else"],
            answer_options=["one", "two"],
        )


def test_dataset_loader_validates_jsonl_and_rejects_duplicate_ids(tmp_path) -> None:
    case = _case()
    dataset_path = tmp_path / "benchmark.jsonl"
    dataset_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")

    raw, cases = _load_dataset(dataset_path)

    assert raw.endswith(b"\n")
    assert [item.id for item in cases] == [case.id]

    dataset_path.write_text(
        case.model_dump_json() + "\n" + case.model_dump_json() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate benchmark case ID"):
        _load_dataset(dataset_path)


def test_synthetic_smoke_fixture_exercises_all_metric_families() -> None:
    fixture = Path(__file__).parent / "fixtures" / "evaluation_smoke.jsonl"
    _, cases = _load_dataset(fixture)
    scores = []
    for case in cases:
        if case.answerable:
            evidence = case.gold_evidence[0]
            point_or_interval = EvidenceSpan(
                video_id=case.video_id,
                start_s=(evidence.start_s + evidence.end_s) / 2,
                end_s=(evidence.start_s + evidence.end_s) / 2,
            )
            scores.append(
                score_case(
                    case,
                    case.acceptable_answers[0],
                    [evidence],
                    [point_or_interval],
                )
            )
        else:
            scores.append(
                score_case(
                    case,
                    "I couldn't find enough evidence.",
                    [],
                    [],
                    abstained=True,
                )
            )

    metrics = aggregate_metrics(cases, scores)

    assert metrics["case_count"] == 3
    assert metrics["answer_exact_match"] == 1.0
    assert metrics["retrieval_recall_at_1"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] == 1.0
    assert metrics["unanswerable_abstention_accuracy"] == 1.0
