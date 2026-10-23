import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from app.config import settings
from app.db.session import SessionFactory
from app.evaluation.metrics import (
    BenchmarkCase,
    CaseMetrics,
    EvidenceSpan,
    aggregate_metrics,
    score_case,
)
from app.evaluation.perception_test import load_perception_test_multiple_choice
from app.qa.answering import ABSTENTION_ANSWER
from app.qa.runtime import create_qa_runtime
from app.qa.service import VideoQuestionService


def _load_dataset(path: Path) -> tuple[bytes, list[BenchmarkCase]]:
    data = path.read_bytes()
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = BenchmarkCase.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(f"Invalid benchmark record on line {line_number}: {error}") from error
        if case.id in seen_ids:
            raise ValueError(f"Duplicate benchmark case ID on line {line_number}: {case.id}")
        seen_ids.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("The benchmark JSONL file contains no cases")
    return data, cases


async def evaluate_dataset(
    dataset_path: Path,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in the interval (0, 1]")
    dataset_bytes, cases = _load_dataset(dataset_path)
    return await _evaluate_cases(
        dataset_name=dataset_path.name,
        dataset_bytes=dataset_bytes,
        cases=cases,
        iou_threshold=iou_threshold,
        dataset_kind="project_jsonl",
    )


async def evaluate_perception_test(
    annotations_archive: Path,
    video_id_map_path: Path,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in the interval (0, 1]")
    archive_bytes, cases = load_perception_test_multiple_choice(
        annotations_archive,
        video_id_map_path,
    )
    return await _evaluate_cases(
        dataset_name=annotations_archive.name,
        dataset_bytes=archive_bytes,
        cases=cases,
        iou_threshold=iou_threshold,
        dataset_kind="perception_test_validation_mcqa",
    )


async def _evaluate_cases(
    dataset_name: str,
    dataset_bytes: bytes,
    cases: list[BenchmarkCase],
    iou_threshold: float,
    dataset_kind: str,
) -> dict[str, object]:
    runtime = create_qa_runtime(settings)
    case_results: list[dict[str, object]] = []
    case_scores: list[CaseMetrics] = []
    try:
        for case in cases:
            async with SessionFactory() as session:
                service = VideoQuestionService(
                    session=session,
                    vector_store=runtime.vector_store,
                    text_encoder=runtime.text_encoder,
                    visual_encoder=runtime.visual_encoder,
                    planner=runtime.planner,
                    answerer=runtime.answerer,
                    media_storage_path=settings.media_storage_path,
                )
                response, retrieved = await service.ask_with_trace(
                    question=_question_with_options(case),
                    collection_id=None,
                    video_id=case.video_id,
                    thread_id=None,
                    top_k=10,
                    persist=False,
                )
            retrieved_spans = [
                EvidenceSpan(
                    video_id=item.video_id,
                    start_s=item.start_s,
                    end_s=item.end_s,
                )
                for item in retrieved
            ]
            cited_spans = [
                EvidenceSpan(
                    video_id=item.video_id,
                    start_s=item.start_s,
                    end_s=item.end_s,
                )
                for item in response.citations
            ]
            score = score_case(
                case,
                response.answer,
                retrieved_spans,
                cited_spans,
                abstained=response.answer == ABSTENTION_ANSWER,
                iou_threshold=iou_threshold,
            )
            case_scores.append(score)
            case_results.append({"id": case.id, "metrics": score.as_dict()})
    finally:
        runtime.close()

    return {
        "format_version": 1,
        "dataset_kind": dataset_kind,
        "dataset_file": dataset_name,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "temporal_iou_threshold": iou_threshold,
        "model_config": {
            "planning_model": settings.anthropic_planning_model,
            "answer_model": settings.anthropic_fast_model,
            "text_embedding_model": settings.text_embedding_model,
            "visual_embedding_model": settings.visual_embedding_model,
        },
        "metrics": aggregate_metrics(cases, case_scores),
        "cases": case_results,
    }


def _question_with_options(case: BenchmarkCase) -> str:
    if not case.answer_options:
        return case.question
    options = "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(case.answer_options)
    )
    return (
        f"{case.question}\nChoose one answer from these options:\n{options}\n"
        "Respond with the exact text of your chosen option."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the video QA evaluator against a reviewed benchmark."
    )
    dataset_input = parser.add_mutually_exclusive_group(required=True)
    dataset_input.add_argument("--dataset", type=Path, help="Reviewed benchmark JSONL")
    dataset_input.add_argument(
        "--perception-test-annotations",
        type=Path,
        help="Official Perception Test annotation ZIP (validation MCQA)",
    )
    parser.add_argument(
        "--video-id-map",
        type=Path,
        help="JSON mapping Perception Test video IDs to ingested application UUIDs",
    )
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Temporal IoU threshold for matching evidence spans (default: 0.5)",
    )
    arguments = parser.parse_args(argv)

    if arguments.perception_test_annotations is not None:
        if arguments.video_id_map is None:
            parser.error("--video-id-map is required with --perception-test-annotations")
        report = asyncio.run(
            evaluate_perception_test(
                arguments.perception_test_annotations,
                arguments.video_id_map,
                arguments.iou_threshold,
            )
        )
    else:
        if arguments.video_id_map is not None:
            parser.error("--video-id-map can only be used with --perception-test-annotations")
        report = asyncio.run(evaluate_dataset(arguments.dataset, arguments.iou_threshold))
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        sys.stdout.write(serialized)
    else:
        arguments.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
