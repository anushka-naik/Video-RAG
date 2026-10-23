from __future__ import annotations

import asyncio
from collections import defaultdict
from time import perf_counter
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from app.evaluation.metrics import BenchmarkCase, _normalize
from app.qa.answering import Answerer
from app.qa.evidence import RetrievedEvidence
from app.research.retrieval import METHODS, QueryEncoder, ResearchRetriever
from app.research.store import ResearchStore


def run_perception_test_experiment(
    store: ResearchStore,
    index: QueryEncoder,
    cases: list[BenchmarkCase],
    answerer: Answerer,
    methods: list[str] | None = None,
    top_k: int = 5,
    limit: int | None = None,
) -> dict[str, object]:
    selected_methods = methods or list(METHODS)
    unknown_methods = set(selected_methods) - set(METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown retrieval methods: {sorted(unknown_methods)}")
    if not selected_methods or len(set(selected_methods)) != len(selected_methods):
        raise ValueError("Select at least one unique retrieval method")
    if top_k <= 0 or (limit is not None and limit <= 0):
        raise ValueError("top_k and limit must be positive")
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No benchmark cases selected")

    retriever = ResearchRetriever(index)
    method_records: dict[str, list[dict[str, object]]] = defaultdict(list)

    async def evaluate() -> None:
        for case in cases:
            external_id = case.id.split(":mc:", maxsplit=1)[0]
            video_items = store.items(external_id)
            video = store.video(external_id)
            if video is None:
                raise ValueError(f"Video {external_id} has not been locally indexed")
            video_uuid = uuid5(NAMESPACE_URL, f"perception-test:{external_id}")
            for method in selected_methods:
                retrieval_start = perf_counter()
                ranked = retriever.search(video_items, case.question, method, top_k)
                retrieval_seconds = perf_counter() - retrieval_start
                evidence = [
                    RetrievedEvidence(
                        id=result.item.id,
                        video_id=video_uuid,
                        video_title=str(video["title"]),
                        start_s=result.item.start_s,
                        end_s=result.item.end_s,
                        source_type=result.item.kind,
                        transcript_text=(
                            result.item.text if result.item.kind == "transcript" else ""
                        ),
                        slide_text=None,
                        ocr_text=None,
                        image_path=result.item.image_path,
                        rank_score=result.score,
                    )
                    for result in ranked
                ]
                answer_start = perf_counter()
                generated = await answerer.answer(_multiple_choice_prompt(case), evidence)
                answer_seconds = perf_counter() - answer_start
                predicted_option = _multiple_choice_option(generated.answer, case.answer_options)
                correct = _normalize(predicted_option or generated.answer) in {
                    _normalize(answer) for answer in case.acceptable_answers
                }
                method_records[method].append(
                    {
                        "case_id": case.id,
                        "video_id": external_id,
                        "question": case.question,
                        "gold_answer": case.acceptable_answers[0],
                        "answer": generated.answer,
                        "predicted_option": predicted_option,
                        "correct": correct,
                        "retrieval_seconds": retrieval_seconds,
                        "answer_seconds": answer_seconds,
                        "input_tokens": generated.usage.get("input_tokens", 0),
                        "output_tokens": generated.usage.get("output_tokens", 0),
                        "retrieved": [
                            {
                                "kind": result.item.kind,
                                "start_s": result.item.start_s,
                                "end_s": result.item.end_s,
                                "score": result.score,
                            }
                            for result in ranked
                        ],
                    }
                )

    asyncio.run(evaluate())
    comparisons: dict[str, object] = {}
    for method, records in method_records.items():
        count = len(records)
        comparisons[method] = {
            "cases": count,
            "accuracy": sum(bool(record["correct"]) for record in records) / count,
            "mean_retrieval_seconds": sum(
                cast(float, record["retrieval_seconds"]) for record in records
            )
            / count,
            "mean_answer_seconds": sum(cast(float, record["answer_seconds"]) for record in records)
            / count,
            "total_input_tokens": sum(cast(int, record["input_tokens"]) for record in records),
            "total_output_tokens": sum(cast(int, record["output_tokens"]) for record in records),
        }
    return {
        "dataset": "Perception Test sample validation multiple-choice QA",
        "case_count": len(cases),
        "top_k": top_k,
        "answer_metric": "exact normalized match to the official correct option",
        "evidence_quality_metrics": None,
        "methods": comparisons,
        "per_case": dict(method_records),
    }


def _multiple_choice_prompt(case: BenchmarkCase) -> str:
    options = "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(case.answer_options)
    )
    return (
        f"{case.question}\nChoose exactly one answer from the options below:\n{options}\n"
        "Set the answer to the full exact text of the selected option, not its letter. "
        "Do not add an explanation."
    )


def _multiple_choice_option(answer: str, options: list[str]) -> str | None:
    normalized_answer = _normalize(answer)
    for option in options:
        if normalized_answer == _normalize(option):
            return option
    letter = normalized_answer.removeprefix("option ")
    if len(letter) == 1 and "a" <= letter <= "z":
        option_index = ord(letter) - ord("a")
        if option_index < len(options):
            return options[option_index]
    for index, option in enumerate(options):
        if normalized_answer == f"{chr(ord('a') + index)} {_normalize(option)}":
            return option
    return None
