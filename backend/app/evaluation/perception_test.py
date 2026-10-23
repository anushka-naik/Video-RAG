import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from uuid import UUID
from zipfile import ZipFile

from app.evaluation.metrics import BenchmarkCase


def perception_test_validation_video_ids(annotations_archive: Path) -> list[str]:
    try:
        with ZipFile(annotations_archive) as archive:
            json_files = [name for name in archive.namelist() if name.endswith(".json")]
            if len(json_files) != 1:
                raise ValueError(
                    "Perception Test annotation archive must contain exactly one JSON file"
                )
            annotation_data = json.loads(archive.read(json_files[0]))
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read Perception Test annotations: {error}") from error
    if not isinstance(annotation_data, dict):
        raise ValueError("Perception Test annotations must be a JSON object keyed by video ID")
    return sorted(
        video_id
        for video_id, record in annotation_data.items()
        if isinstance(record, dict)
        and isinstance(record.get("metadata"), dict)
        and record["metadata"].get("split") == "valid"
    )


def load_perception_test_multiple_choice(
    annotations_archive: Path,
    video_id_map_source: Path | dict[str, str],
) -> tuple[bytes, list[BenchmarkCase]]:
    archive_bytes = annotations_archive.read_bytes()
    video_id_map = (
        _load_video_id_map(video_id_map_source)
        if isinstance(video_id_map_source, Path)
        else _parse_video_id_map(video_id_map_source)
    )
    try:
        with ZipFile(annotations_archive) as archive:
            json_files = [name for name in archive.namelist() if name.endswith(".json")]
            if len(json_files) != 1:
                raise ValueError(
                    "Perception Test annotation archive must contain exactly one JSON file"
                )
            annotation_data = json.loads(archive.read(json_files[0]))
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read Perception Test annotations: {error}") from error
    if not isinstance(annotation_data, dict):
        raise ValueError("Perception Test annotations must be a JSON object keyed by video ID")

    cases: list[BenchmarkCase] = []
    for external_video_id, raw_record in annotation_data.items():
        if not isinstance(raw_record, dict):
            raise ValueError(f"Invalid annotation record for {external_video_id}")
        metadata = raw_record.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("split") != "valid":
            continue
        if external_video_id not in video_id_map:
            raise ValueError(f"Video ID map is missing validation video {external_video_id}")
        questions = raw_record.get("mc_question")
        if not isinstance(questions, list):
            raise ValueError(f"Missing multiple-choice questions for {external_video_id}")
        for question in questions:
            cases.append(
                _parse_question(external_video_id, video_id_map[external_video_id], question)
            )
    if not cases:
        raise ValueError("No validation multiple-choice questions found in Perception Test data")
    return archive_bytes, cases


def _load_video_id_map(path: Path) -> dict[str, UUID]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read video ID map: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Video ID map must be a JSON object")
    return _parse_video_id_map(value)


def _parse_video_id_map(value: Mapping[str, object]) -> dict[str, UUID]:
    mapping: dict[str, UUID] = {}
    for source_id, target_id in value.items():
        if not isinstance(source_id, str) or not isinstance(target_id, str):
            raise ValueError("Video ID map keys and values must be strings")
        try:
            mapping[source_id] = UUID(target_id)
        except ValueError as error:
            raise ValueError(f"Invalid application video UUID for {source_id}") from error
    return mapping


def _parse_question(
    video_id: str,
    application_video_id: UUID,
    raw_question: object,
) -> BenchmarkCase:
    if not isinstance(raw_question, dict):
        raise ValueError(f"Invalid multiple-choice question for {video_id}")
    question_id = raw_question.get("id")
    text = raw_question.get("question")
    options = raw_question.get("options")
    answer_id = raw_question.get("answer_id")
    if not isinstance(question_id, (int, str)) or isinstance(question_id, bool):
        raise ValueError(f"Invalid question ID for {video_id}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Invalid question text for {video_id}:{question_id}")
    if (
        not isinstance(options, list)
        or len(options) < 2
        or not all(isinstance(option, str) and option.strip() for option in options)
    ):
        raise ValueError(f"Invalid answer options for {video_id}:{question_id}")
    if (
        not isinstance(answer_id, int)
        or isinstance(answer_id, bool)
        or not 0 <= answer_id < len(options)
    ):
        raise ValueError(f"Invalid answer index for {video_id}:{question_id}")
    answer_options = cast(list[str], options)
    return BenchmarkCase(
        id=f"{video_id}:mc:{question_id}",
        video_id=application_video_id,
        question=text,
        answerable=True,
        acceptable_answers=[answer_options[answer_id]],
        answer_options=answer_options,
    )
