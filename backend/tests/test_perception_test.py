import json
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_BZIP2, ZipFile

import pytest

from app.evaluation.perception_test import load_perception_test_multiple_choice


def test_load_perception_test_archive_converts_validation_mcqa(
    tmp_path: Path,
) -> None:
    video_uuid = uuid4()
    annotations_path = tmp_path / "annotations.zip"
    mapping_path = tmp_path / "video-map.json"
    records = {
        "video_train": {
            "metadata": {"split": "train"},
            "mc_question": [
                {
                    "id": 0,
                    "question": "Train question?",
                    "options": ["a", "b", "c"],
                    "answer_id": 1,
                }
            ],
        },
        "video_valid": {
            "metadata": {"split": "valid"},
            "mc_question": [
                {
                    "id": 7,
                    "question": "What is visible?",
                    "options": ["a cup", "a book", "a spoon"],
                    "answer_id": 1,
                }
            ],
        },
    }
    with ZipFile(annotations_path, "w", compression=ZIP_BZIP2) as archive:
        archive.writestr("sample.json", json.dumps(records))
    mapping_path.write_text(json.dumps({"video_valid": str(video_uuid)}), encoding="utf-8")

    archive_bytes, cases = load_perception_test_multiple_choice(
        annotations_path,
        mapping_path,
    )

    assert archive_bytes == annotations_path.read_bytes()
    assert len(cases) == 1
    assert cases[0].id == "video_valid:mc:7"
    assert cases[0].video_id == video_uuid
    assert cases[0].question == "What is visible?"
    assert cases[0].answer_options == ["a cup", "a book", "a spoon"]
    assert cases[0].acceptable_answers == ["a book"]
    assert cases[0].gold_evidence == []


def test_perception_test_loader_rejects_missing_validation_video_mapping(
    tmp_path: Path,
) -> None:
    annotations_path = tmp_path / "annotations.zip"
    mapping_path = tmp_path / "video-map.json"
    with ZipFile(annotations_path, "w", compression=ZIP_BZIP2) as archive:
        archive.writestr(
            "valid.json",
            json.dumps(
                {
                    "video_valid": {
                        "metadata": {"split": "valid"},
                        "mc_question": [],
                    }
                }
            ),
        )
    mapping_path.write_text(json.dumps({}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing validation video video_valid"):
        load_perception_test_multiple_choice(annotations_path, mapping_path)


def test_perception_test_loader_rejects_invalid_answer_index(tmp_path: Path) -> None:
    annotations_path = tmp_path / "annotations.zip"
    mapping_path = tmp_path / "video-map.json"
    with ZipFile(annotations_path, "w", compression=ZIP_BZIP2) as archive:
        archive.writestr(
            "valid.json",
            json.dumps(
                {
                    "video_valid": {
                        "metadata": {"split": "valid"},
                        "mc_question": [
                            {
                                "id": 0,
                                "question": "Which one?",
                                "options": ["a", "b"],
                                "answer_id": 2,
                            }
                        ],
                    }
                }
            ),
        )
    mapping_path.write_text(
        json.dumps({"video_valid": str(uuid4())}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid answer index"):
        load_perception_test_multiple_choice(annotations_path, mapping_path)
