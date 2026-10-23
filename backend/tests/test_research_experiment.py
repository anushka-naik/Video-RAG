from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from app.evaluation.metrics import BenchmarkCase
from app.qa.answering import GeneratedAnswer
from app.qa.evidence import RetrievedEvidence
from app.research.experiment import _multiple_choice_option, run_perception_test_experiment
from app.research.store import ResearchItem, ResearchStore


class FakeAnswerer:
    async def answer(self, question: str, evidence: Sequence[RetrievedEvidence]) -> GeneratedAnswer:
        assert "Choose exactly one answer" in question
        assert evidence
        return GeneratedAnswer(
            answer="A person enters the room",
            citation_ids=[],
            abstain=False,
            usage={"input_tokens": 10, "output_tokens": 4},
        )


class FakeIndex:
    def embed_text_query(self, text: str, model: str = "dense") -> list[float]:
        raise AssertionError("BM25 should not use embedding models")


def test_experiment_reports_multiple_choice_accuracy_and_cost(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "index.sqlite3")
    external_id = "video_1"
    video_uuid = uuid5(NAMESPACE_URL, f"perception-test:{external_id}")
    item = ResearchItem(
        id=uuid5(video_uuid, "transcript"),
        video_id=external_id,
        kind="transcript",
        start_s=0,
        end_s=3,
        text="A person enters the room.",
        image_path=None,
        dense_vector=None,
        clip_vector=None,
    )
    store.replace_video(external_id, external_id, tmp_path, tmp_path, 3, [item])
    case = BenchmarkCase(
        id=f"{external_id}:mc:1",
        video_id=video_uuid,
        question="Who enters the room?",
        answerable=True,
        acceptable_answers=["A person enters the room"],
        answer_options=["A person enters the room", "A person leaves the room"],
    )

    report = run_perception_test_experiment(
        store=store,
        index=FakeIndex(),
        cases=[case],
        answerer=FakeAnswerer(),
        methods=["bm25"],
        top_k=1,
    )

    assert report["case_count"] == 1
    assert report["evidence_quality_metrics"] is None
    methods = report["methods"]
    assert isinstance(methods, dict)
    result = methods["bm25"]
    assert isinstance(result, dict)
    assert result["cases"] == 1
    assert result["accuracy"] == 1.0
    assert result["mean_retrieval_seconds"] >= 0
    assert result["mean_answer_seconds"] >= 0
    assert result["total_input_tokens"] == 10
    assert result["total_output_tokens"] == 4


def test_multiple_choice_parser_maps_model_choice_letters_to_gold_option_text() -> None:
    choices = ["A person enters the room.", "A person leaves the room."]

    assert _multiple_choice_option("B", choices) == choices[1]
    assert _multiple_choice_option("Option A", choices) == choices[0]
    assert _multiple_choice_option("A person enters the room.", choices) == choices[0]
    assert _multiple_choice_option("Unlisted response", choices) is None
