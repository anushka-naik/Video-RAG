from pathlib import Path
from uuid import uuid4

from app.research.retrieval import ResearchRetriever
from app.research.store import ResearchItem


class FakeResearchIndex:
    def embed_text_query(self, text: str, model: str = "dense") -> list[float]:
        return [1.0, 0.0] if model == "dense" else [0.0, 1.0]


def _item(
    kind: str,
    start_s: float,
    text: str = "",
    dense_vector: list[float] | None = None,
    clip_vector: list[float] | None = None,
) -> ResearchItem:
    return ResearchItem(
        id=uuid4(),
        video_id="video_1",
        kind=kind,
        start_s=start_s,
        end_s=start_s + (4 if kind == "transcript" else 0),
        text=text,
        image_path=Path("/tmp/frame.jpg") if kind == "keyframe" else None,
        dense_vector=dense_vector,
        clip_vector=clip_vector,
    )


def test_retrieval_strategies_rank_lexical_dense_and_visual_items() -> None:
    lexical_item = _item("transcript", 0, "blue whale swimming", dense_vector=[0.0, 1.0])
    dense_item = _item("transcript", 10, "ocean animal", dense_vector=[1.0, 0.0])
    visual_item = _item("keyframe", 20, clip_vector=[0.0, 1.0])
    retriever = ResearchRetriever(FakeResearchIndex())  # type: ignore[arg-type]
    items = [lexical_item, dense_item, visual_item]

    bm25 = retriever.search(items, "blue whale", "bm25", 1)
    dense = retriever.search(items, "blue whale", "dense_text", 1)
    visual = retriever.search(items, "blue whale", "clip_visual", 1)
    hybrid = retriever.search(items, "blue whale", "hybrid_rrf", 3)

    assert bm25[0].item.id == lexical_item.id
    assert dense[0].item.id == dense_item.id
    assert visual[0].item.id == visual_item.id
    assert {result.item.id for result in hybrid} == {
        lexical_item.id,
        dense_item.id,
        visual_item.id,
    }


def test_bm25_returns_no_results_for_no_lexical_match() -> None:
    item = _item("transcript", 0, "someone walks into a room")
    retriever = ResearchRetriever(FakeResearchIndex())  # type: ignore[arg-type]

    assert retriever.search([item], "blue whale", "bm25", 5) == []


def test_hybrid_mmr_spreads_near_duplicate_temporal_hits() -> None:
    first = _item("keyframe", 0, clip_vector=[0.0, 1.0])
    nearby = _item("keyframe", 0.5, clip_vector=[0.0, 1.0])
    later = _item("keyframe", 20, clip_vector=[0.0, 1.0])
    retriever = ResearchRetriever(FakeResearchIndex())  # type: ignore[arg-type]

    selected = retriever.search([first, nearby, later], "room", "hybrid_mmr", 2)

    assert len(selected) == 2
    assert abs(selected[0].item.start_s - selected[1].item.start_s) >= 15
