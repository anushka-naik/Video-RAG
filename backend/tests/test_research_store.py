from pathlib import Path
from uuid import uuid4

from app.research.store import ResearchItem, ResearchStore


def test_research_store_persists_and_replaces_video_items(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "index.sqlite3")
    video_id = "video_123"
    segment = ResearchItem(
        id=uuid4(),
        video_id=video_id,
        kind="transcript",
        start_s=0,
        end_s=5,
        text="A person enters the room.",
        image_path=None,
        dense_vector=[0.1, 0.2],
        clip_vector=None,
    )
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"frame")
    frame = ResearchItem(
        id=uuid4(),
        video_id=video_id,
        kind="keyframe",
        start_s=2,
        end_s=2,
        text="",
        image_path=image_path,
        dense_vector=None,
        clip_vector=[0.3, 0.4],
    )

    store.replace_video(video_id, video_id, image_path, image_path, 10, [segment, frame])
    store.replace_video(video_id, "updated", image_path, image_path, 11, [frame])

    assert store.video(video_id) == {
        "external_id": video_id,
        "title": "updated",
        "source_path": str(image_path.resolve()),
        "processed_path": str(image_path.resolve()),
        "duration_s": 11.0,
    }
    items = store.items(video_id)
    assert len(items) == 1
    assert items[0].id == frame.id
    assert items[0].image_path == image_path.resolve()
    assert items[0].clip_vector == [0.3, 0.4]
