from app.db import models
from app.db.base import Base


def test_initial_schema_contains_all_entities() -> None:
    assert models.Video.__tablename__ == "videos"
    assert set(Base.metadata.tables) == {
        "collections",
        "videos",
        "chapters",
        "segments",
        "keyframes",
        "threads",
        "messages",
        "jobs",
    }
