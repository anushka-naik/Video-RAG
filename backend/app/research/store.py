from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ResearchItem:
    id: UUID
    video_id: str
    kind: Literal["transcript", "keyframe"]
    start_s: float
    end_s: float
    text: str
    image_path: Path | None
    dense_vector: list[float] | None
    clip_vector: list[float] | None


class ResearchStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS videos (
                    external_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    processed_path TEXT NOT NULL,
                    duration_s REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    external_id TEXT NOT NULL REFERENCES videos(external_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK (kind IN ('transcript', 'keyframe')),
                    start_s REAL NOT NULL,
                    end_s REAL NOT NULL,
                    text TEXT NOT NULL,
                    image_path TEXT,
                    dense_vector TEXT,
                    clip_vector TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_items_video_kind_time
                    ON items (external_id, kind, start_s);
                """
            )
            connection.execute("PRAGMA foreign_keys=ON")

    def replace_video(
        self,
        external_id: str,
        title: str,
        source_path: Path,
        processed_path: Path,
        duration_s: float,
        items: list[ResearchItem],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO videos (external_id, title, source_path, processed_path, duration_s)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(external_id) DO UPDATE SET
                    title=excluded.title,
                    source_path=excluded.source_path,
                    processed_path=excluded.processed_path,
                    duration_s=excluded.duration_s
                """,
                (
                    external_id,
                    title,
                    str(source_path.resolve()),
                    str(processed_path.resolve()),
                    duration_s,
                ),
            )
            connection.execute("DELETE FROM items WHERE external_id = ?", (external_id,))
            connection.executemany(
                """
                INSERT INTO items (
                    id, external_id, kind, start_s, end_s, text, image_path,
                    dense_vector, clip_vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(item.id),
                        external_id,
                        item.kind,
                        item.start_s,
                        item.end_s,
                        item.text,
                        str(item.image_path.resolve()) if item.image_path else None,
                        json.dumps(item.dense_vector) if item.dense_vector is not None else None,
                        json.dumps(item.clip_vector) if item.clip_vector is not None else None,
                    )
                    for item in items
                ],
            )

    def video(self, external_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM videos WHERE external_id = ?",
                (external_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def videos(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM videos ORDER BY external_id").fetchall()
        return [dict(row) for row in rows]

    def items(self, external_id: str) -> list[ResearchItem]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM items WHERE external_id = ? ORDER BY start_s, kind",
                (external_id,),
            ).fetchall()
        return [
            ResearchItem(
                id=UUID(row["id"]),
                video_id=row["external_id"],
                kind=row["kind"],
                start_s=row["start_s"],
                end_s=row["end_s"],
                text=row["text"],
                image_path=Path(row["image_path"]) if row["image_path"] else None,
                dense_vector=json.loads(row["dense_vector"]) if row["dense_vector"] else None,
                clip_vector=json.loads(row["clip_vector"]) if row["clip_vector"] else None,
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
