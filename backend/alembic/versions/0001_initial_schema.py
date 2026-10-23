from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "videos",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("source_url", sa.String(), nullable=True),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=16), server_default="queued", nullable=False),
        sa.Column("media_path", sa.String(), nullable=True),
        sa.Column("summary", sa.String(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'processing', 'ready', 'failed')", name="ck_videos_status"
        ),
        sa.CheckConstraint("duration_s IS NULL OR duration_s >= 0", name="ck_videos_duration"),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_videos_collection_id", "videos", ["collection_id"])
    op.create_table(
        "chapters",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("video_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("start_s", sa.Float(), nullable=False),
        sa.Column("end_s", sa.Float(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("summary", sa.String(), nullable=True),
        sa.CheckConstraint("ordinal >= 0", name="ck_chapters_ordinal"),
        sa.CheckConstraint("start_s >= 0 AND end_s >= start_s", name="ck_chapters_time_range"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("video_id", "ordinal", name="uq_chapters_video_ordinal"),
    )
    op.create_index("ix_chapters_video_id", "chapters", ["video_id"])
    op.create_table(
        "segments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("video_id", sa.Uuid(), nullable=False),
        sa.Column("chapter_id", sa.Uuid(), nullable=True),
        sa.Column("start_s", sa.Float(), nullable=False),
        sa.Column("end_s", sa.Float(), nullable=False),
        sa.Column("transcript_text", sa.String(), nullable=False),
        sa.Column("slide_text", sa.String(), nullable=True),
        sa.CheckConstraint("start_s >= 0 AND end_s >= start_s", name="ck_segments_time_range"),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_segments_video_id", "segments", ["video_id"])
    op.create_index("ix_segments_chapter_id", "segments", ["chapter_id"])
    op.create_table(
        "keyframes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("video_id", sa.Uuid(), nullable=False),
        sa.Column("timestamp_s", sa.Float(), nullable=False),
        sa.Column("image_path", sa.String(), nullable=False),
        sa.Column("phash", sa.String(length=64), nullable=True),
        sa.Column("ocr_text", sa.String(), nullable=True),
        sa.CheckConstraint("timestamp_s >= 0", name="ck_keyframes_timestamp"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_keyframes_video_id", "keyframes", ["video_id"])
    op.create_table(
        "threads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("scope", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("usage", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("role IN ('user', 'assistant', 'tool')", name="ck_messages_role"),
        sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_thread_id_created_at", "messages", ["thread_id", "created_at"])
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("video_id", sa.Uuid(), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("progress", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", sa.String(length=16), server_default="queued", nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="ck_jobs_progress"),
        sa.CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed')", name="ck_jobs_status"
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_video_id", "jobs", ["video_id"])


def downgrade() -> None:
    op.drop_index("ix_jobs_video_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_messages_thread_id_created_at", table_name="messages")
    op.drop_table("messages")
    op.drop_table("threads")
    op.drop_index("ix_keyframes_video_id", table_name="keyframes")
    op.drop_table("keyframes")
    op.drop_index("ix_segments_chapter_id", table_name="segments")
    op.drop_index("ix_segments_video_id", table_name="segments")
    op.drop_table("segments")
    op.drop_index("ix_chapters_video_id", table_name="chapters")
    op.drop_table("chapters")
    op.drop_index("ix_videos_collection_id", table_name="videos")
    op.drop_table("videos")
    op.drop_table("collections")
