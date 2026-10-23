from app.config import Settings
from app.ingestion.errors import IngestionError
from app.ingestion.keyframes import PySceneDetectKeyframeExtractor
from app.ingestion.media import MediaProcessor
from app.ingestion.ocr import TesseractOCRReader
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.providers import (
    FastEmbedTextEncoder,
    QdrantVectorStore,
    QdrantVideoIndexer,
    SiglipImageEncoder,
    create_qdrant_vector_store,
)
from app.ingestion.summarization import AnthropicChapterSummarizer
from app.ingestion.transcription import FasterWhisperTranscriber


def create_ingestion_runtime(settings: Settings) -> tuple[IngestionPipeline, QdrantVectorStore]:
    if settings.anthropic_api_key is None:
        raise IngestionError("VIDEO_RAG_ANTHROPIC_API_KEY is required for the ingestion worker")
    if not settings.anthropic_fast_model:
        raise IngestionError("VIDEO_RAG_ANTHROPIC_FAST_MODEL is required for the ingestion worker")
    vector_store = create_qdrant_vector_store(
        settings.qdrant_url,
        settings.request_timeout_seconds,
    )
    try:
        pipeline = IngestionPipeline(
            media_processor=MediaProcessor(
                ffmpeg_executable=settings.ffmpeg_executable,
                ffprobe_executable=settings.ffprobe_executable,
                timeout_seconds=settings.media_timeout_seconds,
                max_download_bytes=settings.max_video_upload_bytes,
            ),
            transcriber=FasterWhisperTranscriber(
                model_name=settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            ),
            keyframe_extractor=PySceneDetectKeyframeExtractor(
                ffmpeg_executable=settings.ffmpeg_executable,
                interval_seconds=settings.keyframe_interval_seconds,
                timeout_seconds=settings.media_timeout_seconds,
            ),
            ocr_reader=TesseractOCRReader(
                executable=settings.tesseract_executable,
                language=settings.ocr_language,
                timeout_seconds=settings.request_timeout_seconds,
            ),
            indexer=QdrantVideoIndexer(
                vector_store=vector_store,
                text_encoder=FastEmbedTextEncoder(settings.text_embedding_model),
                image_encoder=SiglipImageEncoder(settings.visual_embedding_model),
            ),
            chapter_summarizer=AnthropicChapterSummarizer(
                api_key=(
                    settings.anthropic_api_key.get_secret_value()
                    if settings.anthropic_api_key is not None
                    else None
                ),
                model_name=settings.anthropic_fast_model,
                timeout_seconds=settings.request_timeout_seconds,
                window_seconds=settings.summary_window_seconds,
            ),
            ocr_concurrency=settings.ocr_concurrency,
        )
    except Exception:
        vector_store.close()
        raise
    return pipeline, vector_store
