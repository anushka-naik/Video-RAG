from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from app.config import settings
from app.evaluation.perception_test import (
    load_perception_test_multiple_choice,
    perception_test_validation_video_ids,
)
from app.qa.answering import create_grounded_answerer
from app.research.experiment import run_perception_test_experiment
from app.research.indexing import ResearchIndex
from app.research.retrieval import METHODS
from app.research.store import ResearchStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local, service-free video RAG experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser(
        "index-perception-test",
        help="Index videos from the Perception Test sample validation split",
    )
    index_parser.add_argument("--annotations", type=Path, required=True)
    index_parser.add_argument("--videos-dir", type=Path, required=True)
    index_parser.add_argument("--data-dir", type=Path, default=Path("data/research"))
    index_parser.add_argument("--text-model", default="BAAI/bge-base-en-v1.5")
    index_parser.add_argument("--cross-modal-model", default="jinaai/jina-clip-v1")
    index_parser.add_argument("--whisper-model", default="base.en")
    index_parser.add_argument("--sample-interval", type=float, default=2.0)
    index_parser.add_argument("--no-transcription", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate-perception-test",
        help="Compare retrieval methods on Perception Test validation MCQA",
    )
    evaluate_parser.add_argument("--annotations", type=Path, required=True)
    evaluate_parser.add_argument("--data-dir", type=Path, default=Path("data/research"))
    evaluate_parser.add_argument(
        "--method",
        action="append",
        choices=METHODS,
        help="Retrieval method to include; repeat to compare methods (default: all)",
    )
    evaluate_parser.add_argument("--top-k", type=int, default=5)
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument("--output", type=Path)
    evaluate_parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "ollama"),
        help="Override VIDEO_RAG_ANSWER_PROVIDER for this run",
    )

    arguments = parser.parse_args(argv)
    if arguments.command == "index-perception-test":
        return _index(arguments)
    return _evaluate(arguments)


def _index(arguments: argparse.Namespace) -> int:
    data_directory = arguments.data_dir.resolve()
    store = ResearchStore(data_directory / "index.sqlite3")
    index = ResearchIndex(
        store=store,
        settings=settings,
        data_directory=data_directory,
        text_model_name=arguments.text_model,
        cross_modal_model_name=arguments.cross_modal_model,
        whisper_model_name=arguments.whisper_model,
        sample_interval_s=arguments.sample_interval,
        transcribe=not arguments.no_transcription,
    )
    video_ids = perception_test_validation_video_ids(arguments.annotations)
    results = []
    for external_id in video_ids:
        source_path = arguments.videos_dir / f"{external_id}.mp4"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing validation video file: {source_path}")
        results.append(asyncio.run(index.index_video(external_id, source_path)))
        print(json.dumps(results[-1], sort_keys=True), flush=True)
    print(json.dumps({"indexed_videos": len(results)}, sort_keys=True))
    return 0


def _evaluate(arguments: argparse.Namespace) -> int:
    data_directory = arguments.data_dir.resolve()
    store = ResearchStore(data_directory / "index.sqlite3")
    validation_ids = perception_test_validation_video_ids(arguments.annotations)
    video_id_map = {
        external_id: str(uuid5(NAMESPACE_URL, f"perception-test:{external_id}"))
        for external_id in validation_ids
    }
    cases = load_perception_test_multiple_choice(
        arguments.annotations,
        video_id_map,
    )[1]
    for external_id in validation_ids:
        if store.video(external_id) is None:
            raise ValueError(
                f"Validation video {external_id} is not indexed; run index-perception-test first"
            )
    provider = (arguments.provider or settings.answer_provider).casefold()
    if provider == "openai":
        api_key = (
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key is not None
            else None
        )
        model_name = settings.openai_model
        if api_key is None or not model_name:
            raise ValueError(
                "OpenAI selected but configuration is incomplete. Set OPENAI_API_KEY "
                "(or VIDEO_RAG_OPENAI_API_KEY) and VIDEO_RAG_OPENAI_MODEL in .env. "
                "Also pass --provider openai or set VIDEO_RAG_ANSWER_PROVIDER=openai."
            )
    elif provider == "ollama":
        api_key = None
        model_name = settings.ollama_model
        if not model_name:
            raise ValueError(
                "Ollama selected but VIDEO_RAG_OLLAMA_MODEL is unset. Set it in .env "
                "(the recommended local model is qwen3-vl:8b). Ensure Ollama is running."
            )
    elif provider == "anthropic":
        api_key = (
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key is not None
            else None
        )
        model_name = settings.anthropic_fast_model
        if api_key is None or not model_name:
            raise ValueError(
                "Anthropic selected but configuration is incomplete. Set "
                "VIDEO_RAG_ANTHROPIC_API_KEY and VIDEO_RAG_ANTHROPIC_FAST_MODEL in .env, "
                "or configure OpenAI and pass --provider openai."
            )
    else:
        raise ValueError("VIDEO_RAG_ANSWER_PROVIDER must be 'anthropic', 'openai', or 'ollama'")

    index = ResearchIndex(store, settings, data_directory)
    answerer = create_grounded_answerer(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        timeout_seconds=(
            settings.ollama_timeout_seconds
            if provider == "ollama"
            else settings.request_timeout_seconds
        ),
        base_url=settings.ollama_base_url if provider == "ollama" else None,
        ollama_context_length=settings.ollama_context_length,
    )
    report = run_perception_test_experiment(
        store=store,
        index=index,
        cases=cases,
        answerer=answerer,
        methods=arguments.method,
        top_k=arguments.top_k,
        limit=arguments.limit,
    )
    report["dataset_sha256"] = hashlib.sha256(arguments.annotations.read_bytes()).hexdigest()
    report["model_config"] = {
        "answer_provider": provider,
        "answer_model": model_name,
        "text_embedding_model": index.text_model_name,
        "cross_modal_model": index.cross_modal_model_name,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
