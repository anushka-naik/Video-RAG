from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import streamlit as st

from app.config import settings
from app.qa.answering import create_grounded_answerer
from app.qa.evidence import RetrievedEvidence
from app.research.indexing import ResearchIndex
from app.research.retrieval import METHODS, ResearchRetriever
from app.research.store import ResearchStore

DATA_DIRECTORY = Path("data/research").resolve()
STORE = ResearchStore(DATA_DIRECTORY / "index.sqlite3")


def main() -> None:
    st.set_page_config(page_title="Video RAG Lab", page_icon="🎞️", layout="wide")
    st.title("Video RAG Lab")
    st.caption("Explore transcript, visual, and hybrid retrieval over locally indexed video.")
    videos = STORE.videos()
    if not videos:
        st.info("No local video index yet. Follow the Research setup in the README.")
        return

    external_id = st.selectbox(
        "Indexed video",
        [str(video["external_id"]) for video in videos],
    )
    video = STORE.video(external_id)
    if video is None:
        st.error(f"Video index disappeared for {external_id}")
        return
    duration = video["duration_s"]
    if not isinstance(duration, (int, float)):
        raise TypeError(f"Invalid duration in local video index: {duration!r}")
    st.caption(f"Duration: {duration:.1f}s")
    question = st.text_input("Ask a question about this video")
    method = st.selectbox("Retrieval strategy", METHODS, index=METHODS.index("hybrid_rrf"))
    top_k = st.slider("Evidence items", min_value=1, max_value=10, value=5)
    answer_button = st.button("Retrieve and answer", type="primary", disabled=not question.strip())

    if answer_button:
        index = ResearchIndex(STORE, settings, DATA_DIRECTORY)
        results = ResearchRetriever(index).search(
            STORE.items(external_id),
            question,
            method,
            top_k,
        )
        video_uuid = uuid5(NAMESPACE_URL, f"perception-test:{external_id}")
        evidence = [
            RetrievedEvidence(
                id=result.item.id,
                video_id=video_uuid,
                video_title=str(video["title"]),
                start_s=result.item.start_s,
                end_s=result.item.end_s,
                source_type=result.item.kind,
                transcript_text=result.item.text if result.item.kind == "transcript" else "",
                slide_text=None,
                ocr_text=None,
                image_path=result.item.image_path,
                rank_score=result.score,
            )
            for result in results
        ]
        if not evidence:
            st.warning("This retrieval strategy found no evidence.")
            return
        provider = settings.answer_provider.casefold()
        if provider == "openai":
            api_key = (
                settings.openai_api_key.get_secret_value()
                if settings.openai_api_key is not None
                else None
            )
            model_name = settings.openai_model
            required_settings = "OPENAI_API_KEY and VIDEO_RAG_OPENAI_MODEL"
        elif provider == "ollama":
            api_key = None
            model_name = settings.ollama_model
            required_settings = "VIDEO_RAG_OLLAMA_MODEL and the local Ollama service"
        elif provider == "anthropic":
            api_key = (
                settings.anthropic_api_key.get_secret_value()
                if settings.anthropic_api_key is not None
                else None
            )
            model_name = settings.anthropic_fast_model
            required_settings = "VIDEO_RAG_ANTHROPIC_API_KEY and VIDEO_RAG_ANTHROPIC_FAST_MODEL"
        else:
            st.error("VIDEO_RAG_ANSWER_PROVIDER must be 'anthropic', 'openai', or 'ollama'.")
            return
        if (provider != "ollama" and api_key is None) or not model_name:
            st.error(
                f"Set {required_settings} to generate an answer. "
                "Evidence retrieval can still be inspected below."
            )
        else:
            answerer = create_grounded_answerer(
                provider,
                api_key,
                model_name,
                (
                    settings.ollama_timeout_seconds
                    if provider == "ollama"
                    else settings.request_timeout_seconds
                ),
                base_url=settings.ollama_base_url if provider == "ollama" else None,
                ollama_context_length=settings.ollama_context_length,
            )
            answer = asyncio.run(answerer.answer(question, evidence))
            st.subheader("Answer")
            st.write(answer.answer)
            st.caption(f"Token usage: {answer.usage}")
        st.subheader("Retrieved evidence")
        for rank, (item, result) in enumerate(zip(evidence, results, strict=True), start=1):
            with st.container(border=True):
                st.markdown(
                    f"**{rank}. {item.source_type} — "
                    f"{item.start_s:.1f}s–{item.end_s:.1f}s** "
                    f"(retrieval score: `{result.score:.4f}`)"
                )
                if item.transcript_text:
                    st.write(item.transcript_text)
                if item.image_path is not None:
                    st.image(str(item.image_path), width=360)


if __name__ == "__main__":
    main()
