# Project Overview

Video RAG is for ML practitioners and researchers exploring question answering over video. It retrieves relevant video moments and presents timestamped evidence alongside generated answers.

The project has two execution paths. The API-backed service queues video ingestion, extracts timestamped speech and keyframes, and writes metadata to PostgreSQL and text/image embeddings to separate Qdrant collections. Chat plans and encodes a question, retrieves across both modalities, fuses the rankings, and generates a cited answer from the retrieved evidence. The service uses Anthropic for query planning and answer generation.

The standalone research path uses a local CLI and Streamlit demo to index video and compare BM25, dense-text, visual, and hybrid retrieval in a SQLite-backed corpus. Its default answer model is Qwen3-VL through local Ollama; OpenAI and Anthropic are also selectable. Local inference avoids hosted API charges but model speed depends on available hardware. Automated tests cover retrieval, indexing, scoring, citations, and provider behavior. One Perception Test smoke case passed; that small sample cannot support a meaningful benchmark claim or retrieval-quality score.

Both paths run locally; there is no hosted deployment.
