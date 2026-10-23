# Project Overview

Video RAG is for ML practitioners and researchers exploring question answering over video. It retrieves relevant video moments and presents timestamped evidence alongside generated answers.

The workflow is a single-agent Python RAG system: local video is processed into frames and timed transcript, indexed for lexical and embedding-based retrieval, then passed with retrieved evidence to a vision-language model. Qwen3-VL runs locally through Ollama.

The key tradeoff is local inference: it avoids hosted API charges and keeps video on-device, but model speed depends on available hardware. Automated tests cover retrieval, indexing, scoring, citations, and provider behavior. One Perception Test smoke case passed; that small sample cannot support a meaningful benchmark claim or retrieval-quality score.

The Streamlit demo runs locally. There is no hosted deployment.
