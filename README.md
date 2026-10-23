# Research Video RAG

[![CI](https://github.com/anushka-naik/Video-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/anushka-naik/Video-RAG/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)](https://www.python.org/)

A research-first project for evaluating retrieval and grounded question answering over video. The default workflow is a local Python pipeline with a SQLite index; Docker, a database service, queues, and the API are not required. It compares lexical, dense-transcript, cross-modal visual, reciprocal-rank-fusion, and temporally diversified retrieval on the Perception Test sample.

The repository also retains an optional service-oriented API and ingestion stack, but it is not required for research experiments or the demo.

See the concise [project overview](docs/project-overview.md) for intended users, architecture, design tradeoffs, evaluation, and demo status.

## Requirements

- Python 3.11 or later
- FFmpeg and FFprobe
- `make`

## Local setup

```sh
make install
```

The first index run downloads embedding and speech-recognition models. Install FFmpeg and FFprobe using your system package manager. Transcription can be skipped with `--no-transcription`; this keeps the visual retrieval baseline usable without Whisper model downloads.

### Perception Test sample experiment

The sample archive is a small pilot (four validation clips, 18 multiple-choice questions), not the full benchmark. Download the official sample archives from the [Perception Test README](https://github.com/google-deepmind/perception_test#download-the-data-and-annotations) and extract the four validation videos:

```sh
mkdir -p data/perception-test
curl -fL -o data/perception-test/sample_videos.zip \
  https://storage.googleapis.com/dm-perception-test/zip_data/sample_videos.zip
curl -fL -o data/perception-test/sample_annotations.zip \
  https://storage.googleapis.com/dm-perception-test/zip_data/sample_annotations.zip
bsdtar -xf data/perception-test/sample_videos.zip -C data/perception-test \
  videos/video_6860.mp4 videos/video_9253.mp4 \
  videos/video_8241.mp4 videos/video_1580.mp4
```

Index the clips locally (no server services required):

```sh
make research-index ARGS="--annotations data/perception-test/sample_annotations.zip --videos-dir data/perception-test/videos --data-dir data/research"
```

Compare all retrieval methods with the same answer model and save the raw per-question report:

```sh
cp .env.example .env
# Default provider is local Ollama. For an OpenAI API run, change the provider
# and configure OPENAI_API_KEY plus VIDEO_RAG_OPENAI_MODEL in .env.
make eval-perception ARGS="--annotations data/perception-test/sample_annotations.zip --data-dir data/research --output data/research/perception-report.json"
```

The default is a local Ollama model, so generation makes no paid API calls. Install/start Ollama and download the configured model first:

```sh
ollama serve
ollama pull qwen3-vl:8b
```

The default is Qwen3-VL 8B, a vision-language model suited to question answering over retrieved video frames and text. On this 24 GB Apple Silicon Mac it is a practical local quality/runtime balance; Ollama reports a 6.1 GB model download. The experiment configures a 16,384-token context to fit the evidence images and a 300-second timeout for local inference. Adjust `VIDEO_RAG_OLLAMA_CONTEXT_LENGTH` and `VIDEO_RAG_OLLAMA_TIMEOUT_SECONDS` if needed for your machine. To use another local vision model, set `VIDEO_RAG_OLLAMA_MODEL` in `.env` to a model installed in Ollama. Local generation runs on your computer and may be slower depending on hardware.

The evaluator also supports OpenAI and Anthropic. For OpenAI, change `VIDEO_RAG_ANSWER_PROVIDER=openai`, set `OPENAI_API_KEY` and `VIDEO_RAG_OPENAI_MODEL` (for example `gpt-4o-mini`); for Anthropic select `anthropic` and configure its key/model variables. Hosted API calls may incur charges independently of ChatGPT or Claude subscriptions. Keep keys only in your ignored local `.env`; never commit or paste them into source, issues, or chat. Reports include provider/model configuration, dataset SHA-256, exact-match multiple-choice accuracy, per-method retrieval/answer timings, token usage, and retrieved timestamps. Perception Test sample MCQA has no query-specific evidence spans, so retrieval and citation quality are not scored. Do not present this tiny sample as a robust performance estimate.

Use `--provider ollama`, `--provider openai`, or `--provider anthropic` to override the configured provider for a single evaluation run.

Launch the lightweight local demo after indexing:

```sh
make demo
```

The UI lets you select a video and retrieval strategy, inspect timestamped evidence, and optionally generate a grounded answer using the selected provider. The default Ollama provider requires the local service and configured model to be running.

### Optional API stack

The original service-oriented backend remains available separately. Install `make install-all`, configure `.env`, run `make up`, apply migrations with `cd backend && ../.venv/bin/alembic upgrade head`, then run `make dev` and `make worker` in separate terminals. This API uses PostgreSQL, Qdrant, Redis, and Docker Compose; none are needed for local research indexing or the demo. Replace all development-only credentials before exposing the stack.

Run `make test` for focused research/evaluation tests, `make test-all` for the full backend suite, `make lint`, and `make typecheck` for research code. `make up` requires Docker Compose v2.

## Configuration

Settings use the `VIDEO_RAG_` environment-variable prefix. The local research command accepts embedding, cross-modal, Whisper, frame sampling, and data-directory overrides. Verify model availability and dataset/video usage terms before processing data.

## Evaluation

The project includes an offline scorer, a live-service benchmark runner, and the local retrieval experiment described above. Benchmark options, metrics, licensing caveats, and interpretation are documented in [docs/evaluation.md](./docs/evaluation.md). A stronger RAG benchmark still requires videos with documented usage rights and manually reviewed answer/evidence intervals.

No model-backed benchmark scores or measured quality, latency, or cost results are available yet. Unit tests verify deterministic code paths, annotation parsing, metrics, retrieval logic, SQLite persistence, API behavior, and provider-adapter contracts using synthetic fixtures and fakes; they do not establish transcription accuracy, OCR quality, retrieval relevance, answer faithfulness, or end-to-end performance. Model-backed evaluation requires model downloads and configured credentials. No benchmark quality claims are made.
