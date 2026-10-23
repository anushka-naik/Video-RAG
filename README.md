# Video RAG

[![CI](https://github.com/anushka-naik/Video-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/anushka-naik/Video-RAG/actions/workflows/ci.yml)

Research project for finding and answering questions about moments in video. Compare lexical, dense, visual, and hybrid retrieval over local video, with timestamped evidence and grounded answers.

The default answer model is Qwen3-VL 8B through Ollama. Video indexing and evaluation run locally; no hosted demo is available.

## Try it

Install the project with `make install`, index a local dataset using the instructions in [evaluation](docs/evaluation.md), then start the demo with `make demo`.

Run checks with:

```sh
make test-all
make lint
make typecheck-all
```

See the short [project overview](docs/project-overview.md) for the audience, architecture, design tradeoff, and evaluation notes.
