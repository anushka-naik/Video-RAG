# Video QA evaluation

## Benchmark choice

Use a reviewed, project-specific dataset as the primary benchmark: videos must be owned by the project or covered by permission for this use, and each answerable question must have an accepted answer plus one or more manually checked supporting time intervals. Include unanswerable questions with no gold answer or evidence. This gives the benchmark the evidence annotations needed to evaluate retrieval and citations, which ordinary video-question-answering scores do not provide.

Public benchmark candidates must be reviewed for the exact intended use:

- [Perception Test](https://github.com/google-deepmind/perception_test) links MP4 video and annotation archives, and its README states that non-software materials are CC BY 4.0. Its multiple-choice QA is usable for a small answer-accuracy experiment. Its separate grounded-QA answers reference object tracks, not natural-language answer text or QA-specific evidence intervals. The repository does not itemize rights for each video asset; review the [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) and confirm that they cover the video archive and intended use before relying on it, especially commercially.
- [LongVideoBench](https://arxiv.org/abs/2407.15754) is the closer fit for long-video and temporally dependent questions. Its validation split can test video-QA answer accuracy, but its official answer annotations do not provide gold RAG retrieval or citation spans. Its dataset is CC BY-NC-SA 4.0; the videos are third-party content, and the dataset is noncommercial. Confirm access and terms before use.
- [Video-MME](https://arxiv.org/abs/2405.21075) is a broad video-understanding QA benchmark, not a retrieval or citation benchmark. The official project limits use to academic research and prohibits commercial use and redistribution absent prior approval. Do not bundle or redistribute its data or videos.

The official Perception Test sample split has been downloaded outside the repository for integration work. This is a small sample, not the full benchmark validation split. Public QA labels are not by themselves gold RAG retrieval or citation labels. Do not claim RAG retrieval quality or citation faithfulness without independently reviewed evidence annotations and compliance with all dataset and video rights. Report results from any public set separately from the project-specific benchmark.

Attribute the source as Google DeepMind's Perception Test and link the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) when redistributing or adapting covered materials. This project's adapter converts the original multiple-choice annotations into its evaluation input format. The license does not grant rights beyond those actually held by the licensor, and other rights (including privacy and publicity) may apply.

## Running the project benchmark

For local research without Docker or backend services, install with `make install`, index the Perception Test sample using `make research-index`, and compare retrieval methods using `make eval-perception`. The commands and sample download steps are in the main README. The local evaluator is an answer-accuracy pilot; it does not measure retrieval or citation quality because this dataset has no question-specific evidence intervals.

The local comparison includes:

- `bm25`: lexical matching over transcript chunks.
- `dense_text`: cosine similarity over BGE transcript embeddings.
- `clip_visual`: cross-modal query-to-frame similarity using Jina CLIP.
- `hybrid_rrf`: reciprocal-rank fusion of BM25, dense transcript, and visual rankings.
- `hybrid_mmr`: the same fused ranking with a temporal-overlap penalty to reduce redundant moments.

The default answer provider is local Ollama (`VIDEO_RAG_ANSWER_PROVIDER=ollama`) using Qwen3-VL 8B (`VIDEO_RAG_OLLAMA_MODEL=qwen3-vl:8b`), a vision-language model for reasoning over retrieved frames and text. The local adapter requests a 16,384-token context by default (`VIDEO_RAG_OLLAMA_CONTEXT_LENGTH`) and uses a 300-second timeout (`VIDEO_RAG_OLLAMA_TIMEOUT_SECONDS`); reduce context on memory-constrained machines or increase it if Ollama reports context overflow. Local inference avoids hosted API charges but depends on available hardware and model license terms. Set `VIDEO_RAG_OLLAMA_BASE_URL` if Ollama listens on a non-default local URL. Set `VIDEO_RAG_ANSWER_PROVIDER=openai`, `OPENAI_API_KEY`, and `VIDEO_RAG_OPENAI_MODEL` to use OpenAI; or select `anthropic` and configure its existing key/model variables. Hosted API access and billing are separate from ChatGPT/Claude subscriptions. The report records mean retrieval and answer latency, answer accuracy, token totals, per-case outputs, dataset SHA-256, top-k, and provider/model names. Compare retrieval latency separately from answer latency; answer calls are repeated for each method and may vary with provider/model behavior. This is an ablation harness, not evidence that any method is superior. Pin model versions and rerun on more data before making comparative claims.

Install the project and ingestion dependencies, start PostgreSQL and Qdrant, run migrations, and ensure each benchmark video has completed ingestion. Configure the Anthropic API key and model IDs as described in the main README. Prepare a UTF-8 JSONL file with one record per question, then run:

```sh
make eval ARGS="--dataset /absolute/path/to/benchmark.jsonl"
```

The command uses the live QA pipeline, but does not persist benchmark threads or messages. It prints a JSON report containing the dataset SHA-256, configured model IDs, temporal IoU threshold, aggregate scores, and per-case scores. Use `--output /path/to/report.json` to write the report to a file. Pin provider model versions when available; a configured model alias alone may not guarantee repeatable model behavior.

### Perception Test sample validation MCQA

The evaluator can read the official Perception Test annotation ZIP directly. The included sample archive has eight videos across train and validation; its validation portion contains four videos and 18 multiple-choice questions. This is useful for wiring and a small smoke evaluation, but too small to represent full benchmark performance.

Download the official archives from the links in the [Perception Test README](https://github.com/google-deepmind/perception_test#download-the-data-and-annotations). For example:

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

Ingest these four validation videos using the normal upload flow (`POST /videos/upload`). Record the returned application video UUIDs in a JSON file keyed by the Perception Test video IDs. Start from [the video ID map template](./perception-test-video-id-map.template.json) and replace all placeholders with UUIDs returned by ingestion:

```json
{
  "video_6860": "<application-video-uuid>",
  "video_9253": "<application-video-uuid>",
  "video_8241": "<application-video-uuid>",
  "video_1580": "<application-video-uuid>"
}
```

The above IDs are from the official sample archive, and the UUID values are placeholders. After all four videos are ready, run:

```sh
make eval ARGS="--perception-test-annotations data/perception-test/sample_annotations.zip --video-id-map data/perception-test/video-id-map.json"
```

The adapter selects only annotation records with `metadata.split == "valid"`, supplies the answer choices to QA, and scores exact top-1 option accuracy. The sample MCQA annotations do not give QA-specific gold evidence spans, so retrieval and citation metrics are intentionally reported as unavailable for these cases. The standard RAG service may still return citations, but they are not treated as benchmark-verified citations.

Each JSONL record has this shape:

```json
{
  "id": "case-001",
  "video_id": "00000000-0000-0000-0000-000000000000",
  "question": "What happens in the annotated interval?",
  "answerable": true,
  "acceptable_answers": ["A person enters the room."],
  "gold_evidence": [
    {
      "video_id": "00000000-0000-0000-0000-000000000000",
      "start_s": 12.0,
      "end_s": 18.0
    }
  ]
}
```

The values above are placeholders, not benchmark annotations. `video_id` must refer to a ready video in the configured database. Every answerable example requires at least one accepted answer and gold evidence span, all for that video. An unanswerable example must have empty `acceptable_answers` and `gold_evidence` arrays.

`backend/tests/fixtures/evaluation_smoke.jsonl` is a synthetic scorer fixture with fake video IDs. Unit tests use it to verify metric calculations only; it is not a video dataset, must not be passed to the live runner, and its expected smoke scores are not system-quality results.

## Metrics and interpretation

- Answer exact match and token F1 use normalized text comparison against the accepted answers. They are deterministic lexical metrics, not a semantic judge.
- Perception Test multiple-choice accuracy measures exact normalized match to the gold option, consistent with its documented top-1 accuracy task; it is distinct from free-response answer metrics.
- Retrieval Recall@1/5/10 counts gold intervals matched by at least one retrieved interval within the first *k* results. Retrieval MRR uses the rank of the first relevant result.
- Citation precision and recall measure overlap between the response's cited intervals and the manually annotated gold intervals.
- A match requires the same video and temporal intersection-over-union (IoU) at or above `--iou-threshold` (default `0.5`). A keyframe timestamp is matched as a point inside the gold interval. The threshold is recorded in every report and can be changed explicitly.
- Unanswerable abstention accuracy is reported separately. It is not mixed into answerable answer or retrieval scores.

Scores are fractions in the range `[0, 1]`; higher is better. These scores describe only the supplied dataset, annotation policy, model configuration, and run. They do not prove that all generated claims are supported, and lexical answer scores do not replace human review of correctness and evidence. A benchmark result is not available until the dataset videos are ingested and evaluation has run against live dependencies.
