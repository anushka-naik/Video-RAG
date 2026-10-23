import math
import re
from collections.abc import Sequence

from app.ingestion.schemas import Sentence, TimedWord, TranscriptSegment

SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")


def sentences_from_words(words: Sequence[TimedWord]) -> list[Sentence]:
    sentences: list[Sentence] = []
    current_words: list[TimedWord] = []
    previous_start = -math.inf

    for word in words:
        if not word.text.strip():
            raise ValueError("Word text cannot be empty")
        if not math.isfinite(word.start_s) or not math.isfinite(word.end_s):
            raise ValueError("Word timestamps must be finite")
        if word.start_s < 0 or word.end_s < word.start_s:
            raise ValueError("Word timestamps must be non-negative and ordered")
        if word.start_s < previous_start:
            raise ValueError("Words must be ordered by start time")
        current_words.append(word)
        previous_start = word.start_s

        if SENTENCE_END.search(word.text.strip()):
            sentences.append(_sentence_from_words(current_words))
            current_words = []

    if current_words:
        sentences.append(_sentence_from_words(current_words))
    return sentences


def segment_sentences(
    sentences: Sequence[Sentence],
    target_duration_s: float = 45.0,
    maximum_duration_s: float = 60.0,
    overlap_ratio: float = 0.1,
) -> list[TranscriptSegment]:
    if not math.isfinite(target_duration_s) or target_duration_s <= 0:
        raise ValueError("Target duration must be positive and finite")
    if not math.isfinite(maximum_duration_s) or maximum_duration_s < target_duration_s:
        raise ValueError("Maximum duration must be finite and at least the target duration")
    if not math.isfinite(overlap_ratio) or not 0 <= overlap_ratio < 1:
        raise ValueError("Overlap ratio must be in the range [0, 1)")
    _validate_sentences(sentences)
    if not sentences:
        return []

    segments: list[TranscriptSegment] = []
    start_index = 0
    while start_index < len(sentences):
        end_index = start_index + 1
        while end_index < len(sentences):
            duration_s = sentences[end_index].end_s - sentences[start_index].start_s
            if duration_s > maximum_duration_s or duration_s >= target_duration_s:
                break
            end_index += 1

        segment_sentences = sentences[start_index:end_index]
        segments.append(
            TranscriptSegment(
                start_s=segment_sentences[0].start_s,
                end_s=segment_sentences[-1].end_s,
                transcript_text=" ".join(sentence.text for sentence in segment_sentences),
            )
        )
        if end_index >= len(sentences):
            break

        segment_start_s = segment_sentences[0].start_s
        segment_end_s = segment_sentences[-1].end_s
        overlap_start_s = segment_end_s - (segment_end_s - segment_start_s) * overlap_ratio
        overlapping_index = next(
            (
                index
                for index in range(start_index + 1, end_index)
                if sentences[index].end_s > overlap_start_s
            ),
            end_index,
        )
        start_index = max(start_index + 1, overlapping_index)

    return segments


def _sentence_from_words(words: Sequence[TimedWord]) -> Sentence:
    return Sentence(
        start_s=words[0].start_s,
        end_s=max(word.end_s for word in words),
        text="".join(word.text for word in words).strip(),
    )


def _validate_sentences(sentences: Sequence[Sentence]) -> None:
    previous_start = -math.inf
    for sentence in sentences:
        if not sentence.text.strip():
            raise ValueError("Sentence text cannot be empty")
        if not math.isfinite(sentence.start_s) or not math.isfinite(sentence.end_s):
            raise ValueError("Sentence timestamps must be finite")
        if sentence.start_s < 0 or sentence.end_s < sentence.start_s:
            raise ValueError("Sentence timestamps must be non-negative and ordered")
        if sentence.start_s < previous_start:
            raise ValueError("Sentences must be ordered by start time")
        previous_start = sentence.start_s
