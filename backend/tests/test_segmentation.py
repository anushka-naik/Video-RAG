import pytest

from app.ingestion.schemas import Sentence, TimedWord, TranscriptSegment
from app.ingestion.segmentation import segment_sentences, sentences_from_words


def test_sentences_split_on_punctuation_and_keep_trailing_words() -> None:
    words = [
        TimedWord(0.0, 0.3, "This "),
        TimedWord(0.3, 0.6, "works."),
        TimedWord(0.7, 1.0, "So "),
        TimedWord(1.0, 1.2, "does "),
        TimedWord(1.2, 1.5, "this"),
    ]

    sentences = sentences_from_words(words)

    assert sentences == [
        Sentence(0.0, 0.6, "This works."),
        Sentence(0.7, 1.5, "So does this"),
    ]


def test_segment_windows_respect_duration_and_overlap() -> None:
    sentences = [
        Sentence(float(start), float(start + 5), f"Sentence {start}.") for start in range(0, 120, 5)
    ]

    segments = segment_sentences(sentences)

    assert all(segment.end_s - segment.start_s <= 60 for segment in segments)
    assert all(
        current.start_s < previous.end_s
        for previous, current in zip(segments, segments[1:], strict=False)
    )
    assert all(segment.text for segment in segments)
    assert segments[0].start_s == 0
    assert segments[-1].end_s == sentences[-1].end_s


def test_short_transcript_remains_one_segment() -> None:
    sentences = [Sentence(0.0, 2.0, "Short sentence.")]

    assert segment_sentences(sentences) == [TranscriptSegment(0.0, 2.0, "Short sentence.")]


@pytest.mark.parametrize(
    ("words", "message"),
    [
        ([TimedWord(-1.0, 0.0, "bad")], "non-negative"),
        ([TimedWord(1.0, 0.5, "bad")], "non-negative"),
        ([TimedWord(1.0, 1.5, "first"), TimedWord(0.5, 0.8, "second")], "ordered"),
    ],
)
def test_invalid_word_timestamps_are_rejected(words: list[TimedWord], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sentences_from_words(words)


def test_invalid_overlap_ratio_is_rejected() -> None:
    with pytest.raises(ValueError, match="Overlap ratio"):
        segment_sentences([Sentence(0.0, 1.0, "Text.")], overlap_ratio=1.0)
