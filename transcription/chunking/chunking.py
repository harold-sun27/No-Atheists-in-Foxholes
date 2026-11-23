#!/usr/bin/env python3
"""Combine timestamped transcript segments into topic-sized text chunks."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

TARGET_CHUNK_WORDS = 40
MIN_CHUNK_WORDS = 20

TIMESTAMPED_LINE = re.compile(
    r"^\[(?P<start>\d+:\d{2}:\d{2}\.\d{3})\s+-->\s+"
    r"(?P<end>\d+:\d{2}:\d{2}\.\d{3})\]\s*(?P<text>.*)$"
)


class SegmentLike(Protocol):
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptChunk:
    start: float
    end: float
    text: str
    word_count: int


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_timestamp(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def count_words(text: str) -> int:
    return len(text.split())


def _make_chunk(segments: list[TranscriptSegment]) -> TranscriptChunk:
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return TranscriptChunk(
        start=segments[0].start,
        end=segments[-1].end,
        text=text,
        word_count=count_words(text),
    )


def chunk_segments(
    segments: Iterable[SegmentLike],
    target_words: int = TARGET_CHUNK_WORDS,
    min_words: int = MIN_CHUNK_WORDS,
) -> list[TranscriptChunk]:
    """Group consecutive segments into chunks of approximately target_words."""
    if target_words < 1:
        raise ValueError("target_words must be at least 1")
    if min_words < 1 or min_words > target_words:
        raise ValueError("min_words must be between 1 and target_words")

    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSegment] = []
    current_words = 0

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        segment_words = count_words(text)
        combined_words = current_words + segment_words
        if current and current_words >= min_words:
            current_distance = abs(target_words - current_words)
            combined_distance = abs(target_words - combined_words)
            if combined_distance > current_distance:
                chunks.append(_make_chunk(current))
                current = []
                current_words = 0

        normalized = TranscriptSegment(
            start=float(segment.start),
            end=float(segment.end),
            text=text,
        )
        current.append(normalized)
        current_words += segment_words

        if current_words >= target_words:
            chunks.append(_make_chunk(current))
            current = []
            current_words = 0

    if current:
        tail = _make_chunk(current)
        if chunks and tail.word_count < min_words:
            previous = chunks.pop()
            chunks.append(
                TranscriptChunk(
                    start=previous.start,
                    end=tail.end,
                    text=f"{previous.text} {tail.text}",
                    word_count=previous.word_count + tail.word_count,
                )
            )
        else:
            chunks.append(tail)

    return chunks


def parse_transcript(path: Path) -> list[TranscriptSegment]:
    """Read the timestamped text format written by transcribe.py."""
    segments: list[TranscriptSegment] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = TIMESTAMPED_LINE.fullmatch(line)
        if not match:
            raise ValueError(
                f"Invalid timestamped transcript line {line_number}: {raw_line}"
            )
        segments.append(
            TranscriptSegment(
                start=parse_timestamp(match.group("start")),
                end=parse_timestamp(match.group("end")),
                text=match.group("text").strip(),
            )
        )
    if not segments:
        raise ValueError(f"No timestamped segments found in: {path}")
    return segments


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_transcription"):
        stem = stem[: -len("_transcription")]
    return input_path.with_name(f"{stem}_chunks.txt")


def write_chunks(chunks: Iterable[TranscriptChunk], output_path: Path) -> Path:
    lines = [
        f"[{format_timestamp(chunk.start)} --> {format_timestamp(chunk.end)}] "
        f"{chunk.text}"
        for chunk in chunks
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine timestamped transcript segments into approximately 40-word chunks."
    )
    parser.add_argument("input_file", type=Path, help="Timestamped transcript file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (default: <input>_chunks.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        input_path = args.input_file.expanduser().resolve()
        if not input_path.is_file():
            raise ValueError(f"Input file not found: {input_path}")
        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else default_output_path(input_path)
        )
        chunks = chunk_segments(parse_transcript(input_path))
        write_chunks(chunks, output_path)
        print(f"Created {len(chunks)} chunks: {output_path}")
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
