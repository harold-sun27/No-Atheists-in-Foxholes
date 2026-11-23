#!/usr/bin/env python3
"""Embed timestamped transcript chunks with all-MiniLM-L6-v2."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSIONS = 384

TIMESTAMPED_LINE = re.compile(
    r"^\[(?P<start>\d+:\d{2}:\d{2}\.\d{3})\s+-->\s+"
    r"(?P<end>\d+:\d{2}:\d{2}\.\d{3})\]\s*(?P<text>.*)$"
)


@dataclass(frozen=True)
class TranscriptChunk:
    start: float
    end: float
    text: str


def parse_timestamp(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_chunks(path: Path) -> list[TranscriptChunk]:
    """Read the timestamped text format produced by the chunking stage."""
    chunks: list[TranscriptChunk] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = TIMESTAMPED_LINE.fullmatch(line)
        if not match:
            raise ValueError(f"Invalid chunk line {line_number}: {raw_line}")
        text = match.group("text").strip()
        if not text:
            raise ValueError(f"Chunk line {line_number} contains no text")
        chunks.append(
            TranscriptChunk(
                start=parse_timestamp(match.group("start")),
                end=parse_timestamp(match.group("end")),
                text=text,
            )
        )
    if not chunks:
        raise ValueError(f"No timestamped chunks found in: {path}")
    return chunks


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_chunks"):
        stem = stem[: -len("_chunks")]
    return input_path.with_name(f"{stem}_embeddings.npz")


def embed_chunks(chunks: list[TranscriptChunk], output_path: Path) -> tuple[int, int]:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding dependencies are not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    print(f"Loading embedding model: {MODEL_NAME}")
    try:
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            show_progress_bar=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not create embeddings: {exc}") from exc

    if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_DIMENSIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_DIMENSIONS}-dimensional embeddings, "
            f"received shape {embeddings.shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float32),
        texts=np.asarray([chunk.text for chunk in chunks], dtype=np.str_),
        start_times=np.asarray([chunk.start for chunk in chunks], dtype=np.float64),
        end_times=np.asarray([chunk.end for chunk in chunks], dtype=np.float64),
        model=np.asarray(MODEL_NAME, dtype=np.str_),
    )
    return embeddings.shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed timestamped transcript chunks with all-MiniLM-L6-v2."
    )
    parser.add_argument("input_file", type=Path, help="Timestamped chunks file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (default: <input>_embeddings.npz)",
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
        if output_path.suffix.lower() != ".npz":
            raise ValueError("The embedding output must use the .npz extension")

        chunks = parse_chunks(input_path)
        count, dimensions = embed_chunks(chunks, output_path)
        print(f"Embedded {count} chunks into {dimensions}-dimensional vectors")
        print(f"Saved embeddings: {output_path}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
