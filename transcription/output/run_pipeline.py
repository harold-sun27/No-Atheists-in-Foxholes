#!/usr/bin/env python3
"""Run the local transcription pipeline and build the final topic JSON."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTION_ROOT = REPOSITORY_ROOT / "transcription"

TIMESTAMPED_LINE = re.compile(
    r"^\[(?P<start>\d+:\d{2}:\d{2}\.\d{3})\s+-->\s+"
    r"(?P<end>\d+:\d{2}:\d{2}\.\d{3})\]\s*(?P<text>.*)$"
)
LANGUAGE_LINE = re.compile(r"^Language:\s+(?P<language>[^\s(]+)", re.MULTILINE)
DURATION_LINE = re.compile(r"^Duration:\s+(?P<duration>[\d.]+)", re.MULTILINE)


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_timestamp(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_transcript(path: Path) -> list[TranscriptSegment]:
    segments = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = TIMESTAMPED_LINE.fullmatch(line)
        if not match:
            raise ValueError(
                f"Invalid transcript line {line_number} in {path.name}: {raw_line}"
            )
        segments.append(
            TranscriptSegment(
                start=parse_timestamp(match.group("start")),
                end=parse_timestamp(match.group("end")),
                text=match.group("text").strip(),
            )
        )
    if not segments:
        raise ValueError(f"No transcript segments found in: {path}")
    return segments


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{description} not found: {path}")


def run_stage(name: str, command: list[str]) -> str:
    print(f"\n=== {name} ===")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{name} failed with exit code {return_code}")
    return "".join(output_lines)


def merge_adjacent_chunks(chunks: list[dict]) -> list[dict]:
    sections: list[dict] = []
    for chunk in chunks:
        topic_id = int(chunk["topic_id"])
        if sections and sections[-1]["topic_id"] == topic_id:
            sections[-1]["end_seconds"] = float(chunk["end"])
            sections[-1]["text"] = (
                f"{sections[-1]['text']} {str(chunk['text']).strip()}"
            ).strip()
        else:
            sections.append(
                {
                    "topic_id": topic_id,
                    "start_seconds": float(chunk["start"]),
                    "end_seconds": float(chunk["end"]),
                    "text": str(chunk["text"]).strip(),
                }
            )
    return sections


def section_for_output(section: dict) -> dict:
    return {
        "start": format_timestamp(section["start_seconds"]),
        "end": format_timestamp(section["end_seconds"]),
        "text": section["text"],
    }


def excerpt(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    shortened = text[: limit - 1].rsplit(" ", 1)[0]
    return f"{shortened}…"


def build_final_json(
    media_path: Path,
    transcript_path: Path,
    clusters_path: Path,
    language: str,
    media_duration: float | None = None,
) -> dict:
    transcript_segments = parse_transcript(transcript_path)
    cluster_data = json.loads(clusters_path.read_text(encoding="utf-8"))
    chunks = cluster_data.get("chunks", [])
    if not chunks:
        raise ValueError("Clustering output contains no chunks")

    topic_lookup = {
        int(topic["topic_id"]): topic for topic in cluster_data.get("topics", [])
    }
    sections = merge_adjacent_chunks(chunks)
    transcript_end = max(segment.end for segment in transcript_segments)
    duration = max(media_duration or 0.0, transcript_end)

    topics = []
    for topic_id, topic_info in topic_lookup.items():
        topic_sections = [
            section for section in sections if section["topic_id"] == topic_id
        ]
        if not topic_sections:
            continue
        covered_seconds = sum(
            section["end_seconds"] - section["start_seconds"]
            for section in topic_sections
        )
        topics.append(
            {
                "topic_id": topic_id,
                "label": topic_info["label"],
                "keywords": topic_info["keywords"],
                "share_of_recording": round(
                    (covered_seconds / duration * 100) if duration else 0.0,
                    2,
                ),
                "first_mention": format_timestamp(
                    topic_sections[0]["start_seconds"]
                ),
                "excerpt": excerpt(topic_sections[0]["text"]),
                "sections": [
                    section_for_output(section) for section in topic_sections
                ],
            }
        )
    topics.sort(key=lambda topic: topic["first_mention"])

    labels = {topic["topic_id"]: topic["label"] for topic in topics}
    timeline = [
        {
            "topic_id": section["topic_id"],
            "label": labels[section["topic_id"]],
            "start": format_timestamp(section["start_seconds"]),
            "end": format_timestamp(section["end_seconds"]),
        }
        for section in sections
    ]

    return {
        "metadata": {
            "file_name": media_path.name,
            "duration": format_timestamp(duration),
            "duration_seconds": round(duration, 3),
            "language": language,
            "number_of_topics": len(topics),
        },
        "topics": topics,
        "timeline": timeline,
        "transcript": transcript_path.read_text(encoding="utf-8").strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run transcription, chunking, embedding, clustering, and JSON output."
    )
    parser.add_argument("input_file", type=Path, help="Input audio or video file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Final JSON path (default: <input>_topics.json)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Intermediate artifact directory (default: <input>_pipeline)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        media_path = args.input_file.expanduser().resolve()
        if not media_path.is_file():
            raise ValueError(f"Input file not found: {media_path}")

        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else media_path.with_name(f"{media_path.stem}_topics.json")
        )
        if output_path.suffix.lower() != ".json":
            raise ValueError("The final output must use the .json extension")

        work_dir = (
            args.work_dir.expanduser().resolve()
            if args.work_dir
            else media_path.with_name(f"{media_path.stem}_pipeline")
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = work_dir / f"{media_path.stem}_transcription.txt"
        chunks_path = work_dir / f"{media_path.stem}_chunks.txt"
        embeddings_path = work_dir / f"{media_path.stem}_embeddings.npz"
        clusters_path = work_dir / f"{media_path.stem}_clusters.json"

        speech_python = (
            TRANSCRIPTION_ROOT / "speech_to_text" / ".venv" / "Scripts" / "python.exe"
        )
        embedding_python = (
            TRANSCRIPTION_ROOT / "embedding" / ".venv" / "Scripts" / "python.exe"
        )
        clustering_python = (
            TRANSCRIPTION_ROOT / "clustering" / ".venv" / "Scripts" / "python.exe"
        )
        speech_script = TRANSCRIPTION_ROOT / "speech_to_text" / "transcribe.py"
        chunk_script = TRANSCRIPTION_ROOT / "chunking" / "chunking.py"
        embedding_script = TRANSCRIPTION_ROOT / "embedding" / "embed_chunks.py"
        clustering_script = TRANSCRIPTION_ROOT / "clustering" / "cluster_embeddings.py"

        for path, description in (
            (speech_python, "Speech-to-text Python environment"),
            (embedding_python, "Embedding Python environment"),
            (clustering_python, "Clustering Python environment"),
            (speech_script, "Speech-to-text script"),
            (chunk_script, "Chunking script"),
            (embedding_script, "Embedding script"),
            (clustering_script, "Clustering script"),
        ):
            require_file(path, description)

        transcription_output = run_stage(
            "Transcription",
            [
                str(speech_python),
                str(speech_script),
                str(media_path),
                "--output",
                str(transcript_path),
            ],
        )
        language_match = LANGUAGE_LINE.search(transcription_output)
        language = language_match.group("language") if language_match else "unknown"
        duration_match = DURATION_LINE.search(transcription_output)
        media_duration = (
            float(duration_match.group("duration")) if duration_match else None
        )

        run_stage(
            "Chunking",
            [
                sys.executable,
                str(chunk_script),
                str(transcript_path),
                "--output",
                str(chunks_path),
            ],
        )
        run_stage(
            "Embedding",
            [
                str(embedding_python),
                str(embedding_script),
                str(chunks_path),
                "--output",
                str(embeddings_path),
            ],
        )
        run_stage(
            "Clustering and labelling",
            [
                str(clustering_python),
                str(clustering_script),
                str(embeddings_path),
                "--output",
                str(clusters_path),
            ],
        )

        result = build_final_json(
            media_path,
            transcript_path,
            clusters_path,
            language,
            media_duration,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("\n=== Final JSON ===")
        print(f"Topics: {result['metadata']['number_of_topics']}")
        print(f"Saved output: {output_path}")
        print(f"Intermediate artifacts: {work_dir}")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
