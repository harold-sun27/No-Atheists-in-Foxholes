#!/usr/bin/env python3
"""Transcribe an MP3 or MP4 file locally with faster-whisper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp3", ".mp4"}
DEFAULT_MODEL = "turbo"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an MP3 or MP4 recording into a timestamped text transcript."
    )
    parser.add_argument("input_file", type=Path, help="Path to an .mp3 or .mp4 file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output text file (default: <input>_transcription.txt)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="faster-whisper model name or local model path (default: turbo)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        help="CTranslate2 compute type (default: float16 on CUDA, int8 on CPU)",
    )
    parser.add_argument(
        "--language",
        help="Optional language code such as en; detected automatically when omitted",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Write plain text without timestamps",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable voice-activity detection",
    )
    return parser.parse_args()


def validate_input(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Input file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{path.suffix}'. Use: {supported}")
    return path


def choose_runtime(
    requested_device: str, requested_compute_type: str | None
) -> tuple[str, str]:
    if requested_device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = requested_device

    compute_type = requested_compute_type or (
        "float16" if device == "cuda" else "int8"
    )
    return device, compute_type


def load_model(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    print(f"Loading '{model_name}' on {device} ({compute_type})...")
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        return model, device, compute_type
    except Exception as exc:
        if device != "cuda":
            raise RuntimeError(f"Could not load the model: {exc}") from exc

        print(
            f"CUDA could not be initialized ({exc}). Falling back to CPU int8.",
            file=sys.stderr,
        )
        try:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            return model, "cpu", "int8"
        except Exception as cpu_exc:
            raise RuntimeError(
                f"Could not load the model on CUDA or CPU: {cpu_exc}"
            ) from cpu_exc


def transcribe(args: argparse.Namespace) -> Path:
    input_path = validate_input(args.input_file)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_transcription.txt")
    )
    if output_path.suffix.lower() != ".txt":
        raise ValueError("The output file must use the .txt extension")

    device, compute_type = choose_runtime(args.device, args.compute_type)
    model, device, compute_type = load_model(args.model, device, compute_type)

    print(f"Transcribing: {input_path.name}")
    segments, info = model.transcribe(
        str(input_path),
        beam_size=5,
        language=args.language,
        vad_filter=not args.no_vad,
    )
    segments = list(segments)
    if not segments:
        raise RuntimeError("No speech was detected in the input file")

    if args.plain:
        transcript = " ".join(segment.text.strip() for segment in segments)
    else:
        transcript = "\n".join(
            f"[{format_timestamp(segment.start)} --> "
            f"{format_timestamp(segment.end)}] {segment.text.strip()}"
            for segment in segments
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcript + "\n", encoding="utf-8")

    language = info.language or "unknown"
    probability = getattr(info, "language_probability", None)
    confidence = f" ({probability:.1%})" if probability is not None else ""
    print(f"Language: {language}{confidence}")
    print(f"Runtime: {device} ({compute_type})")
    print(f"Saved transcript: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    try:
        transcribe(args)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
