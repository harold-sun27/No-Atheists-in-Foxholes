#!/usr/bin/env python3
"""
revoice.py - Transcribe an audio file with WhisperX, then have Fish Audio S2-Pro
read the transcript back in a different (cloned) voice.

Pipeline
    input audio --WhisperX--> transcript --S2-Pro (voice clone of --voice)--> output audio

Two ways to drive S2-Pro:

  --backend server  (recommended)
      Talks to a running fish-speech API server, so the 4B model stays loaded.
      Start it once from your fish-speech repo:
          python tools/api_server.py --listen 127.0.0.1:8080 \
              --llama-checkpoint-path checkpoints/s2-pro \
              --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
              --decoder-config-name modded_dac_vq

  --backend cli
      Calls fish-speech's command-line scripts directly. No server needed, but the
      model reloads for every chunk of text, so it's slow on long recordings.

Examples
    # Server backend, voice cloned from a 10-30 s clip (its transcript is auto-detected)
    python revoice.py interview.mp3 --voice narrator.wav -o interview_revoiced.wav

    # CLI backend, fish-speech living in its own venv
    python revoice.py interview.mp3 --voice narrator.wav -o out.wav --backend cli \
        --fish-repo ~/fish-speech --fish-python ~/fish-speech/.venv/bin/python

    # Fix WhisperX mistakes first: run once, edit out.txt, then re-run with it
    python revoice.py interview.mp3 --voice narrator.wav -o out.wav --transcript out.txt
"""

from __future__ import annotations

import argparse
import gc
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


# --------------------------------------------------------------------------- #
# WhisperX
# --------------------------------------------------------------------------- #

def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def transcribe_files(paths, model_name, device, compute_type, batch_size, language):
    """Transcribe each file with one loaded WhisperX model.

    Returns a list of (segments, language) where segments are
    [{"start": float, "end": float, "text": str}, ...].
    """
    import whisperx

    print(f"[whisperx] loading {model_name} on {device} ({compute_type})")
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)

    results = []
    for p in paths:
        print(f"[whisperx] transcribing {p}")
        audio = whisperx.load_audio(str(p))
        res = model.transcribe(audio, batch_size=batch_size)
        segs = [
            {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
            for s in res["segments"]
            if s["text"].strip()
        ]
        results.append((segs, res.get("language")))

    # Release VRAM before S2-Pro (which wants ~24 GB) gets going.
    del model
    free_gpu()
    return results


# --------------------------------------------------------------------------- #
# Text chunking
# --------------------------------------------------------------------------- #

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split one over-long string on sentence ends, then on words if needed."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for sent in _SENTENCE_SPLIT.split(text):
        if len(sent) <= max_chars:
            pieces.append(sent)
            continue
        cur = ""
        for word in sent.split():
            if cur and len(cur) + 1 + len(word) > max_chars:
                pieces.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        if cur:
            pieces.append(cur)
    return pieces


def chunk_text(units: list[str], max_chars: int) -> list[str]:
    """Pack transcript units (segments or paragraphs) into chunks <= max_chars."""
    pieces = [p for u in units for p in _split_long(u.strip(), max_chars) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------- #
# Audio helpers (ffmpeg is already a WhisperX requirement)
# --------------------------------------------------------------------------- #

def ffmpeg(*args: str) -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH (WhisperX needs it too).")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def prepare_reference(voice_path: Path, voice_text: str | None, voice_segments,
                      workdir: Path, max_seconds: float) -> tuple[Path, str]:
    """Return (mono wav path, exact transcript) for the voice-cloning reference.

    If the user gave no transcript, use WhisperX segments and trim the clip at a
    segment boundary so the audio and text match exactly (S2 wants 10-30 s).
    """
    out = workdir / "reference.wav"
    if voice_text:
        ffmpeg("-i", str(voice_path), "-ac", "1", str(out))
        return out, voice_text.strip()

    if not voice_segments:
        sys.exit("WhisperX found no speech in the --voice clip.")

    start = voice_segments[0]["start"]
    keep = [voice_segments[0]]
    for seg in voice_segments[1:]:
        if seg["end"] - start > max_seconds:
            break
        keep.append(seg)
    end = keep[-1]["end"] + 0.15
    text = " ".join(s["text"] for s in keep)

    print(f"[voice] using {end - start:.1f}s of the reference clip: \"{text[:80]}...\"")
    ffmpeg("-i", str(voice_path), "-ss", f"{max(start - 0.05, 0):.3f}", "-to", f"{end:.3f}",
           "-ac", "1", str(out))
    return out, text


# --------------------------------------------------------------------------- #
# S2-Pro backends
# --------------------------------------------------------------------------- #

class ServerBackend:
    """Uses fish-speech's tools/api_server.py (POST /v1/tts, msgpack body)."""

    def __init__(self, url: str, ref_wav: Path, ref_text: str, api_key: str | None,
                 chunk_length: int, temperature: float, top_p: float, timeout: float):
        import requests  # noqa: F401  (fail early if missing)

        self.base = url.rstrip("/")
        self.ref_audio = ref_wav.read_bytes()
        self.ref_text = ref_text
        self.headers = {"content-type": "application/msgpack"}
        if api_key:
            self.headers["authorization"] = f"Bearer {api_key}"
        self.chunk_length = chunk_length
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self._pack = self._get_packer()
        self._check_health()

    @staticmethod
    def _get_packer():
        try:
            import ormsgpack
            return ormsgpack.packb
        except ImportError:
            pass
        try:
            import msgpack
            return lambda obj: msgpack.packb(obj, use_bin_type=True)
        except ImportError:
            sys.exit("Install a msgpack library in this environment: pip install msgpack")

    def _check_health(self):
        import requests
        try:
            r = requests.get(f"{self.base}/v1/health", timeout=10)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Can't reach the fish-speech API server at {self.base} ({e}).\n"
                     "Start it (see the top of this file) or use --backend cli.")

    def synth(self, text: str, idx: int) -> tuple[np.ndarray, int]:
        import requests
        payload = {
            "text": text,
            "references": [{"audio": self.ref_audio, "text": self.ref_text}],
            "reference_id": None,
            "format": "wav",
            "chunk_length": self.chunk_length,
            "normalize": True,
            "streaming": False,
            "use_memory_cache": "on",   # server caches the encoded reference
            "max_new_tokens": 1024,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": None,
        }
        r = requests.post(f"{self.base}/v1/tts", data=self._pack(payload),
                          headers=self.headers, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"/v1/tts returned {r.status_code}: {r.text[:500]}")
        audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        return audio, sr


class CliBackend:
    """Runs fish-speech's 3-step command-line inference in subprocesses."""

    def __init__(self, repo: Path, checkpoint: Path, python: str, ref_wav: Path, ref_text: str,
                 workdir: Path, compile_: bool, half: bool):
        self.repo = repo.expanduser().resolve()
        self.ckpt = (checkpoint if checkpoint.is_absolute() else self.repo / checkpoint).resolve()
        self.codec = self.ckpt / "codec.pth"
        self.dac_script = self.repo / "fish_speech/models/dac/inference.py"
        self.t2s_script = self.repo / "fish_speech/models/text2semantic/inference.py"
        for p in (self.dac_script, self.t2s_script, self.codec):
            if not p.exists():
                sys.exit(f"Not found: {p}  (check --fish-repo / --checkpoint)")
        self.python = python
        self.ref_text = ref_text
        self.workdir = workdir
        self.extra = (["--compile"] if compile_ else []) + (["--half"] if half else [])
        self.env = {**os.environ,
                    "PYTHONPATH": f"{self.repo}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}

        # Step 1 (once): encode the reference voice to VQ tokens -> fake.npy
        ref_dir = workdir / "ref"
        ref_dir.mkdir()
        print("[s2] encoding reference voice")
        self._run([str(self.dac_script), "-i", str(ref_wav),
                   "--checkpoint-path", str(self.codec)], cwd=ref_dir)
        self.ref_tokens = ref_dir / "fake.npy"
        if not self.ref_tokens.exists():
            sys.exit("Reference encoding didn't produce fake.npy - see the output above.")

    def _run(self, args: list[str], cwd: Path) -> None:
        subprocess.run([self.python, *args], cwd=cwd, env=self.env, check=True)

    def synth(self, text: str, idx: int) -> tuple[np.ndarray, int]:
        d = self.workdir / f"chunk_{idx:04d}"
        d.mkdir()
        # Step 2: text -> semantic tokens (codes_0.npy)
        self._run([str(self.t2s_script), "--text", text,
                   "--prompt-text", self.ref_text,
                   "--prompt-tokens", str(self.ref_tokens),
                   "--checkpoint-path", str(self.ckpt), *self.extra], cwd=d)
        codes = d / "codes_0.npy"
        if not codes.exists():
            raise RuntimeError(f"text2semantic produced no codes_0.npy in {d}")
        # Step 3: semantic tokens -> audio (fake.wav)
        self._run([str(self.dac_script), "-i", str(codes),
                   "--checkpoint-path", str(self.codec)], cwd=d)
        audio, sr = sf.read(d / "fake.wav", dtype="float32")
        return audio, sr


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Transcribe audio with WhisperX and re-read it in another voice with Fish Audio S2-Pro.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", type=Path, help="Audio file to transcribe")
    ap.add_argument("--voice", type=Path, required=True,
                    help="Reference clip of the target voice (ideally 10-30 s of clean speech)")
    ap.add_argument("--voice-text", help="Exact transcript of --voice (auto-transcribed if omitted)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output audio (.wav, or any format ffmpeg can write). Default: <input>_revoiced.wav")
    ap.add_argument("--transcript", type=Path,
                    help="Use this text file instead of transcribing the input (blank lines = paragraph breaks)")
    ap.add_argument("--transcript-out", type=Path,
                    help="Where to save the WhisperX transcript. Default: next to --output as .txt")

    g = ap.add_argument_group("WhisperX")
    g.add_argument("--whisper-model", default="large-v3")
    g.add_argument("--whisper-device", default=None, help="cuda or cpu (default: cuda if available)")
    g.add_argument("--compute-type", default=None, help="float16 / int8 (default: float16 on cuda, int8 on cpu)")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--language", default=None, help="Force a language code, e.g. en")

    g = ap.add_argument_group("Fish Audio S2")
    g.add_argument("--backend", choices=["server", "cli"], default="server")
    g.add_argument("--server-url", default="http://127.0.0.1:8080")
    g.add_argument("--api-key", default=None, help="If the server was started with --api-key")
    g.add_argument("--fish-repo", type=Path, default=Path("."), help="[cli] path to your fish-speech checkout")
    g.add_argument("--checkpoint", type=Path, default=Path("checkpoints/s2-pro"),
                   help="[cli] S2-Pro checkpoint dir (relative to --fish-repo unless absolute)")
    g.add_argument("--fish-python", default=sys.executable,
                   help="[cli] Python interpreter of the env where fish-speech is installed")
    g.add_argument("--compile", action="store_true", help="[cli] pass --compile to text2semantic")
    g.add_argument("--half", action="store_true", help="[cli] pass --half (GPUs without bf16)")
    g.add_argument("--max-chars", type=int, default=500,
                   help="Characters of transcript per synthesis request")
    g.add_argument("--chunk-length", type=int, default=200, help="[server] S2 internal chunk length")
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-p", type=float, default=0.7)
    g.add_argument("--max-ref-seconds", type=float, default=30.0,
                   help="Trim an auto-transcribed --voice clip to about this length")
    g.add_argument("--gap", type=float, default=0.25, help="Seconds of silence between chunks")
    g.add_argument("--timeout", type=float, default=600, help="[server] seconds per request")
    g.add_argument("--keep-temp", action="store_true", help="Keep intermediate files for debugging")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for p in (args.input, args.voice):
        if not p.exists():
            sys.exit(f"File not found: {p}")
    output = args.output or args.input.with_name(f"{args.input.stem}_revoiced.wav")

    # ---- 1. Transcribe (input and, if needed, the voice reference) ----
    need_input = args.transcript is None
    need_voice = args.voice_text is None
    input_units: list[str] = []
    voice_segments = None

    if need_input or need_voice:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        device = args.whisper_device or ("cuda" if has_cuda else "cpu")
        compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")

        paths = ([args.input] if need_input else []) + ([args.voice] if need_voice else [])
        results = transcribe_files(paths, args.whisper_model, device, compute_type,
                                   args.batch_size, args.language)
        if need_input:
            segs, lang = results.pop(0)
            if not segs:
                sys.exit("WhisperX found no speech in the input.")
            input_units = [s["text"] for s in segs]
            transcript_out = args.transcript_out or output.with_suffix(".txt")
            transcript_out.write_text("\n".join(input_units) + "\n", encoding="utf-8")
            print(f"[whisperx] {len(segs)} segments, language={lang}; transcript -> {transcript_out}")
        if need_voice:
            voice_segments, _ = results.pop(0)

    if not need_input:
        raw = args.transcript.read_text(encoding="utf-8")
        input_units = [p.strip() for p in re.split(r"\n\s*\n|\n", raw) if p.strip()]

    chunks = chunk_text(input_units, args.max_chars)
    print(f"[s2] {len(chunks)} chunk(s) to synthesize")

    workdir = Path(tempfile.mkdtemp(prefix="revoice_"))
    try:
        # ---- 2. Prepare the voice reference ----
        ref_wav, ref_text = prepare_reference(args.voice, args.voice_text, voice_segments,
                                              workdir, args.max_ref_seconds)

        # ---- 3. Synthesize ----
        if args.backend == "server":
            tts = ServerBackend(args.server_url, ref_wav, ref_text, args.api_key,
                                args.chunk_length, args.temperature, args.top_p, args.timeout)
        else:
            if len(chunks) > 3:
                print("[s2] note: the cli backend reloads the model for every chunk; "
                      "the server backend is much faster for long inputs.")
            tts = CliBackend(args.fish_repo, args.checkpoint, args.fish_python, ref_wav, ref_text,
                             workdir, args.compile, args.half)

        pieces: list[np.ndarray] = []
        sample_rate = None
        for i, text in enumerate(chunks):
            print(f"[s2] chunk {i + 1}/{len(chunks)}: {text[:70]}{'...' if len(text) > 70 else ''}")
            audio, sr = tts.synth(text, i)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise RuntimeError(f"Sample rate changed between chunks ({sample_rate} -> {sr})")
            if pieces and args.gap > 0:
                pieces.append(np.zeros(int(args.gap * sr), dtype=np.float32))
            pieces.append(audio.astype(np.float32))

        # ---- 4. Write the result ----
        full = np.concatenate(pieces)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".wav":
            sf.write(output, full, sample_rate)
        else:
            tmp_wav = workdir / "combined.wav"
            sf.write(tmp_wav, full, sample_rate)
            ffmpeg("-i", str(tmp_wav), str(output))
        print(f"[done] {len(full) / sample_rate:.1f}s of audio -> {output}")
    finally:
        if args.keep_temp:
            print(f"[debug] intermediate files kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()