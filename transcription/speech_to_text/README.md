# Local Speech-to-Text

This folder contains the first stage of the Archivist transcription pipeline. It converts an MP3 or MP4 file into a local text transcript using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Model

The default model is `turbo` (`large-v3-turbo`), an optimized Whisper model with about 809 million parameters. It is comfortably below the 8-billion-parameter limit and provides a strong speed/accuracy balance. You can select another compatible model with `--model`; for example, `large-v3` is approximately 1.55 billion parameters and prioritizes accuracy over speed.

The model is downloaded on the first run and cached locally. Transcription then runs locally. MP3 and MP4 decoding is handled by PyAV, which bundles the required FFmpeg libraries.

## Setup

Python 3.10-3.12 is recommended.

```powershell
cd C:/Users/sarth/No-Atheists-in-Foxholes/transcription/speech_to_text
py -3.11 -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

## Transcribe a file

```powershell
python transcribe.py "C:/path/to/recording.mp3"
python transcribe.py "C:/path/to/video.mp4"
```

By default, the script creates `<input-name>_transcription.txt` beside the input file. Each line contains the segment's start time, end time, and transcription.

Example:

```text
[00:00:00.000 --> 00:00:04.250] This is the first transcribed segment.
```

Use a custom output location:

```powershell
python transcribe.py recording.mp3 --output output/transcription.txt
```

Write plain text without timestamps:

```powershell
python transcribe.py recording.mp3 --plain
```

Force CPU or CUDA explicitly:

```powershell
python transcribe.py recording.mp3 --device cpu
python transcribe.py recording.mp3 --device cuda
```

Show every option:

```powershell
python transcribe.py --help
```
