# Speech-to-Text

This module turns recorded audio or video into a timestamped transcript using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). It accepts MP3,
MP4, and WAV files and performs transcription locally.

## Project scope

The Archivist is an interactive history project built around recorded and
written source material. Transcription is the first step in preparing that
material for later stages such as topic organization, knowledge retrieval,
interactive dialogue, and voice output.

This directory covers transcription only. Topic extraction, language-model
integration, and speech generation belong in separate modules.

## Setup

Python 3.11 is recommended. From the repository root:

```powershell
cd transcription\speech_to_text
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The environment only needs to be created once. After setup, `transcribe.bat`
selects the correct Python interpreter automatically, so manual activation is
not required.

## Usage

```powershell
.\transcribe.bat "C:\path\to\recording.wav"
```

MP3 and MP4 inputs work the same way:

```powershell
.\transcribe.bat "C:\path\to\recording.mp3"
.\transcribe.bat "C:\path\to\video.mp4"
```

The default model is `large-v3`. It runs on CUDA with `int8_float16` compute by
default. The model is downloaded and cached on its first use; subsequent
transcriptions use the local cached copy.

The output is saved beside the input as `<input-name>_transcription.txt`.
Segments include their start and end times:

```text
[00:00:00.000 --> 00:00:04.250] This is the first transcribed segment.
```

## Options

Choose an output file:

```powershell
.\transcribe.bat recording.wav --output output\transcription.txt
```

Write plain text without timestamps:

```powershell
.\transcribe.bat recording.wav --plain
```

Run on the CPU when CUDA is unavailable:

```powershell
.\transcribe.bat recording.wav --device cpu
```

Show all available options:

```powershell
.\transcribe.bat --help
```
