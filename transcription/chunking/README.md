# Transcript Chunking

This pipeline stage combines consecutive timestamped transcript segments into
pieces of approximately 40 words. Larger chunks provide enough context for the
later embedding and topic-clustering stages while retaining the recording's
timestamps.

The chunker keeps Whisper segments intact and chooses boundaries close to the
40-word target. A short final remainder is merged into the preceding chunk.

## Usage

```powershell
cd transcription\chunking
.\chunk_transcript.bat "C:\path\to\recording_transcription.txt"
```

The default output is written beside the transcript as
`<recording-name>_chunks.txt`.

Choose a different output path with:

```powershell
.\chunk_transcript.bat "C:\path\to\recording_transcription.txt" `
    --output "C:\path\to\recording_chunks.txt"
```

Input lines must use the timestamped format produced by the speech-to-text
stage:

```text
[00:00:00.000 --> 00:00:04.250] This is a transcript segment.
```
