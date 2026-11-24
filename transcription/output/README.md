# Final Topic JSON

This component runs the complete local pipeline from an audio or video file:

1. Transcribe the recording into timestamped segments.
2. Combine segments into approximately 40-word chunks.
3. Embed each chunk with `all-MiniLM-L6-v2`.
4. Cluster and label the chunks with BERTopic and agglomerative clustering.
5. Merge adjacent same-topic chunks and write the final JSON.

Set up the speech-to-text, embedding, and clustering environments described in
their respective READMEs before running the combined pipeline.

## Usage

From the repository root:

```powershell
.\transcription\output\run_pipeline.bat "C:\path\to\recording.mp3"
```

The final output is saved beside the recording as
`<recording-name>_topics.json`. Intermediate transcripts, chunks, embeddings,
and cluster assignments are retained in `<recording-name>_pipeline`.

Choose another final output or working directory with:

```powershell
.\transcription\output\run_pipeline.bat "C:\path\to\recording.mp3" `
    --output "C:\path\to\result.json" `
    --work-dir "C:\path\to\pipeline-artifacts"
```

## JSON structure

The final JSON contains:

- `metadata`: file name, duration, detected language, and number of topics
- `topics`: label, keywords, share of recording, first mention, excerpt, and
  every timestamped section for each topic
- `timeline`: the chronological order in which topic sections appear
- `transcript`: the complete timestamped transcript

`share_of_recording` is the percentage of the recording duration covered by a
topic's sections.
