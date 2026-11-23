# Chunk Embeddings

This pipeline stage converts each timestamped transcript chunk into a vector
representing its meaning. It uses the pretrained
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
model through `sentence-transformers`.

The model produces one 384-dimensional embedding per chunk. These vectors are
saved with the chunk text and timestamps so the next stage can cluster related
chunks without parsing the transcript again.

## Setup

Python 3.11 is recommended. From the repository root:

```powershell
cd transcription\embedding
..\speech_to_text\.venv\Scripts\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The environment only needs to be created once. The launcher selects it
automatically after setup.

## Usage

```powershell
.\embed_chunks.bat "C:\path\to\recording_chunks.txt"
```

The model is downloaded and cached on its first use. Later runs use the local
cached copy.

The default output is `<recording-name>_embeddings.npz`. The compressed file
contains:

- `embeddings`: one 384-value vector per chunk
- `texts`: the original chunk text
- `start_times`: chunk start times in seconds
- `end_times`: chunk end times in seconds
- `model`: the embedding model identifier

Choose a different output location with:

```powershell
.\embed_chunks.bat "C:\path\to\recording_chunks.txt" `
    --output "C:\path\to\recording_embeddings.npz"
```
