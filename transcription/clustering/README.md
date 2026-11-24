# Topic Clustering and Labelling

This stage groups semantically similar transcript chunks into topics and labels
each topic with its most distinctive words.

It uses BERTopic for c-TF-IDF topic representations and scikit-learn's
agglomerative clustering over the precomputed `all-MiniLM-L6-v2` embeddings.
The number of topics is discovered automatically from a cosine-distance
threshold instead of being specified in advance.

## Setup

From the repository root:

```powershell
cd transcription\clustering
..\speech_to_text\.venv\Scripts\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The environment only needs to be created once. The launcher selects it
automatically after setup.

## Usage

```powershell
.\cluster_embeddings.bat "C:\path\to\recording_embeddings.npz"
```

The default output is `<recording-name>_clusters.json`. It contains:

- the automatically discovered topic IDs, labels, and keywords
- every chunk's topic assignment, text, and timestamps
- the embedding model and clustering configuration used

Choose a different output location with:

```powershell
.\cluster_embeddings.bat "C:\path\to\recording_embeddings.npz" `
    --output "C:\path\to\recording_clusters.json"
```
