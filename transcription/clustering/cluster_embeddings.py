#!/usr/bin/env python3
"""Cluster transcript embeddings and label the discovered topics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DISTANCE_THRESHOLD = 0.80
TOP_KEYWORDS = 5

FALLBACK_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
}


def load_embeddings(path: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Could not read embedding file: {exc}") from exc

    required = {"embeddings", "texts", "start_times", "end_times", "model"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Embedding file is missing: {', '.join(sorted(missing))}")

    embeddings = data["embeddings"]
    texts = data["texts"].astype(str).tolist()
    starts = data["start_times"].astype(float).tolist()
    ends = data["end_times"].astype(float).tolist()
    model_name = str(data["model"])

    count = len(texts)
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array, received {embeddings.shape}")
    if not count:
        raise ValueError("Embedding file contains no chunks")
    if not (embeddings.shape[0] == len(starts) == len(ends) == count):
        raise ValueError("Embedding vectors, texts, and timestamps have different lengths")
    if any(end < start for start, end in zip(starts, ends)):
        raise ValueError("A chunk ends before it starts")

    return embeddings, texts, starts, ends, model_name


def fallback_keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
    counts = Counter(word for word in words if word not in FALLBACK_STOP_WORDS)
    return [word for word, _ in counts.most_common(TOP_KEYWORDS)] or ["transcript"]


def cluster_topics(texts: list[str], embeddings):
    """Return per-chunk topic IDs and BERTopic keyword representations."""
    if len(texts) == 1:
        keywords = fallback_keywords(texts[0])
        return [0], {0: [(word, 1.0) for word in keywords]}

    try:
        from bertopic import BERTopic
        from bertopic.dimensionality import BaseDimensionalityReduction
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise RuntimeError(
            "Clustering dependencies are not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    cluster_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="average",
        compute_full_tree=True,
    )
    topic_model = BERTopic(
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=cluster_model,
        vectorizer_model=CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        ),
        top_n_words=TOP_KEYWORDS,
        calculate_probabilities=False,
        verbose=True,
    )

    try:
        assignments, _ = topic_model.fit_transform(texts, embeddings)
    except Exception as exc:
        raise RuntimeError(f"Could not cluster embeddings: {exc}") from exc

    representations = {
        int(topic_id): topic_model.get_topic(int(topic_id)) or []
        for topic_id in sorted(set(assignments))
    }
    return [int(topic_id) for topic_id in assignments], representations


def build_result(
    source_path: Path,
    embeddings,
    texts: list[str],
    starts: list[float],
    ends: list[float],
    embedding_model: str,
) -> dict:
    assignments, representations = cluster_topics(texts, embeddings)

    topic_records = []
    for topic_id in sorted(set(assignments)):
        representation = representations.get(topic_id, [])
        keywords = [str(word) for word, _ in representation[:TOP_KEYWORDS]]
        if not keywords:
            topic_text = " ".join(
                text for text, assigned in zip(texts, assignments) if assigned == topic_id
            )
            keywords = fallback_keywords(topic_text)
        topic_records.append(
            {
                "topic_id": topic_id,
                "label": " / ".join(keywords[:3]),
                "keywords": keywords,
            }
        )

    chunks = [
        {
            "index": index,
            "start": start,
            "end": end,
            "text": text,
            "topic_id": topic_id,
        }
        for index, (start, end, text, topic_id) in enumerate(
            zip(starts, ends, texts, assignments)
        )
    ]

    return {
        "source_embeddings": source_path.name,
        "embedding_model": embedding_model,
        "clustering": {
            "method": "AgglomerativeClustering",
            "metric": "cosine",
            "linkage": "average",
            "distance_threshold": DISTANCE_THRESHOLD,
            "topic_representation": "BERTopic c-TF-IDF",
        },
        "number_of_topics": len(topic_records),
        "topics": topic_records,
        "chunks": chunks,
    }


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_embeddings"):
        stem = stem[: -len("_embeddings")]
    return input_path.with_name(f"{stem}_clusters.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster chunk embeddings and label topics with BERTopic."
    )
    parser.add_argument("input_file", type=Path, help="Embedding .npz file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (default: <input>_clusters.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        input_path = args.input_file.expanduser().resolve()
        if not input_path.is_file():
            raise ValueError(f"Input file not found: {input_path}")
        if input_path.suffix.lower() != ".npz":
            raise ValueError("The embedding input must use the .npz extension")

        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else default_output_path(input_path)
        )
        if output_path.suffix.lower() != ".json":
            raise ValueError("The clustering output must use the .json extension")

        embeddings, texts, starts, ends, model_name = load_embeddings(input_path)
        result = build_result(
            input_path,
            embeddings,
            texts,
            starts,
            ends,
            model_name,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Discovered {result['number_of_topics']} topics")
        for topic in result["topics"]:
            print(f"Topic {topic['topic_id']}: {topic['label']}")
        print(f"Saved clusters: {output_path}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
