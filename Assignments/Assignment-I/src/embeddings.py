"""Embeddings for the top-5 WordNet pairs.

Strategy (final iteration): symmetric co-occurrence counts within a sliding
window over the tokenized corpus, restricted to the target words, then
L2-normalised rows. Deterministic (seeded) random fallback keeps early
iterations runnable even when WordNet yields < 5 pairs.

Why co-occurrence and not Word2Vec? Zero extra dependencies (numpy only),
fully explainable for a hard-copy submission, and good enough to show that
WordNet-similar words obtain non-trivial cosine similarity.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def target_words_from_pairs(pairs, top_k: int = 5) -> list[str]:
    """Unique words from the top-K pairs, order-preserving."""
    seen: list[str] = []
    for p in pairs[:top_k]:
        for w in (p.word1, p.word2):
            if w not in seen:
                seen.append(w)
    return seen


def cooccurrence_embeddings(
    tokenized: list[list[str]],
    targets: list[str],
    dim: int | None = None,
    window: int = 2,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Build `|targets|`-dim co-occurrence vectors (one axis per target word).

    `dim` is accepted for API compatibility: if dim != len(targets) we project
    with a fixed random matrix so callers always get `dim`-wide vectors.
    """
    idx = {w: i for i, w in enumerate(targets)}
    n = len(targets)
    mat = np.zeros((n, n), dtype=float)
    for sent in tokenized:
        filt = [t for t in sent if t in idx]
        for i, w in enumerate(filt):
            for j in range(max(0, i - window), min(len(filt), i + window + 1)):
                if i != j:
                    mat[idx[w], idx[filt[j]]] += 1.0
    # Row-normalise (avoid div-by-zero for isolated words).
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    if dim is not None and dim != n:
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((n, dim))
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-12
        mat = mat @ proj
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return {w: mat[idx[w]] for w in targets}


def random_embeddings(words: list[str], dim: int = 50, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((len(words), dim))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return {w: vecs[i] for i, w in enumerate(words)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cosine_table(embeddings: dict[str, np.ndarray]) -> list[tuple[str, str, float]]:
    words = list(embeddings)
    out = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            out.append((words[i], words[j], round(cosine(embeddings[words[i]], embeddings[words[j]]), 4)))
    out.sort(key=lambda r: r[2], reverse=True)
    return out
