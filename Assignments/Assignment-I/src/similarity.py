"""WordNet pairwise similarity with the assignment's `> 0.5` rule."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

logger = logging.getLogger(__name__)


@dataclass
class SimPair:
    word1: str
    word2: str
    score: float
    metric: str


def _ensure_wordnet() -> bool:
    try:
        import nltk

        for pkg, path in (("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
        from nltk.corpus import wordnet  # noqa: F401

        return True
    except Exception as exc:
        logger.warning("WordNet unavailable: %s", exc)
        return False


def _synset(word: str, pos: str | None = None):
    from nltk.corpus import wordnet as wn

    # Use first synset (most frequent sense) — standard baseline.
    return wn.synsets(word, pos=pos)


def pair_score(w1: str, w2: str, metric: str = "path") -> float | None:
    """Similarity in [0, 1] or None when incomparable/missing."""
    from nltk.corpus import wordnet as wn  # noqa

    # WordNet is lowercase; normalise so "First" == "first".
    s1 = _synset(w1.lower())
    s2 = _synset(w2.lower())
    if not s1 or not s2:
        return None
    a, b = s1[0], s2[0]
    try:
        if metric == "wup":
            s = a.wup_similarity(b)
        else:
            s = a.path_similarity(b)
    except Exception:
        return None
    return float(s) if s is not None else None


def pairwise_similarity(
    words: list[str],
    metric: str = "path",
    threshold: float = 0.5,
) -> list[SimPair]:
    """Score every unordered pair; keep only `score > threshold`.

    `words` should already be frequency-capped (e.g. top-80) because this is
    O(N^2) WordNet lookups.
    """
    if not _ensure_wordnet():
        return []
    kept: list[SimPair] = []
    for w1, w2 in combinations(sorted(set(words)), 2):
        s = pair_score(w1, w2, metric=metric)
        if s is not None and s > threshold:
            kept.append(SimPair(w1, w2, round(s, 4), metric))
    kept.sort(key=lambda p: p.score, reverse=True)
    logger.info(
        "WordNet %s: %d pairs above %.2f (from %d words)",
        metric,
        len(kept),
        threshold,
        len(set(words)),
    )
    return kept
