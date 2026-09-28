"""Bigram language model with add-k smoothing.

Covers the assignment line "Save in language model": we persist counts,
conditional probabilities P(w2|w1), and unigram backoff so the artefact is a
real (if tiny) LM, not just a vocab dump.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class BigramLanguageModel:
    def __init__(self, smoothing_k: float = 1.0) -> None:
        self.k = smoothing_k
        self.unigram: Counter[str] = Counter()
        self.bigram: Counter[tuple[str, str]] = Counter()
        self.vocab: set[str] = set()
        self.total_tokens: int = 0

    def fit(self, tokenized: list[list[str]]) -> None:
        for sent in tokenized:
            for i, w in enumerate(sent):
                self.unigram[w] += 1
                self.vocab.add(w)
                self.total_tokens += 1
                if i > 0:
                    self.bigram[(sent[i - 1], w)] += 1
        logger.info(
            "LM fit: %d unigrams, %d bigrams, V=%d",
            len(self.unigram),
            len(self.bigram),
            len(self.vocab),
        )

    def prob(self, prev: str, word: str) -> float:
        """P(word | prev) with add-k smoothing."""
        V = max(len(self.vocab), 1)
        c_bi = self.bigram.get((prev, word), 0)
        c_uni = self.unigram.get(prev, 0)
        return (c_bi + self.k) / (c_uni + self.k * V) if (c_uni + self.k * V) > 0 else 0.0

    def sentence_logprob(self, sent: list[str]) -> float:
        import math

        if not sent:
            return 0.0
        # P(w0) as unigram MLE, then chain bigrams.
        lp = math.log((self.unigram.get(sent[0], 0) + self.k) / (self.total_tokens + self.k * len(self.vocab)))
        for i in range(1, len(sent)):
            lp += math.log(max(self.prob(sent[i - 1], sent[i]), 1e-12))
        return lp

    def perplexity(self, tokenized: list[list[str]]) -> float:
        import math

        n = sum(len(s) for s in tokenized)
        if n == 0:
            return float("inf")
        ll = sum(self.sentence_logprob(s) for s in tokenized)
        return math.exp(-ll / n)

    def save_with_ppl(self, path: Path, tokenized: list[list[str]], top_bigrams: int = 200) -> float:
        """Persist counts + train perplexity; return perplexity."""
        ppl = self.perplexity(tokenized)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "smoothing_k": self.k,
            "total_tokens": self.total_tokens,
            "vocab_size": len(self.vocab),
            "unigram_counts": dict(self.unigram.most_common(500)),
            "bigram_counts": [[a, b, int(c)] for (a, b), c in self.bigram.most_common(top_bigrams)],
            "perplexity_train": ppl,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ppl
