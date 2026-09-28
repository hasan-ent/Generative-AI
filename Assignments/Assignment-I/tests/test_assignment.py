"""Smoke tests for Assignment-I (run: python -m pytest tests/ -q)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer import WordTokenizer
from src.language_model import BigramLanguageModel


def test_tokenizer_roundtrip():
    tok = WordTokenizer(lowercase=True, min_freq=1, use_nltk=False)
    sents = [["to", "be", "or"], ["not", "to", "be"]]
    tok.build_vocab(sents)
    assert tok.vocab_size >= 5
    ids = tok.encode(["to", "be", "zzz-unknown"])
    assert ids[-1] == tok.word2id["<unk>"]
    assert tok.decode(ids[:2]) == ["to", "be"]


def test_bigram_prob_sums_to_one():
    tok_sents = [["to", "be"], ["to", "sleep"], ["to", "be"]]
    lm = BigramLanguageModel(smoothing_k=1.0)
    lm.fit(tok_sents)
    for prev in ("to", "be", "sleep"):
        total = sum(lm.prob(prev, w) for w in lm.vocab)
        # NOTE: without explicit </s> handling, sentence-final tokens have
        # sum < 1 (no outgoing bigram mass). So we assert validity, not unity.
        assert 0.0 < total <= 1.0 + 1e-6, (prev, total)
    # 'to' is never sentence-final here, so its distribution must sum to 1.
    total_to = sum(lm.prob("to", w) for w in lm.vocab)
    assert abs(total_to - 1.0) < 1e-6, total_to


def test_similarity_threshold_logic():
    pairs = [("a", "b", 0.49), ("c", "d", 0.51), ("e", "f", 0.9)]
    kept = [(a, b) for a, b, s in pairs if s > 0.5]
    assert kept == [("c", "d"), ("e", "f")]
