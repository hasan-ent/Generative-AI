"""Word-level tokenizer + vocabulary with encode/decode and persistence."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

PAD, UNK = "<pad>", "<unk>"


class WordTokenizer:
    """Minimal word tokenizer.

    - `use_nltk=False`: regex/whitespace baseline (iteration 1).
    - `use_nltk=True` : `nltk.word_tokenize` + alphabetic filtering.
    Vocabulary is frequency-pruned by `min_freq`, with <pad>=0, <unk>=1.
    """

    def __init__(
        self,
        lowercase: bool = True,
        min_freq: int = 2,
        use_nltk: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.min_freq = min_freq
        self.use_nltk = use_nltk
        self.word2id: dict[str, int] = {PAD: 0, UNK: 1}
        self.id2word: dict[int, str] = {0: PAD, 1: UNK}
        self.freq: Counter[str] = Counter()

    # -- tokenization -----------------------------------------------------
    def _nltk_tokens(self, sentence: str) -> list[str]:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
        from nltk.tokenize import word_tokenize

        toks = word_tokenize(sentence)
        # Keep alphabetic tokens only (drops punctuation/numbers).
        toks = [t for t in toks if re.search(r"[A-Za-z]", t)]
        return toks

    def tokenize(self, sentence: str) -> list[str]:
        if self.lowercase:
            sentence = sentence.lower()
        if self.use_nltk:
            try:
                toks = self._nltk_tokens(sentence)
            except Exception as exc:
                logger.warning("word_tokenize failed (%s); regex fallback", exc)
                toks = re.findall(r"[A-Za-z']+", sentence)
        else:
            toks = re.findall(r"\S+", sentence)
            toks = [re.sub(r"^[^A-Za-z']+|[^A-Za-z']+$", "", t) for t in toks]
            toks = [t for t in toks if t]
        if self.lowercase:
            toks = [t.lower() for t in toks]
        return toks

    # -- vocabulary -------------------------------------------------------
    def build_vocab(self, tokenized: list[list[str]]) -> None:
        self.freq = Counter(t for sent in tokenized for t in sent)
        kept = sorted(
            [w for w, c in self.freq.items() if c >= self.min_freq],
            key=lambda w: (-self.freq[w], w),
        )
        self.word2id = {PAD: 0, UNK: 1}
        self.word2id.update({w: i + 2 for i, w in enumerate(kept)})
        self.id2word = {i: w for w, i in self.word2id.items()}
        logger.info(
            "vocab: %d types (%d tokens), min_freq=%d",
            len(self.word2id),
            sum(self.freq.values()),
            self.min_freq,
        )

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.word2id.get(t, self.word2id[UNK]) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id2word.get(i, UNK) for i in ids]

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    def most_common(self, n: int) -> list[str]:
        return [w for w, _ in self.freq.most_common(n)]

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "word2id": self.word2id,
            "freq": dict(self.freq),
            "config": {
                "lowercase": self.lowercase,
                "min_freq": self.min_freq,
                "use_nltk": self.use_nltk,
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        tok = cls(
            lowercase=cfg.get("lowercase", True),
            min_freq=cfg.get("min_freq", 2),
            use_nltk=cfg.get("use_nltk", True),
        )
        tok.word2id = {k: int(v) for k, v in payload["word2id"].items()}
        tok.id2word = {i: w for w, i in tok.word2id.items()}
        from collections import Counter as _C

        tok.freq = _C({k: int(v) for k, v in payload.get("freq", {}).items()})
        return tok
