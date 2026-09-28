"""Dataset helpers: download + load + sentence-split Tiny Shakespeare."""
from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

FALLBACK_TEXT = """\
To be, or not to be: that is the question.
Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.
Or to take arms against a sea of troubles, and by opposing end them.
To die: to sleep; no more; and by a sleep to say we end.
The heart-ache and the thousand natural shocks that flesh is heir to.
'Tis a consummation devoutly to be wish'd.
To die, to sleep; to sleep: perchance to dream.
For in that sleep of death what dreams may come.
When we have shuffled off this mortal coil, must give us pause.
There is the respect that makes calamity of so long life.
"""


def download_if_missing(url: str, dest: Path, timeout: int = 30) -> Path:
    """Download `url` to `dest` unless it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("dataset exists: %s (%d bytes)", dest, dest.stat().st_size)
        return dest
    logger.info("downloading %s -> %s", url, dest)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # offline / blocked -> fallback sample
        logger.warning("download failed (%s); writing fallback sample", exc)
        dest.write_text(FALLBACK_TEXT, encoding="utf-8")
    return dest


def load_raw_text(path: Path, max_chars: int = 20000) -> str:
    """Read raw text, capped at `max_chars` to keep the assignment small."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        logger.info("truncating raw text %d -> %d chars", len(text), max_chars)
        text = text[:max_chars]
    return text


def split_sentences_nltk(text: str) -> list[str]:
    """Split into sentences with NLTK punkt; fallback to regex if missing."""
    try:
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
        from nltk.tokenize import sent_tokenize

        sents = sent_tokenize(text)
        sents = [s.strip() for s in sents if s.strip()]
        if sents:
            return sents
    except Exception as exc:
        logger.warning("NLTK sent_tokenize failed (%s); using regex", exc)
    # Regex fallback: split on .!? followed by whitespace/newline.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_sentences(
    raw_path: Path, url: str, max_sentences: int = 200, max_chars: int = 20000
) -> list[str]:
    """End-to-end: ensure data, load, sentence-split, cap count."""
    download_if_missing(url, raw_path)
    text = load_raw_text(raw_path, max_chars=max_chars)
    # Normalise whitespace but keep sentence punctuation for sent_tokenize.
    text = re.sub(r"\s+", " ", text).strip()
    sentences = split_sentences_nltk(text)
    if len(sentences) > max_sentences:
        logger.info("capping sentences %d -> %d", len(sentences), max_sentences)
        sentences = sentences[:max_sentences]
    logger.info("loaded %d sentences", len(sentences))
    return sentences
