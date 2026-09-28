"""Central configuration for Assignment-I.

Best practice: single source of truth for hyper-parameters, paths and
thresholds. Each iteration overrides a subset of these defaults
(see ITERATIONS.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
SRC_DIR = BASE_DIR / "src"

# Canonical tiny-Shakespeare source (Karpathy char-rnn mirror, GitHub).
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
RAW_DATA_PATH = DATA_DIR / "tinyshakespeare.txt"

# Output artefacts (final iteration).
TOKENS_CSV = OUTPUTS_DIR / "token_ids.csv"
VOCAB_JSON = OUTPUTS_DIR / "vocab.json"
LM_JSON = OUTPUTS_DIR / "language_model.json"
SIMILARITY_CSV = OUTPUTS_DIR / "wordnet_similarity.csv"
EMBEDDING_CSV = OUTPUTS_DIR / "top5_embeddings.csv"
EMBEDDING_NPY = OUTPUTS_DIR / "top5_embeddings.npy"
STATS_JSON = OUTPUTS_DIR / "stats.json"


@dataclass(frozen=True)
class PipelineConfig:
    """All tunable knobs for one pipeline run."""

    # data
    max_sentences: int = 200  # keep assignment "small" + WordNet O(V^2) tractable
    max_chars: int = 20000  # safety cap on raw text read
    # tokenizer
    lowercase: bool = True
    min_freq: int = 2  # drop hapax/singletons from vocab
    use_nltk_word_tokenize: bool = True
    # language model
    smoothing_k: float = 1.0  # add-k for bigram
    # wordnet
    max_vocab_for_similarity: int = 80  # top-N frequent words only
    similarity_threshold: float = 0.5  # assignment condition: > 0.5
    similarity_metric: str = "path"  # "path" | "wup"
    top_k_pairs: int = 20  # keep top-K pairs above threshold for inspection
    top_k_embed: int = 5  # assignment: embedding for top-5 values
    # embeddings
    embedding_dim: int = 50
    embedding_window: int = 2  # co-occurrence window
    random_seed: int = 42

    # derived artefact prefix for iteration logging
    tag: str = "final"


# Five documented iterations (oldest -> newest). Each entry maps to a
# PipelineConfig override + a hypothesis about what will improve.
ITERATION_OVERRIDES: list[dict] = field(default_factory=list)  # placeholder


def iteration_configs() -> list[PipelineConfig]:
    """Return the 5+ iteration configs executed in `main.py --all-iterations`.

    Key learning (documented in ITERATIONS.md): `path_similarity > 0.5`
    is almost never satisfied (requires distance 0), so iterations 1-2
    demonstrate the failure, and 3+ switch to `wup_similarity`, the standard
    WordNet measure where > 0.5 is meaningful.
    """
    return [
        # Iter-1: naive baseline. Whitespace split, no lowercasing, unigram-ish.
        # Uses path_similarity -> expect ~0 pairs (documents the pitfall).
        PipelineConfig(
            tag="iter1_baseline",
            lowercase=False,
            min_freq=1,
            use_nltk_word_tokenize=False,
            smoothing_k=0.0,
            max_vocab_for_similarity=30,
            similarity_threshold=0.5,
            similarity_metric="path",
            embedding_dim=16,
            embedding_window=1,
        ),
        # Iter-2: hygiene. Lowercase + NLTK sentence/word tokenize, min_freq.
        # Still path -> still ~0 pairs, but vocab/LM improve.
        PipelineConfig(
            tag="iter2_hygiene",
            lowercase=True,
            min_freq=2,
            use_nltk_word_tokenize=True,
            smoothing_k=1.0,
            max_vocab_for_similarity=50,
            similarity_threshold=0.5,
            similarity_metric="path",
            embedding_dim=32,
            embedding_window=2,
        ),
        # Iter-3: fix similarity metric -> wup. First iteration with real pairs.
        PipelineConfig(
            tag="iter3_wordnet",
            lowercase=True,
            min_freq=2,
            use_nltk_word_tokenize=True,
            smoothing_k=1.0,
            max_vocab_for_similarity=80,
            similarity_threshold=0.5,
            similarity_metric="wup",
            embedding_dim=50,
            embedding_window=2,
        ),
        # Iter-4: same metric, co-occurrence embeddings (vs random).
        PipelineConfig(
            tag="iter4_cooccur",
            lowercase=True,
            min_freq=2,
            use_nltk_word_tokenize=True,
            smoothing_k=1.0,
            max_vocab_for_similarity=80,
            similarity_threshold=0.5,
            similarity_metric="wup",
            top_k_pairs=20,
            embedding_dim=50,
            embedding_window=2,
        ),
        # Iter-5 (final): same as iter-4 but full artefacts + eval + plots.
        PipelineConfig(
            tag="iter5_final",
            lowercase=True,
            min_freq=2,
            use_nltk_word_tokenize=True,
            smoothing_k=1.0,
            max_vocab_for_similarity=80,
            similarity_threshold=0.5,
            similarity_metric="wup",
            top_k_pairs=20,
            top_k_embed=5,
            embedding_dim=50,
            embedding_window=2,
        ),
    ]
