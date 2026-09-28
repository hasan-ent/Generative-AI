"""Assignment-I end-to-end pipeline.

Covers every line of the handwritten brief:
  1. Small dataset (Shakespeare @ GitHub)
  2. Tokenize (sentences)
  3. Token IDs saved in CSV
  4. Save in language model
  5. import wordnet in python
  6. read each word; identify similarity against each word
  7. Apply condition > 0.5 similarity
  8. Create embedding for top-5 values

Usage:
    python main.py --iteration iter5_final   # single run
    python main.py --all-iterations          # all 5 documented iterations
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    DATA_DIR,
    OUTPUTS_DIR,
    RAW_DATA_PATH,
    TINY_SHAKESPEARE_URL,
    PipelineConfig,
    iteration_configs,
)
from src.dataset import load_sentences  # noqa: E402
from src.embeddings import (  # noqa: E402
    cooccurrence_embeddings,
    cosine_table,
    random_embeddings,
    target_words_from_pairs,
)
from src.language_model import BigramLanguageModel  # noqa: E402
from src.similarity import pairwise_similarity  # noqa: E402
from src.tokenizer import WordTokenizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger("assignment1")


def run_pipeline(cfg: PipelineConfig) -> dict:
    """Execute one full pass; write artefacts under outputs/<tag>/ + latest."""
    iter_dir = OUTPUTS_DIR / cfg.tag
    iter_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset -----------------------------------------------------------
    sentences = load_sentences(
        RAW_DATA_PATH, TINY_SHAKESPEARE_URL,
        max_sentences=cfg.max_sentences, max_chars=cfg.max_chars,
    )

    # 2. Tokenize ----------------------------------------------------------
    tok = WordTokenizer(
        lowercase=cfg.lowercase,
        min_freq=cfg.min_freq,
        use_nltk=cfg.use_nltk_word_tokenize,
    )
    tokenized = [tok.tokenize(s) for s in sentences]
    tokenized = [t for t in tokenized if t]  # drop empties after filtering
    tok.build_vocab(tokenized)

    # 3. Token IDs -> CSV --------------------------------------------------
    rows = [
        {
            "sentence_id": i,
            "sentence": sentences[i] if i < len(sentences) else "",
            "tokens": " ".join(toks),
            "token_ids": " ".join(map(str, tok.encode(toks))),
            "length": len(toks),
        }
        for i, toks in enumerate(tokenized)
    ]
    df_tokens = pd.DataFrame(rows)
    tok_csv = iter_dir / "token_ids.csv"
    df_tokens.to_csv(tok_csv, index=False)
    df_tokens.to_csv(OUTPUTS_DIR / "token_ids.csv", index=False)  # latest
    tok.save(iter_dir / "vocab.json")
    tok.save(OUTPUTS_DIR / "vocab.json")

    # 4. Language model ----------------------------------------------------
    # Best practice: LM shares the tokenizer vocab (OOV -> <unk>) so
    # token_ids.csv, vocab.json and language_model.json stay consistent.
    lm_tokens = [
        [t if t in tok.word2id else "<unk>" for t in sent] for sent in tokenized
    ]
    lm = BigramLanguageModel(smoothing_k=cfg.smoothing_k)
    lm.fit(lm_tokens)
    ppl = lm.save_with_ppl(iter_dir / "language_model.json", lm_tokens)
    lm.save_with_ppl(OUTPUTS_DIR / "language_model.json", lm_tokens)
    logger.info("[%s] LM train perplexity=%.2f", cfg.tag, ppl)

    # 5-7. WordNet similarity ---------------------------------------------
    # Best practice: exclude stopwords/function words (no WordNet synsets,
    # and they dominate raw frequency). This is the iter2->iter3 fix.
    try:
        import nltk

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as _sw

        stop = set(_sw.words("english"))
    except Exception:
        stop = {
            "the", "you", "to", "and", "i", "he", "it", "a", "in", "of",
            "is", "that", "your", "what", "not", "they", "his", "for",
            "we", "are", "with", "have", "all", "but", "my", "him",
            "this", "us", "our", "their", "was", "be", "me", "than",
            "as", "more", "one", "would", "if", "well", "at", "no",
            "which", "will", "on", "shall", "upon", "were", "them",
        }
    cand_words = tok.most_common(cfg.max_vocab_for_similarity * 4)
    # Keep alphabetic content words, len>2, not stopwords.
    cand_words = [
        w for w in cand_words
        if w.isalpha() and len(w) > 2 and w.lower() not in stop
    ][: cfg.max_vocab_for_similarity]
    pairs = pairwise_similarity(
        cand_words, metric=cfg.similarity_metric, threshold=cfg.similarity_threshold
    )
    top_pairs = pairs[: cfg.top_k_pairs]
    df_sim = pd.DataFrame(
        [{"word1": p.word1, "word2": p.word2, "score": p.score, "metric": p.metric} for p in top_pairs]
    )
    sim_csv = iter_dir / "wordnet_similarity.csv"
    if not df_sim.empty:
        df_sim.to_csv(sim_csv, index=False)
        df_sim.to_csv(OUTPUTS_DIR / "wordnet_similarity.csv", index=False)
    else:
        # Still write header-only CSV so downstream steps stay uniform.
        pd.DataFrame(columns=["word1", "word2", "score", "metric"]).to_csv(sim_csv, index=False)
        pd.DataFrame(columns=["word1", "word2", "score", "metric"]).to_csv(
            OUTPUTS_DIR / "wordnet_similarity.csv", index=False
        )
        logger.warning("[%s] no pairs above threshold — embeddings will use fallback", cfg.tag)

    # 8. Embeddings for top-5 ---------------------------------------------
    targets = target_words_from_pairs(top_pairs, top_k=cfg.top_k_embed)
    if len(targets) < 2:  # fallback: most frequent content words
        logger.warning("[%s] WordNet shortfall; fallback targets", cfg.tag)
        _fb = [
            w for w in tok.most_common(40)
            if w.isalpha() and len(w) > 2 and w.lower() not in stop
        ][: max(5, cfg.top_k_embed * 2)]
        # target_words_from_pairs returns up to 2*top_k words; mirror that.
        targets = _fb[: max(5, cfg.top_k_embed * 2)] or ["king", "queen"]
        use_cooccur = cfg.tag in ("iter4_cooccur", "iter5_final")
        embeddings = (
            cooccurrence_embeddings(tokenized, targets, dim=cfg.embedding_dim,
                                    window=cfg.embedding_window, seed=cfg.random_seed)
            if use_cooccur else random_embeddings(targets, dim=cfg.embedding_dim, seed=cfg.random_seed)
        )
        method = "cooccur-fallback" if use_cooccur else "random-fallback"
    else:
        # Iterations 1-2 used random as a baseline; 3+ use co-occurrence.
        if cfg.tag in ("iter1_baseline", "iter2_hygiene"):
            embeddings = random_embeddings(targets, dim=cfg.embedding_dim, seed=cfg.random_seed)
            method = "random"
        else:
            embeddings = cooccurrence_embeddings(
                tokenized, targets, dim=cfg.embedding_dim,
                window=cfg.embedding_window, seed=cfg.random_seed,
            )
            method = "cooccurrence"
    import numpy as np

    emb_mat = np.stack([embeddings[w] for w in targets])
    np.save(iter_dir / "top5_embeddings.npy", emb_mat)
    np.save(OUTPUTS_DIR / "top5_embeddings.npy", emb_mat)
    df_emb = pd.DataFrame(emb_mat, index=targets)
    df_emb.index.name = "word"
    df_emb.to_csv(iter_dir / "top5_embeddings.csv")
    df_emb.to_csv(OUTPUTS_DIR / "top5_embeddings.csv")
    cos_table = cosine_table(embeddings)

    # Stats ----------------------------------------------------------------
    stats = {
        "tag": cfg.tag,
        "n_sentences": len(tokenized),
        "n_tokens": int(sum(len(t) for t in tokenized)),
        "vocab_size": tok.vocab_size,
        "train_perplexity": round(float(ppl), 3),
        "n_candidate_words": len(cand_words),
        "n_pairs_above_threshold": len(pairs),
        "top_pairs": [{"word1": p.word1, "word2": p.word2, "score": p.score} for p in top_pairs[:5]],
        "embedding_words": targets,
        "embedding_method": method,
        "embedding_dim": cfg.embedding_dim,
        "top_cosine": [{"word1": a, "word2": b, "cosine": c} for a, b, c in cos_table[:5]],
        "config": cfg.__dict__,
    }
    (iter_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(
        "[%s] done: sents=%d toks=%d V=%d pairs>%s=%d embed=%s(%s)",
        cfg.tag, stats["n_sentences"], stats["n_tokens"], stats["vocab_size"],
        cfg.similarity_threshold, len(pairs), targets, method,
    )

    # Optional plot (final iteration only, matplotlib if available).
    if cfg.tag == "iter5_final":
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if len(targets) >= 2 and len(emb_mat) >= 2:
                # Cosine heatmap.
                norm = emb_mat / (np.linalg.norm(emb_mat, axis=1, keepdims=True) + 1e-12)
                cos = norm @ norm.T
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cos, vmin=-1, vmax=1)
                ax.set_xticks(range(len(targets)), targets, rotation=30, ha="right")
                ax.set_yticks(range(len(targets)), targets)
                ax.set_title("Top-word embedding cosine (co-occurrence)")
                fig.colorbar(im, ax=ax, label="cosine")
                fig.tight_layout()
                fig.savefig(iter_dir / "embedding_cosine.png", dpi=150)
                fig.savefig(OUTPUTS_DIR / "embedding_cosine.png", dpi=150)
                plt.close(fig)
                logger.info("saved cosine heatmap")
        except Exception as exc:
            logger.warning("plot skipped: %s", exc)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Assignment-I pipeline")
    ap.add_argument("--iteration", default="iter5_final",
                    help="one tag from iteration_configs()")
    ap.add_argument("--all-iterations", action="store_true",
                    help="run all 5 documented iterations in order")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    cfgs = iteration_configs()
    if args.all_iterations:
        all_stats = [run_pipeline(c) for c in cfgs]
        print(json.dumps([{k: s[k] for k in (
            "tag", "n_sentences", "n_tokens", "vocab_size", "train_perplexity",
            "n_pairs_above_threshold", "embedding_words", "embedding_method")} for s in all_stats], indent=2))
    else:
        match = [c for c in cfgs if c.tag == args.iteration]
        cfg = match[0] if match else PipelineConfig(tag=args.iteration)
        stats = run_pipeline(cfg)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
