# Iterations — at least 5, end-to-end

All runs: `python main.py --all-iterations` (code in `config.iteration_configs()`).
Dataset fixed (200 sents). Only pipeline logic changes per iteration.

| # | Tag | Change from previous | Sent/Tok/V | Train PPL | Pairs > 0.5 | Embedding |
|---|-----|----------------------|------------|-----------|-------------|-----------|
| 1 | `iter1_baseline` | Whitespace split, keep case, `min_freq=1`, `k=0`, `path`, random-16d, window=1 | 200 / 3194 / 1168 | 7.18 | **1** (`First–first`, 1.0 — case-duplicate artefact) | random (`First, first`) |
| 2 | `iter2_hygiene` | Lowercase + NLTK `sent/word_tokenize`, `min_freq=2`, `k=1`, stopword filter for candidates, random-32d | 200 / 3272 / 339 | 107.55 | **0** (`path` too strict: needs distance 0) → fallback | random-fallback (`first, citizen, marcius, …`) |
| 3 | `iter3_wordnet` | **Metric fix: `path` → `wup_similarity`** (standard measure where > 0.5 is meaningful), vocab 50→80, dim 50 | 200 / 3272 / 339 | 107.55 | **110** | co-occurrence (first real top-5) |
| 4 | `iter4_cooccur` | Embeddings: random → **co-occurrence** (window=2, L2-norm, seeded projection to 50-d) | same | 107.55 | 110 | co-occurrence (`done, made, man, sir, city, rome, …`) |
| 5 | `iter5_final` | Same as 4 + full artefacts: `embedding_cosine.png`, `stats.json`, latest `outputs/` mirror | same | 107.55 | 110 | co-occurrence + heatmap |

## What each iteration taught

**Iter-1 (baseline, deliberately naive).**
Whitespace tokenisation keeps punctuation (`speak.` ≠ `speak`), no lowercasing
splits `First`/`first`, no smoothing (`k=0`) overfits train (PPL 7.18 — misleadingly low).
`path_similarity` yields a single trivial pair: the case duplicate. Lesson: need hygiene.

**Iter-2 (hygiene).**
Lowercasing + NLTK + `min_freq=2` collapses vocab 1168 → 339 and stabilises tokens
(3272). Add-1 smoothing raises train PPL to 107.55 (correct: smoothing reserves mass
for unseen events). Stopword filtering cleans candidates, but `path > 0.5` still gives
0 pairs. Lesson: the metric itself is the bottleneck (`path = 1/(dist+1)`; `> 0.5`
requires identical synsets).

**Iter-3 (WordNet fix).**
Switch to `wup_similarity` (Wu-Palmer, depth-normalised, routinely > 0.5 for related
words). Verified: `king–queen path=0.10 / wup=0.57`. Result jumps 0 → 110 pairs.
Top: `done–made 1.0`, `man–sir 0.947`, `city–rome 0.9`. Lesson: match the measure to
the threshold.

**Iter-4 (embeddings).**
Random vectors cannot reflect corpus context. Replace with symmetric co-occurrence
counts (window=2) over the tokenised corpus, L2-normalised, projected to 50-d with a
fixed seed. Cosine table becomes interpretable (e.g. `sir–like 0.84`). Lesson:
embeddings must derive from the same corpus as the LM.

**Iter-5 (final, submission).**
Freeze iter-4 config, add `embedding_cosine.png` heatmap, per-iteration folders
(`outputs/<tag>/`) plus latest mirror (`outputs/*.csv|json|npy`), `stats.json` with
top pairs + top cosines. All 8 brief lines verified in `outputs/`.

## Reproduce

```bash
python main.py --all-iterations
ls outputs/iter*/stats.json
cat outputs/iter5_final/stats.json
```

## Files per iteration

Each `outputs/<tag>/` contains:
`token_ids.csv`, `vocab.json`, `language_model.json`, `wordnet_similarity.csv`,
`top5_embeddings.{csv,npy}`, `stats.json` (+ `embedding_cosine.png` for final).
