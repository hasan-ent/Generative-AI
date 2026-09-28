# Assignment-I: Shakespeare → Tokens → LM → WordNet → Embeddings

Tiny-Shakespeare (GitHub) → sentence/word tokenize → `token_ids.csv` → bigram LM → WordNet all-pairs `wup > 0.5` → top-5 co-occurrence embeddings. See `ITERATIONS.md`.

## Run
```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python main.py --all-iterations  # or --iteration iter5_final
```

## Layout
`config.py` `main.py` `src/` `tests/` `data/` `outputs/` (`token_ids.csv`, `vocab.json`, `language_model.json`, `wordnet_similarity.csv`, `top5_embeddings.{csv,npy}`, `iter*/`)

## Result (final)
200 sents, 3272 toks, V=339, PPL=107.55, 110 pairs >0.5. Top: `done-made 1.0`, `man-sir 0.947`, `city-rome 0.9`.
