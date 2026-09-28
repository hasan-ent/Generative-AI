# Assignment-I: Shakespeare → Tokens → LM → WordNet → Embeddings

Tiny-Shakespeare (GitHub) → sentence/word tokenize → `token_ids.csv` → bigram LM → WordNet all-pairs `wup > 0.5` → top-5 co-occurrence embeddings. See `ITERATIONS.md`.

## Run (venv only — all flows use `.venv`)
```bash
make venv        # python3 -m venv --system-site-packages .venv
make install     # pip install -r requirements.txt (offline-tolerant)
make test        # .venv/bin/python -m pytest tests/ -q
make run         # .venv/bin/python main.py --all-iterations
make latex       # pdflatex handwritten.tex -> handwritten.pdf
```

## Layout
`config.py` `main.py` `src/` `tests/` `data/` `outputs/` (`token_ids.csv`, `vocab.json`, `language_model.json`, `wordnet_similarity.csv`, `top5_embeddings.{csv,npy}`, `iter*/`) `handwritten.{tex,pdf}`

## Result (final)
200 sents, 3272 toks, V=339, PPL=107.55, 110 pairs >0.5. Top: `done-made 1.0`, `man-sir 0.947`, `city-rome 0.9`.
