VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# All flows MUST use the venv interpreter ($(PY)).
# Venv inherits system site-packages so it works offline;
# `make install` only upgrades pip and installs requirements
# (best-effort — skipped gracefully when offline).

venv:
	python3 -m venv --system-site-packages $(VENV)
	$(PY) -c "import nltk, pandas, numpy; print('venv ok')"

install: venv
	$(PIP) install --upgrade pip || true
	$(PIP) install -r requirements.txt || echo "offline: using system site-packages"

test: venv
	$(PY) -m pytest tests/ -q

run: venv
	$(PY) main.py --all-iterations

final: venv
	$(PY) main.py --iteration iter5_final

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
	rm -f handwritten.aux handwritten.log handwritten.out handwritten.synctex.gz handwritten.fls handwritten.fdb_latexmk

latex:
	pdflatex -interaction=nonstopmode handwritten.tex
	rm -f handwritten.aux handwritten.log handwritten.out handwritten.synctex.gz handwritten.fls handwritten.fdb_latexmk
