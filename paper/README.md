# Paper source

`prioritykv.tex` is the standalone arXiv source. It uses only tracked PDF figures from
`figures/`; no generated result or external image is required at compile time.

Build from this directory with a standard TeX Live installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error prioritykv.tex
```

`prioritykv_manuscript.md` is the readable source manuscript. When its scientific text is
changed, update the TeX source in the same commit and regenerate figures from the repository
root with `uv run python scripts/make_publication_figures.py`.

Before submission, both authors must verify name spelling, affiliation, author order, and
the final PDF produced by the arXiv compiler.
