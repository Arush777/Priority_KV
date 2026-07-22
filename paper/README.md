# Paper source

Two build targets share one body, so content cannot drift between them.

| File | Target | Layout |
|---|---|---|
| `prioritykv_arxiv.tex` | arXiv preprint | single column, no page limit |
| `prioritykv_conference.tex` | conference submission | two column, page-limited |
| `body.tex` | shared prose, tables, floats | edit **here** |
| `body_conf.tex` | generated from `body.tex` | wide floats promoted to `figure*`/`table*` |

**Edit `body.tex` only.** Regenerate the conference body afterwards:

```bash
uv run python scripts/make_conference_body.py
```

## Build

```bash
export PATH="$PRAJNA_ROOT/texenv/bin:$PATH"
export TECTONIC_CACHE_DIR="$PRAJNA_ROOT/tectonic-cache"
mkdir -p build
tectonic -X compile prioritykv_arxiv.tex      --outdir build --keep-logs
tectonic -X compile prioritykv_conference.tex --outdir build --keep-logs
```

Verified with `tectonic 0.15.0` on 2026-07-22:

| Target | Pages | Errors | Overfull |
|---|---:|---:|---:|
| arXiv | 14 | 0 | 0 |
| conference | 13 | 0 | 4 |

The conference target uses a neutral two-column layout so it compiles anywhere.
When the venue is fixed, drop its official `.sty` beside the file and swap the
`\documentclass`/`geometry` block — the header comment in
`prioritykv_conference.tex` shows the exact replacement for ICLR and NeurIPS.
At 13 two-column pages it is over a typical 9-page main-text limit; trim from
Sections 5 and 6, which carry the most detail already covered by the appendix
material in `docs/`.

## Figures

One command regenerates and validates all ten figures.  The two external plots
use the tracked submission snapshot because the originating cluster filesystem
is not part of this checkout:

```bash
uv run python scripts/make_publication_figures.py
```

`\graphicspath{{figures/}{paper/figures/}}` resolves figures whether `paper/`
or the repository root is the project root.

Before submission, both authors must verify name spelling, affiliation, author
order, the compiled PDF, and the claim boundary in `../docs/EVIDENCE.md`.
