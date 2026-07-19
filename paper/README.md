# Paper source

`prioritykv.tex` is the canonical handwritten arXiv source. It is synchronized with
`prioritykv_manuscript.md` and uses only tracked PDF figures under `figures/`.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error prioritykv.tex
```

Or build from the repository root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/prioritykv.tex
```

The source intentionally contains:

```tex
\graphicspath{{figures/}{paper/figures/}}
```

This resolves figures when either `paper/` or the repository root is the TeX/Overleaf
project root.

Regenerate all eight SVG/PDF figure pairs and both PNG review scales from the repository
root:

```bash
uv run python scripts/make_publication_figures.py
```

The required figure set is:

- `agent_trace_failure_mode`
- `page_allocation_architecture`
- `decode_memory_lifetime`
- `hypothesis_split`
- `eviction_and_baselines`
- `budget_and_transfer`
- `lock240_quality_by_length`
- `systems_tradeoff`

`prioritykv.pdf` is the compiled canonical manuscript.  The former pre-P0–P3
draft export has been retired so GitHub exposes only the evidence-current paper.

Before submission, both authors must verify name spelling, affiliation, author order,
the compiled PDF, and the claim boundary in `../docs/EVIDENCE.md`.
