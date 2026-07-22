# Building the manuscript

No LaTeX toolchain ships on Prajna (no `pdflatex`, `xelatex`, or system TeX).
`tectonic` works without root and fetches only the packages the document needs:

```bash
conda create -y -p "$PRAJNA_ROOT/texenv" -c conda-forge tectonic
export PATH="$PRAJNA_ROOT/texenv/bin:$PATH"
export TECTONIC_CACHE_DIR="$PRAJNA_ROOT/tectonic-cache"

cd paper
mkdir -p build
tectonic -X compile prioritykv.tex --outdir build --keep-logs
```

First run downloads fonts and maps (a few minutes); later runs hit the cache.

## Verified build

`tectonic 0.16.9`, 2026-07-23:

- 14 pages
- 0 errors, 0 undefined references, 0 missing figures
- 0 overfull/underfull boxes

## Checks worth repeating after edits

```bash
# structure, before spending a compile
python3 - <<'PY'
import re
s = open("paper/prioritykv.tex").read()
assert s.count("{") == s.count("}"), "unbalanced braces"
for env in ("figure", "table", "tabular", "equation", "abstract", "document"):
    assert s.count(rf"\begin{{{env}}}") == s.count(rf"\end{{{env}}}"), env
labels = set(re.findall(r"\\label\{([^}]+)\}", s))
refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
assert not (refs - labels), f"undefined refs: {refs - labels}"
print("structure OK")
PY

# then the log
grep -iE "^! |LaTeX Error|Undefined|Overfull" build/prioritykv.log
```

External figures are regenerated from tracked summary JSON, never hand-edited:

```bash
uv run python scripts/make_external_figures.py --tag primary --llama-tag llama
```

It writes both PNG and PDF; the manuscript includes the PDF.
