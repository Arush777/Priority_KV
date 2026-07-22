#!/usr/bin/env python3
"""Generate `paper/body_conf.tex` from `paper/body.tex`.

The two build targets share one body so the arXiv and conference PDFs cannot
disagree. The only difference is layout: a two-column measure is too narrow for
the wide figures and tables, so those floats are promoted to their starred
full-width forms and the ADAPT equation is split across two lines.

Edit `body.tex`; never edit `body_conf.tex` by hand.

    uv run python scripts/make_conference_body.py
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"

WIDE_FIGURES = (
    "protected_fraction_boundary", "external_bfcl_arms", "eviction_and_baselines",
    "budget_and_transfer", "lock240_quality_by_length", "systems_tradeoff",
    "decode_memory_lifetime", "page_allocation_architecture",
    "agent_trace_failure_mode", "hypothesis_split",
)

# A row wider than this cannot fit a ~3.4in column without overfulling.
MAX_ROW_CHARS = 78


def promote(text: str, env: str, opt: str, should) -> str:
    """Convert whole environments to their starred form, matching begin/end pairs."""
    out, i = [], 0
    pattern = re.compile(r"\\begin\{%s\}(\[[^\]]*\])?(.*?)\\end\{%s\}" % (env, env), re.S)
    for m in pattern.finditer(text):
        out.append(text[i:m.start()])
        block = m.group(0)
        if should(block):
            block = re.sub(r"^\\begin\{%s\}(\[[^\]]*\])?" % env,
                           r"\\begin{%s*}%s" % (env, opt), block)
            block = block.replace(r"\end{%s}" % env, r"\end{%s*}" % env)
        out.append(block)
        i = m.end()
    out.append(text[i:])
    return "".join(out)


def table_is_wide(block: str) -> bool:
    rows = [ln for ln in block.splitlines() if r"\\" in ln and "&" in ln]
    return any(len(ln) > MAX_ROW_CHARS for ln in rows)


SINGLE_COL_EQUATION = r"""\begin{equation}
  \alpha = \min\!\left(1, \frac{\text{keep budget}}{\text{protected mass}}\right),
  \qquad
  s = \alpha \cdot \mathrm{rank}(s_{\text{struct}}) + (1-\alpha)\cdot
      \mathrm{rank}(s_{\text{attn}}),
  \label{eq:adapt}
\end{equation}"""

TWO_COL_EQUATION = r"""\begin{align}
  \alpha &= \min\!\left(1,\ \frac{\text{keep budget}}{\text{protected mass}}\right),
  \label{eq:adapt-alpha}\\[2pt]
  s &= \alpha\,\mathrm{rank}(s_{\text{struct}})
     + (1-\alpha)\,\mathrm{rank}(s_{\text{attn}}).
  \label{eq:adapt}
\end{align}"""


def main() -> int:
    body = (PAPER / "body.tex").read_text()

    out = promote(body, "figure", "[!tbp]",
                  lambda b: any(w in b for w in WIDE_FIGURES))
    out = promote(out, "table", "[!tbp]", table_is_wide)

    if SINGLE_COL_EQUATION in out:
        out = out.replace(SINGLE_COL_EQUATION, TWO_COL_EQUATION, 1)
    elif r"\label{eq:adapt-alpha}" not in out:
        raise SystemExit("ADAPT equation not found in body.tex; update this script")

    for env in ("figure*", "table*"):
        b, e = out.count(r"\begin{%s}" % env), out.count(r"\end{%s}" % env)
        if b != e:
            raise SystemExit(f"unbalanced {env}: {b} begin vs {e} end")

    dest = PAPER / "body_conf.tex"
    dest.write_text(out)
    print(f"wrote {dest}")
    n_fig = out.count(chr(92) + "begin{figure*}")
    n_tab = out.count(chr(92) + "begin{table*}")
    print(f"  figure*: {n_fig}  table*: {n_tab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
