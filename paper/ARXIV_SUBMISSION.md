# arXiv submission handoff

## Ready artifacts

- Compiled paper: `prioritykv_arxiv.pdf`
- Upload source: `dist/prioritykv_arxiv_source.tar.gz`
- Source checksum manifest: `dist/SHA256SUMS`
- Root TeX file inside the archive: `prioritykv_arxiv.tex`

The source archive contains `prioritykv_arxiv.tex`, `body.tex`, and only the ten
PDF figures referenced by the manuscript.  It intentionally does not contain a
precompiled PDF, auxiliary files, SVG/PNG review assets, conference sources, or
unused figures.

## Suggested arXiv metadata

- Title: **PriorityKV: Structure-Aware KV Retention with a Measurable Operating Boundary**
- Primary category: `cs.CL`
- Possible cross-list: `cs.LG`

The category and license are author choices.  Before upload, both authors should
confirm the title, author order, affiliation spelling, abstract, category,
license, and any workshop anonymity requirement.

## Upload check

1. Upload `dist/prioritykv_arxiv_source.tar.gz` and select
   `prioritykv_arxiv.tex` as the main file if arXiv does not detect it.
2. Inspect arXiv's processed PDF, especially Figures 3--5 and 9--10.
3. Confirm that the processed PDF is 14 pages, has no missing figures, and shows
   the same title and author metadata as `prioritykv_arxiv.pdf`.
4. Do not upload `prioritykv_conference.tex`; it is a neutral workshop-layout
   build, not an official venue style.
