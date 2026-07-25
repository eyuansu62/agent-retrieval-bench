# Agent Retrieval Bench Paper

This directory contains a lightweight LaTeX skeleton converted from the V1.1 Markdown draft.

Files:

- `main.tex`: continuous paper draft.
- `references.bib`: BibTeX references copied from `docs/paper_references_v1_1.bib`.
- `Makefile`: local build helper.

Build:

```bash
cd paper
make
```

The current environment may not have a TeX distribution installed. If `latexmk`, `pdflatex`, or `bibtex` is missing, install a standard TeX Live distribution or compile the files in Overleaf.

Open TODOs:

- Replace `article` with the target venue template.
- Convert tables to the final venue width.
- Add task-definition and result figures.
- Fill the runtime/cache table once GPU wall-clock logs are available.
