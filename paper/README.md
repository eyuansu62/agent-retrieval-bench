# Agent Retrieval Bench Paper

The paper is available on arXiv:

- [Abstract](https://arxiv.org/abs/2607.24882)
- [PDF](https://arxiv.org/pdf/2607.24882)

This directory contains the LaTeX source used to build the paper.

Files:

- `main.tex`: paper source.
- `references.bib`: BibTeX references.
- `Makefile`: local build helper.

Build:

```bash
cd paper
make
```

The current environment may not have a TeX distribution installed. If `latexmk`, `pdflatex`, or `bibtex` is missing, install a standard TeX Live distribution or compile the files in Overleaf.
