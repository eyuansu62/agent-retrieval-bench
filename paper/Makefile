LATEXMK ?= latexmk
PDFLATEX ?= pdflatex

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex; \
	else \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex; \
		bibtex main; \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex; \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex; \
	fi

clean:
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.pdf
