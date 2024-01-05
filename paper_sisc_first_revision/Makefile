FILE = localpsf
BIBS = 

LATEX=pdflatex
LATEXOPT=--shell-escape
NONSTOP=--interaction=nonstopmode

LATEXMK=latexmk
LATEXMKOPT=-pdf
CONTINUOUS=

$(FILE).pdf: $(FILE).tex $(BIBS)
	pdflatex $(FILE)
	bibtex $(FILE)
	pdflatex $(FILE)
	bibtex $(FILE)
	pdflatex $(FILE)

clean:
	rm -f  $(FILE).pdf $(FILE).aux $(FILE).cb2 $(FILE).bbl $(FILE).blg $(FILE).log $(FILE).out $(FILE).toc $(FILE).ind $(FILE).ilg $(FILE).cb $(FILE).idx $(FILE).dvi $(FILE).ps *~

diff:
	latexdiff $(FILE)_submitted.tex $(FILE).tex > $(FILE)_diff.tex
	$(LATEXMK) $(LATEXMKOPT) $(CONTINUOUS) \
            -pdflatex="$(LATEX) $(LATEXOPT) $(NONSTOP) %O %S" $(FILE)_diff
