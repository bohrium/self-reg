%.pdf: %.tex sammath.sty
	pdflatex $<
	rm *.aux *.log
	evince $@

all: bound.pdf landscape.pdf perturb.pdf surd.pdf notes.pdf

notes: notes.pdf
