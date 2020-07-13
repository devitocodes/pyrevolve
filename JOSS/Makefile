all: pyrevolve.pdf

KNIT = Rscript -e "require(rmarkdown); render('$<')"

pyrevolve.pdf: pyrevolve.md
	$(KNIT)
