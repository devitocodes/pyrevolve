all: paper.html

KNIT = Rscript -e "require(rmarkdown); render('$<')"

paper.html: paper.md
	$(KNIT)
