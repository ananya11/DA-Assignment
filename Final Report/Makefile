CS5525TARGETS: CS5525.pdf
.PHONY: clean

tex_files = $(wildcard *.tex */*.tex */*/*.tex */*/*/*.tex)
bib_files = $(wildcard *.bib */*.bib */*/*.bib */*/*/*.bib)
pdf_files = $(wildcard */*.pdf */*/*.pdf */*/*/*.pdf)

all: CS5525.pdf

CS5525.pdf: $(tex_files) $(bib_files) $(pdf_files)
	pdflatex CS5525.tex | tee latex.out ; \
	if grep -q '^LaTeX Warning: Citation.*undefined' latex.out; \
	then \
		bibtex CS5525; \
		touch .rebuild; \
	fi ;
	while [ -f .rebuild -o -n "`grep '^LaTeX Warning:.*Rerun' latex.out`" ]; \
	do \
		rm -f .rebuild; \
		pdflatex CS5525.tex | tee latex.out; \
	done ; \
	rm -f latex.out ;
	find . \( -name '*.blg' -print \) -or \( -name '*.aux' -print \) -or \
		\( -name '*.bbl' -print \) -or \( -name '*~' -print \) -or \
		\( -name '*.lof' -print \) -or \( -name '*.lot' -print \) -or \
		\( -name '*.toc' -print \) | xargs rm -f; \
	rm -f CS5525.log CS5525.out
	
clean:
	find . \( -name '*.blg' -print \) -or \( -name '*.aux' -print \) -or \
		\( -name '*.bbl' -print \) -or \( -name '*~' -print \) -or \
		\( -name '*.lof' -print \) -or \( -name '*.lot' -print \) -or \
		\( -name '*.toc' -print \) | xargs rm -f; \
	rm -f CS5525.log CS5525.out CS5525.pdf

camera_ready: CS5525.pdf
	gs -dSAFER -dNOPAUSE -dBATCH -sDEVICE=pdfwrite \
	    -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress          \
	    -dCompatibilityLevel=1.4               \
	    -dSubsetFonts=true -dEmbedAllFonts=true                  \
	    -sOutputFile=camera_ready.pdf -f CS5525.pdf
	@if  pdfinfo camera_ready.pdf | grep version | grep 1.4 > /dev/null ; \
	then \
	  if pdffonts camera_ready.pdf 2>&1 | grep -v Error | cut -c 56-62 | grep no > /dev/null ; \
	  then \
	    echo pdf failed test on fonts ; \
	  else \
	    echo pdf passed test, all fonts embedded and Acrobat 5 compatible ; \
	  fi \
	else \
	  echo pdf failed test on version ; \
	fi
