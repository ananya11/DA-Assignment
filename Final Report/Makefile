TARGETS: icpe2015-power.pdf
.PHONY: clean

tex_files = $(wildcard *.tex */*.tex */*/*.tex */*/*/*.tex)
bib_files = $(wildcard *.bib */*.bib */*/*.bib */*/*/*.bib)
pdf_files = $(wildcard */*.pdf */*/*.pdf */*/*/*.pdf)

all: icpe2015-power.pdf

icpe2015-power.pdf: $(tex_files) $(bib_files) $(pdf_files)
	pdflatex icpe2015-power.tex | tee latex.out ; \
	if grep -q '^LaTeX Warning: Citation.*undefined' latex.out; \
	then \
		bibtex icpe2015-power; \
		touch .rebuild; \
	fi ;
	while [ -f .rebuild -o -n "`grep '^LaTeX Warning:.*Rerun' latex.out`" ]; \
	do \
		rm -f .rebuild; \
		pdflatex icpe2015-power.tex | tee latex.out; \
	done ; \
	rm -f latex.out ;
	find . \( -name '*.blg' -print \) -or \( -name '*.aux' -print \) -or \
		\( -name '*.bbl' -print \) -or \( -name '*~' -print \) -or \
		\( -name '*.lof' -print \) -or \( -name '*.lot' -print \) -or \
		\( -name '*.toc' -print \) | xargs rm -f; \
	rm -f icpe2015-power.log icpe2015-power.out
	
clean:
	find . \( -name '*.blg' -print \) -or \( -name '*.aux' -print \) -or \
		\( -name '*.bbl' -print \) -or \( -name '*~' -print \) -or \
		\( -name '*.lof' -print \) -or \( -name '*.lot' -print \) -or \
		\( -name '*.toc' -print \) | xargs rm -f; \
	rm -f icpe2015-power.log icpe2015-power.out icpe2015-power.pdf

camera_ready: icpe2015-power.pdf
	gs -dSAFER -dNOPAUSE -dBATCH -sDEVICE=pdfwrite \
	    -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress          \
	    -dCompatibilityLevel=1.4               \
	    -dSubsetFonts=true -dEmbedAllFonts=true                  \
	    -sOutputFile=camera_ready.pdf -f icpe2015-power.pdf
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
