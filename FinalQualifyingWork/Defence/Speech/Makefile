DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

SRC := $(DIR)Protection.md
PDF := $(DIR)Protection.pdf

.PHONY: all pdf clean

all: pdf

pdf:
	pandoc $(SRC) -o $(PDF) \
	  --pdf-engine=xelatex \
	  -V documentclass=extarticle \
	  -V mainfont="Times New Roman" \
	  -V lang=en \
	  -V papersize=a4 \
	  -V geometry:margin=2cm \
	  -V linestretch=1.15 \
	  -V indent=false \
	  -V fontsize=14pt

clean:
	rm -f $(PDF)
