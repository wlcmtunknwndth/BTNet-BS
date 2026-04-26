DIR      := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PRES_SRC := $(DIR)Presentation.md
PRES_PDF := $(DIR)Presentation.pdf

RPT_SRC  := $(DIR)Report.md
RPT_PDF  := $(DIR)Report.pdf

.PHONY: all pdf pptx report clean

all: presentation report

presentation:
	marp $(PRES_SRC) --pdf --allow-local-files -o $(PRES_PDF)

report:
	pandoc $(RPT_SRC) -o $(RPT_PDF) \
	  --pdf-engine=xelatex \
	  -V mainfont="Times New Roman" \
	  -V lang=ru \
	  -V geometry:margin=2cm \
	  -V fontsize=14pt

clean:
	rm -f $(PRES_PDF) $(RPT_PDF)
