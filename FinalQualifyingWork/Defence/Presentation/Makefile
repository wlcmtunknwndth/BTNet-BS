DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PRES_SRC := $(DIR)Presentation.md
PRES_PDF := $(DIR)Presentation.pdf
PRES_PPTX := $(DIR)Presentation.pptx
PRES_EDITABLE_PPTX := $(DIR)Presentation.editable.pptx

.PHONY: all presentation pptx pptx-editable clean

all: presentation

presentation:
	marp $(PRES_SRC) --pdf --allow-local-files -o $(PRES_PDF)

pptx:
	marp $(PRES_SRC) --pptx --allow-local-files -o $(PRES_PPTX)

pptx-editable:
	marp $(PRES_SRC) --pptx-editable --allow-local-files -o $(PRES_EDITABLE_PPTX)

clean:
	rm -f $(PRES_PDF) $(PRES_PPTX) $(PRES_EDITABLE_PPTX)
