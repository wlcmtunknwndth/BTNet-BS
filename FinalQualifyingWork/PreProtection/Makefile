DIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRC  := $(DIR)Presentation.md
PDF  := $(DIR)Presentation.pdf
PPTX := $(DIR)Presentation.pptx

.PHONY: pdf pptx clean

pdf:
	marp $(SRC) --pdf --allow-local-files -o $(PDF)

pptx:
	marp $(SRC) --pptx --allow-local-files -o $(PPTX)

clean:
	rm -f $(PDF) $(PPTX)
