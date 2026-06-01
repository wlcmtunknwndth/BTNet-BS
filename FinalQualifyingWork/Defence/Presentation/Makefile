DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PRES_SRC := $(DIR)Presentation.md
PRES_PDF := $(DIR)Presentation.pdf

.PHONY: all presentation clean

all: presentation

presentation:
	marp $(PRES_SRC) --pdf --allow-local-files -o $(PRES_PDF)

clean:
	rm -f $(PRES_PDF)
