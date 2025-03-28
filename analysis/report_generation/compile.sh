#!/bin/bash
set -e
pdflatex -interaction=nonstopmode -file-line-error -shell-escape --output-directory=build $1.tex
pdflatex -interaction=nonstopmode -file-line-error -shell-escape --output-directory=build $1.tex