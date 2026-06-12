TEX := experiment_report.tex
PDF := build/experiment_report.pdf
CONDA_ENV ?= aitrader-latex

.PHONY: pdf check clean

pdf:
	@mkdir -p build
	@if command -v latexmk >/dev/null 2>&1 && command -v xelatex >/dev/null 2>&1; then \
		latexmk -xelatex -interaction=nonstopmode -file-line-error -synctex=1 -outdir=build $(TEX); \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic --outdir build $(TEX); \
	elif command -v conda >/dev/null 2>&1 && conda run -n $(CONDA_ENV) tectonic --version >/dev/null 2>&1; then \
		conda run -n $(CONDA_ENV) tectonic --outdir build $(TEX); \
	else \
		echo "No LaTeX compiler found."; \
		echo "Run: conda env create -f environment.yml"; \
		exit 1; \
	fi
	@echo "Wrote $(PDF)"

check:
	@if command -v latexmk >/dev/null 2>&1 && command -v xelatex >/dev/null 2>&1; then \
		latexmk -v | head -1; \
		xelatex --version | head -1; \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic --version; \
	elif command -v conda >/dev/null 2>&1; then \
		conda run -n $(CONDA_ENV) tectonic --version; \
	else \
		echo "No LaTeX compiler found."; \
		exit 1; \
	fi

clean:
	rm -rf build
