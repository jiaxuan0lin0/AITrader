# Report LaTeX

This directory contains the LaTeX source for the experiment report.

## One-time setup

```bash
conda env create -f environment.yml
```

The environment used here is `aitrader-latex` and provides `tectonic`, a
lightweight XeLaTeX-compatible compiler that downloads TeX packages on demand.

## Build

```bash
make pdf
```

The generated PDF is written to:

```text
build/experiment_report.pdf
```

If a full TeX Live installation with `latexmk` and `xelatex` is available,
`make pdf` will use it automatically. Otherwise it uses `tectonic` from the
`aitrader-latex` conda environment.

## Clean

```bash
make clean
```
