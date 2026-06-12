# 实验报告

`report/` 保存 AITrader 实验报告的 LaTeX 源码、图表和构建脚本。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `experiment_report.tex` | 报告主文件。 |
| `figures/` | 报告图表。 |
| `scripts/generate_figures.py` | 图表生成脚本。 |
| `environment.yml` | 报告构建环境。 |
| `Makefile` | PDF 构建和清理入口。 |
| `build/` | 构建输出目录。 |

## 环境

报告构建环境由 `environment.yml` 定义，环境名为 `aitrader-latex`。

```bash
conda env create -f environment.yml
conda activate aitrader-latex
```

该环境包含 `tectonic`。如果本机已有包含 `latexmk` 和 `xelatex` 的 TeX Live，`make pdf` 会优先使用本机 TeX Live；否则使用 `tectonic`。

## 构建

```bash
make pdf
```

生成文件：

```text
build/experiment_report.pdf
```

## 清理

```bash
make clean
```
