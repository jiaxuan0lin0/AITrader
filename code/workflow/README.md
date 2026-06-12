# AITrader 工作流

本文档说明端到端工作流的模块边界和主要入口。所有命令默认从 `code/` 目录执行，数据目录默认解析为 `../data`。

## 流程总览

```text
raw_market_data
-> datasets/processed
-> datasets/factors
-> datasets/features
-> datasets/factors/evaluation/final
-> data/experiments/msgca
-> competition_signals
```

## 模块边界

| 模块 | 路径 | 输出 |
| --- | --- | --- |
| 数据处理 | `data/` | `../data/datasets/processed/*.parquet` |
| 因子工程 | `FactorMiner/` | `factor_registry.json`、`feature_registry.json`、特征筛选结果 |
| 模型训练 | `model/msgca/` | checkpoint、预测文件、回测指标 |
| Live 推理 | `workflow/run_live_pipeline.py` | `signals_YYYYMMDD.csv`、`buy_list_YYYYMMDD.csv`、`sell_list_YYYYMMDD.csv` |

## 路径配置

默认路径通过 `aitrader_paths.py` 从项目根目录推导：

```text
AITRADER_ROOT        = code/..
AITRADER_DATA_ROOT   = ../data
AITRADER_DATASETS_ROOT = ../data/datasets
AITRADER_EXPERIMENTS_ROOT = ../data/experiments
```

跨机器部署时可以用环境变量覆盖：

```bash
export AITRADER_ROOT=/path/to/AItrader
export AITRADER_DATA_ROOT=/path/to/AItrader/data
```

## 常用入口

构建标准数据表：

```bash
python3 -m data.a_share_pipeline
```

构建因子和样本特征：

```bash
python3 -m FactorMiner.run_pipeline --prepare-review
```

训练 MSGCA：

```bash
python3 -m model.msgca.train --config model/msgca/config.yaml
```

生成 live 信号：

```bash
python3 -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD
```

## 详细文档

| 文档 | 内容 |
| --- | --- |
| `../README.md` | 项目总览、上传边界和最终模型位置 |
| `../data/README.md` | 数据目录说明 |
| `data/README.md` | 数据同步和标准化 |
| `FactorMiner/README.md` | 因子工程和特征筛选 |
| `FactorMiner/news_scoring/README.md` | 新闻 LLM 打分 |
| `model/msgca/README.md` | MSGCA 模型训练、评估、回测和推理 |
| `workflow/README_LIVE_PIPELINE.md` | live 推理流程 |

