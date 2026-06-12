# AITrader 工作流

`code/workflow/` 负责编排 live 流程：数据更新、新闻 LLM 打分、因子构建、样本特征组装、GPT-mining live 特征组装和 MSGCA 推理。

命令默认在 `aitrader` conda 环境中从 `code/` 目录执行。

## 流程总览

```text
raw_market_data
-> datasets/processed
-> datasets/factors
-> datasets/features
-> inference feature panel
-> MSGCA predictions
-> competition_signals
```

## 输入输出

| 阶段 | 主要输入 | 主要输出 |
| --- | --- | --- |
| 数据更新 | `data/raw_market_data/` | `../data/datasets/processed/*.parquet` |
| 新闻评分 | `processed/news.parquet` | `../data/datasets/factors/news_llm_scores.parquet` |
| 因子构建 | `processed/`、`news_llm_scores.parquet` | `factor_registry.json`、`feature_registry.json` |
| 特征组装 | `selected_features*.json`、feature blocks | live feature panel |
| 模型推理 | MSGCA config 和 checkpoint | `signals_YYYYMMDD.csv`、`buy_list_YYYYMMDD.csv`、`sell_list_YYYYMMDD.csv` |

## 路径配置

默认路径通过 `aitrader_paths.py` 从项目根目录推导：

```text
AITRADER_ROOT             = code/..
AITRADER_DATA_ROOT        = ../data
AITRADER_DATASETS_ROOT    = ../data/datasets
AITRADER_EXPERIMENTS_ROOT = ../data/experiments
```

跨机器部署时可用环境变量覆盖：

```bash
export AITRADER_ROOT=/path/to/AItrader
export AITRADER_DATA_ROOT=/path/to/AItrader/data
```

## 常用入口

执行完整 live 流程：

```bash
conda activate aitrader
cd code
python -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD
```

复用已构建数据并跳过部分阶段：

```bash
python -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD \
  --skip-data-update \
  --skip-news-scoring
```

只组装特征、不执行模型推理：

```bash
python -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD \
  --skip-model-inference
```

## 关键参数

| 参数 | 说明 |
| --- | --- |
| `--target-date` | 预测目标交易日；省略时使用数据更新后的最新样本日期。 |
| `--live-sample-mode` | 当目标日期不在 `samples.parquet` 中时的 live 样本构建策略。 |
| `--news-service` | 新闻评分服务策略：`auto`、`assume-running` 或 `off`。 |
| `--news-scope` | 新闻样本因子输出范围：`split`、`all`、`market` 或 `stock`。 |
| `--daily-blocks` | 需要构建的日频因子块，默认 `metric,moneyflow,alpha158`。 |
| `--sample-blocks` | 需要对齐的 sample feature blocks。 |
| `--selected-features-path` | 推理使用的 selected features JSON。 |
| `--model-config` | MSGCA live 推理配置。 |
| `--checkpoint` | MSGCA checkpoint。 |

## 详细文档

| 文档 | 内容 |
| --- | --- |
| `../README.md` | 项目总览、数据边界和最终模型位置 |
| `../data/README.md` | 数据目录说明 |
| `data/README.md` | 数据同步和标准化 |
| `FactorMiner/README.md` | 因子工程和特征筛选 |
| `FactorMiner/news_scoring/README.md` | 新闻 LLM 打分 |
| `model/msgca/README.md` | MSGCA 模型训练、评估、回测和推理 |
