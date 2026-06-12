# AITrader

AITrader 是一个面向 A 股 Top20 选股任务的研究与模拟交易系统，包含数据同步、因子构建、特征筛选、MSGCA 深度排序模型、策略回测和实盘信号生成流程。

## 目录结构

```text
AItrader/
  code/      代码、训练脚本、工作流和测试
  data/      本地数据、实验摘要和模型产物目录
  report/    实验报告源码和绘图脚本
```

仓库只提交代码、说明文档和轻量级实验摘要。原始行情、特征矩阵、checkpoint、外部模型权重、日志和运行时文件均保留在本地，并由 `.gitignore` 排除。

## 主要模块

| 模块 | 路径 | 说明 |
| --- | --- | --- |
| 数据处理 | `code/data/` | 将原始 A 股文件转换为标准 parquet 表。 |
| 因子工程 | `code/FactorMiner/` | 构建日频因子、新闻因子、样本级特征和特征筛选产物。 |
| MSGCA | `code/model/msgca/` | 训练、评估、回测和推理多源门控交叉注意力排序模型。 |
| 实盘流程 | `code/workflow/` | 更新数据、新闻打分、重建特征、模型推理并导出买卖列表。 |
| 实验记录 | `data/experiments/msgca/` | 保存轻量实验摘要、配置和最终运行元数据。 |

## 环境变量

路径默认值由 `code/aitrader_paths.py` 解析：

| 变量 | 默认值 |
| --- | --- |
| `AITRADER_ROOT` | 项目根目录 |
| `AITRADER_DATA_ROOT` | `data/` |
| `AITRADER_RAW_DATA_DIR` | `data/raw_market_data/` |
| `AITRADER_DATASETS_ROOT` | `data/datasets/` |
| `AITRADER_EXPERIMENTS_ROOT` | `data/experiments/` |
| `AITRADER_MODELS_ROOT` | `data/models/` |
| `AITRADER_LOG_DIR` | `data/logs/` |
| `AITRADER_RUNTIME_DIR` | `data/runtime/` |
| `AITRADER_SECRETS_DIR` | `data/secrets/` |

Python 命令建议从 `code/` 目录执行：

```bash
cd code
python3 -m pytest -q tests
```

## 最终 MSGCA 运行

实盘流程默认使用最终的动态聚类上下文模型：

```text
data/experiments/msgca/final/model
```

本地只保留最终 checkpoint：

```text
checkpoints/msgca_best.pt
checkpoints/msgca_best.json
```

checkpoint 是大体积二进制产物，不提交到 Git。跨机器复现推理时需要通过外部 artifact 存储单独传输。

## 数据边界

以下目录属于本地资产，不提交到 Git：

```text
data/raw_market_data/
data/datasets/
data/models/
data/logs/
data/runtime/
data/secrets/
```

实验目录中可以提交体积较小的 CSV、JSON、YAML 摘要。checkpoint、parquet、日志、预测明细和比赛信号导出文件均不提交。

## 常用命令

运行 MSGCA 相关测试：

```bash
cd code
python3 -m pytest -q tests/test_live_pipeline.py tests/test_msgca_modules_losses.py tests/test_msgca_strategy_backtest.py
```

执行实盘流程：

```bash
cd code
python3 -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD
```

