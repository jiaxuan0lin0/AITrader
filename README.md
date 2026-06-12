# AITrader

AITrader 是面向 A 股 Top20 选股任务的研究与模拟交易系统，覆盖数据同步、标准表构建、因子工程、MSGCA 排序模型、策略回测和 live 信号生成流程。

本文档只描述项目级信息。各模块的输入、输出、脚本和参数说明见对应目录 README。

## 目录结构

```text
AItrader/
  code/      Python 代码、训练脚本、工作流和测试
  data/      本地数据、实验摘要和模型产物目录
  report/    实验报告源码、图表和构建脚本
```

仓库提交代码、说明文档和轻量级实验摘要。原始行情、特征矩阵、checkpoint、外部模型权重、日志和运行时文件保留在本地，并由 `.gitignore` 排除。

## 核心模块

| 模块 | 路径 | 说明 |
| --- | --- | --- |
| 数据处理 | `code/data/` | 同步原始 A 股文件并生成标准 `parquet` 表。 |
| 因子工程 | `code/FactorMiner/` | 构建日频因子、新闻样本因子、样本级特征和特征筛选产物。 |
| MSGCA | `code/model/msgca/` | 训练、评估、回测和推理多源门控交叉注意力排序模型。 |
| 工作流 | `code/workflow/` | 编排数据更新、新闻打分、因子构建、特征组装和 live 推理。 |
| 实验记录 | `data/experiments/msgca/` | 保存轻量实验摘要、配置和最终运行元数据。 |
| 报告 | `report/` | 保存实验报告源码和图表生成脚本。 |

## 环境

本文档中的命令默认在已配置的 `aitrader` conda 环境中执行，并从 `code/` 目录运行 Python 模块。

```bash
conda activate aitrader
cd code
```

依赖清单位于 `code/requirements.txt`。

## 路径变量

路径默认值由 `code/aitrader_paths.py` 解析。

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

## 快速开始

运行测试：

```bash
conda activate aitrader
cd code
python -m pytest -q tests
```

执行 live 工作流：

```bash
conda activate aitrader
cd code
python -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD
```

## 总体流程

```text
raw_market_data
-> datasets/processed
-> datasets/factors
-> datasets/features
-> datasets/factors/evaluation
-> data/experiments/msgca
-> competition_signals
```

## 数据和模型资产

以下目录属于本地资产，不提交到 Git：

```text
data/raw_market_data/
data/datasets/
data/models/
data/logs/
data/runtime/
data/secrets/
```

最终 MSGCA 运行目录：

```text
data/experiments/msgca/final/model
```

本地最终 checkpoint：

```text
data/experiments/msgca/final/model/checkpoints/msgca_best.pt
data/experiments/msgca/final/model/checkpoints/msgca_best.json
```

checkpoint 是大体积二进制产物，不提交到 Git。跨机器部署时需要通过外部 artifact 存储单独传输。

## 子文档索引

| 文档 | 内容 |
| --- | --- |
| `code/README.md` | 代码目录结构和通用入口 |
| `code/data/README.md` | 数据同步和标准表构建 |
| `code/FactorMiner/README.md` | 因子工程总览 |
| `code/model/msgca/README.md` | MSGCA 模型训练、评估、回测和推理 |
| `code/workflow/README.md` | 端到端工作流 |
| `data/README.md` | 本地数据目录边界 |
| `data/experiments/msgca/README.md` | MSGCA 实验登记 |
| `report/README.md` | 报告构建 |

## License

代码目录包含 `code/LICENSE`，当前许可证文本为 Apache License 2.0。

## Citation

TODO: 补充论文或项目引用方式。
