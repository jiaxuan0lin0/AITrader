# 代码目录

`code/` 是 AITrader 的 Python 代码根目录。训练、因子构建、回测、推理和测试命令默认从本目录执行。

## 环境

```bash
conda activate aitrader
cd code
```

## 模块索引

| 路径 | 说明 |
| --- | --- |
| `data/` | 原始数据同步与标准 `parquet` 表构建。 |
| `FactorMiner/` | 日频因子、新闻因子、样本特征、因子评估和特征筛选。 |
| `model/msgca/` | MSGCA 模型训练、评估、回测和 live 推理。 |
| `workflow/` | 端到端工作流入口。 |
| `tests/` | 单元测试和关键流程测试。 |
| `aitrader_paths.py` | 项目路径和环境变量解析。 |
| `requirements.txt` | Python 依赖清单。 |

## 路径配置

默认项目根目录为 `code/..`，数据目录为 `../data`。可通过环境变量覆盖：

```bash
export AITRADER_ROOT=/path/to/AItrader
export AITRADER_DATA_ROOT=/path/to/AItrader/data
```

## 常用命令

运行测试：

```bash
python -m pytest -q tests
```

执行 live 工作流：

```bash
python -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD
```

## 子文档

| 文档 | 内容 |
| --- | --- |
| `data/README.md` | 数据同步和预处理 |
| `FactorMiner/README.md` | 因子工程 |
| `model/msgca/README.md` | MSGCA 模型 |
| `workflow/README.md` | 工作流编排 |
