# 代码目录

`code/` 是 AITrader 的 Python 代码根目录。建议从本目录执行训练、因子构建、回测和推理命令。

## 模块

| 路径 | 说明 |
| --- | --- |
| `data/` | 原始数据同步与标准 parquet 表构建。 |
| `FactorMiner/` | 日频因子、新闻因子、样本特征、因子评估和特征筛选。 |
| `model/msgca/` | MSGCA 模型训练、评估、回测和 live 推理。 |
| `workflow/` | 端到端流程入口和运行手册。 |
| `tests/` | 单元测试和关键流程测试。 |

## 路径

默认项目根目录是 `code/..`，数据目录是 `../data`。也可以通过环境变量覆盖：

```bash
export AITRADER_ROOT=/path/to/AItrader
export AITRADER_DATA_ROOT=/path/to/AItrader/data
```

## 测试

```bash
python3 -m pytest -q tests
```

## Live 推理

```bash
python3 -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD
```

