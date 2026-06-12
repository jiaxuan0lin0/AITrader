# 数据目录

`data/` 是 AITrader 的本地数据根目录，用于存放原始行情、标准数据表、因子、特征矩阵、模型产物、日志和运行状态。

大体积文件不提交到 Git。仓库只保留少量说明文档、实验摘要和必要配置。

## 目录说明

| 目录 | 说明 | Git 策略 |
| --- | --- | --- |
| `raw_market_data/` | 外部存储同步得到的原始 A 股文件。 | 忽略 |
| `datasets/processed/` | 标准化后的价格、估值、资金流、新闻、样本和标签表。 | 忽略 |
| `datasets/factors/` | 日频因子、因子注册表、新闻评分和因子评估结果。 | 忽略 |
| `datasets/features/` | 样本级特征块和特征注册表。 | 忽略 |
| `models/` | 新闻评分使用的外部大模型权重。 | 忽略 |
| `experiments/` | MSGCA 实验摘要、配置和最终运行元数据。 | 部分保留 |
| `logs/` | 数据同步、训练、推理和新闻评分日志。 | 忽略 |
| `runtime/` | live workflow 工作区、pid 文件和临时状态。 | 忽略 |
| `secrets/` | 本机账号、密钥和私有环境变量。 | 忽略 |

## 路径变量

默认值由 `code/aitrader_paths.py` 解析：

| 变量 | 默认值 |
| --- | --- |
| `AITRADER_DATA_ROOT` | `data/` |
| `AITRADER_RAW_DATA_DIR` | `data/raw_market_data/` |
| `AITRADER_DATASETS_ROOT` | `data/datasets/` |
| `AITRADER_EXPERIMENTS_ROOT` | `data/experiments/` |
| `AITRADER_MODELS_ROOT` | `data/models/` |
| `AITRADER_LOG_DIR` | `data/logs/` |
| `AITRADER_RUNTIME_DIR` | `data/runtime/` |
| `AITRADER_SECRETS_DIR` | `data/secrets/` |

## 最终 checkpoint

本地只保留最终 MSGCA checkpoint：

```text
data/experiments/msgca/20260605_cluster_train/runs/cluster_inrank020_scratch_seed2031/checkpoints/msgca_best.pt
data/experiments/msgca/20260605_cluster_train/runs/cluster_inrank020_scratch_seed2031/checkpoints/msgca_best.json
```

该 checkpoint 不提交到 Git。跨机器部署时需要通过外部 artifact 存储单独传输。

