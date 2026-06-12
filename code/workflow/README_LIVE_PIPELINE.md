# Live Pipeline

本文档说明 AITrader 的 live 推理流程。该流程面向指定目标交易日，依次完成数据更新、新闻评分、因子重建、样本特征对齐、MSGCA 推理和买卖列表导出。

## 流程

```text
数据更新
-> 新闻 LLM 打分
-> 日频因子重建
-> 样本级特征重建
-> 目标日特征快照
-> MSGCA 推理
-> signals_YYYYMMDD.csv / buy_list_YYYYMMDD.csv / sell_list_YYYYMMDD.csv
```

目标日 `features.parquet` 是审计产物；模型推理仍通过 `feature_registry.json` 和 `selected_features_reviewed.json` 读取特征，保证训练、评估和 live 推理路径一致。

## 默认模型

默认 live 模型为动态聚类上下文 MSGCA：

```text
../data/experiments/msgca/final/model
```

默认 checkpoint：

```text
../data/experiments/msgca/final/model/checkpoints/msgca_best.pt
```

## 执行命令

从 `code/` 目录执行：

```bash
python3 -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD
```

如果新闻评分服务已经启动：

```bash
python3 -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD \
  --news-service assume-running
```

如果只更新数据和特征，不执行模型推理：

```bash
python3 -m workflow.run_live_pipeline \
  --target-date YYYY-MM-DD \
  --skip-model-inference
```

## 输出

```text
../data/runtime/live_pipeline/<run_id>/summary.json
../data/runtime/live_pipeline/<run_id>/model_live_config.yaml
../data/datasets/model_features/inference/target_date=YYYYMMDD/features.parquet
../data/experiments/msgca/<run>/competition_signals/signals_YYYYMMDD.csv
../data/experiments/msgca/<run>/competition_signals/buy_list_YYYYMMDD.csv
../data/experiments/msgca/<run>/competition_signals/sell_list_YYYYMMDD.csv
```

## 路径配置

默认路径由 `aitrader_paths.py` 从项目根目录推导。跨机器部署时可以设置：

```bash
export AITRADER_ROOT=/path/to/AItrader
export AITRADER_DATA_ROOT=/path/to/AItrader/data
```

