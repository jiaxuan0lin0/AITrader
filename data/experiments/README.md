# AITrader 实验目录

`data/experiments/` 用于存放实验配置、轻量摘要和最终运行元数据，并与代码目录、标准数据处理产物分开管理。

## 目录结构

```text
data/experiments/
  msgca/
```

## 管理规则

- 实验组按模型或任务类型归档。
- 轻量 CSV、JSON、YAML 摘要可纳入版本控制。
- checkpoint、预测 `parquet`、训练日志和大体积运行产物按外部资产管理。
- MSGCA 实验登记见 `msgca/README.md`。
