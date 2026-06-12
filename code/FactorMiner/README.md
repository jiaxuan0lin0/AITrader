# FactorMiner

`FactorMiner/` 在 `code/data/` 产出的标准数据表之上构建因子、样本级特征和可审计的特征筛选结果。它不负责原始数据清洗、样本标签生成、模型训练或模型回测。

命令默认在 `aitrader` conda 环境中从 `code/` 目录执行。

## 模块边界

| 上游输入 | 默认路径 | 用途 |
| --- | --- | --- |
| 价格表 | `data/datasets/processed/price.parquet` | 量价、趋势、波动率和流动性因子。 |
| 指标表 | `data/datasets/processed/metric.parquet` | 估值、市值、换手和股息类因子。 |
| 资金流表 | `data/datasets/processed/moneyflow.parquet` | 主力资金、小单压力和资金动量因子。 |
| 股票基础表 | `data/datasets/processed/basic.parquet` | 股票名称、行业、上市日期和市场信息。 |
| 新闻表 | `data/datasets/processed/news.parquet` | 新闻 item、新闻-股票映射和新闻评分输入。 |
| 样本表 | `data/datasets/processed/samples.parquet` | `sample_id`、`stock_code`、`feature_asof_date`、`decision_ts` 和标签。 |
| 新闻评分表 | `data/datasets/factors/news_llm_scores.parquet` | 新闻样本因子聚合输入。 |

| 输出 | 默认路径 | 说明 |
| --- | --- | --- |
| 日频因子块 | `data/datasets/factors/blocks/daily/` | `stock_code + trade_date` 粒度。 |
| 因子 manifest | `data/datasets/factors/manifests/` | 因子元信息。 |
| 因子注册表 | `data/datasets/factors/factor_registry.json` | 日频因子块注册表。 |
| 样本特征块 | `data/datasets/features/blocks/sample/` | `sample_id` 粒度。 |
| 特征 manifest | `data/datasets/features/manifests/` | 样本特征元信息。 |
| 特征注册表 | `data/datasets/features/feature_registry.json` | 样本特征块注册表。 |
| 评估与筛选结果 | `data/datasets/factors/evaluation/` | 质量检查、单因子评估和 selected features。 |

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `operators.py` | 通用时序、横截面和安全数值算子。 |
| `core/` | `FactorSpec`、`FactorResult`、`FactorBlock`、注册表和样本对齐规则。 |
| `build/` | 日频因子、新闻样本因子和 sample feature block 构建入口。 |
| `pools/` | 手工日频因子池和新闻样本因子池。 |
| `news_scoring/` | 新闻 LLM 打分、schema 校验和 vLLM 服务脚本。 |
| `evaluation/` | 质量检查、单因子评估、自动筛选和可选复核。 |
| `mining/` | GPT 辅助候选因子材料包、候选校验和候选物化。 |
| `scripts/` | 常用筛选工作流 shell 封装。 |
| `run_pipeline.py` | 从 processed 数据和新闻评分缓存到自动筛选的一键入口。 |
| `run_factor_workflow.py` | 固定因子块后的训练期筛选和推理特征组装入口。 |

## 子文档索引

| 文档 | 内容 |
| --- | --- |
| `core/README.md` | 核心合同、block、registry 和样本对齐规则 |
| `build/README.md` | daily/sample/news feature 构建入口和命令 |
| `pools/README.md` | 因子池输入输出合同和新增因子检查表 |
| `news_scoring/README.md` | 新闻 LLM 打分服务、执行和输出字段 |
| `evaluation/README.md` | 质量检查、单因子评估、筛选和复核 |
| `mining/README.md` | GPT 候选因子 packet、候选校验和候选物化 |
| `scripts/README.md` | 训练期和比赛版筛选脚本 |
| `README_RUNBOOK.md` | 从已有数据到筛选结果的执行流程 |

## 核心流程

```text
processed data
-> cached news LLM scores
-> daily factor blocks
-> sample feature blocks
-> quality reports
-> single-factor reports
-> selected_features.json
-> optional selected_features_reviewed.json
```

## 常用命令

运行 FactorMiner 相关测试：

```bash
conda activate aitrader
cd code
python -m pytest -q \
  tests/test_operators.py \
  tests/test_factor_block_registry.py \
  tests/test_build_daily.py \
  tests/test_build_sample_features.py \
  tests/test_build_news_sample.py \
  tests/test_evaluation_quality.py \
  tests/test_selection.py
```

从 processed 数据和新闻评分缓存执行完整 FactorMiner 流程：

```bash
python -m FactorMiner.run_pipeline
```

构建日频因子：

```bash
python -m FactorMiner.build.daily --block all
```

把日频因子对齐到样本级特征：

```bash
python -m FactorMiner.build.sample_features --blocks all
```

构建新闻样本特征：

```bash
python -m FactorMiner.build.news_sample --scope split
```

运行质量检查、单因子评估和自动筛选：

```bash
python -m FactorMiner.evaluation.quality
python -m FactorMiner.evaluation.single_factor
python -m FactorMiner.evaluation.selection
```

固定因子块后按训练窗口重新筛选：

```bash
python -m FactorMiner.run_factor_workflow \
  --mode select \
  --select-engine slice \
  --select-since 2016-01-05 \
  --select-until 2025-09-30 \
  --review-profile research \
  --prepare-review
```

按目标交易日组装推理特征：

```bash
python -m FactorMiner.run_factor_workflow \
  --mode inference \
  --target-date YYYY-MM-DD \
  --selected-features-path data/datasets/factors/evaluation/final/selected_features_reviewed.json
```

## 数据可见性约束

- 日频因子必须通过 `feature_asof_date` 对齐样本，不能使用 `target_trade_date` 当日收盘后数据。
- 新闻因子必须满足 `window_start < publish_time <= decision_ts`。
- `samples.parquet` 中一行表示某只股票在某个 `target_trade_date` 开盘前的一次预测事件。
- `1/3/5/10/20/60` 等窗口表示同一样本上的不同历史观察长度，不改变样本粒度。
- 标签列只来自 `samples.parquet`，FactorMiner 不重新生成标签。
- 训练和推理应通过 `selected_features.json` 或 `selected_features_reviewed.json` 读取特征清单。
