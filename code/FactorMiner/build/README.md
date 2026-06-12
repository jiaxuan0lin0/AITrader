# FactorMiner Build 使用手册

`FactorMiner/build/` 是因子和样本特征的构建入口。这里负责读取标准 processed 数据、调用 `pools/` 中的因子池、写出 block 文件，并更新注册表。

## 目录职责

| 文件 | 职责 |
| --- | --- |
| `daily.py` | 构建日频因子块：`alpha158`、`metric`、`moneyflow` |
| `sample_features.py` | 把日频因子按 `feature_asof_date` 对齐到 `sample_id` |
| `news_sample.py` | 把新闻 LLM 评分聚合成样本级新闻特征 |

## 输入和输出

默认输入：

```text
data/datasets/processed/price.parquet
data/datasets/processed/metric.parquet
data/datasets/processed/moneyflow.parquet
data/datasets/processed/basic.parquet
data/datasets/processed/samples.parquet
data/datasets/processed/news.parquet
data/datasets/factors/news_llm_scores.parquet
```

默认输出：

```text
data/datasets/factors/blocks/daily/*.parquet
data/datasets/factors/manifests/*.json
data/datasets/factors/factor_registry.json
data/datasets/features/blocks/sample/*.parquet
data/datasets/features/manifests/*.json
data/datasets/features/feature_registry.json
```

## 常用命令

构建全部日频因子：

```bash
cd code
python3 -m FactorMiner.build.daily
```

只构建一个日频块：

```bash
python3 -m FactorMiner.build.daily --block metric
python3 -m FactorMiner.build.daily --block moneyflow
python3 -m FactorMiner.build.daily --block alpha158
```

把日频块对齐到样本：

```bash
python3 -m FactorMiner.build.sample_features --blocks all
```

构建新闻样本特征：

```bash
python3 -m FactorMiner.build.news_sample --scope split
```

校验注册表：

```bash
python3 -m FactorMiner.build.daily --validate-only
python3 -m FactorMiner.build.sample_features --validate-only
python3 -m FactorMiner.build.news_sample --validate-only
```

## 分块规则

`daily.py` 支持的逻辑块：

| 参数 | 注册 block |
| --- | --- |
| `--block metric` | `manual_metric` |
| `--block moneyflow` | `manual_moneyflow` |
| `--block alpha158 --alpha-layout split` | `manual_alpha158_kbar` 等多个子块 |
| `--block alpha158 --alpha-layout single` | `manual_alpha158` |

默认使用 `--alpha-layout split`，避免单个 Alpha158 parquet 过大，也便于中断后续跑。

`news_sample.py` 默认 `--scope split`，写出两个样本级 block：

```text
news_llm_market_sample
news_llm_stock_sample
```

只有兼容旧流程时才使用 `--scope all`。

## 小范围验证

小范围验证必须写到非默认输出目录，避免污染正式注册表：

```bash
python3 -m FactorMiner.build.daily \
  --block metric \
  --since 2025-01-01 \
  --until 2025-01-31 \
  --stock-limit 50 \
  --output-root /tmp/factorminer_daily_check \
  --registry-path /tmp/factorminer_daily_check/factor_registry.json
```

```bash
python3 -m FactorMiner.build.sample_features \
  --limit 1000 \
  --output-root /tmp/factorminer_feature_check \
  --feature-registry-path /tmp/factorminer_feature_check/feature_registry.json
```

## 修改守则

- build 层只做编排、路径解析、读写和注册，不新增因子公式。
- 新增日频因子池时，在 `DAILY_BLOCKS`、`BLOCK_NAMES`、`BLOCK_DESCRIPTIONS` 和 `_build_results()` 中同时登记。
- 新增输出 block 后，必须写 manifest，并通过 registry 校验。
- `--limit`、`--stock-limit` 只用于小范围验证；默认路径下禁止使用这类截断参数。
- 样本对齐只能调用 `core.alignment.align_daily_factors_to_samples()`，不要绕过 `feature_asof_date`。
