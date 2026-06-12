# FactorMiner Pools 使用手册

`FactorMiner/pools/` 放具体因子池和新闻样本特征计算逻辑。这里是公式层：输入是已经标准化的 `DataFrame`，输出必须是带完整元信息的 `FactorResult`。

## 目录职责

| 文件 | 职责 |
| --- | --- |
| `alpha158.py` | Alpha158 风格价量日频因子 |
| `metric.py` | 估值、市值、换手、股息等日频因子 |
| `moneyflow.py` | 主力资金、小单压力、价量资金确认类日频因子 |
| `neutral.py` | 横截面百分位、横截面 zscore、行业中性派生 |
| `news_llm.py` | 从 `news.parquet` 生成新闻 item 和新闻-股票映射 |
| `news_sample.py` | 用新闻 item、映射和 LLM 评分生成样本级新闻因子 |

## 输入合同

日频因子池必须至少包含：

```text
stock_code
trade_date
```

样本新闻因子必须至少包含：

```text
samples: sample_id, stock_code, decision_ts
news_items: news_id, news_text_hash, publish_time, matched_stock_count
news_stock_map: news_id, stock_code
news_scores: news_text_hash, sentiment_score, impact_score, risk_score, relevance_score, novelty_score, event_type
```

行业中性派生需要 `industry`。如果启用中性化但缺少 `industry`，应直接报错或在 build 层显式关闭，不要自动构造伪行业。

## 输出合同

每个 build 函数返回：

```python
FactorResult(factors=<DataFrame>, specs=<list[FactorSpec]>)
```

要求：

- `factors` 包含主键列和所有 `FactorSpec.name` 对应列。
- `FactorSpec.name` 在同一个结果内唯一。
- 输出因子不能含 `inf` 或 `-inf`。
- 缺失值可以保留，缺失填充由 quality/evaluation 或下游模型决定。
- 日频因子的 `availability` 保持 `feature_asof_date`。

## 因子池说明

### Alpha158

入口：

```python
from FactorMiner.pools.alpha158 import Alpha158Config, build_alpha158_factors
```

主要产生：

- K 线形态类：`KMID`、`KLEN`、`KUP`、`KLOW`、`KSFT`
- 相对价格类：`OPEN0`、`HIGH1`、`VWAP3` 等
- 收益和滚动统计类：`ROC`、均值、标准差、相关、斜率、残差等

### Metric

入口：

```python
from FactorMiner.pools.metric import MetricConfig, build_metric_factors
```

主要产生：

- 原始估值和市值字段
- 缺失标记
- 倒数估值派生：earnings/book/sales yield
- delta 和 rolling mean

### Moneyflow

入口：

```python
from FactorMiner.pools.moneyflow import MoneyflowConfig, build_moneyflow_factors
```

主要产生：

- 净流入占成交额比例
- 主力、超大单、大单、小单压力
- 资金滚动均值和斜率
- 价格上涨与资金流入确认
- 价格收益和资金流相关

### News Sample

入口：

```python
from FactorMiner.pools.news_llm import prepare_news_items
from FactorMiner.pools.news_sample import NewsSampleConfig, build_news_sample_factors
```

新闻窗口使用自然日：

```text
decision_ts - window < publish_time <= decision_ts
```

它不会调用 LLM，只消费已经缓存的 `news_llm_scores.parquet`。

## 新增因子检查表

1. 在对应池中新增公式，优先复用 `FactorMiner.operators`。
2. 给每个输出列补 `FactorSpec`，写清 `source`、`category`、`inputs`、`expression`、`window`、`lookback`。
3. 对除法、log、sqrt 使用安全算子。
4. 对 rolling 因子设置完整窗口，避免 warmup 不足导致口径漂移。
5. 若加入中性派生，确认缺行业股票不会被合并成同一伪行业。
6. 补或更新 `code/tests/test_factor_pools.py`。
7. 用 build 层命令写出 block，再跑 registry validate。

## 修改守则

- pools 层不读取 parquet、不写文件、不更新注册表。
- 不在这里做日期切片和截断参数；这些由 build 层负责。
- 不使用 `target_trade_date`、标签列或未来收益列。
- 不把模型训练逻辑放入因子池。
