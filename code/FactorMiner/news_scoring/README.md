# 新闻 LLM 打分

`FactorMiner/news_scoring/` 负责把 `news.parquet` 中的新闻转换为可复用的 LLM 评分记录。股票级和市场级新闻样本因子不在本目录直接展开，而是在 `FactorMiner.build.news_sample` 中按样本可见窗口聚合。

命令默认在 `aitrader` conda 环境中从 `code/` 目录执行。

## 路径约定

| 项 | 默认路径 |
| --- | --- |
| 输入新闻表 | `data/datasets/processed/news.parquet` |
| 模型目录 | `data/models/Qwen3-32B-AWQ` |
| 评分输出 | `data/datasets/factors/news_llm_scores.parquet` |
| 下载日志 | `data/logs/qwen3_32b_awq_modelscope_download.log` |
| 全量评分日志 | `data/logs/news_score_all.log` |

## 脚本列表

| 文件 | 说明 |
| --- | --- |
| `download_qwen3_32b_awq.sh` | 从默认模型源下载 `Qwen/Qwen3-32B-AWQ`。 |
| `download_qwen3_32b_awq_modelscope.sh` | 通过 ModelScope 下载 `Qwen/Qwen3-32B-AWQ`。 |
| `serve_qwen3_32b_awq_vllm.sh` | 使用 vLLM 启动 OpenAI-compatible 本地服务。 |
| `score_news_items.py` | 调用 OpenAI-compatible 服务并写出新闻评分表。 |
| `score_all_news.sh` | 全量评分封装脚本。 |
| `schema.py` | 新闻评分 JSON schema 校验。 |
| `prompt.py` | 新闻评分 prompt。 |

## 模型下载

使用 ModelScope 下载：

```bash
conda activate aitrader
cd code
bash FactorMiner/news_scoring/download_qwen3_32b_awq_modelscope.sh
```

查看下载日志：

```bash
tail -80 data/logs/qwen3_32b_awq_modelscope_download.log
```

TODO: 补充新闻评分推理依赖的推荐版本范围。

## 启动推理服务

`serve_qwen3_32b_awq_vllm.sh` 需要可用的 `vllm` 命令，或可通过 `python3 -m vllm.entrypoints.openai.api_server` 启动服务。

```bash
bash FactorMiner/news_scoring/serve_qwen3_32b_awq_vllm.sh
```

默认服务参数：

| 变量 | 默认值 |
| --- | --- |
| `MODEL_DIR` | `data/models/Qwen3-32B-AWQ` |
| `SERVED_MODEL_NAME` | `qwen3-news` |
| `HOST` | `0.0.0.0` |
| `PORT` | `8000` |
| `MAX_MODEL_LEN` | `4096` |
| `MAX_NUM_SEQS` | `32` |
| `GPU_MEMORY_UTILIZATION` | `0.90` |
| `QUANTIZATION` | `awq` |

示例：

```bash
MAX_MODEL_LEN=3072 bash FactorMiner/news_scoring/serve_qwen3_32b_awq_vllm.sh
```

## 执行评分

小批量评分：

```bash
python -m FactorMiner.news_scoring.score_news_items \
  --since "2026-05-18" \
  --limit 5
```

全量评分封装脚本：

```bash
bash FactorMiner/news_scoring/score_all_news.sh
```

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CONCURRENCY` | `20` | 并发请求数。 |
| `REQUEST_BATCH_SIZE` | `4` | 单次请求包含的新闻条数。 |
| `CHECKPOINT_SIZE` | `1000` | 周期性落盘间隔。 |
| `MAX_TOKENS` | `1024` | 单次请求最大生成 token 数。 |
| `USE_GUIDED_JSON` | `0` | 是否启用 guided JSON。 |
| `SINCE` | 空 | `publish_time` 起始时间。 |
| `UNTIL` | 空 | `publish_time` 截止时间。 |
| `LIMIT` | 空 | 评分条数上限。 |

全量评分示例：

```bash
CONCURRENCY=20 REQUEST_BATCH_SIZE=4 CHECKPOINT_SIZE=5000 MAX_TOKENS=1024 \
  bash FactorMiner/news_scoring/score_all_news.sh
```

指定日期区间：

```bash
SINCE="2024-01-01" UNTIL="2024-12-31 23:59:59" \
  bash FactorMiner/news_scoring/score_all_news.sh
```

小批量校验：

```bash
SINCE="2026-05-18" LIMIT=20 \
  bash FactorMiner/news_scoring/score_all_news.sh
```

查看评分日志：

```bash
tail -f data/logs/news_score_all.log
```

评分脚本按 `(news_text_hash, llm_model, prompt_version)` 跳过已评分新闻，并按 `--checkpoint-size` 周期性写出结果。

## 评分字段

评分表每行对应一条去重新闻。

| 字段 | 说明 |
| --- | --- |
| `sentiment_score` | 方向分，范围 `[-1, 1]`。 |
| `impact_score` | 短期影响强度，范围 `[0, 1]`。 |
| `risk_score` | 风险强度，范围 `[0, 1]`。 |
| `relevance_score` | 与 A 股市场、行业或个股的相关度，范围 `[0, 1]`。 |
| `novelty_score` | 信息新鲜度，范围 `[0, 1]`。 |
| `event_type` | 事件类型枚举。 |
| `horizon` | 模型判断的影响周期。 |
| `summary` | 中文摘要，用于审计。 |

`event_type` 可取值：

```text
company
earnings
policy
macro
rates
fx
geopolitics
commodity
shipping
industry
market
litigation
contract
other
```

## 生成新闻样本因子

评分完成后，使用 `FactorMiner.build.news_sample` 将单条新闻评分聚合为 `sample_id` 粒度的数值因子。该步骤不调用 LLM，只读取已经缓存的评分表。

输入表：

| 表 | 默认路径 |
| --- | --- |
| 样本表 | `data/datasets/processed/samples.parquet` |
| 新闻明细表 | `data/datasets/processed/news.parquet` |
| 新闻评分表 | `data/datasets/factors/news_llm_scores.parquet` |

默认输出：

| 文件 | 默认路径 |
| --- | --- |
| 市场级新闻样本因子 | `data/datasets/features/blocks/sample/news_llm_market_sample.parquet` |
| 个股级新闻样本因子 | `data/datasets/features/blocks/sample/news_llm_stock_sample.parquet` |
| 市场级 manifest | `data/datasets/features/manifests/news_llm_market_sample.json` |
| 个股级 manifest | `data/datasets/features/manifests/news_llm_stock_sample.json` |
| 特征注册表 | `data/datasets/features/feature_registry.json` |

执行命令：

```bash
python -m FactorMiner.build.news_sample --scope split
```

小样本验证：

```bash
python -m FactorMiner.build.news_sample \
  --since 2019-01-16 \
  --until 2019-01-16 \
  --limit 200 \
  --output-root /tmp/news_sample_check \
  --feature-registry-path /tmp/news_sample_check/feature_registry.json
```

自定义窗口：

```bash
python -m FactorMiner.build.news_sample --windows 1,3,5,10
```

窗口规则：

```text
1d  = decision_ts - 24h  到 decision_ts
3d  = decision_ts - 72h  到 decision_ts
5d  = decision_ts - 120h 到 decision_ts
10d = decision_ts - 240h 到 decision_ts
```

新闻可见性约束：

```text
window_start < publish_time <= decision_ts
```

市场级和个股级新闻分别聚合：

| 类型 | 前缀 | 说明 |
| --- | --- | --- |
| 市场级新闻 | `news_market_*` | `matched_stock_count == 0` 的新闻，同一 `decision_ts` 下样本共享。 |
| 个股级新闻 | `news_stock_*` | 通过 `news_stock_map` 映射到股票，只进入对应股票样本。 |

训练表不写入新闻原文和 `summary`。模型训练使用聚合后的数值因子。

## 验证

```bash
python -m py_compile FactorMiner/news_scoring/*.py
python -m py_compile FactorMiner/build/news_sample.py FactorMiner/pools/news_sample.py
python -m pytest tests/test_news_llm.py tests/test_news_scoring.py tests/test_news_sample.py -q
```
