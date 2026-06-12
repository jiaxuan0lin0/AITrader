# 新闻 LLM 打分模块说明

本模块负责把 `news.parquet` 中的一条新闻转换为一条可复用的 LLM 评分记录。股票级和市场级新闻因子不在这里直接展开，后续通过 `news_items`、`news_stock_map` 和评分表按自然日窗口聚合。

## 路径约定

- 输入新闻表：`data/datasets/processed/news.parquet`
- 模型目录：`data/models/Qwen3-32B-AWQ`
- 评分输出：`data/datasets/factors/news_llm_scores.parquet`
- 下载日志：`data/logs/qwen3_32b_awq_modelscope_download.log`

## 下载模型

优先使用 ModelScope 镜像：

```bash
bash FactorMiner/news_scoring/download_qwen3_32b_awq_modelscope.sh
```

查看进度：

```bash
tail -80 data/logs/qwen3_32b_awq_modelscope_download.log
find data/models/Qwen3-32B-AWQ -maxdepth 2 -type f -printf '%p %s\n' | sort | tail -40
```

## 安装推理依赖

推理服务建议固定安装 `vllm==0.10.2`，避免直接安装 latest vLLM 触发 PyTorch/CUDA 基座升级。

```bash
python3 -m pip install "vllm==0.10.2" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

推荐版本组合：

- `vllm==0.10.2`
- `torch==2.8.0+cu128`
- `transformers==4.55.2`
- `tokenizers==0.21.4`
- `numpy==2.2.6`

如果启动时报 `Qwen2Tokenizer has no attribute all_special_tokens_extended`，说明 `transformers` 被装到了 5.x，按上面的版本固定回 `4.55.2` 即可。

## 启动推理服务

模型下载完成后，用 vLLM 启动本地 OpenAI-compatible 服务：

```bash
bash FactorMiner/news_scoring/serve_qwen3_32b_awq_vllm.sh
```

默认服务名为 `qwen3-news`，接口地址为 `http://127.0.0.1:8000/v1`。如需降低资源占用，可以降低 `MAX_MODEL_LEN`：

```bash
MAX_MODEL_LEN=3072 bash FactorMiner/news_scoring/serve_qwen3_32b_awq_vllm.sh
```

## 执行评分

小批量评分：

```bash
python3 -m FactorMiner.news_scoring.score_news_items --since "2026-05-18" --limit 5
```

全量评分时去掉 `--limit`。脚本会按 `(news_text_hash, llm_model, prompt_version)` 跳过已评分新闻，并按 `--checkpoint-size` 周期性落盘。

也可以使用封装脚本执行全量评分：

```bash
bash FactorMiner/news_scoring/score_all_news.sh
```

默认参数：

```text
CONCURRENCY=20
REQUEST_BATCH_SIZE=4
CHECKPOINT_SIZE=1000
MAX_TOKENS=1024
USE_GUIDED_JSON=0
```

`CONCURRENCY` 控制同时提交给 vLLM 的请求数。`REQUEST_BATCH_SIZE` 控制每个请求中携带几条新闻；它不会过滤新闻，只减少 HTTP 请求和 prompt 重复开销。批量输出采用固定 key 对象格式，形如 `scores.0`、`scores.1`、`scores.2`、`scores.3`，比数组格式更不容易漏项。

推荐全量命令：

```bash
CONCURRENCY=20 REQUEST_BATCH_SIZE=4 CHECKPOINT_SIZE=5000 MAX_TOKENS=1024 \
  bash FactorMiner/news_scoring/score_all_news.sh
```

如果日志中出现 `batch_failed ... fallback=split` 或 `fallback=single`，说明该批次的模型返回没有通过校验。脚本会自动拆小重试，最终只写入字段完整、枚举合法的记录。

如果 `batch_failed` 很频繁，可以把并发降到 `16`，仍然保持 `REQUEST_BATCH_SIZE=4`：

```bash
CONCURRENCY=16 REQUEST_BATCH_SIZE=4 CHECKPOINT_SIZE=5000 MAX_TOKENS=1024 \
  bash FactorMiner/news_scoring/score_all_news.sh
```

如果需要最强结构约束，可以打开 vLLM guided JSON；该模式会强制 JSON schema，但通常会降低吞吐，更适合小批量核验，不建议全量默认开启：

```bash
USE_GUIDED_JSON=1 CONCURRENCY=4 REQUEST_BATCH_SIZE=4 CHECKPOINT_SIZE=100 MAX_TOKENS=1024 \
  bash FactorMiner/news_scoring/score_all_news.sh --since "2026-05-18" --limit 100
```

`CHECKPOINT_SIZE` 控制每多少条落盘一次。数值越大，写 parquet 越少，速度越快，但中断时需要重跑的尾部越多：

```bash
CONCURRENCY=20 REQUEST_BATCH_SIZE=4 CHECKPOINT_SIZE=10000 MAX_TOKENS=1024 \
  bash FactorMiner/news_scoring/score_all_news.sh
```

该命令默认断点续跑，不会覆盖已有评分。如果需要清空已有评分并重新全量打分，先删除评分文件：

```bash
rm data/datasets/factors/news_llm_scores.parquet
bash FactorMiner/news_scoring/score_all_news.sh
```

指定起始日期：

```bash
SINCE="2024-01-01" bash FactorMiner/news_scoring/score_all_news.sh
```

指定日期区间：

```bash
SINCE="2024-01-01" UNTIL="2024-12-31 23:59:59" bash FactorMiner/news_scoring/score_all_news.sh
```

小批量验证：

```bash
SINCE="2026-05-18" LIMIT=20 bash FactorMiner/news_scoring/score_all_news.sh
```

查看进度：

```bash
tail -f data/logs/news_score_all.log
```

评分客户端会绕开 `HTTP_PROXY`、`HTTPS_PROXY` 等环境代理，直接访问本地 vLLM 服务。若用其他脚本访问本地服务，需要设置：

```bash
export NO_PROXY=127.0.0.1,localhost
```

## 输出字段

评分表每行对应一条去重新闻，核心字段包括：

- `sentiment_score`：方向分，范围 `[-1, 1]`
- `impact_score`：短期影响强度，范围 `[0, 1]`
- `risk_score`：风险强度，范围 `[0, 1]`
- `relevance_score`：与 A 股市场、行业或个股的相关度，范围 `[0, 1]`
- `novelty_score`：信息新鲜度，范围 `[0, 1]`
- `event_type`：事件类型，固定为 `company`、`earnings`、`policy`、`macro`、`rates`、`fx`、`geopolitics`、`commodity`、`shipping`、`industry`、`market`、`litigation`、`contract`、`other` 之一
- `horizon`：模型判断的影响周期
- `summary`：短中文摘要，低相关新闻也需要保留摘要以便审计

## 生成新闻样本因子

评分完成后，使用 `python3 -m FactorMiner.build.news_sample` 将单条新闻评分聚合为 `sample_id` 粒度的数值因子。该步骤不再次调用大模型，只读取已经缓存的评分表。

输入表：

| 表 | 默认路径 |
| --- | --- |
| 样本表 | `data/datasets/processed/samples.parquet` |
| 新闻明细表 | `data/datasets/processed/news.parquet` |
| 新闻评分表 | `data/datasets/factors/news_llm_scores.parquet` |

输出表：

| 文件 | 默认路径 |
| --- | --- |
| 新闻样本因子 | `data/datasets/features/blocks/sample/news_llm_sample.parquet` |
| 因子说明 | `data/datasets/features/manifests/news_llm_sample.json` |
| 特征注册表 | `data/datasets/features/feature_registry.json` |

执行命令：

```bash
python3 -m FactorMiner.build.news_sample
```

小样本验证：

```bash
python3 -m FactorMiner.build.news_sample \
  --since 2019-01-16 --until 2019-01-16 --limit 200 \
  --output-root /tmp/news_sample_check \
  --feature-registry-path /tmp/news_sample_check/feature_registry.json
```

窗口参数：

```bash
python3 -m FactorMiner.build.news_sample --windows 1,3,5,10
```

窗口含义：

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

输出主键：

```text
sample_id
```

市场级新闻和个股级新闻分开聚合：

| 类型 | 前缀 | 说明 |
| --- | --- | --- |
| 市场级新闻 | `news_market_*` | `matched_stock_count == 0` 的新闻，同一 `decision_ts` 下样本共享 |
| 个股级新闻 | `news_stock_*` | 通过 `news_stock_map` 映射到股票，只进入对应股票样本 |

训练表中不写入新闻原文和 `summary`。新闻原文只用于 LLM 打分，`summary` 只用于审计。模型训练使用的是固定列数的聚合数值因子，例如：

```text
news_market_count_1d
news_market_impact_weighted_sentiment_3d
news_market_max_risk_5d
news_stock_count_1d
news_stock_latest_impact_1d
news_stock_hours_since_latest_10d
```

## 验证

本模块的基础验证命令：

```bash
python3 -m py_compile FactorMiner/news_scoring/*.py
python3 -m py_compile FactorMiner/build/news_sample.py FactorMiner/pools/news_sample.py
python3 -m pytest tests/test_news_llm.py tests/test_news_scoring.py tests/test_news_sample.py -q
```
