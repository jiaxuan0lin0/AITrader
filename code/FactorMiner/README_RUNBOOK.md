# FactorMiner 执行流程 README

本文档只说明后续如何执行现有代码。设计解释见 `FactorMiner/README.md`；模块手册见 `FactorMiner/core/README.md`、`FactorMiner/build/README.md`、`FactorMiner/pools/README.md`、`FactorMiner/news_scoring/README.md`、`FactorMiner/evaluation/README.md`、`FactorMiner/mining/README.md` 和 `FactorMiner/scripts/README.md`。

执行目标：

```text
原始数据
-> 标准 processed 数据
-> 新闻 LLM 评分
-> daily 因子块
-> sample feature blocks
-> 质量检查
-> 单因子评估
-> 自动筛选
-> 可选 ChatGPT 复核
-> selected_features.json / selected_features_reviewed.json
```

本流程聚焦因子构建、样本特征生成、质量检查、单因子评估和因子筛选。模型训练和模型评估由模型工程目录负责。

## 1. 路径约定

默认输入路径：

```text
data/raw_market_data
data/datasets/processed
data/datasets/factors/news_llm_scores.parquet
```

默认输出路径：

```text
data/datasets/processed/
data/datasets/factors/
data/datasets/features/
data/datasets/factors/evaluation/
```

核心注册表：

```text
data/datasets/factors/factor_registry.json
data/datasets/features/feature_registry.json
```

最终给下游模型使用的清单：

```text
data/datasets/factors/evaluation/final/selected_features.json
data/datasets/factors/evaluation/final/selected_features_reviewed.json
```

使用规则：

```text
没有人工复核时：使用 selected_features.json
做过人工复核时：优先使用 selected_features_reviewed.json
```

## 2. 执行前检查

在项目根目录执行：

```bash
cd code
python3 -m pytest -q
```

检查原始数据目录：

```bash
find data/raw_market_data -maxdepth 3 -type f | head -50
```

检查 processed 数据是否已有：

```bash
find data/datasets/processed -maxdepth 1 -type f -printf '%f\n' | sort
```

至少应关注：

```text
price.parquet
metric.parquet
moneyflow.parquet
basic.parquet
news.parquet
samples.parquet
```

## 2.5 推荐：一键执行脚本

如果 `data/` 模块已经生成 processed 数据，且新闻 LLM 评分表已经完整存在，直接运行：

```bash
cd code
python3 -m FactorMiner.run_pipeline
```

该命令默认执行：

```text
precheck
-> build.daily --block metric
-> build.daily --block moneyflow
-> build.daily --block alpha158
-> daily registry validate
-> build.sample_features --blocks all
-> build.news_sample --scope split
-> feature registry validate
-> evaluation.quality
-> evaluation.single_factor
-> evaluation.selection
```

默认不会执行：

```text
data.a_share_pipeline
news_scoring.score_news_items
review_selection --apply
model training/evaluation
```

原因：

- `data.a_share_pipeline` 属于数据标准化阶段，不应该在因子一键脚本里默认重写 processed 数据。
- 新闻 LLM 评分已假定完成，脚本只读取 `news_llm_scores.parquet` 聚合新闻样本因子。
- ChatGPT 复核需要人工返回 JSON，不能完全自动闭环。
- 模型训练和评估属于模型工程阶段，不由 `run_pipeline.py` 默认执行。
- `alpha158` 计算量最大，一键流程会把它作为 daily 阶段的第三个逻辑 block 执行，summary 中会显示为 `daily_alpha158`。内部默认使用 `--alpha-layout split` 输出多个物理子块，降低单文件内存峰值。
- 新闻样本因子默认使用 `--news-scope split`，输出 `news_llm_market_sample` 和 `news_llm_stock_sample` 两个 sample block，避免把 market+stock 合成一个 176 列大 parquet 导致内存峰值过高。需要单文件输出时才显式传 `--news-scope all`。

## 2.6 固定因子块后：只做筛选和推理组装

如果 daily block、sample feature block、新闻样本因子都已经构建完成，就不要再刷新因子块。后续只需要两件事：

```text
select:
  在指定训练日期内做 quality -> factor_summary 切片/重聚合 -> selection

inference:
  按 target_date 读取 selected_features，并组装当天模型输入特征
```

统一脚本：

```text
FactorMiner/run_factor_workflow.py
```

### select：按训练日期筛选因子

推荐切分：

```text
训练集: target_trade_date <= 2025-09-30
验证集: 2025-10-01 <= target_trade_date <= 2025-12-31
测试集: target_trade_date >= 2026-01-01
```

正式筛选只允许使用训练集日期，避免用验证集和测试集信息选因子：

推荐直接用封装脚本：

```bash
cd code
FactorMiner/scripts/run_train_selection_slice.sh
```

等价的 Python 命令：

```bash
cd code
python3 -m FactorMiner.run_factor_workflow \
  --mode select \
  --select-engine slice \
  --select-since 2016-01-05 \
  --select-until 2025-09-30 \
  --review-profile research \
  --prepare-review
```

封装脚本支持环境变量覆盖：

```bash
SELECT_SINCE=2016-01-05 \
SELECT_UNTIL=2025-09-30 \
REVIEW_PROFILE=research \
EVALUATION_DIR=data/datasets/factors/evaluation/experiment/select_20160105_20250930_slice \
FactorMiner/scripts/run_train_selection_slice.sh
```

正式比赛最终版使用到 2026-05-20 的历史材料，并生成 competition 复核 prompt：

```bash
cd code
FactorMiner/scripts/run_competition_selection_slice.sh
```

等价于：

```bash
SELECT_UNTIL=2026-05-20 \
REVIEW_PROFILE=competition \
EVALUATION_DIR=data/datasets/factors/evaluation/final \
FactorMiner/scripts/run_train_selection_slice.sh
```

日志默认写到：

```text
data/logs/factorminer_select_20160105_20250930_slice.log
```

`--select-engine slice` 表示复用已有全量逐日明细：

```text
data/datasets/factors/evaluation/factor_ic.csv
data/datasets/factors/evaluation/factor_rankic.csv
data/datasets/factors/evaluation/group_return.csv
```

脚本会按 `--select-since/--select-until` 重新聚合训练期 `factor_summary.csv`，再做 selection。这样避免重新跑最慢的 `single_factor` 全计算，同时不会用验证/测试期收益指标选因子。

如果要完全重算单因子评估，把 `--select-engine slice` 去掉或改成：

```bash
--select-engine full
```

默认输出目录会按日期命名：

```text
data/datasets/factors/evaluation/experiment/select_20160105_20250930_slice/
```

重要参数：

| 参数 | 说明 |
| --- | --- |
| `--select-since` | 因子筛选使用样本的起始 `target_trade_date`，闭区间 |
| `--select-until` | 因子筛选使用样本的结束 `target_trade_date`，闭区间 |
| `--select-engine slice` | 从已有逐日评估明细切片重聚合，推荐正式使用 |
| `--select-engine full` | 重跑 quality、single_factor、selection，耗时更长 |
| `--source-evaluation-dir` | `slice` 模式读取已有逐日明细的目录 |
| `--review-profile research` | 研究验证 prompt，强调不能看验证/测试表现 |
| `--review-profile competition` | 比赛最终 prompt，允许使用 cutoff 前历史并更关注近期市场 |
| `--blocks` | 参与筛选的 sample block，默认 `all` |
| `--corr-row-limit` | 相关性去重抽样行数；默认 `0` 表示全量，内存紧张时再设为 `200000` |
| `--prepare-review` | 额外生成 ChatGPT 复核 prompt 和 response template |
| `--evaluation-dir` | 手动指定筛选结果输出目录 |

如果不做 ChatGPT 复核，下游使用：

```text
selected_features.json
```

如果做过复核，下游使用：

```text
selected_features_reviewed.json
```

### inference：按目标日期组装模型输入

推理只读取 `samples.parquet` 中 `target_trade_date == --target-date` 的样本，并按最终因子清单从 feature blocks 中取列。默认不输出任何 `label_` 列。

```bash
cd code
python3 -m FactorMiner.run_factor_workflow \
  --mode inference \
  --target-date 2026-05-20 \
  --selected-features-path data/datasets/factors/evaluation/final/selected_features_reviewed.json
```

默认输出：

```text
data/datasets/model_features/inference/target_date=20260520/features.parquet
data/datasets/model_features/inference/target_date=20260520/features.json
```

输出表结构：

```text
sample_id
stock_code
stock_name
industry
target_trade_date
feature_asof_date
decision_ts
selected factor columns...
```

如需指定输出位置：

```bash
python3 -m FactorMiner.run_factor_workflow \
  --mode inference \
  --target-date 2026-05-20 \
  --selected-features-path data/datasets/factors/evaluation/final/selected_features_reviewed.json \
  --output-path data/datasets/model_features/inference/target_date=20260520/features.parquet
```

如果只想单独跑 alpha158：

```bash
python3 -m FactorMiner.run_pipeline \
  --daily-block alpha158 \
  --skip-sample-features \
  --skip-news-sample \
  --skip-feature-validate \
  --skip-quality \
  --skip-single-factor \
  --skip-selection
```

输出 summary：

```text
data/datasets/factors/evaluation/pipeline_summary.json
```

先看计划但不执行：

```bash
python3 -m FactorMiner.run_pipeline
```

执行完自动筛选后，同时生成 ChatGPT 复核 prompt：

```bash
python3 -m FactorMiner.run_pipeline --prepare-review
```

如果只想从某个中间阶段续跑，可以用 skip 参数。例如 daily 因子已经构建好，只重跑样本对齐和评估：

```bash
python3 -m FactorMiner.run_pipeline --skip-daily --skip-daily-validate
```

如果 daily 和 feature blocks 都已经构建好，只重跑 evaluation：

```bash
python3 -m FactorMiner.run_pipeline \
  --skip-daily \
  --skip-daily-validate \
  --skip-sample-features \
  --skip-news-sample
```

如果 daily 和 sample feature blocks 已经构建好，但新闻样本因子还没完成，只跑新闻聚合：

```bash
python3 -m FactorMiner.run_pipeline \
  --skip-daily \
  --skip-sample-features \
  --skip-feature-validate \
  --skip-quality \
  --skip-single-factor \
  --skip-selection
```

如果新闻样本因子已经构建好，跳过新闻聚合：

```bash
python3 -m FactorMiner.run_pipeline --skip-news-sample
```

`sample_features` 支持断点续跑：默认会跳过 registry 中已经完整存在、行数和因子数匹配的 sample block。只有显式传 `FactorMiner.build.sample_features --overwrite` 才会重建已完成 block。

正式流程默认使用 parquet metadata 做 registry 校验，不整表读取 70G 级 parquet。只有显式传 `--full-registry-validate` 才会执行重校验。

常用筛选参数也可以直接传给一键脚本：

```bash
python3 -m FactorMiner.run_pipeline \
  --min-rank-ic-days 60 \
  --min-coverage 0.05 \
  --min-abs-rank-ic 0.01 \
  --corr-threshold 0.95
```

默认相关性去重使用全量样本；如果计算太慢或内存紧张，再临时抽样：

```bash
python3 -m FactorMiner.run_pipeline \
  --corr-row-limit 200000 \
  --min-corr-pairs 5000
```

正式结果应去掉 `--corr-row-limit` 或显式设为 `0` 再跑一次 selection。

## 3. 第一步：生成标准 processed 数据

如果 `data/datasets/processed/` 已经是最新的，可以跳过本步。

正式执行：

```bash
python3 -m data.a_share_pipeline
```

指定原始数据目录和输出目录：

```bash
python3 -m data.a_share_pipeline \
  --raw-dir data/raw_market_data \
  --output-dir data/datasets
```

输出位置：

```text
data/datasets/processed/price.parquet
data/datasets/processed/metric.parquet
data/datasets/processed/moneyflow.parquet
data/datasets/processed/basic.parquet
data/datasets/processed/panel.parquet
data/datasets/processed/samples.parquet
data/datasets/processed/news.parquet
data/datasets/meta/run_summary.json
```

检查重点：

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("data/datasets/processed")
for name in ["price", "metric", "moneyflow", "basic", "samples", "news"]:
    path = root / f"{name}.parquet"
    frame = pd.read_parquet(path)
    print(name, frame.shape, list(frame.columns)[:12])
PY
```

必须确认：

- `samples.parquet` 已存在，并且 `sample_id` 唯一。
- `metric.parquet` 和 `moneyflow.parquet` 已带 `industry`。
- `news.parquet` 的新闻字段未被破坏。
- 如果新闻 LLM 已经在评分，避免无意删除或重写 `news.parquet` 导致评分映射变化。

## 4. 第二步：新闻 LLM 评分

如果 `news_llm_scores.parquet` 已存在，可以跳过本步，直接进入第 5 步。

评分输出：

```text
data/datasets/factors/news_llm_scores.parquet
```

### 4.1 启动本地 Qwen 服务

如果模型还没下载：

```bash
bash FactorMiner/news_scoring/download_qwen3_32b_awq_modelscope.sh
```

启动 vLLM：

```bash
bash FactorMiner/news_scoring/serve_qwen3_32b_awq_vllm.sh
```

如需降低上下文长度：

```bash
MAX_MODEL_LEN=3072 bash FactorMiner/news_scoring/serve_qwen3_32b_awq_vllm.sh
```

服务默认地址：

```text
http://127.0.0.1:8000/v1
```

### 4.2 小批量执行

```bash
python3 -m FactorMiner.news_scoring.score_news_items \
  --since "2026-05-18" \
  --limit 5
```

### 4.3 全量评分

推荐命令：

```bash
CONCURRENCY=20 REQUEST_BATCH_SIZE=4 CHECKPOINT_SIZE=5000 MAX_TOKENS=1024 \
  bash FactorMiner/news_scoring/score_all_news.sh
```

如果失败回退很多，降低并发：

```bash
CONCURRENCY=16 REQUEST_BATCH_SIZE=4 CHECKPOINT_SIZE=5000 MAX_TOKENS=1024 \
  bash FactorMiner/news_scoring/score_all_news.sh
```

查看日志：

```bash
tail -f data/logs/news_score_all.log
```

断点续跑规则：

- 脚本按 `(news_text_hash, llm_model, prompt_version)` 跳过已评分新闻。
- 默认不会覆盖已有评分。
- 不要删除 `news_llm_scores.parquet`，除非明确要重打全部新闻。

### 4.4 评分完成检查

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("data/datasets/factors/news_llm_scores.parquet")
scores = pd.read_parquet(path)
print(scores.shape)
print(scores["event_type"].value_counts(dropna=False).head(20))
print(scores[["sentiment_score", "impact_score", "risk_score", "relevance_score", "novelty_score"]].describe())
PY
```

## 5. 第三步：构建 daily 因子块

daily 因子块包括：

```text
manual_alpha158_kbar
manual_alpha158_price
manual_alpha158_return
manual_alpha158_rolling3
manual_alpha158_rolling5
manual_alpha158_rolling10
manual_alpha158_rolling20
manual_alpha158_rolling60
manual_metric
manual_moneyflow
```

正式执行：

```bash
python3 -m FactorMiner.build.daily --block all
```

输出：

```text
data/datasets/factors/blocks/daily/manual_alpha158_kbar.parquet
data/datasets/factors/blocks/daily/manual_alpha158_price.parquet
data/datasets/factors/blocks/daily/manual_alpha158_return.parquet
data/datasets/factors/blocks/daily/manual_alpha158_rolling3.parquet
data/datasets/factors/blocks/daily/manual_alpha158_rolling5.parquet
data/datasets/factors/blocks/daily/manual_alpha158_rolling10.parquet
data/datasets/factors/blocks/daily/manual_alpha158_rolling20.parquet
data/datasets/factors/blocks/daily/manual_alpha158_rolling60.parquet
data/datasets/factors/blocks/daily/manual_metric.parquet
data/datasets/factors/blocks/daily/manual_moneyflow.parquet
data/datasets/factors/manifests/manual_alpha158_kbar.json
data/datasets/factors/manifests/manual_alpha158_price.json
data/datasets/factors/manifests/manual_alpha158_return.json
data/datasets/factors/manifests/manual_alpha158_rolling3.json
data/datasets/factors/manifests/manual_alpha158_rolling5.json
data/datasets/factors/manifests/manual_alpha158_rolling10.json
data/datasets/factors/manifests/manual_alpha158_rolling20.json
data/datasets/factors/manifests/manual_alpha158_rolling60.json
data/datasets/factors/manifests/manual_metric.json
data/datasets/factors/manifests/manual_moneyflow.json
data/datasets/factors/factor_registry.json
```

`alpha158` 默认拆分为多个物理 block。因子名仍然是 `alpha158_*`，筛选和模型训练阶段会从 registry 统一读取这些 block。需要单文件输出时：

```bash
python3 -m FactorMiner.build.daily --block alpha158 --alpha-layout single
```

校验 registry：

```bash
python3 -m FactorMiner.build.daily --validate-only
```

只跑某个 block：

```bash
python3 -m FactorMiner.build.daily --block metric
python3 -m FactorMiner.build.daily --block moneyflow
python3 -m FactorMiner.build.daily --block alpha158
```

关闭中性派生：

```bash
python3 -m FactorMiner.build.daily --block all --disable-neutral
```

行业缺失率阈值：

```bash
python3 -m FactorMiner.build.daily \
  --block all \
  --max-industry-missing-rate 0.20
```

注意：

- 正式全量不要传 `--stock-limit`。
- `--stock-limit` 只能用于小范围验证，并且必须写到非默认输出目录。
- `--since` 只过滤最终输出，不截断历史计算。
- `--until` 会限制输入和输出上界。

小范围验证示例：

```bash
python3 -m FactorMiner.build.daily \
  --block all \
  --since 2019-01-01 \
  --until 2019-01-31 \
  --stock-limit 10 \
  --output-root /tmp/factor_daily_check \
  --registry-path /tmp/factor_daily_check/factor_registry.json
```

## 6. 第四步：把 daily 因子对齐到 samples

本步骤读取 daily factor registry，把 daily 粒度因子转换成 `sample_id` 粒度 feature block。

正式执行：

```bash
python3 -m FactorMiner.build.sample_features \
  --blocks all \
  --validate-source
```

输出：

```text
data/datasets/features/blocks/sample/manual_alpha158_kbar_sample.parquet
data/datasets/features/blocks/sample/manual_alpha158_price_sample.parquet
data/datasets/features/blocks/sample/manual_alpha158_return_sample.parquet
data/datasets/features/blocks/sample/manual_alpha158_rolling3_sample.parquet
data/datasets/features/blocks/sample/manual_alpha158_rolling5_sample.parquet
data/datasets/features/blocks/sample/manual_alpha158_rolling10_sample.parquet
data/datasets/features/blocks/sample/manual_alpha158_rolling20_sample.parquet
data/datasets/features/blocks/sample/manual_alpha158_rolling60_sample.parquet
data/datasets/features/blocks/sample/manual_metric_sample.parquet
data/datasets/features/blocks/sample/manual_moneyflow_sample.parquet
data/datasets/features/manifests/manual_alpha158_kbar_sample.json
data/datasets/features/manifests/manual_alpha158_price_sample.json
data/datasets/features/manifests/manual_alpha158_return_sample.json
data/datasets/features/manifests/manual_alpha158_rolling3_sample.json
data/datasets/features/manifests/manual_alpha158_rolling5_sample.json
data/datasets/features/manifests/manual_alpha158_rolling10_sample.json
data/datasets/features/manifests/manual_alpha158_rolling20_sample.json
data/datasets/features/manifests/manual_alpha158_rolling60_sample.json
data/datasets/features/manifests/manual_metric_sample.json
data/datasets/features/manifests/manual_moneyflow_sample.json
data/datasets/features/feature_registry.json
```

校验 feature registry：

```bash
python3 -m FactorMiner.build.sample_features --validate-only
```

只对齐某些 block：

```bash
python3 -m FactorMiner.build.sample_features \
  --blocks manual_metric,manual_moneyflow \
  --validate-source
```

小范围验证示例：

```bash
python3 -m FactorMiner.build.sample_features \
  --source-registry-path /tmp/factor_daily_check/factor_registry.json \
  --output-root /tmp/sample_features_check \
  --feature-registry-path /tmp/sample_features_check/feature_registry.json \
  --blocks manual_metric \
  --since 2019-01-03 \
  --until 2019-01-31 \
  --limit 1000 \
  --validate-source
```

注意：

- 正式全量不要传 `--limit`。
- `--limit` 只能用于小范围验证，并且必须写到非默认输出目录。
- 对齐使用 `samples.stock_code + samples.feature_asof_date = daily.stock_code + daily.trade_date`。
- 不能用 `target_trade_date` 对齐日频因子。

## 7. 第五步：构建新闻样本因子

本步骤不调用大模型，只读取已存在的：

```text
data/datasets/factors/news_llm_scores.parquet
```

正式执行：

```bash
python3 -m FactorMiner.build.news_sample
```

输出：

```text
data/datasets/features/blocks/sample/news_llm_market_sample.parquet
data/datasets/features/blocks/sample/news_llm_stock_sample.parquet
data/datasets/features/manifests/news_llm_market_sample.json
data/datasets/features/manifests/news_llm_stock_sample.json
data/datasets/features/feature_registry.json
```

默认窗口：

```text
1,3,5,10 natural days
```

指定窗口：

```bash
python3 -m FactorMiner.build.news_sample --windows 1,3,5,10
```

默认 `--scope split` 会分开写市场级和个股级新闻 block。单文件输出可以这样跑，但全量不推荐：

```bash
python3 -m FactorMiner.build.news_sample --scope all
```

校验 feature registry：

```bash
python3 -m FactorMiner.build.news_sample --validate-only
```

小范围验证示例：

```bash
python3 -m FactorMiner.build.news_sample \
  --since 2019-01-16 \
  --until 2019-01-16 \
  --limit 200 \
  --output-root /tmp/news_sample_check \
  --feature-registry-path /tmp/news_sample_check/feature_registry.json
```

注意：

- 新闻样本因子的主键是 `sample_id`。
- 默认输出两个 block：`news_llm_market_sample` 包含 `news_market_*`，`news_llm_stock_sample` 包含 `news_stock_*`。
- 可见性规则是 `window_start < publish_time <= decision_ts`。
- 正式全量不要传 `--limit`。
- 如果新闻评分还没完成，本步骤只能聚合已评分部分，不会自动调用 LLM。

## 8. 第六步：检查 feature registry

daily 对齐和新闻聚合都完成后，检查 feature registry：

```bash
python3 -m FactorMiner.build.sample_features --validate-only
```

查看 sample feature blocks：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("data/datasets/features/feature_registry.json")
blocks = json.loads(path.read_text(encoding="utf-8"))
for block in blocks:
    print(block["name"], block["granularity"], block["factor_count"], block["row_count"], block["factor_path"])
PY
```

期望至少看到：

```text
manual_alpha158_kbar_sample
manual_alpha158_price_sample
manual_alpha158_return_sample
manual_alpha158_rolling3_sample
manual_alpha158_rolling5_sample
manual_alpha158_rolling10_sample
manual_alpha158_rolling20_sample
manual_alpha158_rolling60_sample
manual_metric_sample
manual_moneyflow_sample
news_llm_market_sample
news_llm_stock_sample
```

只跑人工日频因子时，可以不包含新闻 sample block；正式筛选建议同时纳入 `news_llm_market_sample` 和 `news_llm_stock_sample`。

## 9. 第七步：样本特征质量检查

正式执行：

```bash
python3 -m FactorMiner.evaluation.quality
```

输出：

```text
data/datasets/factors/evaluation/sample_feature_quality.csv
data/datasets/factors/evaluation/sample_feature_block_quality.csv
data/datasets/factors/evaluation/sample_feature_quality_summary.json
```

只检查某些 block：

```bash
python3 -m FactorMiner.evaluation.quality \
  --blocks manual_metric_sample,manual_moneyflow_sample
```

常用阈值：

```bash
python3 -m FactorMiner.evaluation.quality \
  --max-missing-rate 0.98 \
  --min-non-missing 100 \
  --min-year-coverage 0.01
```

检查结果：

```bash
python3 - <<'PY'
import pandas as pd

q = pd.read_csv("data/datasets/factors/evaluation/sample_feature_quality.csv")
print(q.shape)
print(q["quality_pass"].value_counts(dropna=False))
print(q["quality_flags"].fillna("").value_counts().head(20))
print(q.groupby("block")["factor_name"].count())
PY
```

## 10. 第八步：单因子评估

正式执行：

```bash
python3 -m FactorMiner.evaluation.single_factor
```

输出：

```text
data/datasets/factors/evaluation/factor_ic.csv
data/datasets/factors/evaluation/factor_rankic.csv
data/datasets/factors/evaluation/group_return.csv
data/datasets/factors/evaluation/factor_summary.csv
data/datasets/factors/evaluation/single_factor_summary.json
```

默认标签：

```text
label_next_open_return
label_next_vwap_return
```

只评估主标签：

```bash
python3 -m FactorMiner.evaluation.single_factor \
  --labels label_next_open_return
```

指定日期区间：

```bash
python3 -m FactorMiner.evaluation.single_factor \
  --since 2019-01-01 \
  --until 2024-12-31
```

只评估某些 block：

```bash
python3 -m FactorMiner.evaluation.single_factor \
  --blocks manual_alpha158_kbar_sample,manual_alpha158_price_sample,manual_alpha158_return_sample,manual_alpha158_rolling3_sample,manual_alpha158_rolling5_sample,manual_alpha158_rolling10_sample,manual_alpha158_rolling20_sample,manual_alpha158_rolling60_sample,manual_metric_sample,manual_moneyflow_sample
```

检查结果：

```bash
python3 - <<'PY'
import pandas as pd

s = pd.read_csv("data/datasets/factors/evaluation/factor_summary.csv")
print(s.shape)
print(s.groupby("label")["factor_name"].nunique())
print(
    s.sort_values("rank_ic_mean", key=lambda x: x.abs(), ascending=False)
     [["block", "factor_name", "label", "rank_ic_mean", "rank_ic_ir", "rank_ic_day_count", "coverage_mean"]]
     .head(30)
)
PY
```

## 11. 第九步：自动筛选可用因子

正式执行：

```bash
python3 -m FactorMiner.evaluation.selection
```

输出：

```text
data/datasets/factors/evaluation/candidate_features.csv
data/datasets/factors/evaluation/rejected_features.csv
data/datasets/factors/evaluation/selected_features.csv
data/datasets/factors/evaluation/final/selected_features.json
data/datasets/factors/evaluation/correlation_conflicts.csv
data/datasets/factors/evaluation/correlation_clusters.csv
data/datasets/factors/evaluation/review_packet.json
data/datasets/factors/evaluation/selection_summary.json
```

默认筛选逻辑：

```text
quality_pass == true
constant_flag == false
rank_ic_day_count >= 60
coverage_mean >= 0.05
abs(rank_ic_mean) >= 0.01
高相关阈值 abs(corr) >= 0.95
```

更宽松：

```bash
python3 -m FactorMiner.evaluation.selection \
  --min-rank-ic-days 30 \
  --min-coverage 0.02 \
  --min-abs-rank-ic 0.005
```

更严格：

```bash
python3 -m FactorMiner.evaluation.selection \
  --min-rank-ic-days 120 \
  --min-coverage 0.20 \
  --min-abs-rank-ic 0.015 \
  --min-abs-rank-ic-ir 0.10
```

默认相关性去重使用全量样本；如果计算太慢或内存紧张，再临时抽样：

```bash
python3 -m FactorMiner.evaluation.selection \
  --corr-row-limit 200000 \
  --min-corr-pairs 5000
```

最终全量复核应去掉 `--corr-row-limit` 或显式设为 `0`。

限制最终因子数量：

```bash
python3 -m FactorMiner.evaluation.selection \
  --max-selected 300
```

检查结果：

```bash
python3 - <<'PY'
import json
import pandas as pd
from pathlib import Path

root = Path("data/datasets/factors/evaluation")
c = pd.read_csv(root / "candidate_features.csv")
print(c["selection_status"].value_counts(dropna=False))
print(c["selected"].value_counts(dropna=False))
print(c["final_reject_reason"].fillna("").value_counts().head(20))

selected = json.loads((root / "selected_features.json").read_text(encoding="utf-8"))
print("selected_count", len(selected["selected_features"]))
print("blocks", {k: len(v) for k, v in selected["blocks"].items()})
PY
```

## 12. 第十步：可选 ChatGPT 复核

如果接受自动筛选结果，可以跳过本步，直接使用：

```text
data/datasets/factors/evaluation/final/selected_features.json
```

如果需要人工/ChatGPT 复核，先生成 prompt：

```bash
python3 -m FactorMiner.evaluation.review_selection --prepare
```

输出：

```text
data/datasets/factors/evaluation/review_prompt.md
data/datasets/factors/evaluation/review_inputs.txt
data/datasets/factors/evaluation/review_response_template.json
```

操作：

1. 打开 `review_inputs.txt`。
2. 按里面列出的绝对路径，把 `review_prompt.md` 和辅助材料提供给 ChatGPT。
3. 要求只返回 JSON。
4. 保存返回结果到：

```text
data/datasets/factors/evaluation/review_response.json
```

应用复核结果：

```bash
python3 -m FactorMiner.evaluation.review_selection \
  --apply \
  --response-path data/datasets/factors/evaluation/review_response.json
```

输出：

```text
data/datasets/factors/evaluation/final/selected_features_reviewed.json
data/datasets/factors/evaluation/final/selected_features_reviewed.csv
data/datasets/factors/evaluation/final/selection_review_audit.csv
data/datasets/factors/evaluation/final/selection_review_report.md
```

硬约束：

- `remove` 只能删除自动入选因子。
- `add_back` 只能从已知候选因子中加回。
- 默认禁止加回 `quality_failed` 或 `constant` 因子。
- 每个操作必须有 reason。
- 未知因子直接报错。

不建议使用，但确有必要时可强制允许加回质量失败因子：

```bash
python3 -m FactorMiner.evaluation.review_selection \
  --apply \
  --response-path data/datasets/factors/evaluation/review_response.json \
  --allow-quality-failed-add-back
```

## 13. 最终产物确认

执行完成后，至少确认这些文件存在：

```bash
ls -lh data/datasets/factors/factor_registry.json
ls -lh data/datasets/features/feature_registry.json
ls -lh data/datasets/factors/evaluation/final/selected_features.json
```

如果做了复核：

```bash
ls -lh data/datasets/factors/evaluation/final/selected_features_reviewed.json
```

最终清单检查：

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("data/datasets/factors/evaluation")
path = root / "selected_features_reviewed.json"
if not path.exists():
    path = root / "selected_features.json"

data = json.loads(path.read_text(encoding="utf-8"))
print("using", path)
print("mode", data.get("selection_mode"))
print("selected_count", len(data.get("selected_features", [])))
print("blocks")
for block, factors in data.get("blocks", {}).items():
    print(" ", block, len(factors))
PY
```

## 14. 推荐全流程命令清单

如果 processed 数据和新闻评分都已存在，从这里开始：

```bash
cd code
python3 -m FactorMiner.run_pipeline --prepare-review
```

这条命令默认跑 `metric,moneyflow,alpha158,news` 到 evaluation/selection。

大规模全量更推荐分段跑，方便观察和续跑：

```bash
python3 -m FactorMiner.run_pipeline \
  --daily-block alpha158 \
  --skip-sample-features \
  --skip-news-sample \
  --skip-feature-validate \
  --skip-quality \
  --skip-single-factor \
  --skip-selection

python3 -m FactorMiner.run_pipeline \
  --skip-daily \
  --skip-news-sample \
  --skip-feature-validate \
  --skip-quality \
  --skip-single-factor \
  --skip-selection

python3 -m FactorMiner.run_pipeline \
  --skip-daily \
  --skip-sample-features \
  --skip-feature-validate \
  --skip-quality \
  --skip-single-factor \
  --skip-selection

python3 -m FactorMiner.run_pipeline \
  --skip-daily \
  --skip-sample-features \
  --skip-news-sample \
  --prepare-review
```

最后一段会跑 feature 校验、quality、single_factor、selection 和 review prompt。默认 selection 相关性去重使用全量样本；如果临时传 `--corr-row-limit`，它只限制 selection 的相关性去重抽样，不影响前面的单因子 IC 计算。

如果要应用 ChatGPT 复核：

```bash
python3 -m FactorMiner.evaluation.review_selection \
  --apply \
  --response-path data/datasets/factors/evaluation/review_response.json
```

如果要从原始数据开始：

```bash
cd code

python3 -m data.a_share_pipeline

CONCURRENCY=20 REQUEST_BATCH_SIZE=4 CHECKPOINT_SIZE=5000 MAX_TOKENS=1024 \
  bash FactorMiner/news_scoring/score_all_news.sh

python3 -m FactorMiner.run_pipeline --prepare-review
```

## 15. 常见失败处理

### 15.1 daily 构建报 industry 缺失

处理顺序：

1. 确认 `basic.parquet` 存在且有 `industry`。
2. 重新运行 `data.a_share_pipeline`。
3. 检查 `metric.parquet` 和 `moneyflow.parquet` 是否有 `industry`。
4. 如需只跑非中性因子，使用 `--disable-neutral`。

检查命令：

```bash
python3 - <<'PY'
import pandas as pd
root = "data/datasets/processed"
for name in ["metric", "moneyflow"]:
    df = pd.read_parquet(f"{root}/{name}.parquet", columns=["industry"])
    print(name, df["industry"].isna().mean(), df["industry"].head().tolist())
PY
```

### 15.2 feature registry 校验失败

通常原因：

- block parquet 被删除。
- manifest 被删除。
- registry 指向旧路径。
- 同名因子在不同 block 中重复。

处理：

```bash
python3 -m FactorMiner.build.daily --validate-only
python3 -m FactorMiner.build.sample_features --validate-only
```

必要时重新构建相关 block。

### 15.3 selection 选出因子过少

先查看原因：

```bash
python3 - <<'PY'
import pandas as pd
c = pd.read_csv("data/datasets/factors/evaluation/candidate_features.csv")
print(c["selection_status"].value_counts(dropna=False))
print(c["reject_reason"].fillna("").value_counts().head(30))
print(c["final_reject_reason"].fillna("").value_counts().head(30))
PY
```

常见解释：

- `rank_ic_days_too_low`：评估区间太短或因子覆盖太稀疏。
- `coverage_too_low`：样本对齐不足或事件因子天然稀疏。
- `rank_ic_too_weak`：单因子信号弱。
- `high_corr_with:*`：被同簇更高分因子替代，不代表因子本身错误。

### 15.4 新闻评分中断

直接重新运行全量评分命令即可。脚本会跳过已评分新闻。

```bash
CONCURRENCY=16 REQUEST_BATCH_SIZE=4 CHECKPOINT_SIZE=5000 MAX_TOKENS=1024 \
  bash FactorMiner/news_scoring/score_all_news.sh
```

不要先删除评分文件，除非明确要重打全部。

### 15.5 相关性计算太慢

先抽样跑通：

```bash
python3 -m FactorMiner.evaluation.selection \
  --corr-row-limit 200000 \
  --min-corr-pairs 5000
```

最终正式结果再全量运行：

```bash
python3 -m FactorMiner.evaluation.selection
```

## 16. 流程输出

FactorMiner 流程的主要输出是：

```text
selected_features.json
或
selected_features_reviewed.json
```

这两个文件只定义“使用哪些因子”和“从哪些 block 读取”。它们不是训练矩阵本身。

下游模型阶段需要做：

1. 读取 `feature_registry.json`。
2. 读取 selected JSON 中的 `blocks`。
3. 按 block 只加载入选列。
4. 按 `sample_id` 合并到样本标签。
5. 做时间切分训练和评估。
