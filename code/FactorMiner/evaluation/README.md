# FactorMiner Evaluation 使用说明

`FactorMiner/evaluation/` 负责把已经对齐到 `sample_id` 的样本特征块，转换成可以被下游模型使用的因子筛选结果。

本目录包含四类入口：

```text
quality.py
  样本特征质量检查

single_factor.py
  单因子 IC / RankIC / 分组收益评估

selection.py
  机械候选筛选 + 高相关去冗余 + 自动版 selected_features

review_selection.py
  可选 ChatGPT 复核包生成和复核结果应用
```

本目录不训练最终模型，也不承担 MSGCA 等复杂模型逻辑；它只负责产出“哪些因子可以进入模型”的可审计清单。

## 1. 目录边界

Evaluation 的输入来自两类文件。

第一类是样本和样本特征块：

```text
data/datasets/processed/samples.parquet
data/datasets/features/feature_registry.json
data/datasets/features/blocks/sample/*.parquet
data/datasets/features/manifests/*.json
```

第二类是 evaluation 自己上一步输出的报告：

```text
data/datasets/factors/evaluation/sample_feature_quality.csv
data/datasets/factors/evaluation/factor_summary.csv
data/datasets/factors/evaluation/final/selected_features.json
```

默认输出目录：

```text
data/datasets/factors/evaluation/
```

完整流程：

```text
feature_registry.json + sample feature blocks + samples.parquet
-> quality.py
-> single_factor.py
-> selection.py
-> review_selection.py 可选
-> selected_features.json 或 selected_features_reviewed.json
-> 下游模型训练
```

核心约束：

- 所有输入特征必须已经是 `sample_id` 粒度。
- 标签只从 `samples.parquet` 读取，evaluation 不重新生成标签。
- 日频因子的未来函数问题应在 `build/sample_features.py` 对齐阶段解决。
- 新闻因子的可见性问题应在 `build/news_sample.py` 聚合阶段解决。
- evaluation 不修改 feature block，只读取、评估、筛选并输出报告。

## 2. 一键推荐流程

在 feature blocks 已经构建完成后，按下面顺序运行：

```bash
python3 -m FactorMiner.evaluation.quality
python3 -m FactorMiner.evaluation.single_factor
python3 -m FactorMiner.evaluation.selection
```

如果需要 ChatGPT 复核：

```bash
python3 -m FactorMiner.evaluation.review_selection --prepare
```

然后把生成的：

```text
data/datasets/factors/evaluation/review_prompt.md
```

粘贴给 ChatGPT。把 ChatGPT 返回的 JSON 保存为：

```text
data/datasets/factors/evaluation/review_response.json
```

再应用复核结果：

```bash
python3 -m FactorMiner.evaluation.review_selection \
  --apply \
  --response-path data/datasets/factors/evaluation/review_response.json
```

最终给下游模型使用的文件优先级：

```text
如果做了复核：selected_features_reviewed.json
如果没做复核：selected_features.json
```

## 3. quality.py

`quality.py` 做样本特征质量检查。它不关心因子收益效果，只回答一个问题：这个特征列从数据质量上能不能进入后续评估。

运行命令：

```bash
python3 -m FactorMiner.evaluation.quality
```

### 3.1 输入

默认输入：

```text
--samples-path data/datasets/processed/samples.parquet
--feature-registry-path data/datasets/features/feature_registry.json
```

`samples.parquet` 至少需要：

```text
sample_id
target_trade_date
```

`feature_registry.json` 里必须注册 sample 粒度 block。每个 sample block 的 parquet 至少需要：

```text
sample_id
factor columns
```

### 3.2 输出

默认输出：

```text
sample_feature_quality.csv
sample_feature_block_quality.csv
sample_feature_quality_summary.json
```

`sample_feature_quality.csv` 是逐因子质量报告，核心字段：

| 字段 | 含义 |
| --- | --- |
| `block` | 因子所在 sample feature block |
| `factor_name` | 因子名 |
| `source` / `category` | 来自 FactorSpec 的来源和类别 |
| `row_count` | block 中行数 |
| `sample_count` | samples 中样本总数 |
| `block_sample_match_rate` | block 的样本匹配率 |
| `non_missing_count` | 有效非缺失数量 |
| `missing_rate` | 缺失率 |
| `zero_rate` | 有效值中等于 0 的比例 |
| `unique_count` | 有效值去重数量 |
| `constant_flag` | 是否常数列 |
| `mean/std/min/p01/p05/p50/p95/p99/max` | 分布统计 |
| `year_count` | 覆盖年份数量 |
| `year_coverage_min` | 最差年份覆盖率 |
| `worst_year` | 覆盖率最差年份 |
| `quality_pass` | 是否通过质量检查 |
| `quality_flags` | 不通过原因 |

`sample_feature_block_quality.csv` 是逐 block 质量报告，核心字段：

| 字段 | 含义 |
| --- | --- |
| `block` | block 名称 |
| `row_count` | block 行数 |
| `factor_count` | block 内因子数量 |
| `sample_count` | samples 行数 |
| `matched_sample_count` | block 与 samples 能匹配上的样本数 |
| `missing_sample_count` | samples 中缺失的样本数 |
| `extra_sample_count` | block 中多出来、不在 samples 的样本数 |
| `sample_match_rate` | 样本匹配率 |
| `factor_path` | block parquet 路径 |
| `manifest_path` | manifest 路径 |

### 3.3 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--samples-path` | `data/datasets/processed/samples.parquet` | 样本表路径 |
| `--feature-registry-path` | `data/datasets/features/feature_registry.json` | sample feature registry 路径 |
| `--output-dir` | `data/datasets/factors/evaluation` | 输出目录 |
| `--blocks` | `all` | 要检查的 sample block，逗号分隔；`all` 表示全部 |
| `--since` | 无 | 按 `target_trade_date` 过滤起始日期，闭区间 |
| `--until` | 无 | 按 `target_trade_date` 过滤结束日期，闭区间 |
| `--max-missing-rate` | `0.98` | 最大允许缺失率，超过则标记 `missing_rate_high` |
| `--min-non-missing` | `100` | 最少有效样本数，低于则标记 `non_missing_too_low` |
| `--min-year-coverage` | `0.01` | 最差年份最小覆盖率，低于则标记 `year_coverage_low` |
| `--skip-registry-validate` | false | 跳过 registry 完整性校验 |

### 3.4 quality_flags

| flag | 含义 |
| --- | --- |
| `non_numeric_values` | 非缺失值无法转成数值 |
| `has_inf` | 存在 `inf` 或 `-inf` |
| `missing_rate_high` | 缺失率超过阈值 |
| `non_missing_too_low` | 有效样本数过少 |
| `constant` | 有效值只有一个取值 |
| `year_coverage_low` | 至少一个年份覆盖率过低 |

质量筛选是“底线过滤”。新闻事件因子可能天然稀疏，所以 selection 里对部分新闻因子有 `borderline` 机制，但如果已经是 `quality_failed` 或 `constant`，默认仍不应进入最终清单。

## 4. single_factor.py

`single_factor.py` 做单因子有效性评估。它按 `sample_id` 合并样本标签和特征 block，然后在每个 `target_trade_date` 的横截面上计算 IC、RankIC 和分组收益。

运行命令：

```bash
python3 -m FactorMiner.evaluation.single_factor
```

### 4.1 它在评估什么

对每个因子、每个标签、每个交易日：

```text
x = 当天横截面上的因子值
y = 当天横截面上的未来收益标签
```

然后计算：

| 指标 | 含义 |
| --- | --- |
| `ic` | Pearson correlation，衡量因子值和标签的线性相关 |
| `rank_ic` | Spearman-like rank correlation，先排序再相关，更适合横截面因子 |
| `group_spread` | 因子分组后最高组平均收益减最低组平均收益 |
| `coverage` | 当日有效因子和有效标签配对比例 |
| `pair_count` | 当日有效配对样本数 |

默认标签：

```text
label_next_open_return
label_next_vwap_return
```

### 4.2 输出

默认输出：

```text
factor_ic.csv
factor_rankic.csv
group_return.csv
factor_summary.csv
single_factor_summary.json
```

`factor_ic.csv`：逐日 IC 明细。

| 字段 | 含义 |
| --- | --- |
| `block` | 因子所在 block |
| `factor_name` | 因子名 |
| `label` | 标签名 |
| `target_trade_date` | 评估日期 |
| `row_count` | 当日样本行数 |
| `pair_count` | 当日有效配对数 |
| `coverage` | 当日覆盖率 |
| `ic` | 当日 IC |

`factor_rankic.csv`：逐日 RankIC 明细，字段和 `factor_ic.csv` 类似，只是最后一列为 `rank_ic`。

`group_return.csv`：逐日分组收益。

| 字段 | 含义 |
| --- | --- |
| `group` | 分组编号，默认 1 到 5 |
| `count` | 该组样本数 |
| `mean_return` | 该组平均标签收益 |

`factor_summary.csv`：逐因子汇总，是后续 `selection.py` 的主要输入。

| 字段 | 含义 |
| --- | --- |
| `target_day_count` | 参与评估的交易日数量 |
| `ic_day_count` | 有效 IC 天数 |
| `rank_ic_day_count` | 有效 RankIC 天数 |
| `pair_count_mean` | 平均有效配对数 |
| `coverage_mean` | 平均覆盖率 |
| `ic_mean` / `rank_ic_mean` | 平均 IC / RankIC |
| `ic_std` / `rank_ic_std` | IC / RankIC 的日度标准差 |
| `ic_ir` / `rank_ic_ir` | 均值除以标准差 |
| `ic_positive_rate` / `rank_ic_positive_rate` | IC / RankIC 为正的日期占比 |
| `group_spread_mean` | 平均最高组减最低组收益 |
| `group_spread_positive_rate` | 分组收益差为正的日期占比 |

### 4.3 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--samples-path` | `data/datasets/processed/samples.parquet` | 样本表路径 |
| `--feature-registry-path` | `data/datasets/features/feature_registry.json` | sample feature registry 路径 |
| `--output-dir` | `data/datasets/factors/evaluation` | 输出目录 |
| `--blocks` | `all` | 要评估的 sample block，逗号分隔 |
| `--labels` | `label_next_open_return,label_next_vwap_return` | 要评估的标签列 |
| `--since` | 无 | 按 `target_trade_date` 过滤起始日期，闭区间 |
| `--until` | 无 | 按 `target_trade_date` 过滤结束日期，闭区间 |
| `--min-pairs` | `30` | 单日横截面最少有效配对数，低于则该日 IC 为空 |
| `--groups` | `5` | 分组收益的分组数 |
| `--skip-registry-validate` | false | 跳过 registry 完整性校验 |

### 4.4 常用跑法

只评估某两个 block：

```bash
python3 -m FactorMiner.evaluation.single_factor \
  --blocks manual_metric_sample,manual_moneyflow_sample
```

只评估一段时间：

```bash
python3 -m FactorMiner.evaluation.single_factor \
  --since 2019-01-01 \
  --until 2020-12-31
```

只看一个标签：

```bash
python3 -m FactorMiner.evaluation.single_factor \
  --labels label_next_open_return
```

## 5. selection.py

`selection.py` 把质量报告和单因子报告合并，输出一版可以直接给下游模型使用的自动因子清单。

运行命令：

```bash
python3 -m FactorMiner.evaluation.selection
```

### 5.1 筛选逻辑

它分四层做事。

第一层：合并质量和效果指标。

```text
sample_feature_quality.csv
+ factor_summary.csv
-> candidate_features.csv
```

第二层：机械候选判断。

一个因子要成为 `candidate`，默认需要满足：

```text
quality_pass == true
constant_flag == false
rank_ic_day_count >= 60
coverage_mean >= 0.05
abs(rank_ic_mean) >= 0.01
```

如果设置了 `--min-abs-rank-ic-ir`，还需要：

```text
abs(rank_ic_ir) >= min_abs_rank_ic_ir
```

新闻稀疏因子有一个特殊缓冲：如果不是质量失败或常数列，只是覆盖率或有效 RankIC 天数偏低，但 RankIC 强度够，可能被标记为 `borderline`。`borderline` 不会自动进入 selected，但会进入 review material，方便人工判断事件信号是否值得保留。

第三层：候选打分。

打分公式近似为：

```text
selection_score =
abs(rank_ic_mean)
* sqrt(rank_ic_day_count)
* sqrt(coverage_mean)
* (1 + clipped_abs_rank_ic_ir)
* directional_rank_ic_hit_rate
```

解释：

- `abs(rank_ic_mean)`：信号强度。
- `rank_ic_day_count`：有效天数越多越可信。
- `coverage_mean`：覆盖率越高越可用。
- `rank_ic_ir`：稳定性越高越好。
- `directional_rank_ic_hit_rate`：如果因子方向为正，看 RankIC 为正的比例；如果方向为负，看 RankIC 为负的比例。

第四层：相关性去冗余。

对所有 `candidate` 因子计算相关性。如果两个因子绝对相关超过阈值，就放进同一相关性簇。每个簇只自动保留 `selection_score` 最高的代表因子，其余候选因子标记为：

```text
high_corr_with:<representative_factor>
```

### 5.2 输出

默认输出：

```text
candidate_features.csv
rejected_features.csv
selected_features.csv
selected_features.json
correlation_conflicts.csv
correlation_clusters.csv
review_packet.json
selection_summary.json
```

`candidate_features.csv` 是全量筛选工作台，不只包含最终入选因子。核心字段：

| 字段 | 含义 |
| --- | --- |
| `selection_status` | `candidate` / `borderline` / `rejected` |
| `reject_reason` | 初步拒绝原因 |
| `selection_score` | 自动评分 |
| `direction` | 因子方向，RankIC 均值正为 1，负为 -1 |
| `selected` | 是否最终自动入选 |
| `cluster_id` | 高相关簇编号 |
| `cluster_size` | 所在簇大小 |
| `final_reject_reason` | 最终未入选原因 |
| `high_corr_with` | 因高相关被哪个代表因子替代 |
| `corr_with_selected` | 与代表因子的相关系数 |
| `secondary_direction_consistent` | 主标签和辅助标签方向是否一致 |

`selected_features.json` 是下游模型最重要的自动版输出，结构类似：

```json
{
  "version": "auto_20260520T000000Z",
  "selection_mode": "auto",
  "primary_label": "label_next_open_return",
  "selected_features": ["factor_a", "factor_b"],
  "directions": {
    "factor_a": 1,
    "factor_b": -1
  },
  "blocks": {
    "manual_metric_sample": ["factor_a"],
    "news_llm_market_sample": ["factor_b"],
    "news_llm_stock_sample": ["factor_c"]
  },
  "config": {}
}
```

下游组装训练特征时，应该根据 `blocks` 到对应 sample feature block 里读取列，而不是盲目扫描所有 parquet。

`review_packet.json` 是给人工或 ChatGPT 复核用的材料，不是必须步骤。

### 5.3 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--quality-path` | `evaluation/sample_feature_quality.csv` | 质量报告路径 |
| `--factor-summary-path` | `evaluation/factor_summary.csv` | 单因子汇总路径 |
| `--feature-registry-path` | `data/datasets/features/feature_registry.json` | sample feature registry 路径 |
| `--output-dir` | `data/datasets/factors/evaluation` | 输出目录 |
| `--primary-label` | `label_next_open_return` | 主筛选标签 |
| `--secondary-label` | `label_next_vwap_return` | 辅助参考标签 |
| `--min-rank-ic-days` | `60` | 最少有效 RankIC 天数 |
| `--min-coverage` | `0.05` | 最小平均覆盖率 |
| `--min-abs-rank-ic` | `0.01` | 最小绝对 RankIC 均值 |
| `--min-abs-rank-ic-ir` | `0.0` | 最小绝对 RankIC IR，0 表示不启用 |
| `--corr-threshold` | `0.95` | 高相关去冗余阈值 |
| `--min-corr-pairs` | `10000` | 计算相关性时两个因子的最少共同有效样本数 |
| `--corr-method` | `spearman` | 相关性方法，`spearman` 或 `pearson` |
| `--corr-row-limit` | `0` | 相关性计算抽样行数，0 表示全量 |
| `--max-selected` | `0` | 最多保留多少因子，0 表示不限制 |
| `--skip-registry-validate` | false | 跳过 registry 完整性校验 |

### 5.4 常用跑法

更宽松地保留候选：

```bash
python3 -m FactorMiner.evaluation.selection \
  --min-rank-ic-days 30 \
  --min-coverage 0.02 \
  --min-abs-rank-ic 0.005
```

更严格地筛选：

```bash
python3 -m FactorMiner.evaluation.selection \
  --min-rank-ic-days 120 \
  --min-coverage 0.20 \
  --min-abs-rank-ic 0.015 \
  --min-abs-rank-ic-ir 0.10
```

默认相关性去重使用全量样本；全量复核比较慢或内存紧张时，再临时抽样：

```bash
python3 -m FactorMiner.evaluation.selection \
  --corr-row-limit 200000 \
  --min-corr-pairs 5000
```

限制最终数量：

```bash
python3 -m FactorMiner.evaluation.selection \
  --max-selected 300
```

## 6. review_selection.py

`review_selection.py` 是可选复核层。它的设计目标是：即使没有 API key，只要能使用 ChatGPT 网页订阅，也能把高级审查接入流程。

它分两步：

```text
--prepare
  生成 review_prompt.md 和 review_response_template.json

--apply
  读取 review_response.json，应用到 selected_features.json
```

### 6.1 prepare

运行：

```bash
python3 -m FactorMiner.evaluation.review_selection --prepare
```

默认读取：

```text
selected_features.json
candidate_features.csv
review_packet.json
correlation_clusters.csv
correlation_conflicts.csv
```

默认输出：

```text
review_prompt.md
review_response_template.json
```

`review_prompt.md` 会包含：

- 自动入选因子预览。
- 高分但被拒绝的因子预览。
- borderline 因子预览。
- 相关性簇预览。
- 复核规则和严格 JSON schema。

### 6.2 ChatGPT 返回格式

返回 JSON 支持四个顶层字段：

```json
{
  "remove": [
    {
      "factor_name": "factor_to_remove",
      "reason": "为什么从自动入选中删除"
    }
  ],
  "add_back": [
    {
      "factor_name": "factor_to_add_back",
      "reason": "为什么从候选或 rejected 中加回"
    }
  ],
  "flags": [
    {
      "factor_name": "factor_to_flag",
      "flag": "state_factor",
      "reason": "为什么标记"
    }
  ],
  "global_notes": [
    "整体复核备注"
  ]
}
```

允许的 `flag`：

| flag | 含义 |
| --- | --- |
| `state_factor` | 更像市场状态因子，不一定是个股 alpha |
| `interaction_candidate` | 适合后续和其他因子做交互 |
| `watchlist` | 进入观察清单 |
| `redundancy_risk` | 有冗余风险 |
| `sparse_event_signal` | 稀疏事件信号 |
| `manual_review` | 人工明确复核过 |
| `other` | 其他 |

### 6.3 apply

保存 ChatGPT JSON 后运行：

```bash
python3 -m FactorMiner.evaluation.review_selection \
  --apply \
  --response-path data/datasets/factors/evaluation/review_response.json
```

默认输出：

```text
selected_features_reviewed.json
selected_features_reviewed.csv
selection_review_audit.csv
selection_review_report.md
```

硬约束：

- `remove` 只能删除自动 selected 中已有的因子。
- `add_back` 只能从 `candidate_features.csv` 已知因子中加回。
- 默认禁止加回 `quality_failed` 或 `constant` 因子。
- 每个 remove/add_back/flag 都必须写 reason。
- 未知因子会直接报错。
- 同一个因子不能重复 remove 或重复 add_back。
- 同一个因子的同一个 flag 不能重复。

如果非常确定要加回质量失败因子，可以显式使用：

```bash
python3 -m FactorMiner.evaluation.review_selection \
  --apply \
  --response-path data/datasets/factors/evaluation/review_response.json \
  --allow-quality-failed-add-back
```

这个参数只应该用于人工明确知道原因的特殊情况。正常不建议开启。

### 6.4 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--output-dir` | `data/datasets/factors/evaluation` | 默认输入和输出目录 |
| `--selected-features-path` | `output-dir/selected_features.json` | 自动筛选结果 |
| `--candidate-features-path` | `output-dir/candidate_features.csv` | 候选工作台 |
| `--review-packet-path` | `output-dir/review_packet.json` | selection 生成的复核材料 |
| `--correlation-clusters-path` | `output-dir/correlation_clusters.csv` | 相关性簇 |
| `--correlation-conflicts-path` | `output-dir/correlation_conflicts.csv` | 高相关冲突 |
| `--response-path` | 无 | ChatGPT 返回 JSON，`--apply` 时必须提供 |
| `--prompt-path` | `output-dir/review_prompt.md` | prompt 输出路径 |
| `--response-template-path` | `output-dir/review_response_template.json` | response 模板输出路径 |
| `--review-inputs-path` | `output-dir/review_inputs.txt` | AI 复核输入文件清单，包含关键材料的绝对路径 |
| `--reviewed-json-path` | `output-dir/selected_features_reviewed.json` | 复核后 JSON 输出 |
| `--reviewed-csv-path` | `output-dir/selected_features_reviewed.csv` | 复核后 CSV 输出 |
| `--audit-path` | `output-dir/selection_review_audit.csv` | 审计表输出 |
| `--report-path` | `output-dir/selection_review_report.md` | Markdown 报告输出 |
| `--review-profile` | `research` | `research` 强调验证隔离；`competition` 强调用于正式比赛最终版 |
| `--prepare` | false | 生成 prompt/template |
| `--apply` | false | 应用 response |
| `--max-selected-preview` | `250` | prompt 中最多展示多少个自动入选因子 |
| `--max-rejected-preview` | `150` | prompt 中最多展示多少个高分 rejected 因子 |
| `--max-borderline-preview` | `120` | prompt 中最多展示多少个 borderline 因子 |
| `--max-cluster-preview` | `80` | prompt 中最多展示多少个相关性簇 |
| `--allow-quality-failed-add-back` | false | 是否允许加回质量失败或常数因子 |

如果不传 `--prepare` 或 `--apply`，默认行为是 `--prepare`。

## 7. 模型评估边界

模型训练和模型评估应消费 evaluation 产出的 selected feature 清单。实现位置可以放在模型工程目录，也可以放在独立评估模块，但不应混入 quality、single-factor 或 selection 流程。

模型评估入口建议使用以下文件命名：

```text
FactorMiner/evaluation/model_test.py
tests/test_model_test.py
```

它应该消费：

```text
selected_features.json 或 selected_features_reviewed.json
feature_registry.json
samples.parquet
sample feature blocks
```

它不应该消费未筛选的全量因子，也不应该绕过 `selected_features` 清单直接读所有列。

建议功能：

- 按时间切分 train/valid/test，禁止随机切分。
- 根据 selected feature list 从多个 block 里按需组装训练矩阵。
- 支持 baseline 模型，例如线性模型、LightGBM、CatBoost。
- 支持深度模型评估入口。
- 输出模型预测、模型 IC/RankIC、分组收益、特征重要性、训练配置。
- 支持 walk-forward 评估，避免只看单次切分。

建议默认输出：

```text
model_metrics.csv
model_predictions.parquet
model_feature_importance.csv
model_test_summary.json
```

它和 evaluation 的关系应该是：

```text
quality/single_factor/selection/review_selection
  负责选出可用因子

model_test
  负责验证这一组因子合在一起是否能提升预测
```

换句话说，单因子强不代表组合模型一定强；模型测试要回答的是“这批筛出来的因子作为一个集合，是否真的对下游预测有贡献”。

## 8. 常见问题

### 8.1 quality 报 No sample feature blocks selected

说明 `feature_registry.json` 里没有 `granularity == "sample"` 的 block，或者 `--blocks` 指定了不存在的 block。

先检查：

```bash
python3 -m FactorMiner.build.sample_features --validate-only
```

新闻因子则检查：

```bash
python3 -m FactorMiner.build.news_sample --validate-only
```

### 8.2 single_factor 输出很多空 IC

常见原因：

- 单日有效样本低于 `--min-pairs`。
- 因子在某天是常数。
- 标签在某天是常数或缺失严重。
- 因子覆盖率太低。

排查时可以降低：

```bash
python3 -m FactorMiner.evaluation.single_factor --min-pairs 10
```

但正式筛选不建议把阈值调得太低。

### 8.3 selection 选出来很少

先看 `candidate_features.csv` 的：

```text
selection_status
reject_reason
final_reject_reason
```

如果大量是 `rank_ic_days_too_low`，说明评估区间太短或因子太稀疏。

如果大量是 `coverage_too_low`，说明对齐或事件因子覆盖不足。

如果大量是 `rank_ic_too_weak`，说明单因子效果确实弱。

如果大量是 `high_corr_with:*`，说明自动去冗余生效，不代表因子坏，只是被同簇代表因子替代。

### 8.4 相关性计算很慢

默认相关性去重使用全量样本；如果计算很慢，可以用抽样参数先跑：

```bash
python3 -m FactorMiner.evaluation.selection \
  --corr-row-limit 200000 \
  --min-corr-pairs 5000
```

最终全量复核时再去掉 `--corr-row-limit` 或显式设为 `0`。

### 8.5 新闻因子覆盖率低

新闻因子天然稀疏，低覆盖不一定等于没价值。selection 对新闻相关因子保留 `borderline` 通道，方便进入 ChatGPT 或人工复核。

但以下情况仍应谨慎：

- `quality_failed`
- `constant`
- 大量年份完全无覆盖
- 只在极短时间段有效
- 语义上更像市场状态而非个股 alpha

市场级新闻因子可以在 review 中标记为：

```text
state_factor
```

而不是简单删除。

## 9. 推荐验收命令

单测：

```bash
python3 -m pytest tests/test_evaluation_quality.py -q
python3 -m pytest tests/test_single_factor.py -q
python3 -m pytest tests/test_selection.py -q
python3 -m pytest tests/test_review_selection.py -q
```

全量测试：

```bash
python3 -m pytest -q
```

真实流程验收：

```bash
python3 -m FactorMiner.evaluation.quality
python3 -m FactorMiner.evaluation.single_factor
python3 -m FactorMiner.evaluation.selection
python3 -m FactorMiner.evaluation.review_selection --prepare
```

以上命令用于验收 evaluation 流程。
