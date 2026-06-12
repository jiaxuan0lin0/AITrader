# FactorMiner Evaluation

`FactorMiner/evaluation/` 负责评估已经对齐到 `sample_id` 的样本特征块，并输出下游模型使用的特征清单。

命令默认在 `aitrader` conda 环境中从 `code/` 目录执行。

## 模块边界

Evaluation 只读取样本、样本特征块和前序评估结果，不修改 feature block，不训练最终模型，也不重新生成标签。

## 脚本列表

| 文件 | 说明 |
| --- | --- |
| `quality.py` | 样本特征质量检查。 |
| `single_factor.py` | 单因子 IC、RankIC 和分组收益评估。 |
| `selection.py` | 自动候选筛选、相关性去冗余和 selected features 输出。 |
| `review_selection.py` | 可选人工/ChatGPT 复核材料生成和复核结果应用。 |
| `slice_summary.py` | 从已有逐日评估明细按日期窗口重聚合 summary。 |

## 输入输出

默认输入：

```text
data/datasets/processed/samples.parquet
data/datasets/features/feature_registry.json
data/datasets/features/blocks/sample/*.parquet
data/datasets/features/manifests/*.json
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
```

## 推荐流程

```bash
conda activate aitrader
cd code
python -m FactorMiner.evaluation.quality
python -m FactorMiner.evaluation.single_factor
python -m FactorMiner.evaluation.selection
```

如果需要复核：

```bash
python -m FactorMiner.evaluation.review_selection --prepare
python -m FactorMiner.evaluation.review_selection \
  --apply \
  --response-path data/datasets/factors/evaluation/review_response.json
```

下游模型使用文件优先级：

```text
selected_features_reviewed.json
selected_features.json
```

## Quality

`quality.py` 检查样本特征的数据质量。

```bash
python -m FactorMiner.evaluation.quality
```

默认输入参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--samples-path` | `data/datasets/processed/samples.parquet` | 样本表路径。 |
| `--feature-registry-path` | `data/datasets/features/feature_registry.json` | sample feature registry 路径。 |
| `--output-dir` | `data/datasets/factors/evaluation` | 输出目录。 |
| `--blocks` | `all` | 要检查的 sample block，逗号分隔。 |
| `--since` | 无 | `target_trade_date` 起始日期，闭区间。 |
| `--until` | 无 | `target_trade_date` 截止日期，闭区间。 |
| `--max-missing-rate` | `0.98` | 最大允许缺失率。 |
| `--min-non-missing` | `100` | 最少有效样本数。 |
| `--min-year-coverage` | `0.01` | 最差年份最小覆盖率。 |
| `--workers` | `1` | 并行检查的 block 数。 |
| `--skip-registry-validate` | false | 跳过注册表校验。 |
| `--full-registry-validate` | false | 使用全量 parquet 读取校验注册表。 |

默认输出：

| 文件 | 内容 |
| --- | --- |
| `sample_feature_quality.csv` | 逐因子质量报告。 |
| `sample_feature_block_quality.csv` | 逐 block 质量报告。 |
| `sample_feature_quality_summary.json` | 质量检查摘要。 |

主要质量标记：

| 标记 | 含义 |
| --- | --- |
| `non_numeric_values` | 非缺失值无法转成数值。 |
| `has_inf` | 存在 `inf` 或 `-inf`。 |
| `missing_rate_high` | 缺失率超过阈值。 |
| `non_missing_too_low` | 有效样本数过少。 |
| `constant` | 有效值只有一个取值。 |
| `year_coverage_low` | 至少一个年份覆盖率过低。 |

## Single Factor

`single_factor.py` 按 `sample_id` 合并样本标签和特征 block，并在每个 `target_trade_date` 横截面上计算 IC、RankIC 和分组收益。

```bash
python -m FactorMiner.evaluation.single_factor
```

默认标签：

```text
label_next_open_return
label_next_vwap_return
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--samples-path` | `data/datasets/processed/samples.parquet` | 样本表路径。 |
| `--feature-registry-path` | `data/datasets/features/feature_registry.json` | sample feature registry 路径。 |
| `--output-dir` | `data/datasets/factors/evaluation` | 输出目录。 |
| `--blocks` | `all` | 要评估的 sample block，逗号分隔。 |
| `--labels` | `label_next_open_return,label_next_vwap_return` | 标签列。 |
| `--since` | 无 | `target_trade_date` 起始日期，闭区间。 |
| `--until` | 无 | `target_trade_date` 截止日期，闭区间。 |
| `--min-pairs` | `30` | 单日横截面最少有效配对数。 |
| `--groups` | `5` | 分组收益的分组数。 |
| `--workers` | `1` | 并行评估的 block 数。 |
| `--skip-registry-validate` | false | 跳过注册表校验。 |
| `--full-registry-validate` | false | 使用全量 parquet 读取校验注册表。 |

默认输出：

| 文件 | 内容 |
| --- | --- |
| `factor_ic.csv` | 逐日 IC 明细。 |
| `factor_rankic.csv` | 逐日 RankIC 明细。 |
| `group_return.csv` | 逐日分组收益。 |
| `factor_summary.csv` | 逐因子汇总。 |
| `single_factor_summary.json` | 单因子评估摘要。 |

## Selection

`selection.py` 合并质量报告和单因子评估报告，输出自动筛选结果。

```bash
python -m FactorMiner.evaluation.selection
```

筛选流程：

```text
sample_feature_quality.csv
+ factor_summary.csv
-> candidate_features.csv
-> correlation de-duplication
-> selected_features.json
-> review_packet.json
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--quality-path` | `evaluation/sample_feature_quality.csv` | 质量报告路径。 |
| `--factor-summary-path` | `evaluation/factor_summary.csv` | 单因子汇总路径。 |
| `--feature-registry-path` | `data/datasets/features/feature_registry.json` | sample feature registry 路径。 |
| `--samples-path` | `data/datasets/processed/samples.parquet` | 样本表路径。 |
| `--output-dir` | `data/datasets/factors/evaluation` | 输出目录。 |
| `--primary-label` | `label_next_open_return` | 主筛选标签。 |
| `--secondary-label` | `label_next_vwap_return` | 辅助参考标签。 |
| `--since` | 无 | 相关性样本起始日期。 |
| `--until` | 无 | 相关性样本截止日期。 |
| `--min-rank-ic-days` | `60` | 最少有效 RankIC 天数。 |
| `--min-coverage` | `0.05` | 最小平均覆盖率。 |
| `--min-abs-rank-ic` | `0.01` | 最小绝对 RankIC 均值。 |
| `--min-abs-rank-ic-ir` | `0.0` | 最小绝对 RankIC IR；`0` 表示不启用。 |
| `--corr-threshold` | `0.95` | 高相关去冗余阈值。 |
| `--min-corr-pairs` | `10000` | 相关性计算的最少共同有效样本数。 |
| `--corr-method` | `spearman` | 相关性方法：`spearman` 或 `pearson`。 |
| `--corr-row-limit` | `0` | 相关性计算抽样行数；`0` 表示全量。 |
| `--max-selected` | `0` | 最多保留因子数；`0` 表示不限制。 |
| `--skip-registry-validate` | false | 跳过注册表校验。 |
| `--full-registry-validate` | false | 使用全量 parquet 读取校验注册表。 |

默认输出：

| 文件 | 内容 |
| --- | --- |
| `candidate_features.csv` | 全量候选工作台。 |
| `rejected_features.csv` | 未入选因子。 |
| `selected_features.csv` | 自动入选因子表。 |
| `selected_features.json` | 下游模型读取的自动版特征清单。 |
| `correlation_conflicts.csv` | 高相关冲突明细。 |
| `correlation_clusters.csv` | 相关性簇。 |
| `review_packet.json` | 复核材料。 |
| `selection_summary.json` | 筛选摘要。 |

`selected_features.json` 通过 `blocks` 字段记录每个 sample feature block 中应读取的列。下游训练不应扫描所有 parquet 列。

## Review Selection

`review_selection.py` 是可选复核层，支持先生成 prompt 和 JSON 模板，再应用人工或 ChatGPT 返回的 JSON。

生成复核材料：

```bash
python -m FactorMiner.evaluation.review_selection --prepare
```

默认输出：

```text
review_prompt.md
review_inputs.txt
review_response_template.json
```

应用复核结果：

```bash
python -m FactorMiner.evaluation.review_selection \
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

复核 JSON 支持字段：

```json
{
  "remove": [
    {
      "factor_name": "factor_to_remove",
      "reason": "reason"
    }
  ],
  "add_back": [
    {
      "factor_name": "factor_to_add_back",
      "reason": "reason"
    }
  ],
  "flags": [
    {
      "factor_name": "factor_to_flag",
      "flag": "watchlist",
      "reason": "reason"
    }
  ],
  "global_notes": [
    "review note"
  ]
}
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--output-dir` | `data/datasets/factors/evaluation` | 默认输入和输出目录。 |
| `--selected-features-path` | `output-dir/selected_features.json` | 自动筛选结果。 |
| `--candidate-features-path` | `output-dir/candidate_features.csv` | 候选工作台。 |
| `--review-packet-path` | `output-dir/review_packet.json` | 复核材料。 |
| `--correlation-clusters-path` | `output-dir/correlation_clusters.csv` | 相关性簇。 |
| `--correlation-conflicts-path` | `output-dir/correlation_conflicts.csv` | 高相关冲突。 |
| `--response-path` | 无 | 复核返回 JSON，`--apply` 时必须提供。 |
| `--review-profile` | `research` | `research` 或 `competition`。 |
| `--prepare` | false | 生成 prompt/template。 |
| `--apply` | false | 应用 response。 |
| `--allow-quality-failed-add-back` | false | 是否允许加回质量失败或常数因子。 |

## 注意事项

- 所有输入特征必须已经是 `sample_id` 粒度。
- 标签只从 `samples.parquet` 读取。
- 日频因子的可见性由 `build/sample_features.py` 通过 `feature_asof_date` 控制。
- 新闻因子的可见性由 `build/news_sample.py` 通过 `decision_ts` 控制。
- `quality_failed` 或 `constant` 因子默认不进入最终清单。
- 新闻事件因子可能较稀疏，相关因子可通过 `borderline` 进入复核材料。

## 验证

```bash
python -m pytest tests/test_evaluation_quality.py -q
python -m pytest tests/test_single_factor.py -q
python -m pytest tests/test_selection.py -q
python -m pytest tests/test_review_selection.py -q
```
