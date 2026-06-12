# Factor Mining

`FactorMiner/mining/` 用于外部 LLM 辅助候选因子挖掘。该目录生成候选因子材料包、校验 LLM 返回的候选 JSON，并可将已校验候选物化为 sample feature blocks。

命令默认在 `aitrader` conda 环境中从 `code/` 目录执行。

## 模块边界

本模块读取本项目已有数据、因子池、新闻打分字段、泄露约束和候选输出 schema，生成外部 LLM 可消费的小型材料包。

本模块不执行以下操作：

- 不重新进行新闻 LLM 打分。
- 不修改既有 `selected_features.json` 或 `selected_features_reviewed.json`。
- 不训练 MSGCA 或其他模型。
- 不使用原始 parquet 大表作为 LLM 上传材料。

## 脚本列表

| 文件 | 说明 |
| --- | --- |
| `build_packet.py` | 生成候选因子材料包。 |
| `validate_candidates.py` | 校验外部 LLM 返回的候选 JSON。 |
| `materialize_candidates.py` | 将校验通过的候选因子物化为 sample feature blocks。 |

## 生成材料包

```bash
conda activate aitrader
cd code
python -m FactorMiner.mining.build_packet \
  --profile competition \
  --cutoff-date 2026-05-20 \
  --round-name next_regime_20260520 \
  --candidate-count 150
```

默认输出目录：

```text
data/datasets/factors/gpt_mining/experiment/next_regime_20260520/packet/
```

主要输出文件：

| 文件 | 内容 |
| --- | --- |
| `gpt_inputs.txt` | 建议提供给外部 LLM 的文件清单。 |
| `00_context.md` | 项目和任务上下文。 |
| `01_available_fields.json` | 可用字段清单。 |
| `02_existing_factor_summary.csv` | 既有因子摘要。 |
| `03_selected_features_reviewed.json` | 已复核特征清单。 |
| `04_existing_news_features.md` | 新闻特征说明。 |
| `05_market_regime_instruction.md` | 市场结构调研要求。 |
| `06_allowed_operators.md` | 允许使用的算子。 |
| `07_leakage_rules.md` | 防泄露规则。 |
| `candidate_schema.json` | 候选因子 JSON schema。 |
| `prompt_generate_candidates.md` | 候选生成 prompt。 |
| `packet_manifest.json` | 材料包 manifest。 |

## 候选 JSON 要求

外部 LLM 返回内容必须符合 `candidate_schema.json`。顶层字段包含：

```text
web_research_summary
candidates
```

`web_research_summary` 用于记录 cutoff 前公开市场结构、市场情绪、资金偏好、行业主线和因子设计依据。候选因子必须写明 `regime_link`，说明公式试图捕捉的市场结构或交易行为。

建议保存路径：

```text
data/datasets/factors/gpt_mining/experiment/<round>/gpt_response.json
```

## 校验候选

```bash
python -m FactorMiner.mining.validate_candidates \
  --round-dir data/datasets/factors/gpt_mining/experiment/next_regime_20260520
```

默认输入：

```text
data/datasets/factors/gpt_mining/experiment/<round>/gpt_response.json
data/datasets/factors/gpt_mining/experiment/<round>/packet/
```

默认输出：

```text
data/datasets/factors/gpt_mining/experiment/<round>/validated/candidates_validated.json
data/datasets/factors/gpt_mining/experiment/<round>/validated/candidates_rejected_by_parser.csv
data/datasets/factors/gpt_mining/experiment/<round>/validated/candidate_dependency_report.csv
data/datasets/factors/gpt_mining/experiment/<round>/validated/validation_summary.json
```

校验内容：

| 检查项 | 说明 |
| --- | --- |
| JSON schema | 是否符合 `candidate_schema.json`。 |
| 调研摘要 | 是否包含 `web_research_summary`。 |
| 命名 | `factor_name` 是否重复，是否与已有 selected/registered feature 重名。 |
| 输入字段 | `inputs` 是否存在于 packet 字段清单。 |
| 算子 | `formula` 是否只使用白名单算子。 |
| 窗口 | 公式窗口是否只使用 `1/3/5/10/20/60`。 |
| 防泄露 | 是否使用标签、未来收益、比赛期或 post-cutoff 字段。 |
| 新闻评分 | 是否要求重新进行新闻 LLM 打分。 |
| 依赖分类 | `raw_only`、`existing_feature` 或 `mixed`。 |
| 计算分类 | `simple`、`rolling`、`cross_sectional`、`industry_neutral` 或 `interaction`。 |

## 物化候选

校验通过后，可将候选因子写入 sample feature blocks：

```bash
python -m FactorMiner.mining.materialize_candidates \
  --round-dir data/datasets/factors/gpt_mining/experiment/next_regime_20260520
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--validated-path` | 指定 `candidates_validated.json`。 |
| `--categories` | 需要物化的候选类别，逗号分隔；`all` 表示全部。 |
| `--block-prefix` | 输出 block 前缀。 |
| `--limit` | 样本行数上限，仅用于小范围验证。 |
| `--max-candidates` | 候选数量上限。 |
| `--overwrite` | 覆盖已存在输出。 |
| `--skip-registry-validate` | 跳过注册表校验。 |
| `--full-registry-validate` | 使用全量 parquet 读取做注册表校验。 |

## 固定产物

当前固定 GPT-mining 产物目录：

```text
data/datasets/factors/gpt_mining/final/
```

该目录包含材料包、校验结果和已物化候选。

## 约束

- 外部 LLM 只能提出候选公式，本地校验结果决定候选是否保留。
- 候选因子只能使用 packet 字段清单中的字段或已注册 feature。
- 候选因子不能使用未来收益、未来成交、未来新闻或 post-cutoff 信息。
- 候选因子不能要求重新进行 Qwen/GPT 新闻打分。
