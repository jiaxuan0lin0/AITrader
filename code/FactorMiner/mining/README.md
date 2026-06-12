# Factor Mining

本目录用于 GPT 辅助候选因子挖掘，包含：

- 模块一：`Mining Packet` 生成
- 模块二：`Candidate Validation` 候选校验

## 模块一：Mining Packet

目标：把本地已有数据、因子池、新闻打分字段、泄露约束和候选输出 schema 压缩成一组小文件，交给 GPT5.5Pro 做联网调研和候选因子设计。

这个模块只生成材料包，不会：

- 重新新闻打分
- 生成候选因子
- 计算新因子
- 修改现有 `selected_features_reviewed.json`
- 训练模型

## 默认命令

```bash
cd code

python3 -m FactorMiner.mining.build_packet \
  --profile competition \
  --cutoff-date 2026-05-20 \
  --round-name next_regime_20260520 \
  --candidate-count 150
```

默认输出目录：

```text
data/datasets/factors/gpt_mining/experiment/next_regime_20260520/packet/
```

## 输出文件

```text
gpt_inputs.txt
00_context.md
01_available_fields.json
02_existing_factor_summary.csv
03_selected_features_reviewed.json
04_existing_news_features.md
05_market_regime_instruction.md
06_allowed_operators.md
07_leakage_rules.md
candidate_schema.json
prompt_generate_candidates.md
packet_manifest.json
```

给 GPT 上传时优先看 `gpt_inputs.txt`，它会列出建议上传文件的绝对路径。不要上传原始 parquet 大表。

## GPT 输出要求

GPT 应只返回符合 `candidate_schema.json` 的 JSON 对象，顶层包含：

```text
web_research_summary
candidates
```

`web_research_summary` 是强制联网调研摘要，用来说明 GPT 对 cutoff 前 A 股市场结构、市场情绪、资金偏好、行业主线和前沿因子设计的调研结论。它不要求每个候选因子逐条绑定来源，但候选必须写 `regime_link`，说明该公式试图捕捉哪类市场结构或交易行为。

建议保存为：

```text
data/datasets/factors/gpt_mining/experiment/<round>/gpt_response.json
```

后续模块二会读取这个文件做本地候选解析和校验。

## 模块二：Candidate Validation

目标：把 GPT 返回的候选 JSON 转成本地认可的候选清单。这个模块仍然不计算新因子，只做结构校验、字段校验、算子校验、泄露检查和依赖分类。

默认命令：

```bash
cd code

python3 -m FactorMiner.mining.validate_candidates \
  --round-dir data/datasets/factors/gpt_mining/experiment/next_regime_20260520
```

默认输入：

```text
data/datasets/factors/gpt_mining/experiment/<round>/gpt_response.json
data/datasets/factors/gpt_mining/experiment/<round>/packet/
```

如果 `gpt_response.json` 仍放在 `packet/` 下，模块二也会兼容读取；但推荐放在轮次根目录。

默认输出：

```text
data/datasets/factors/gpt_mining/experiment/<round>/validated/candidates_validated.json
data/datasets/factors/gpt_mining/experiment/<round>/validated/candidates_rejected_by_parser.csv
data/datasets/factors/gpt_mining/experiment/<round>/validated/candidate_dependency_report.csv
data/datasets/factors/gpt_mining/experiment/<round>/validated/validation_summary.json
```

当前最终采用版本位于：

```text
data/datasets/factors/gpt_mining/final/
```

校验内容：

- JSON 是否符合 `candidate_schema.json`
- 是否包含强制联网调研摘要 `web_research_summary`
- `factor_name` 是否重复
- 是否和已有 selected/registered feature 重名
- `inputs` 是否都存在于 packet 字段清单
- `formula` 是否只使用白名单算子
- 公式窗口是否只使用 `1/3/5/10/20/60`
- 是否使用标签、未来收益、比赛期或 post-cutoff 字段
- 是否要求重新新闻打分
- 依赖类型分类：`raw_only` / `existing_feature` / `mixed`
- 计算类型分类：`simple` / `rolling` / `cross_sectional` / `industry_neutral` / `interaction`

## 重要边界

- GPT 可以联网调研 cutoff 前的公开市场结构，用于提出研究假设。
- GPT 不能生成本地不可计算字段。
- GPT 不能要求重新 Qwen/GPT 新闻打分。
- GPT 不能使用未来收益、未来成交、未来新闻。
- 本地验证结果才决定候选因子是否保留。
