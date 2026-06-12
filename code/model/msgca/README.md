# MSGCA 模型使用手册

`model/msgca/` 是 AITrader 的模型工程目录。它消费 `data/` 和 `FactorMiner/` 产出的标准表与筛选因子，负责构造张量、训练排序模型、评估预测、执行回测、生成报告素材和比赛信号。

## 1. 模块边界

| 上游 | 输出 | 本模块用途 |
| --- | --- | --- |
| `data/` | `processed/samples.parquet` | 样本、标签、决策时间和切分字段 |
| `data/` | `processed/price.parquet` | 价格窗口、执行价格和回测价格 |
| `data/` | `processed/moneyflow.parquet` | 资金流窗口 |
| `FactorMiner/` | `features/feature_registry.json` | sample feature block 注册表 |
| `FactorMiner/` | `selected_features*.json` | 模型使用的因子清单 |

`model/msgca/` 不生成标签、不重新构造因子、不调用 LLM、不修改 feature registry。

### 代码结构

| 文件 | 职责 |
| --- | --- |
| `config.py` | 配置加载、路径解析和 resolved config 写出 |
| `feature_set.py` | 从 feature registry 和 selected features 解析模型输入列 |
| `dataset.py` | 样本切分、窗口构造、batch sampler 和 collate |
| `modules.py` | MSGCA、factor encoder 和基线网络结构 |
| `losses.py` | 排序、收益、方向和组合损失 |
| `trainer.py` | 训练循环、checkpoint 写出和训练期验证 |
| `inference.py` | 模型构造、checkpoint 加载和预测 DataFrame 生成 |
| `metrics.py` | 预测表、RankIC、TopK 组合和评估指标输出 |
| `strategy.py` | 比赛信号生成和持仓规则 |
| `train.py` | 训练命令入口 |
| `evaluate.py` | 评估命令入口 |
| `backtest.py` | 回测命令入口和回测输出 |

## 2. 模型框架与实验谱系

MSGCA 是一个面向横截面选股的多源门控交叉注意力模型。每个样本对应一个 `stock_code` 和一个 `target_trade_date`，模型只读取 `feature_asof_date` 及之前可见的信息，输出用于当日横截面排序的 `y_score`，并附带收益预测、方向预测和模态门控权重。

本目录不只有一个固定框架，而是一组围绕“价格窗口 + 筛选因子”的模型和消融：

| 框架/变体 | 入口 | 说明 |
| --- | --- | --- |
| MSGCA full | `modules.MSGCA` | 价格、新闻/文本因子、基本面/资金流/alpha 因子三路输入，跨模态注意力后门控融合 |
| Factor-aware MSGCA | `model.factor_encoder=factor_aware` | 将 selected factors 先按 factor block/group 聚合成 group tokens，再参与跨模态融合；这是最终候选使用的主结构 |
| Simple-factor MSGCA | `model.factor_encoder=simple` | 每个标量因子直接作为 feature token，不做组内 factor gate |
| Price-only MSGCA | `enable_news=false`、`enable_fundamental=false` | 只用价格窗口，作为无因子基线 |
| Selected-factors no-price | `enable_price=false` | 只用筛选因子，验证因子本身是否有横截面有效性 |
| No-news / no-fundamental | `enable_news=false` 或 `enable_fundamental=false` | 检查新闻类因子、基本面/资金流/alpha 因子的边际贡献 |
| StrongFactorMLP | `modules.StrongFactorMLP` | 不走价格 encoder 和跨注意力，只用 selected factors 的残差 MLP 基线 |
| Single-factor TopK | `baselines.simple_factor_topk_predictions` | 单因子排序基线，用于快速 sanity check |

当前最终候选不是纯价格模型，也不是只跑一个通用 MSGCA。最终候选使用 `data/datasets/factors/evaluation/final/selected_features_reviewed.json` 中的 reviewed selected factors，当前文件包含 548 个入选因子、21 个 factor blocks；模型配置为 `factor_aware`，`enable_price=true`、`enable_news=true`、`enable_fundamental=true`，即“价格窗口 + 新闻/文本因子 + 基本面/资金流/alpha/GPT 挖掘因子”的因子增强 MSGCA。

已跑过的主要实验矩阵包括：

| 矩阵 | 代表变体 | 目的 |
| --- | --- | --- |
| `first_wave` | `price_only_no_factors_h64`、`selected_factors_no_price_h64`、`fa_*` | 对比价格-only、因子-only、factor-aware full MSGCA |
| `factor_clean` | `clean_selected_*`、`clean_price_plus_*` | 清理后的因子组合消融 |
| `gpt_final` | `gpt_final_upgrade_h48_proto2_topk005_sparsegate_train20260520` | 使用 final reviewed factors 的最终 sparse gate 候选 |
| `gpt_final_soft` | `gpt_final_upgrade_h48_proto2_topk005_softgate*` | 使用 softmax factor gate 的最终候选对照 |

最终候选的关键结构参数来自 `run_systematic_ablations.py` 的 `gpt_final*` 变体和对应 run 目录下的 `config.resolved.yaml`：`hidden_dim=48`、`price_layers=1`、`factor_layers=1`、`factor_group_layers=1`、`factor_group_prototypes=2`、`topk_return_loss_weight=0.05`，训练窗口扩到 `2019-01-01` 至 `2026-05-20`。

### 2.1 端到端数据流

```text
samples.parquet
price.parquet / moneyflow.parquet
feature_registry.json + selected_features*.json
        |
        v
FeatureLayout
  - price_columns
  - text_columns
  - fundamental_columns
  - factor group ids / names
        |
        v
MSGCASampleDataset
  - 价格窗口按 stock_code 截断到 feature_asof_date
  - sample 级因子按 selected features 装载
  - scaler 只在训练集拟合
        |
        v
batch tensors
  - price_window: [B, P, L]
  - price_mask: [B, P, L]
  - text_features: [B, T]
  - text_mask: [B, T]
  - fundamental_features: [B, F]
  - fundamental_mask: [B, F]
        |
        v
MSGCA
  PriceEncoder
  Text/News FactorEncoder
  Fundamental FactorEncoder
  GatedCrossAttentionFusion
  prediction heads
        |
        v
predictions.parquet
  - y_score / rank
  - return_pred
  - direction_prob
  - g_price / g_text / g_fundamental
```

其中 `B` 是 batch 内样本数，`P` 是价格变量数，`L` 是 lookback 窗口长度，`T` 是新闻/文本类因子数，`F` 是基本面、资金流和技术类因子数。训练 batch 由 `DayBatchSampler` 按 `target_trade_date` 组织，便于排序损失在同一交易日横截面内计算。

### 2.2 输入与特征分层

价格输入来自 `data.price_columns`，默认包括 OHLC、VWAP、成交量、成交额和资金流派生比例。`dataset.py` 会按股票取 `feature_asof_date` 前最近 `lookback` 个交易日，得到 `[P, L]` 价格窗口和同形状缺失掩码；`strict_lookback=true` 时会丢弃历史窗口不足的样本。

sample 级因子来自 `selected_features_reviewed.json` 或 `selected_features.json`。`feature_set.py` 使用 `data.text_prefixes` 和 selected block 名称把新闻/文本类因子分到 `text_columns`，使用 `data.fundamental_prefixes` 把基本面、资金流和 alpha 因子分到 `fundamental_columns`，其他未匹配因子默认归入 fundamental 路。缺失值先用训练集 median 填补，再按训练集 mean/std 标准化，同时保留原始 finite mask 供模型屏蔽缺失 token。

当 `model.factor_encoder=factor_aware` 时，因子还会被分配到组。组名优先利用 selected feature 文件里的 `blocks` 做语义归类：手工资金流和 `gpt_mined_moneyflow` 都归入 `moneyflow`，`gpt_mined_news_state` 归入 `news`，其他 GPT 挖掘块归入 `liquidity`、`valuation`、`reversal` 等主题组；无法从 block 判断时再按特征名规则归类为 `news`、`moneyflow`、`metric`、`alpha158_*` 或 `other`。这些组会在模型中形成 group token，并在预测输出里可选写出 `__factor_group__*` 权重。

### 2.3 编码器

`PriceEncoder` 使用 iTransformer 风格的“变量作为 token”结构：

```text
price_window [B, P, L]
  -> MaskedRevIN 按变量在时间维归一化
  -> Linear(L -> hidden_dim)
  -> 加 variable embedding
  -> TransformerEncoder(price_layers)
  -> price_tokens [B, P, H]
```

这种结构把每个价格变量看成一个 token，token 内包含最近 `L` 天的历史轨迹；`price_mask` 会同时用于 RevIN 统计和 Transformer padding mask。

文本和基本面两路因子有两种编码方式：

| `model.factor_encoder` | 行为 |
| --- | --- |
| `simple` | 每个标量因子经 `Linear(1 -> H)` 投影，加 feature embedding，再过 LayerNorm、Dropout 和 residual MLP，输出 feature tokens |
| `factor_aware` | 每个标量因子先形成 feature token，再加 feature embedding 和 group embedding；经过 residual MLP 后由 factor gate 为组内因子分配权重，池化成 group/prototype tokens，并可选经过 group-level Transformer |

`factor_aware` 的 `factor_gate_activation` 支持 `softmax` 和 `sparsemax`。`softmax` 给出平滑权重，`sparsemax` 会产生精确 0 权重，更适合需要稀疏解释的实验。

### 2.4 交叉注意力与门控融合

`GatedCrossAttentionFusion` 接收三路 token：

```text
price_tokens
text_tokens
fundamental_tokens
```

当 `model.use_cross_attention=true` 时，价格 token 会依次 attend 文本和基本面 token，文本和基本面 token 也会 attend 更新后的价格 token。这样价格走势可以吸收新闻、资金流和基本面上下文，因子 token 也能感知近期价格状态。缺失模态通过 mask 安全跳过，不会因为某一路全缺失而产生无效 attention。

交互后，每一路 token 先做 masked mean pooling，得到：

```text
pooled_price
pooled_text
pooled_fundamental
```

若 `model.use_gate=true`，模型会基于三路 pooled 表征和三路可用性标记计算 softmax 门控：

```text
fused = g_price * pooled_price
      + g_text * pooled_text
      + g_fundamental * pooled_fundamental
```

若关闭 gate，则对可用模态做均匀平均。`g_price`、`g_text`、`g_fundamental` 会写入预测表，可用于观察某个样本主要依赖价格、新闻还是基本面/资金流信息。

### 2.5 输出头与训练目标

融合后的 `fused` 表征经过共享 MLP 后接三个输出头：

| 输出 | 用途 |
| --- | --- |
| `y_score` | 主输出，用于同一 `target_trade_date` 内降序排名 |
| `return_pred` | 辅助收益回归，默认监督 `label_next_open_return` |
| `direction_logit` | 辅助方向分类，sigmoid 后得到上涨概率 |

训练损失由 `msgca_loss` 组合：

```text
total =
  rank_loss_weight * lambda_rankic_loss
  + return_loss_weight * return_mse
  + direction_loss_weight * direction_bce
  + topk_return_loss_weight * soft_topk_return_loss
  - gate_entropy_weight * gate_entropy
```

其中 `lambda_rankic_loss` 按 `target_trade_date` 分组构造样本对，优化日内横截面排序；`soft_topk_return_loss` 是可选的多头偏多收益目标；`gate_entropy` 项鼓励模态门控保持一定分散度，避免过早塌缩到单一路输入。

### 2.6 推理、解释与消融

推理阶段 `inference.py` 会使用训练时保存的 layout 构造同形状模型，输出预测表后按 `target_trade_date` 对 `y_score` 排名。回测和比赛信号只消费预测表中的排序分数、日期和股票信息，不重新调用模型内部模块。

常用结构开关：

| 配置 | 作用 |
| --- | --- |
| `enable_price` | 打开或关闭价格窗口输入 |
| `enable_news` | 打开或关闭新闻/文本因子输入 |
| `enable_fundamental` | 打开或关闭基本面、资金流和 alpha 因子输入 |
| `use_cross_attention` | 控制三路 token 是否先做跨模态交互 |
| `use_gate` | 控制融合时使用学习门控还是可用模态均值 |
| `factor_encoder` | 在 `simple` 与 `factor_aware` 因子编码器之间切换 |
| `use_factor_gate` | 控制 factor-aware 编码器是否学习组内因子权重 |

这些开关也是 `run_systematic_ablations.py` 做系统消融的主要入口。

## 3. 配置

默认配置文件：

```text
model/msgca/config.yaml
```

主要配置项：

| 配置段 | 说明 |
| --- | --- |
| `paths` | processed 数据、feature registry、因子筛选目录和模型输出根目录 |
| `data` | lookback、标签列、时间切分、价格列、特征前缀和快速加载选项 |
| `model` | MSGCA 网络结构、隐藏维度、层数、dropout 和 factor-aware 设置 |
| `train` | epoch、按交易日 batch、学习率、损失权重、AMP 和验证策略 |
| `strategy` | TopN、手续费、滑点、初始资金、换仓数和满仓规则 |

模型特征优先级：

```text
selected_features_reviewed.json
selected_features.json
```

注意：`model/msgca/config.yaml` 是通用入口配置，便于直接训练和调试；最终候选不是只依赖这个默认文件，而是由 `run_systematic_ablations.py` 生成带覆盖项的 run config。复现最终候选时，以 `gpt_final*` 矩阵生成的 `config.resolved.yaml` 为准。

## 4. 训练

常规训练：

```bash
cd code
python3 -m model.msgca.train --config model/msgca/config.yaml
```

小范围入口检查：

```bash
python3 -m model.msgca.train \
  --config model/msgca/config.yaml \
  --limit 512 \
  --epochs 1
```

使用当前配置中的训练窗口训练完整模型：

```bash
python3 -m model.msgca.train \
  --config model/msgca/config.yaml \
  --train-only
```

`--train-only` 使用配置中的 `data.train_start` 到 `data.train_end` 区间，丢弃主标签缺失样本，并跳过验证集预测输出。需要保留验证集输出时不要使用该参数。

复现最终 factor-aware 候选：

```bash
python3 -m model.msgca.run_systematic_ablations \
  --base-config model/msgca/config.yaml \
  --matrix gpt_final \
  --epochs 5
```

复现 softmax gate 对照候选：

```bash
python3 -m model.msgca.run_systematic_ablations \
  --base-config model/msgca/config.yaml \
  --matrix gpt_final_soft \
  --epochs 5
```

## 5. 评估

评估 checkpoint：

```bash
python3 -m model.msgca.evaluate \
  --config model/msgca/config.yaml \
  --checkpoint data/experiments/msgca/<run>/checkpoints/msgca_latest.pt \
  --split validation
```

可选切分：

```text
train
validation
holdout
all
```

评估输出通常包括：

```text
<run>/validation_predictions.parquet
<run>/validation_daily_rankic.csv
<run>/validation_portfolio_topk.csv
<run>/evaluation_summary.json
```

## 6. 回测

从 checkpoint 生成预测并回测：

```bash
python3 -m model.msgca.backtest \
  --config model/msgca/config.yaml \
  --checkpoint data/experiments/msgca/<run>/checkpoints/msgca_latest.pt \
  --split validation \
  --save-predictions
```

基于已有预测文件回测：

```bash
python3 -m model.msgca.backtest \
  --predictions-path data/experiments/msgca/<run>/validation_predictions.parquet \
  --output-root data/experiments/msgca/<run>
```

滚动窗口回测：

```bash
python3 -m model.msgca.backtest \
  --predictions-path data/experiments/msgca/<run>/train_predictions.parquet \
  --output-root data/experiments/msgca/<run> \
  --output-prefix train_rolling_10d \
  --initial-cash 1000000 \
  --rolling-window-days 10 \
  --rolling-step-days 1
```

## 7. 比赛信号

生成指定交易日信号：

```bash
python3 -m model.msgca.predict_live \
  --config model/msgca/config.yaml \
  --checkpoint data/experiments/msgca/<run>/checkpoints/msgca_latest.pt \
  --target-date YYYY-MM-DD
```

默认输出：

```text
<run>/competition_signals/signals_YYYYMMDD.csv
```

信号文件应至少包含：

```text
target_trade_date
stock_code
stock_name
y_score
rank
weight
```

## 8. 报告素材

基于预测文件生成报告素材：

```bash
python3 -m model.msgca.report_assets \
  --model-root data/experiments/msgca/<run> \
  --predictions-path data/experiments/msgca/<run>/predictions.parquet
```

常见输出：

```text
<run>/report_assets/
<run>/daily_rankic.csv
<run>/portfolio_topk.csv
<run>/backtest_metrics.json
```

## 9. 系统消融

运行一组消融实验：

```bash
python3 -m model.msgca.run_systematic_ablations \
  --base-config model/msgca/config.yaml \
  --matrix first_wave \
  --epochs 5
```

限制运行指定变体：

```bash
python3 -m model.msgca.run_systematic_ablations \
  --base-config model/msgca/config.yaml \
  --matrix first_wave \
  --only price_only_no_factors_h64 selected_factors_no_price_h64 \
  --epochs 5
```

## 10. 输出结构

推荐每个训练任务写入独立 run 目录：

```text
data/experiments/msgca/<run>/
  configs/
  checkpoints/
  predictions.parquet
  daily_rankic.csv
  portfolio_topk.csv
  backtest_nav.csv
  backtest_trades.csv
  backtest_positions.csv
  backtest_daily_orders.csv
  backtest_metrics.json
  competition_signals/
  report_assets/
```

## 11. 防泄露约束

- 日频价格和资金流窗口必须截断在 `feature_asof_date`，不能使用 `target_trade_date` 当日或之后的数据。
- 非窗口类 scaler 只能在训练集切分上拟合。
- 新闻特征必须是预先生成的样本级特征；训练阶段不能调用 LLM 或 embedding 服务。
- train、validation、holdout 和 live prediction 均按时间切分。
- 最终比赛训练可以使用截止日期前所有已标注样本，但不能把该训练窗口内表现当作严格测试结论。
