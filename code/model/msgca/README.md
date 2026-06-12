# MSGCA 模型

`model/msgca/` 是 AITrader 的模型工程目录。该模块消费 `code/data/` 和 `FactorMiner/` 产出的标准表与筛选因子，负责构造张量、训练排序模型、评估预测、执行回测并生成 live 信号。

命令默认在 `aitrader` conda 环境中从 `code/` 目录执行。

## 模块边界

| 上游 | 输出 | 本模块用途 |
| --- | --- | --- |
| `code/data/` | `processed/samples.parquet` | 样本、标签、决策时间和切分字段。 |
| `code/data/` | `processed/price.parquet` | 价格窗口、执行价格和回测价格。 |
| `code/data/` | `processed/moneyflow.parquet` | 资金流窗口。 |
| `FactorMiner/` | `features/feature_registry.json` | sample feature block 注册表。 |
| `FactorMiner/` | `selected_features*.json` | 模型输入因子清单。 |

`model/msgca/` 不生成标签、不重新构造因子、不调用 LLM、不修改 feature registry。

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `config.py` | 配置加载、路径解析和 resolved config 写出。 |
| `feature_set.py` | 从 feature registry 和 selected features 解析模型输入列。 |
| `dataset.py` | 样本切分、窗口构造、batch sampler 和 collate。 |
| `modules.py` | MSGCA、factor encoder 和基线网络结构。 |
| `losses.py` | 排序、收益、方向、TopK 和上下文相关损失。 |
| `trainer.py` | 训练循环、checkpoint 写出和验证。 |
| `inference.py` | 模型构造、checkpoint 加载和预测表生成。 |
| `metrics.py` | 预测表、RankIC、TopK 组合和评估指标输出。 |
| `strategy.py` | live 信号生成和持仓规则。 |
| `train.py` | 训练入口。 |
| `evaluate.py` | 评估入口。 |
| `backtest.py` | 回测入口。 |
| `predict_live.py` | 单模型 live 信号入口。 |
| `predict_live_ensemble.py` | 多模型 ensemble live 信号入口。 |
| `run_systematic_ablations.py` | 系统消融实验入口。 |
| `competition_metrics.py` | 面向比赛指标的预测文件评估入口。 |

## 模型概览

MSGCA 是面向横截面选股的多源门控交叉注意力模型。每个样本对应一个 `stock_code` 和一个 `target_trade_date`，模型只读取截至 `feature_asof_date` 可见的信息，输出用于当日横截面排序的 `y_score`，并附带收益预测、方向预测和模态门控权重。

主要输入：

| 输入 | 来源 | 说明 |
| --- | --- | --- |
| 价格窗口 | `price.parquet` | 按 `stock_code` 截断到 `feature_asof_date` 前的 lookback 窗口。 |
| 文本/新闻因子 | sample feature blocks | 由 `data.text_prefixes` 和 selected feature blocks 划分。 |
| 基本面/资金流/alpha 因子 | sample feature blocks | 由 `data.fundamental_prefixes` 和 selected feature blocks 划分。 |

主要输出：

| 输出 | 用途 |
| --- | --- |
| `y_score` | 当日横截面排序分数。 |
| `return_pred` | 辅助收益预测。 |
| `direction_prob` | 辅助方向概率。 |
| `g_price`、`g_text`、`g_fundamental` | 模态门控权重。 |
| `__factor_group__*` | 可选 factor group 权重。 |

## 数据流

```text
samples.parquet
price.parquet / moneyflow.parquet
feature_registry.json + selected_features*.json
        |
        v
FeatureLayout
        |
        v
MSGCASampleDataset
        |
        v
batch tensors
        |
        v
MSGCA / baseline model
        |
        v
predictions.parquet
        |
        v
evaluation / backtest / live signals
```

`DayBatchSampler` 按 `target_trade_date` 组织 batch，排序损失在同一交易日横截面内计算。

## 配置

默认配置文件：

```text
model/msgca/config.yaml
```

主要配置段：

| 配置段 | 说明 |
| --- | --- |
| `paths` | processed 数据、feature registry、因子筛选目录和模型输出根目录。 |
| `data` | lookback、标签列、时间切分、价格列、特征前缀和加载选项。 |
| `model` | 网络结构、隐藏维度、层数、dropout 和 factor-aware 设置。 |
| `train` | epoch、按交易日 batch、学习率、损失权重、AMP 和验证策略。 |
| `strategy` | TopN、手续费、滑点、初始资金、换仓数和满仓规则。 |

模型特征清单优先级：

```text
selected_features_reviewed.json
selected_features.json
```

`model/msgca/config.yaml` 是通用入口配置。复现系统消融或最终候选时，以 `run_systematic_ablations.py` 生成的 run config 及对应 `config.resolved.yaml` 为准。

## 训练

常规训练：

```bash
conda activate aitrader
cd code
python -m model.msgca.train \
  --config model/msgca/config.yaml
```

小范围入口检查：

```bash
python -m model.msgca.train \
  --config model/msgca/config.yaml \
  --limit 512 \
  --epochs 1
```

仅使用配置中的训练窗口训练：

```bash
python -m model.msgca.train \
  --config model/msgca/config.yaml \
  --train-only
```

从 checkpoint 继续训练：

```bash
python -m model.msgca.train \
  --config model/msgca/config.yaml \
  --resume-checkpoint data/experiments/msgca/<run>/checkpoints/msgca_latest.pt
```

## 评估

评估 checkpoint：

```bash
python -m model.msgca.evaluate \
  --config model/msgca/config.yaml \
  --checkpoint data/experiments/msgca/<run>/checkpoints/msgca_latest.pt \
  --split validation
```

可选切分：

```text
train
validation
valid
val
holdout
all
```

常见输出：

```text
<run>/validation_predictions.parquet
<run>/validation_daily_rankic.csv
<run>/validation_portfolio_topk.csv
<run>/validation_validation_metrics.json
```

## 回测

从 checkpoint 生成预测并回测：

```bash
python -m model.msgca.backtest \
  --config model/msgca/config.yaml \
  --checkpoint data/experiments/msgca/<run>/checkpoints/msgca_latest.pt \
  --split validation \
  --save-predictions
```

基于已有预测文件回测：

```bash
python -m model.msgca.backtest \
  --predictions-path data/experiments/msgca/<run>/validation_predictions.parquet \
  --output-root data/experiments/msgca/<run>
```

滚动窗口回测：

```bash
python -m model.msgca.backtest \
  --predictions-path data/experiments/msgca/<run>/train_predictions.parquet \
  --output-root data/experiments/msgca/<run> \
  --output-prefix train_rolling_10d \
  --initial-cash 1000000 \
  --rolling-window-days 10 \
  --rolling-step-days 1
```

## Live 信号

单模型 live 推理：

```bash
python -m model.msgca.predict_live \
  --config model/msgca/config.yaml \
  --checkpoint data/experiments/msgca/<run>/checkpoints/msgca_latest.pt \
  --target-date YYYY-MM-DD
```

多模型 ensemble live 推理：

```bash
python -m model.msgca.predict_live_ensemble \
  --config data/experiments/msgca/<run_a>/config.resolved.yaml \
  --checkpoint data/experiments/msgca/<run_a>/checkpoints/msgca_best.pt \
  --config data/experiments/msgca/<run_b>/config.resolved.yaml \
  --checkpoint data/experiments/msgca/<run_b>/checkpoints/msgca_best.pt \
  --target-date YYYY-MM-DD
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

## 系统消融

运行一组消融实验：

```bash
python -m model.msgca.run_systematic_ablations \
  --base-config model/msgca/config.yaml \
  --matrix first_wave \
  --epochs 5
```

限制运行指定变体：

```bash
python -m model.msgca.run_systematic_ablations \
  --base-config model/msgca/config.yaml \
  --matrix first_wave \
  --only price_only_no_factors_h64 selected_factors_no_price_h64 \
  --epochs 5
```

可选矩阵：

```text
first_wave
extended
factor
factor_clean
gpt_final
gpt_final_soft
topk
time
interaction
scale
```

## 报告素材

```bash
python -m model.msgca.report_assets \
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

## 输出结构

训练任务建议写入独立 run 目录：

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

## 防泄露约束

- 日频价格和资金流窗口必须截断在 `feature_asof_date`，不能使用 `target_trade_date` 当日或之后的数据。
- 非窗口类 scaler 只能在训练集切分上拟合。
- 新闻特征必须是预先生成的样本级特征；训练阶段不能调用 LLM 或 embedding 服务。
- train、validation、holdout 和 live prediction 均按时间切分。
- 最终比赛训练可以使用截止日期前所有已标注样本，但该训练窗口内表现不能作为严格测试结论。

## 验证

```bash
python -m pytest tests/test_msgca_dataset.py -q
python -m pytest tests/test_msgca_feature_set.py -q
python -m pytest tests/test_msgca_modules_losses.py -q
python -m pytest tests/test_msgca_strategy_backtest.py -q
```
