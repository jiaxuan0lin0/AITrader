# MSGCA 实验登记

最后更新：2026-06-06

本文件记录 `data/experiments/msgca` 下 MSGCA 相关实验的目录、配置、指标和归档状态。

## 归档状态定义

| 状态 | 含义 |
| --- | --- |
| `final_selected` | 该实验包含已归档的预测产物 |
| `reference` | 该实验作为历史对照或诊断参考保留 |
| `not_selected` | 该实验未进入 `final_selected/` |
| `engineering` | 该实验主要用于工程吞吐、数据管线或 bug 修复 |

## 已归档预测产物

| 字段 | 内容 |
| --- | --- |
| 目录 | `20260605_cluster_train/final_selected/cluster_inrank020_direct_theme_soft_live_20260605/` |
| buy list | `20260605_cluster_train/final_selected/cluster_inrank020_direct_theme_soft_live_20260605/competition_signals/buy_list_20260605.csv` |
| 训练 run | `cluster_inrank020_scratch_seed2031` |
| score | `direct_theme_soft` |
| 状态 | `final_selected` |

当前代码口径下的 holdout 对比：

| 方案 | holdout period_return | excess_equal | rolling10 | latest10 | max_drawdown | competition_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cluster_inrank020 + direct_theme_soft` | 12.56% | 16.36% | 4.32% | 0.70% | -5.68% | 0.05848 |
| `theme baseline + direct_theme_medium` | 7.73% | 11.53% | 3.19% | 4.13% | -5.13% | 0.04517 |
| `theme baseline + direct_theme_soft` | 8.67% | 12.48% | 2.96% | 2.61% | -5.87% | 0.04353 |
| `cluster multi-seed ensemble + direct_theme_soft` | 7.34% | 11.15% | 3.43% | -0.39% | -6.75% | 0.04403 |

`20260604_theme_train` 原始评估中，`theme_ctx008 + direct_theme_medium` 的 holdout score 为 0.06506，latest10 为 4.62%。该结果来自 cluster 训练代码合入前的评估口径。`20260605_cluster_train/diagnostics/recomputed_baseline_vs_cluster_summary.csv` 记录当前代码口径下的 baseline-vs-cluster 重算结果。

## 评估字段

| 字段 | 含义 |
| --- | --- |
| `period_return` | 整段每日 TopN 等权组合收益 |
| `period_excess_equal` | 相对全市场等权收益的超额收益 |
| `rolling10_mean_return` | rolling 10 个交易日窗口收益均值 |
| `latest10_return` | 最近 10 个交易日窗口收益 |
| `win_rate` | 日收益为正的比例 |
| `max_drawdown` | 组合收益曲线最大回撤 |
| `competition_score` | 实验内用于排序模型和策略的综合分 |

跨实验比较字段：

- validation 和 holdout 同时记录。
- 代码口径和数据切分日期同时记录。
- 原始评估结果和重算结果分开记录。
- `final_selected/` 状态单独记录，目录日期不替代实验结果。

## 实验时间线

| 实验组 | 类型 | 目标 | 关键产物 | 状态 |
| --- | --- | --- | --- | --- |
| `20260525_noleak_baseline` | model | 建立无泄漏 MSGCA baseline | `runs/`、评估输出 | `reference` |
| `20260525_msgca_training_iterations` | engineering | 训练工程迭代和吞吐测试 | 多个 `run_*` 目录 | `engineering` |
| `20260526_msgca_factor_aware_vs_mlp` | model | 比较 factor-aware MSGCA 和 MLP | factor-aware/MLP runs | `reference` |
| `20260528_msgca_systematic_ablation` | ablation | 消融价格、文本、基本面、hidden size、topk | `h48/h64/h96`、topk 结果 | `reference` |
| `20260528_factor_evaluation_fullrerun` | factor | 全量因子重跑和筛选 | 因子评估 summary | `reference` |
| `20260529_gpt_factor_clean_final` | factor | GPT/mined 因子复核后清洗 | final feature summaries | `reference` |
| `20260531_final_refit_scaler_resume` | engineering/model | checkpoint scaler 保存/加载修复，final 方案重训 | `final_refit_e20` 等 | `reference` |
| `20260531_epoch_sweep_final` | ablation | 比较 5/8/10/20 epoch 和续训 | epoch checkpoint/eval | `reference` |
| `20260601_loss_ablation` | ablation | return loss、topk loss、top10/top20 策略网格 | `strategy_recheck/` summaries | `reference` |
| `20260602_strategy_window` | model | stagewise 多头训练和 random 10d window 监督 | `stagewise_top20_h48_swloss_v3` | `reference` |
| `20260603_strict_direct_local` | model/score | strict score/loss 变量和直接训练 | strict context/loss/score 文件 | `reference` |
| `20260604_pasted_full_local` | ablation | 外部 loss/score 方案集补测 | `eval_all_txt/combined_competition_summary.csv` | `reference` |
| `20260604_theme_train` | model | 主题强度上下文与主题训练 | `eval/combined_competition_summary.csv`、`final_selected/` | `final_selected` |
| `20260604_theme_overlay_diagnostics` | diagnostics | 主题覆盖、行业覆盖、score overlay 诊断 | diagnostics 输出 | `reference` |
| `20260605_cluster_train` | model | 动态聚类上下文、聚类内排序 loss、层级策略测试 | `eval/`、`diagnostics/`、`strategy_layering/`、`final_selected/` | `final_selected` |

## 关键实验记录

### 20260604 loss/score design set

目标：按外部 loss/score 方案集进行补测。

结果文件：

`20260604_pasted_full_local/eval_all_txt/combined_competition_summary.csv`

| 方案 | holdout period_return | rolling10 | latest10 | competition_score | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `A2 L1 + direct_multihead` | 2.57% | 4.35% | 0.08% | 0.04838 | `reference` |
| `A4/A5 final_score` | 5.71% | 2.11% | 1.53% | 0.04528 | `reference` |
| `A4 exact_a4` | 3.54% | 1.98% | 1.60% | 0.03876 | `reference` |

记录：该组未产生 `final_selected/` 预测产物。

### 20260604 theme train

目标：将主题强度上下文纳入训练。

结果文件：

`20260604_theme_train/eval/combined_competition_summary.csv`

| 方案 | holdout period_return | excess_equal | rolling10 | latest10 | max_drawdown | competition_score | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `theme_ctx008 + direct_theme_medium` | 11.54% | 15.35% | 5.02% | 4.62% | -4.64% | 0.06506 | `final_selected` |
| `theme_ctx008 + direct_theme_soft` | 12.27% | 16.08% | 2.91% | 2.39% | -4.97% | 0.04760 | `reference` |
| `theme_ctx002 + direct_theme_medium` | 10.89% | 14.70% | 5.00% | 未记录 | 未记录 | 0.06332 | `reference` |

预测产物：

`20260604_theme_train/final_selected/theme_ctx008_direct_theme_medium_live_20260604/`

### 20260605 cluster train

目标：将动态聚类上下文和聚类内排序 loss 纳入训练。

结果文件：

- `20260605_cluster_train/eval/combined_competition_summary.csv`
- `20260605_cluster_train/eval/multiseed_competition_summary.csv`
- `20260605_cluster_train/diagnostics/recomputed_baseline_vs_cluster_summary.csv`
- `20260605_cluster_train/strategy_layering/strategy_layering_summary.csv`
- `20260605_cluster_train/final_selected/cluster_inrank020_direct_theme_soft_live_20260605/competition_signals/buy_list_20260605.csv`

记录：

| 项 | 结果 |
| --- | --- |
| 单 seed 2031 | 已进入 `final_selected/` |
| 2031/2041/2051 ensemble | `reference`，未进入 `final_selected/` |
| hard cluster rerank / cluster boost | `reference`，未进入 `final_selected/` |

## 未归档为预测产物的实验记录

| 方案 | 状态 | 记录依据 |
| --- | --- | --- |
| baseline `final_score/y_score` 排序 | `reference` | 与 Top20/rolling 10-day 指标对齐不足 |
| `stagewise_top20_h48_swloss_v3` | `reference` | validation/holdout 指标波动较大 |
| strict A3/A4 单独变体 | `reference` | 早期只覆盖部分外部方案集 |
| `20260604_pasted_full_local` A2/A4/A5 | `reference` | 未生成 `final_selected/` |
| hard cluster rerank / cluster boost | `reference` | `strategy_layering_summary.csv` 记录 holdout score 低于 direct_theme_soft |
| 2031/2041/2051 ensemble | `reference` | holdout score 0.04403，单 seed 2031 score 0.05848 |

## 产物结构

实验目录结构：

```text
data/experiments/msgca/YYYYMMDD_experiment_name/
  README.md                         # 实验目的、配置、结果、状态
  runs/                             # 训练 run、checkpoint、训练日志
  eval/                             # validation/holdout/strategy summary
  diagnostics/                      # 分桶、主题、行业、head calibration 诊断
  strategy_layering/                # 策略网格或后处理实验
  final_selected/                   # 已归档预测产物
  scripts/                          # 队列脚本或复现脚本
```

每个实验 README 记录字段：

- 数据切分：train/validation/holdout 日期。
- 模型来源：scratch、warm-start、checkpoint 路径。
- 核心配置：loss 权重、score 公式、TopN、seed。
- 结果表：validation 和 holdout 分开列。
- 归档状态：`final_selected`、`reference`、`not_selected` 或 `engineering`。

## 数据口径

- 历史实验可能使用不同代码口径，跨实验对比需注明评估脚本和数据切分。
- 预测文件按日期目录归档。
