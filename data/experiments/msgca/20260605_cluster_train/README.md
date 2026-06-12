# 20260605 Cluster Train

本实验记录动态聚类上下文和聚类内排序训练在 MSGCA TopN 选股任务中的效果。

## 实验目标

| 项 | 内容 |
| --- | --- |
| 训练对象 | MSGCA |
| 训练方式 | scratch |
| 主要变量 | 动态聚类上下文 |
| 主要 loss | cluster topk、cluster rank、in-cluster rank |
| 输出目录 | `20260605_cluster_train/` |

## 新增上下文字段

| 字段 | 含义 |
| --- | --- |
| `context_cluster_id` | 动态聚类编号 |
| `context_cluster_size` | 所在簇规模 |
| `context_cluster_strength` | 簇整体强度 |
| `context_cluster_mf` | 簇资金确认强度 |
| `context_cluster_hp` | 簇健康回撤/主题修复强度 |

## 新增训练约束

| loss | 作用 |
| --- | --- |
| `cluster_topk_return_loss` | 簇维度 TopK return 约束 |
| `cluster_rank_loss` | 簇之间排序约束 |
| `in_cluster_rank_loss` | 同簇股票内部排序约束 |

代码与测试记录：

`../../../report/msgca_cluster_training_test_20260605.md`

## Current-code baseline vs cluster

结果来源：

`diagnostics/recomputed_baseline_vs_cluster_summary.csv`

| 方案 | split | score | period_return | excess_equal | rolling10 | latest10 | max_drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline medium | validation | -0.00021 | 14.10% | 4.66% | -0.59% | 11.95% | -25.88% |
| baseline soft | validation | 0.00308 | 13.98% | 4.55% | -0.38% | 11.31% | -25.36% |
| cluster soft | validation | 0.04428 | 35.30% | 25.87% | 1.67% | 14.09% | -21.48% |
| baseline medium | holdout | 0.04517 | 7.73% | 11.53% | 3.19% | 4.13% | -5.13% |
| baseline soft | holdout | 0.04353 | 8.67% | 12.48% | 2.96% | 2.61% | -5.87% |
| cluster soft | holdout | 0.05848 | 12.56% | 16.36% | 4.32% | 0.70% | -5.68% |

记录：

- validation 中，cluster soft 的 `score`、`period_return`、`excess_equal` 高于 baseline medium 和 baseline soft。
- holdout 中，cluster soft 的 `score`、`period_return`、`excess_equal`、`rolling10` 高于 baseline medium 和 baseline soft。
- holdout 中，cluster soft 的 `latest10` 低于 baseline medium 和 baseline soft。

## Multi-seed ensemble

结果来源：

`eval/multiseed_competition_summary.csv`

| 方案 | split | score | period_return | excess_equal | rolling10 | latest10 | max_drawdown | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| seed2031 cluster soft | holdout | 0.05848 | 12.56% | 16.36% | 4.32% | 0.70% | -5.68% | `final_selected` |
| 2031/2041/2051 ensemble soft | holdout | 0.04403 | 7.34% | 11.15% | 3.43% | -0.39% | -6.75% | `reference` |
| seed2051 cluster medium | holdout | 0.02086 | 0.06% | 3.86% | 1.26% | 1.69% | -7.97% | `reference` |

## Strategy layering

结果来源：

`strategy_layering/strategy_layering_summary.csv`

| 推理策略 | split | score | period_return | 状态 |
| --- | --- | ---: | ---: | --- |
| `direct_theme_soft` | holdout | 0.05848 | 12.56% | `final_selected` |
| `cluster_boostlight` | holdout | 0.02723 | 5.33% | `reference` |
| `cluster_boost` | holdout | 0.00920 | 2.64% | `reference` |
| `cluster_booststrong` | holdout | -0.01475 | -0.26% | `reference` |
| `cluster12` | holdout | -0.06083 | -16.10% | `reference` |
| `cluster6` | holdout | -0.06952 | -17.49% | `reference` |

## final_selected

目录：

`final_selected/cluster_inrank020_direct_theme_soft_live_20260605/`

关键文件：

| 文件 | 用途 |
| --- | --- |
| `competition_signals/buy_list_20260605.csv` | 2026-06-05 buy list |
| `competition_signals/` | 选股和下单辅助输出 |

历史对照目录：

`../20260604_theme_train/final_selected/theme_ctx008_direct_theme_medium_live_20260604/`

## 测试记录

| 测试 | 状态 |
| --- | --- |
| loss/module 单测 | passed |
| prediction pipeline 单测 | passed |
| strategy/backtest 单测 | passed |

本 README 为实验记录，不触发训练或回测。
