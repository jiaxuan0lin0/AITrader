# 20260604 Theme Train

本实验记录主题上下文训练在 MSGCA TopN 选股任务中的效果。

## 实验目标

| 项 | 内容 |
| --- | --- |
| 训练对象 | MSGCA |
| 训练方式 | scratch |
| 主要变量 | theme context |
| 输出目录 | `20260604_theme_train/` |

## 主要结果

结果来源：

`eval/combined_competition_summary.csv`

| 方案 | split | period_return | excess_equal | rolling10 | latest10 | max_drawdown | competition_score | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `theme_ctx008 + direct_theme_medium` | holdout | 11.54% | 15.35% | 5.02% | 4.62% | -4.64% | 0.06506 | `final_selected` |
| `theme_ctx008 + direct_theme_soft` | holdout | 12.27% | 16.08% | 2.91% | 2.39% | -4.97% | 0.04760 | `reference` |
| `theme_ctx002 + direct_theme_medium` | holdout | 10.89% | 14.70% | 5.00% | 未记录 | 未记录 | 0.06332 | `reference` |

## final_selected

目录：

`final_selected/theme_ctx008_direct_theme_medium_live_20260604/`

关键输出：

`final_selected/theme_ctx008_direct_theme_medium_live_20260604/competition_signals/buy_list_20260605.csv`

## 对照记录

`20260605_cluster_train/diagnostics/recomputed_baseline_vs_cluster_summary.csv` 对本实验相关 baseline 和 cluster 分支进行了当前代码口径下的重算。

本实验的 `final_selected/` 目录作为历史预测产物保留。
