# MSGCA Cluster-Aware Training Test 2026-06-05

## Scope

Tested the idea of making dynamic clusters part of training, instead of only using cluster/industry information as a post-hoc rerank.

The cluster context is label-free and same-day/as-of only. It is derived from existing strict context variables:

- `strict_theme_strength`
- `strict_tr`
- `strict_industry`
- `strict_regime`
- `strict_mf`
- `strict_oh`
- `strict_br`
- `strict_hp`

No external industry/theme labels were introduced.

## Implemented

- Added dynamic cluster context columns:
  - `strict_cluster_id`
  - `strict_cluster_size`
  - `strict_cluster_strength`
  - `strict_cluster_mf`
  - `strict_cluster_hp`
- Added cluster-aware training losses:
  - `cluster_topk_return_loss`
  - `cluster_rank_loss`
  - `in_cluster_rank_loss`
- Wired the new losses through config and trainer.
- Built full context cache:
  - `data/datasets/features/blocks/sample/strict_context_theme_cluster_sample.parquet`
- Added live workflow support for:
  - `--model-score-variant`
  - disabling historical strict context cache in runtime live configs.

## Tests

- `PYTHONPATH=code /home/sutai/home/envs/aitrader/bin/python -m pytest -q code/tests/test_msgca_modules_losses.py`
  - `19 passed`
- `PYTHONPATH=code /home/sutai/home/envs/aitrader/bin/python -m pytest -q code/tests/test_live_pipeline.py`
  - `4 passed`

## Experiments

Root:

- `data/experiments/msgca/20260605_cluster_train`

Runs:

- `cluster_ctx010_scratch_seed2031`
  - stopped after epoch 6 because validation trend was weak.
- `cluster_ctx020_scratch_seed2031`
  - best validation epoch 3, but holdout did not beat the stronger cluster-inrank variant.
- `cluster_inrank020_scratch_seed2031`
  - best validation epoch 3.
  - best score variant: `direct_theme_soft`.

## Current-Code Recomputed Comparison

The older `20260604_theme_train/eval/combined_competition_summary.csv` was generated before the cluster-context code path existed, so I recomputed the baseline with the current code before comparing.

Summary file:

- `data/experiments/msgca/20260605_cluster_train/diagnostics/recomputed_baseline_vs_cluster_summary.csv`

Key results:

| case | split | period_return | period_excess_equal | rolling_return_mean | latest_window_return | max_drawdown | competition_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline `theme_ctx008 + direct_theme_medium` | validation | 14.10% | 4.66% | -0.59% | 11.95% | -25.88% | -0.0002 |
| baseline `theme_ctx008 + direct_theme_soft` | validation | 13.98% | 4.55% | -0.38% | 11.31% | -23.00% | 0.0031 |
| `cluster_inrank020 + direct_theme_soft` | validation | 35.30% | 25.87% | 1.67% | 14.09% | -15.62% | 0.0443 |
| baseline `theme_ctx008 + direct_theme_medium` | holdout | 7.73% | 11.53% | 3.19% | 4.13% | -5.13% | 0.0452 |
| baseline `theme_ctx008 + direct_theme_soft` | holdout | 8.67% | 12.48% | 2.96% | 2.61% | -5.87% | 0.0435 |
| `cluster_inrank020 + direct_theme_soft` | holdout | 12.56% | 16.36% | 4.32% | 0.70% | -5.68% | 0.0585 |

Holdout rolling window comparison:

- `data/experiments/msgca/20260605_cluster_train/diagnostics/holdout_rolling_window_cluster_vs_baseline.csv`
- Cluster wins vs baseline medium: 6 / 10 windows.
- Cluster wins vs baseline soft: 7 / 10 windows.
- Weak point: the latest window, where cluster return is `0.70%` vs baseline medium `4.13%`.

## Exposure Check

Exposure files:

- `data/experiments/msgca/20260605_cluster_train/diagnostics/holdout_position_context_exposure_summary.csv`
- `data/experiments/msgca/20260605_cluster_train/diagnostics/holdout_position_industry_top.csv`
- `data/experiments/msgca/20260605_cluster_train/diagnostics/holdout_position_cluster_top.csv`

Compared with baseline medium, the cluster version has:

- higher `strict_theme_strength_mean`: `2.68` vs `2.56`
- higher `strict_tr_mean`: `2.72` vs `1.98`
- higher `strict_mf_mean`: `0.293` vs `0.247`
- higher `strict_hp_mean`: `0.107` vs `0.087`
- lower `strict_br_mean`: `0.022` vs `0.039`
- higher `strict_cluster_strength_mean`: `2.30` vs `2.11`

The selected industries also shift toward the intended technology line:

- semi conductor holding-row return improved from `0.36%` to `1.57%`.
- communication equipment improved from `-1.24%` to `2.15%`.

## Candidate Live Output

Candidate output was generated without overwriting the old final selected model.

- Summary:
  - `data/runtime/live_pipeline/20260605_cluster_inrank020_live/summary.json`
- Signals:
  - `data/experiments/msgca/20260605_cluster_train/final_selected/cluster_inrank020_direct_theme_soft_live_20260605/competition_signals/signals_20260605.csv`
- Buy list:
  - `data/experiments/msgca/20260605_cluster_train/final_selected/cluster_inrank020_direct_theme_soft_live_20260605/competition_signals/buy_list_20260605.csv`

Top 20 includes:

- `600487.SH` 亨通光电
- `688515.SH` 裕太微-U
- `300747.SZ` 锐科激光
- `301366.SZ` 一博科技
- `688352.SH` 颀中科技
- `603773.SH` 沃格光电

Overlap with previous `theme_ctx008_direct_theme_medium_live_20260604` buy list:

- 11 / 20 names overlap.

## Decision

`cluster_inrank020 + direct_theme_soft` is a real improvement under the current code/evaluation path, especially on validation and most holdout rolling windows.

I would treat it as a candidate replacement, not automatically overwrite the previous final yet, because the latest holdout window is weaker than the old medium strategy. If deploying for the current mainline/technology regime, this candidate is more aligned with the desired exposure.
