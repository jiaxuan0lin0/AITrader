# Loss Ablation Strategy Recheck

This directory re-ranks saved loss-ablation epoch checkpoints under one fixed strategy.

## Strategy

```json
{
  "initial_cash": 1000000.0,
  "top_n": 10,
  "daily_replace_k": 3,
  "fee_rate": 0.0003,
  "slippage_rate": 0.0005,
  "full_investment": true,
  "score_variant": "weighted",
  "score_weight_y": 0.25,
  "score_weight_return": 1.5,
  "score_weight_direction": 0.5,
  "score_weight_cap": 0.75,
  "cap_min_pct": 0.0,
  "cap_bonus": 0.0,
  "exclude_st": true,
  "exclude_bj": true
}
```

## Windows

- val2026_strict: 2026-01-01 to 2026-05-20
- hold2026_recent: 2026-05-21 to 2026-05-31
- same_calendar_2025: 2025-05-21 to 2025-05-31
- same_comp_2025: 2025-06-02 to 2025-06-13

## Selection Score

- val2026_strict weight: 0.45
- hold2026_recent weight: 0.30
- same_calendar_2025 weight: 0.125
- same_comp_2025 weight: 0.125
- Each window uses competition_score when available, then period_excess_equal, then period_return.

## Current Best By Run

```text
              run  epoch  selection_score  selection_score_weight_sum  val2026_strict_selection_component  hold2026_recent_selection_component  same_calendar_2025_selection_component  same_comp_2025_selection_component
r020_topk005_open     11         0.026031                         1.0                            0.024439                             0.077839                               -0.060579                           -0.005964
r010_topk005_open     12         0.009852                         1.0                           -0.013405                             0.036298                               -0.007699                            0.047660
r030_topk005_open      7         0.008589                         1.0                           -0.016353                             0.058761                                0.004092                           -0.017536
```
