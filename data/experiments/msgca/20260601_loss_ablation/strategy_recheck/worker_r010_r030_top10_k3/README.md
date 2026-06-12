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

- same_comp_2025: 2025-06-02 to 2025-06-13

## Current Best By Run

```text
              run  epoch  selection_score  val2026_strict_competition_score
r010_topk005_open     15         0.000643                          0.000643
```
