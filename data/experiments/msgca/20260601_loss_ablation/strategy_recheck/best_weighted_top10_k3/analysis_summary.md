# Loss Ablation Strategy Recheck Summary

Generated: 2026-06-01 22:05 Asia/Shanghai

## Inputs

- Runs: `r010_topk005_open`, `r020_topk005_open`, `r030_topk005_open`
- Epochs per run: 20
- Windows per epoch: 4
- Final rows: 240, with 0 duplicate `(run, epoch, window)` keys
- Strategy: `weighted`, `top_n=10`, `daily_replace_k=3`, `exclude_st=true`, `exclude_bj=true`

## Selection Score

- `val2026_strict`: weight 0.45, uses `competition_score`
- `hold2026_recent`: weight 0.30, falls back to `period_excess_equal`
- `same_calendar_2025`: weight 0.125, falls back to `period_excess_equal`
- `same_comp_2025`: weight 0.125, falls back to `period_excess_equal`

Short windows do not have enough rolling 10-day subwindows to produce `competition_score`, so the fallback is intentional.

## Best By Run

| Run | Best epoch | Selection score | Val component | Holdout component | 2025 calendar component | 2025 comp component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r020_topk005_open` | 11 | 0.026031 | 0.024439 | 0.077839 | -0.060579 | -0.005964 |
| `r010_topk005_open` | 12 | 0.009852 | -0.013405 | 0.036298 | -0.007699 | 0.047660 |
| `r030_topk005_open` | 7 | 0.008589 | -0.016353 | 0.058761 | 0.004092 | -0.017536 |

## Top Candidate

Recommended candidate from this recheck: `r020_topk005_open`, epoch 11.

Key metrics for epoch 11:

| Window | Strategy return | Market equal return | Excess |
| --- | ---: | ---: | ---: |
| `val2026_strict` | 19.36% | 11.90% | 7.46% |
| `hold2026_recent` | 3.58% | -4.21% | 7.78% |
| `same_calendar_2025` | -8.28% | -2.22% | -6.06% |
| `same_comp_2025` | 0.74% | 1.34% | -0.60% |

## Interpretation

- `r020` is still the best `r` level after including holdout and 2025 short-window evidence.
- `r020` epoch 11 has the strongest validation competition score and the strongest latest holdout excess among the top candidates.
- `r030` does not improve over `r020`; based on this run, expanding to `r050` is not supported as the next priority.
- Raising latest holdout above the combined 2025 reference windows moves the recommended epoch from 12 back to 11.
