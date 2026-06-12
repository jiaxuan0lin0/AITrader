# Next Steps

Completed topk-return queue:
- r020_top10_topk008
- r020_top10_topk010
- r020_top10_topk012

Decision rule:
- Train each to epoch 14 and run unified recheck.
- If the winning run's best checkpoint is epoch 14 and the selection/validation curve is still improving, extend that winning run beyond 14 before moving on.
- After selecting the best topk_return_loss_weight, create follow-up runs using that winning loss setup:
  - <best>_vwap50: return_secondary_weight=0.5, topk_secondary_weight=0.5
  - <best>_tw2025: stronger 2025 time weights
  - <best>_twrecent: smoother recent-year time weights

Keep final strategy fixed at Top10 / daily_replace_k=3 / current weighted score.

## Result

Unified recheck finished with 168 rows.

Top selection-score checkpoints:
- r020_top10_topk010 epoch 11: 0.032885
- r020_top10_topk008 epoch 11: 0.029237
- r020_top10_topk012 epoch 13: 0.023249

No run was still improving at epoch 14, so no epoch extension was triggered.

Comparison against previous 0.05 run under the same final Top10 / daily_replace_k=3 recheck:
- r020_top10_topk010 epoch 11: 0.032885
- r020_topk005_open epoch 11: 0.026031

Note: r020_topk005_open was trained with topk_return_loss_weight=0.05 and topk_return_k=20,
then rechecked under the fixed Top10 strategy. The current winner uses
topk_return_loss_weight=0.10 and topk_return_k=10, so it is both stronger topk loss weight
and a loss target better aligned to the final Top10 strategy.

## Follow-up Queue

Winner setup:
- topk_return_loss_weight: 0.10
- topk_return_k: 10
- return_loss_weight: 0.20

Follow-up runs:
- r020_top10_topk010_vwap50
- r020_top10_topk010_tw2025
- r020_top10_topk010_twrecent

## Follow-up Status

The follow-up queue was stopped on 2026-06-02 before completion to free GPU
memory for live prediction. `r020_top10_topk010_vwap50` reached epoch 11; its
best validation checkpoint stayed at epoch 6 with competition_score 0.009771,
well below the completed baseline `r020_top10_topk010` epoch 11 score 0.044122.

Current live workflow default:
- model config: `runs/r020_top10_topk010/config.resolved.yaml`
- checkpoint: `runs/r020_top10_topk010/checkpoints/msgca_best.pt`
- best checkpoint epoch: 11
- strategy: Top10, daily_replace_k=3, weighted y/return/direction/cap =
  0.25/1.5/0.5/0.75, exclude ST and BJ.
