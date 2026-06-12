# stagewise_top20_h48_swloss_v3 results

Run directory:

`data/experiments/msgca/20260602_strategy_window/runs/stagewise_top20_h48_swloss_v3`

Completed at local time: 2026-06-03 02:10 CST.

## Decision

Use the joint finetune best checkpoint as the current candidate:

`data/experiments/msgca/20260602_strategy_window/runs/stagewise_top20_h48_swloss_v3/checkpoints/msgca_best_candidate.pt`

This is a copy of:

`data/experiments/msgca/20260602_strategy_window/runs/stagewise_top20_h48_swloss_v3/checkpoints/msgca_stage_joint_finetune_best.pt`

Do not use `msgca_latest.pt` as the selected checkpoint for this run. It is the final epoch checkpoint, not the validation-best model state.

## Best Results

Best `final_top20` checkpoint:

- epoch: 58
- stage_epoch: 17
- competition_score: 0.034517
- competition_period_return: 67.99%
- competition_rolling_return_mean: 2.1340%
- competition_recent_return_mean: 3.2761%
- competition_max_drawdown: -13.09%
- top10_return_mean: 0.2452%
- top20_return_mean: 0.2312%
- top40_return_mean: 0.2010%

Best `joint_finetune` checkpoint:

- epoch: 68
- stage_epoch: 2
- competition_score: 0.039118
- competition_period_return: 71.39%
- competition_rolling_return_mean: 2.1840%
- competition_recent_return_mean: 3.1882%
- competition_max_drawdown: -11.60%
- top10_return_mean: 0.2896%
- top20_return_mean: 0.2838%
- top40_return_mean: 0.2215%
- return_pred_blend_mse: 0.001009
- direction_pred_accuracy: 51.50%

## Training Notes

- `final_top20` improved materially after epoch 10 and peaked at stage_epoch 17.
- Extending `final_top20` beyond the configured 25 epochs is not supported by this run: stage_epochs 18-25 did not exceed stage_epoch 17.
- `joint_finetune` improved over `final_top20` best at stage_epoch 2, then degraded; early stopping restored stage_epoch 2.
- The run completed final validation and wrote `validation_predictions.parquet`.

## Next Verification

Before using this candidate for deployment or replacing any final library, run the normal holdout/strategy validation using `msgca_best_candidate.pt`, then compare against the previous production candidate under the same workflow.
