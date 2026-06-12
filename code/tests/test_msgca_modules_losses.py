from __future__ import annotations

import pandas as pd
import torch

from model.msgca.losses import (
    AuxTopKLoss,
    LossWeights,
    cluster_rank_loss,
    cluster_topk_return_loss,
    in_cluster_rank_loss,
    lambda_rankic_loss,
    msgca_loss,
    soft_topk_return_loss,
    strategy_window_return_loss,
)
from model.msgca.modules import FactorAwareEncoder, MSGCA, MSGCAOutput, MaskedRevIN, NumericTokenEncoder, StrongFactorMLP, sparsemax


def test_revin_ignores_masked_values_and_outputs_finite() -> None:
    revin = MaskedRevIN(num_variables=2, affine=False)
    x = torch.tensor([[[1.0, 2.0, 1000.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, False], [False, False, False]]])

    out = revin(x, mask)

    assert torch.isfinite(out).all()
    assert out[0, 0, 2].item() == 0.0
    assert out[0, 1].abs().sum().item() == 0.0


def test_msgca_missing_text_gate_is_zero_and_no_nan() -> None:
    model = MSGCA(
        price_variables=2,
        lookback=4,
        text_features=3,
        fundamental_features=2,
        hidden_dim=16,
        n_heads=4,
        price_layers=1,
        dropout=0.0,
    )
    price = torch.randn(5, 2, 4)
    price_mask = torch.ones(5, 2, 4, dtype=torch.bool)
    text = torch.zeros(5, 3)
    text_mask = torch.zeros(5, dtype=torch.bool)
    fundamental = torch.randn(5, 2)
    fundamental_mask = torch.ones(5, dtype=torch.bool)

    output = model(price, price_mask, text, text_mask, fundamental, fundamental_mask)

    assert torch.isfinite(output.y_score).all()
    assert output.final_score is not None
    assert torch.isfinite(output.final_score).all()
    assert torch.allclose(output.g_text, torch.zeros_like(output.g_text), atol=1e-6)
    assert torch.allclose(output.gates.sum(dim=1), torch.ones(5), atol=1e-6)


def test_numeric_token_encoder_accepts_sample_level_mask() -> None:
    encoder = NumericTokenEncoder(max_features=3, hidden_dim=8, dropout=0.0)
    values = torch.randn(2, 3)
    mask = torch.tensor([True, False])

    tokens, token_mask = encoder(values, mask)

    assert tokens.shape == (2, 3, 8)
    assert token_mask.tolist() == [[True, True, True], [False, False, False]]
    assert torch.isfinite(tokens).all()


def test_factor_aware_encoder_masks_missing_features_and_outputs_groups() -> None:
    encoder = FactorAwareEncoder(
        max_features=4,
        hidden_dim=16,
        n_heads=4,
        dropout=0.0,
        feature_count=4,
        group_ids=[0, 0, 1, 1],
        group_names=["metric", "moneyflow"],
        factor_layers=1,
        group_layers=1,
    )
    values = torch.randn(2, 4)
    mask = torch.tensor([[True, False, True, False], [False, False, False, False]])

    tokens, group_mask, group_weights = encoder(values, mask)

    assert tokens.shape == (2, 2, 16)
    assert group_mask.tolist() == [[True, True], [False, False]]
    assert group_weights.shape == (2, 2)
    assert torch.allclose(group_weights[1], torch.zeros(2), atol=1e-6)
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(group_weights).all()


def test_factor_aware_encoder_group_prototypes_expand_tokens_not_attribution() -> None:
    encoder = FactorAwareEncoder(
        max_features=4,
        hidden_dim=16,
        n_heads=4,
        dropout=0.0,
        feature_count=4,
        group_ids=[0, 0, 1, 1],
        group_names=["metric", "moneyflow"],
        factor_layers=1,
        group_layers=1,
        group_prototypes=2,
    )
    values = torch.randn(2, 4)
    mask = torch.tensor([[True, False, True, True], [False, False, False, False]])

    tokens, group_mask, group_weights = encoder(values, mask)

    assert tokens.shape == (2, 4, 16)
    assert group_mask.tolist() == [[True, True, True, True], [False, False, False, False]]
    assert group_weights.shape == (2, 2)
    assert torch.allclose(group_weights[1], torch.zeros(2), atol=1e-6)
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(group_weights).all()


def test_sparsemax_gate_can_produce_exact_zero_weights() -> None:
    weights = sparsemax(torch.tensor([[2.0, 0.0, -1.0]]), dim=1)

    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-6)
    assert weights[0, 2].item() == 0.0


def test_msgca_factor_aware_outputs_group_attribution_without_nan() -> None:
    model = MSGCA(
        price_variables=2,
        lookback=4,
        text_features=2,
        fundamental_features=3,
        hidden_dim=16,
        n_heads=4,
        price_layers=1,
        dropout=0.0,
        factor_encoder="factor_aware",
        factor_layers=1,
        factor_group_layers=1,
        text_group_ids=[0, 0],
        text_group_names=["news"],
        fundamental_group_ids=[0, 1, 1],
        fundamental_group_names=["metric", "moneyflow"],
    )
    price = torch.randn(3, 2, 4)
    price_mask = torch.ones(3, 2, 4, dtype=torch.bool)
    text = torch.randn(3, 2)
    text_mask = torch.tensor([[True, False], [False, False], [True, True]])
    fundamental = torch.randn(3, 3)
    fundamental_mask = torch.tensor([[True, True, False], [False, False, False], [True, False, True]])

    output = model(price, price_mask, text, text_mask, fundamental, fundamental_mask)

    assert torch.isfinite(output.y_score).all()
    assert output.factor_group_weights is not None
    assert output.factor_group_weights.shape == (3, 3)
    assert torch.isfinite(output.factor_group_weights).all()


def test_strong_factor_mlp_uses_factor_masks_and_outputs_heads() -> None:
    model = StrongFactorMLP(text_features=2, fundamental_features=3, hidden_dim=16, dropout=0.0)
    text = torch.randn(4, 2)
    text_mask = torch.tensor([[True, False], [False, False], [True, True], [True, False]])
    fundamental = torch.randn(4, 3)
    fundamental_mask = torch.tensor([[True, True, False], [True, False, True], [False, False, False], [True, True, True]])

    output = model(text, text_mask, fundamental, fundamental_mask)

    assert output.y_score.shape == (4,)
    assert torch.isfinite(output.return_pred).all()
    assert torch.isfinite(output.direction_logit).all()
    assert torch.allclose(output.gates.sum(dim=1), torch.ones(4), atol=1e-6)


def test_lambda_rankic_groups_by_trade_date() -> None:
    scores = torch.tensor([3.0, 2.0, 1.0, 0.0], requires_grad=True)
    labels = torch.tensor([0.03, 0.02, -0.01, 0.04])
    dates = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"])

    loss = lambda_rankic_loss(scores, labels, dates, max_pairs_per_day=10)
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_time_weighted_rankic_loss_changes_day_importance() -> None:
    scores = torch.tensor([2.0, 1.0, 2.0, 1.0], requires_grad=True)
    labels = torch.tensor([0.03, 0.01, 0.01, 0.03])
    dates = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"])

    equal = lambda_rankic_loss(scores, labels, dates, max_pairs_per_day=10)
    weighted = lambda_rankic_loss(
        scores,
        labels,
        dates,
        max_pairs_per_day=10,
        sample_weights=torch.tensor([0.1, 0.1, 2.0, 2.0]),
    )

    assert torch.isfinite(weighted)
    assert weighted > equal


def test_topk_loss_zero_weight_preserves_total_loss() -> None:
    scores = torch.tensor([0.2, -0.1, 0.4, 0.0])
    labels = torch.tensor([0.01, -0.02, 0.03, 0.0])
    dates = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"])
    gates = torch.full((4, 3), 1 / 3)
    output = MSGCAOutput(
        y_score=scores,
        return_pred=scores,
        direction_logit=scores,
        gates=gates,
        g_price=gates[:, 0],
        g_text=gates[:, 1],
        g_fundamental=gates[:, 2],
    )

    base, _ = msgca_loss(output, labels, labels, labels.gt(0).float(), dates, LossWeights(topk_return=0.0), 10)
    changed_temperature, _ = msgca_loss(
        output,
        labels,
        labels,
        labels.gt(0).float(),
        dates,
        LossWeights(topk_return=0.0, topk_temperature=0.001),
        10,
    )

    assert torch.isfinite(soft_topk_return_loss(scores, labels, dates))
    assert torch.allclose(base, changed_temperature, atol=1e-8)


def test_topk_loss_soft_equal_topk_matches_topn_portfolio() -> None:
    scores = torch.tensor([4.0, 3.0, 1.0, 0.0])
    labels = torch.tensor([0.08, 0.02, -0.10, -0.20])
    dates = pd.to_datetime(["2020-01-02"] * 4)

    loss = soft_topk_return_loss(scores, labels, dates, topk_k=2, temperature=0.01)

    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(-0.05), atol=1e-3)


def test_strategy_window_loss_rewards_topn_excess_return() -> None:
    labels = torch.tensor([0.08, 0.02, -0.10, -0.20, 0.04, 0.03, -0.04, -0.05])
    dates = pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 4)
    good_scores = torch.tensor([4.0, 3.0, 1.0, 0.0, 4.0, 3.0, 1.0, 0.0], requires_grad=True)
    bad_scores = torch.tensor([0.0, 1.0, 3.0, 4.0, 0.0, 1.0, 3.0, 4.0], requires_grad=True)

    good_loss, good_parts = strategy_window_return_loss(good_scores, labels, dates, topk_k=2, temperature=0.01)
    bad_loss, _ = strategy_window_return_loss(bad_scores, labels, dates, topk_k=2, temperature=0.01)
    good_loss.backward()

    assert torch.isfinite(good_loss)
    assert good_loss < bad_loss
    assert good_parts["strategy_window_excess_return"] > 0
    assert good_scores.grad is not None
    assert torch.isfinite(good_scores.grad).all()


def test_cluster_topk_loss_rewards_strong_cluster() -> None:
    labels = torch.tensor([0.06, 0.04, 0.03, -0.02, -0.03, -0.04])
    clusters = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    dates = pd.to_datetime(["2020-01-02"] * 6)
    good_scores = torch.tensor([3.0, 2.5, 2.0, 0.0, -0.5, -1.0], requires_grad=True)
    bad_scores = torch.tensor([0.0, -0.5, -1.0, 3.0, 2.5, 2.0], requires_grad=True)

    good_loss = cluster_topk_return_loss(good_scores, labels, dates, clusters, topk_k=1, min_size=2)
    bad_loss = cluster_topk_return_loss(bad_scores, labels, dates, clusters, topk_k=1, min_size=2)
    good_loss.backward()

    assert torch.isfinite(good_loss)
    assert good_loss < bad_loss
    assert good_scores.grad is not None
    assert torch.isfinite(good_scores.grad).all()


def test_cluster_rank_loss_backpropagates() -> None:
    labels = torch.tensor([0.06, 0.04, 0.03, -0.02, -0.03, -0.04])
    clusters = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    dates = pd.to_datetime(["2020-01-02"] * 6)
    scores = torch.tensor([3.0, 2.5, 2.0, 0.0, -0.5, -1.0], requires_grad=True)

    loss = cluster_rank_loss(scores, labels, dates, clusters, min_size=2)
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_in_cluster_rank_loss_backpropagates() -> None:
    labels = torch.tensor([0.06, 0.04, -0.02, 0.05, 0.01, -0.03])
    clusters = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    dates = pd.to_datetime(["2020-01-02"] * 6)
    scores = torch.tensor([3.0, 2.0, 0.0, 2.5, 1.0, -0.5], requires_grad=True)

    loss = in_cluster_rank_loss(scores, labels, dates, clusters, topk_k=1, min_size=2)
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_msgca_loss_can_use_secondary_return_target() -> None:
    scores = torch.tensor([0.2, -0.1, 0.4, 0.0])
    primary = torch.zeros(4)
    secondary = torch.tensor([0.2, -0.2, 0.4, 0.0])
    dates = pd.to_datetime(["2020-01-02"] * 2 + ["2020-01-03"] * 2)
    gates = torch.full((4, 3), 1 / 3)
    output = MSGCAOutput(
        y_score=scores,
        return_pred=secondary,
        direction_logit=scores,
        gates=gates,
        g_price=gates[:, 0],
        g_text=gates[:, 1],
        g_fundamental=gates[:, 2],
    )

    _, parts = msgca_loss(
        output,
        primary,
        secondary,
        primary.gt(0).float(),
        dates,
        LossWeights(rank=0.0, return_mse=1.0, direction_bce=0.0, topk_return=0.0, gate_entropy=0.0, return_secondary_weight=1.0),
        10,
    )

    assert torch.allclose(parts["return_loss"], torch.tensor(0.0), atol=1e-8)


def test_msgca_loss_can_train_final_score_with_aux_topk() -> None:
    y_score = torch.tensor([0.0, 0.0, 0.0, 0.0])
    final_score = torch.tensor([4.0, 3.0, 1.0, 0.0], requires_grad=True)
    labels = torch.tensor([0.08, 0.02, -0.10, -0.20])
    dates = pd.to_datetime(["2020-01-02"] * 4)
    gates = torch.full((4, 3), 1 / 3)
    output = MSGCAOutput(
        y_score=y_score,
        return_pred=y_score,
        direction_logit=y_score,
        gates=gates,
        g_price=gates[:, 0],
        g_text=gates[:, 1],
        g_fundamental=gates[:, 2],
        final_score=final_score,
    )

    loss, parts = msgca_loss(
        output,
        labels,
        labels,
        labels.gt(0).float(),
        dates,
        LossWeights(
            rank=0.0,
            return_mse=0.0,
            direction_bce=0.0,
            topk_return=1.0,
            topk_score_field="final_score",
            topk_k=2,
            aux_topk=(AuxTopKLoss(weight=0.5, k=1, score_field="final_score"),),
            gate_entropy=0.0,
        ),
        10,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert "topk_aux_final_score_k1_loss" in parts
    assert final_score.grad is not None
    assert torch.isfinite(final_score.grad).all()


def test_msgca_loss_uses_cluster_context_losses() -> None:
    y_score = torch.zeros(6)
    final_score = torch.tensor([3.0, 2.5, 2.0, 0.0, -0.5, -1.0], requires_grad=True)
    labels = torch.tensor([0.06, 0.04, 0.03, -0.02, -0.03, -0.04])
    dates = pd.to_datetime(["2020-01-02"] * 6)
    gates = torch.full((6, 3), 1 / 3)
    output = MSGCAOutput(
        y_score=y_score,
        return_pred=y_score,
        direction_logit=y_score,
        gates=gates,
        g_price=gates[:, 0],
        g_text=gates[:, 1],
        g_fundamental=gates[:, 2],
        final_score=final_score,
    )
    context = torch.tensor([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])

    loss, parts = msgca_loss(
        output,
        labels,
        labels,
        labels.gt(0).float(),
        dates,
        LossWeights(
            rank=0.0,
            return_mse=0.0,
            direction_bce=0.0,
            topk_return=0.0,
            gate_entropy=0.0,
            cluster_topk=1.0,
            cluster_rank=0.5,
            in_cluster_rank=0.5,
            cluster_topk_min_size=2,
            cluster_rank_min_size=2,
            in_cluster_rank_min_size=2,
            cluster_topk_score_field="final_score",
            cluster_rank_score_field="final_score",
            in_cluster_rank_score_field="final_score",
        ),
        10,
        context=context,
        context_columns=["context_cluster_id"],
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert "cluster_topk_loss" in parts
    assert "cluster_rank_loss" in parts
    assert "in_cluster_rank_loss" in parts
    assert final_score.grad is not None
    assert torch.isfinite(final_score.grad).all()
