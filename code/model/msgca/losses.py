from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model.msgca.modules import MSGCAOutput, gate_entropy


@dataclass(frozen=True)
class AuxRankLoss:
    weight: float
    score_field: str = "y_score"
    label_mode: str | None = None
    topk_k: int | None = None


@dataclass(frozen=True)
class AuxTopKLoss:
    weight: float
    k: int
    score_field: str = "y_score"
    secondary_weight: float | None = None
    mode: str | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class LossWeights:
    rank: float = 1.0
    return_mse: float = 0.1
    secondary_return: float = 0.0
    direction_bce: float = 0.05
    return_loss_type: str = "mse"
    return_huber_delta: float = 0.02
    rank_label_mode: str = "raw"
    topk_label_mode: str = "raw"
    rank_topk_k: int = 0
    direction_label_mode: str = "raw"
    direction_label_epsilon: float = 0.0
    direction_label_std_fraction: float = 0.0
    topk_return: float = 0.0
    return_secondary_weight: float = 0.0
    topk_secondary_weight: float = 0.0
    topk_k: int = 20
    topk_mode: str = "soft_equal_topk"
    topk_temperature: float = 0.1
    gate_entropy: float = 0.001
    rank_score_field: str = "y_score"
    topk_score_field: str = "y_score"
    aux_rank: tuple[AuxRankLoss, ...] = ()
    aux_topk: tuple[AuxTopKLoss, ...] = ()
    combined_topk: float = 0.0
    combined_topk_k: int = 20
    combined_topk_mode: str = "soft_equal_topk"
    combined_topk_temperature: float = 0.1
    combined_weight_final: float = 0.60
    combined_weight_return: float = 0.22
    combined_weight_direction: float = 0.08
    combined_weight_y: float = 0.10
    consistency: float = 0.0
    consistency_score_field: str = "final_score"
    consistency_topk_k: int = 40
    consistency_temperature: float = 0.1
    consistency_return_weight: float = 0.7
    consistency_direction_weight: float = 0.3
    context_score_topk: float = 0.0
    context_score_variant: str = "context_s5"
    context_score_topk_k: int = 20
    context_score_topk_mode: str = "soft_equal_topk"
    context_score_topk_temperature: float = 0.1
    trend_adjusted_topk: float = 0.0
    trend_adjusted_topk_k: int = 20
    trend_adjusted_topk_mode: str = "soft_equal_topk"
    trend_adjusted_topk_temperature: float = 0.1
    trend_adjusted_score_field: str = "final_score"
    trend_adjusted_positive_weight: float = 0.25
    trend_overheat_negative_weight: float = 0.35
    trend_broken_negative_weight: float = 0.45
    direction_healthy_pullback_discount: float = 0.0
    cluster_topk: float = 0.0
    cluster_column: str = "context_cluster_id"
    cluster_topk_score_field: str = "final_score"
    cluster_topk_label_mode: str = "excess"
    cluster_topk_k: int = 6
    cluster_topk_mode: str = "soft_equal_topk"
    cluster_topk_temperature: float = 0.1
    cluster_topk_min_size: int = 8
    cluster_topk_member_k: int = 5
    cluster_topk_member_temperature: float = 0.1
    cluster_rank: float = 0.0
    cluster_rank_score_field: str = "final_score"
    cluster_rank_label_mode: str = "excess"
    cluster_rank_min_size: int = 8
    cluster_rank_member_k: int = 5
    cluster_rank_member_temperature: float = 0.1
    cluster_rank_max_pairs_per_day: int = 512
    in_cluster_rank: float = 0.0
    in_cluster_rank_score_field: str = "final_score"
    in_cluster_rank_label_mode: str = "excess"
    in_cluster_rank_topk_k: int = 5
    in_cluster_rank_min_size: int = 8
    in_cluster_rank_max_clusters_per_day: int = 32
    in_cluster_rank_max_pairs_per_cluster: int = 512


def lambda_rankic_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_trade_dates: Sequence[object],
    max_pairs_per_day: int = 4096,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise LambdaRank-style surrogate grouped by target_trade_date."""
    device = scores.device
    total = torch.zeros((), dtype=scores.dtype, device=device)
    weight_total = torch.zeros((), dtype=scores.dtype, device=device)
    date_keys = _date_keys(target_trade_dates)
    sample_weights = _prepare_sample_weights(sample_weights, scores)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor([i for i, value in enumerate(date_keys) if value == key], device=device)
        if index.numel() < 2:
            continue
        day_scores = scores[index]
        day_labels = labels[index]
        day_weights = sample_weights[index]
        finite = torch.isfinite(day_scores) & torch.isfinite(day_labels)
        day_scores = day_scores[finite]
        day_labels = day_labels[finite]
        day_weights = day_weights[finite]
        if day_scores.numel() < 2 or torch.unique(day_labels).numel() < 2:
            continue
        left, right = _sample_pairs(day_labels, max_pairs_per_day)
        if left.numel() == 0:
            continue
        score_diff = day_scores[left] - day_scores[right]
        label_diff = day_labels[left] - day_labels[right]
        pair_sign = torch.sign(label_diff)
        valid = pair_sign.ne(0)
        if not bool(valid.any()):
            continue
        score_diff = score_diff[valid]
        pair_sign = pair_sign[valid]
        weights = _rankic_delta_weights(day_labels, left[valid], right[valid])
        day_weight = day_weights.mean().clamp_min(0.0)
        total = total + (F.softplus(-pair_sign * score_diff) * weights).mean() * day_weight
        weight_total = weight_total + day_weight
    if float(weight_total.detach().cpu()) <= 0:
        return scores.sum() * 0.0
    return total / weight_total


def lambda_topk_rank_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_trade_dates: Sequence[object],
    *,
    topk_k: int = 20,
    max_pairs_per_day: int = 4096,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """TopK-focused pairwise ranking surrogate grouped by target_trade_date."""
    device = scores.device
    total = torch.zeros((), dtype=scores.dtype, device=device)
    weight_total = torch.zeros((), dtype=scores.dtype, device=device)
    date_keys = _date_keys(target_trade_dates)
    sample_weights = _prepare_sample_weights(sample_weights, scores)
    k = max(int(topk_k or 0), 1)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor([i for i, value in enumerate(date_keys) if value == key], device=device)
        if index.numel() < 2:
            continue
        day_scores = scores[index]
        day_labels = labels[index]
        day_weights = sample_weights[index]
        finite = torch.isfinite(day_scores) & torch.isfinite(day_labels)
        day_scores = day_scores[finite]
        day_labels = day_labels[finite]
        day_weights = day_weights[finite]
        n = day_scores.numel()
        if n < 2 or torch.unique(day_labels).numel() < 2:
            continue
        effective_k = min(k, max(n - 1, 1))
        order = torch.argsort(day_labels, descending=True)
        positives = order[:effective_k]
        negatives = order[effective_k:]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        left = positives.repeat_interleave(negatives.numel())
        right = negatives.repeat(positives.numel())
        label_diff = day_labels[left] - day_labels[right]
        valid = label_diff.gt(0)
        if not bool(valid.any()):
            continue
        left = left[valid]
        right = right[valid]
        label_diff = label_diff[valid]
        if left.numel() > max_pairs_per_day:
            perm = torch.randperm(left.numel(), device=device)[:max_pairs_per_day]
            left = left[perm]
            right = right[perm]
            label_diff = label_diff[perm]
        score_diff = day_scores[left] - day_scores[right]
        scale = day_labels.max() - day_labels.min()
        weights = (label_diff.abs() / scale.clamp_min(1e-8)).clamp_min(1.0 / float(max(n - 1, 1)))
        day_weight = day_weights.mean().clamp_min(0.0)
        total = total + (F.softplus(-score_diff) * weights).mean() * day_weight
        weight_total = weight_total + day_weight
    if float(weight_total.detach().cpu()) <= 0:
        return scores.sum() * 0.0
    return total / weight_total


def msgca_loss(
    output: MSGCAOutput,
    labels: torch.Tensor,
    secondary_labels: torch.Tensor,
    direction_labels: torch.Tensor,
    target_trade_dates: Sequence[object],
    weights: LossWeights,
    max_pairs_per_day: int,
    sample_weights: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
    context_columns: Sequence[str] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sample_weights = _prepare_sample_weights(sample_weights, output.y_score)
    context_map = _context_map(context, context_columns)
    excess_labels = _daily_excess(labels, target_trade_dates)
    return_labels = _blend_labels(labels, secondary_labels, weights.return_secondary_weight)
    rank_labels = _labels_for_mode(labels, excess_labels, weights.rank_label_mode)
    topk_base_labels = _labels_for_mode(labels, excess_labels, weights.topk_label_mode)
    topk_labels = _blend_labels(topk_base_labels, secondary_labels, weights.topk_secondary_weight)
    rank_scores = _score_tensor(output, weights.rank_score_field)
    topk_scores = _score_tensor(output, weights.topk_score_field)
    if int(weights.rank_topk_k or 0) > 0:
        rank_loss = lambda_topk_rank_loss(
            rank_scores,
            rank_labels,
            target_trade_dates,
            topk_k=int(weights.rank_topk_k),
            max_pairs_per_day=max_pairs_per_day,
            sample_weights=sample_weights,
        )
    else:
        rank_loss = lambda_rankic_loss(
            rank_scores,
            rank_labels,
            target_trade_dates,
            max_pairs_per_day=max_pairs_per_day,
            sample_weights=sample_weights,
        )
    topk_loss = soft_topk_return_loss(
        topk_scores,
        topk_labels,
        target_trade_dates,
        topk_k=weights.topk_k,
        mode=weights.topk_mode,
        temperature=weights.topk_temperature,
        sample_weights=sample_weights,
    )
    aux_rank_losses: dict[str, torch.Tensor] = {}
    aux_rank_total = output.y_score.sum() * 0.0
    for aux in weights.aux_rank:
        aux_weight = float(aux.weight)
        if aux_weight == 0.0:
            continue
        aux_labels = _labels_for_mode(labels, excess_labels, weights.rank_label_mode if aux.label_mode is None else aux.label_mode)
        if int(aux.topk_k or 0) > 0:
            aux_rank_loss = lambda_topk_rank_loss(
                _score_tensor(output, aux.score_field),
                aux_labels,
                target_trade_dates,
                topk_k=int(aux.topk_k or 0),
                max_pairs_per_day=max_pairs_per_day,
                sample_weights=sample_weights,
            )
        else:
            aux_rank_loss = lambda_rankic_loss(
                _score_tensor(output, aux.score_field),
                aux_labels,
                target_trade_dates,
                max_pairs_per_day=max_pairs_per_day,
                sample_weights=sample_weights,
            )
        aux_rank_total = aux_rank_total + aux_weight * aux_rank_loss
        aux_rank_losses[_aux_rank_loss_name(aux)] = aux_rank_loss.detach()
    aux_losses: dict[str, torch.Tensor] = {}
    aux_total = output.y_score.sum() * 0.0
    for aux in weights.aux_topk:
        aux_weight = float(aux.weight)
        if aux_weight == 0.0:
            continue
        aux_labels = _blend_labels(
            labels,
            secondary_labels,
            weights.topk_secondary_weight if aux.secondary_weight is None else aux.secondary_weight,
        )
        aux_loss = soft_topk_return_loss(
            _score_tensor(output, aux.score_field),
            aux_labels,
            target_trade_dates,
            topk_k=aux.k,
            mode=weights.topk_mode if aux.mode is None else aux.mode,
            temperature=weights.topk_temperature if aux.temperature is None else aux.temperature,
            sample_weights=sample_weights,
        )
        aux_total = aux_total + aux_weight * aux_loss
        aux_losses[_aux_loss_name(aux)] = aux_loss.detach()
    valid_return = torch.isfinite(return_labels) & torch.isfinite(output.return_pred)
    if bool(valid_return.any()):
        if str(weights.return_loss_type).lower() == "huber":
            return_errors = F.huber_loss(
                output.return_pred[valid_return],
                return_labels[valid_return],
                reduction="none",
                delta=max(float(weights.return_huber_delta), 1e-8),
            )
        else:
            return_errors = F.mse_loss(output.return_pred[valid_return], return_labels[valid_return], reduction="none")
        return_loss = _weighted_mean(
            return_errors,
            sample_weights[valid_return],
        )
    else:
        return_loss = output.return_pred.sum() * 0.0
    valid_secondary_return = torch.isfinite(secondary_labels) & torch.isfinite(output.return_pred)
    if bool(valid_secondary_return.any()):
        if str(weights.return_loss_type).lower() == "huber":
            secondary_errors = F.huber_loss(
                output.return_pred[valid_secondary_return],
                secondary_labels[valid_secondary_return],
                reduction="none",
                delta=max(float(weights.return_huber_delta), 1e-8),
            )
        else:
            secondary_errors = F.mse_loss(
                output.return_pred[valid_secondary_return],
                secondary_labels[valid_secondary_return],
                reduction="none",
            )
        secondary_return_loss = _weighted_mean(secondary_errors, sample_weights[valid_secondary_return])
    else:
        secondary_return_loss = output.return_pred.sum() * 0.0
    direction_base_labels = _labels_for_mode(labels, excess_labels, weights.direction_label_mode)
    if str(weights.direction_label_mode).lower() == "excess":
        direction_targets, direction_valid = _direction_targets_from_excess(
            direction_base_labels,
            target_trade_dates,
            fixed_epsilon=float(weights.direction_label_epsilon),
            std_fraction=float(weights.direction_label_std_fraction),
        )
    else:
        direction_targets = direction_labels
        direction_valid = torch.isfinite(direction_labels) & torch.isfinite(direction_base_labels)
    direction_epsilon = max(float(weights.direction_label_epsilon), 0.0)
    valid_direction = direction_valid & torch.isfinite(output.direction_logit)
    if str(weights.direction_label_mode).lower() != "excess" and direction_epsilon > 0:
        valid_direction = valid_direction & direction_base_labels.abs().gt(direction_epsilon)
    if bool(valid_direction.any()):
        direction_weights = sample_weights[valid_direction]
        if float(weights.direction_healthy_pullback_discount) > 0.0 and "context_h" in context_map:
            h = context_map["context_h"].to(device=output.direction_logit.device, dtype=output.direction_logit.dtype).clamp(0.0, 1.0)
            direction_weights = direction_weights * (
                1.0 - min(max(float(weights.direction_healthy_pullback_discount), 0.0), 1.0) * h[valid_direction]
            )
        direction_loss = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                output.direction_logit[valid_direction],
                direction_targets[valid_direction],
                reduction="none",
            ),
            direction_weights,
        )
    else:
        direction_loss = output.direction_logit.sum() * 0.0
    combined_topk_loss = topk_scores.sum() * 0.0
    if float(weights.combined_topk) != 0.0:
        combined_topk_loss = soft_topk_return_loss(
            _combined_score_tensor(output, target_trade_dates, weights),
            topk_labels,
            target_trade_dates,
            topk_k=weights.combined_topk_k,
            mode=weights.combined_topk_mode,
            temperature=weights.combined_topk_temperature,
            sample_weights=sample_weights,
        )
    consistency_loss = topk_scores.sum() * 0.0
    if float(weights.consistency) != 0.0:
        consistency_loss = topk_consistency_loss(
            _score_tensor(output, weights.consistency_score_field),
            output.return_pred,
            torch.sigmoid(output.direction_logit),
            target_trade_dates,
            topk_k=weights.consistency_topk_k,
            temperature=weights.consistency_temperature,
            return_weight=weights.consistency_return_weight,
            direction_weight=weights.consistency_direction_weight,
            sample_weights=sample_weights,
        )
    context_score_topk_loss = topk_scores.sum() * 0.0
    if float(weights.context_score_topk) != 0.0:
        context_score_topk_loss = soft_topk_return_loss(
            _context_score_tensor(output, target_trade_dates, context_map, weights.context_score_variant),
            topk_labels,
            target_trade_dates,
            topk_k=weights.context_score_topk_k,
            mode=weights.context_score_topk_mode,
            temperature=weights.context_score_topk_temperature,
            sample_weights=sample_weights,
        )
    trend_adjusted_topk_loss = topk_scores.sum() * 0.0
    if float(weights.trend_adjusted_topk) != 0.0:
        adjusted_labels = _trend_adjusted_labels(excess_labels, context_map, weights)
        trend_adjusted_topk_loss = soft_topk_return_loss(
            _score_tensor(output, weights.trend_adjusted_score_field),
            adjusted_labels,
            target_trade_dates,
            topk_k=weights.trend_adjusted_topk_k,
            mode=weights.trend_adjusted_topk_mode,
            temperature=weights.trend_adjusted_topk_temperature,
            sample_weights=sample_weights,
        )
    cluster_topk_loss = topk_scores.sum() * 0.0
    in_cluster_rank_loss_value = topk_scores.sum() * 0.0
    cluster_rank_loss_value = topk_scores.sum() * 0.0
    if float(weights.cluster_topk) != 0.0 or float(weights.cluster_rank) != 0.0 or float(weights.in_cluster_rank) != 0.0:
        _require_context(context_map, (weights.cluster_column,))
        cluster_ids = context_map[weights.cluster_column].to(device=topk_scores.device, dtype=topk_scores.dtype)
        if float(weights.cluster_topk) != 0.0:
            cluster_topk_loss = cluster_topk_return_loss(
                _score_tensor(output, weights.cluster_topk_score_field),
                _labels_for_mode(labels, excess_labels, weights.cluster_topk_label_mode),
                target_trade_dates,
                cluster_ids,
                topk_k=weights.cluster_topk_k,
                mode=weights.cluster_topk_mode,
                temperature=weights.cluster_topk_temperature,
                min_size=weights.cluster_topk_min_size,
                member_topk_k=weights.cluster_topk_member_k,
                member_temperature=weights.cluster_topk_member_temperature,
                sample_weights=sample_weights,
            )
        if float(weights.cluster_rank) != 0.0:
            cluster_rank_loss_value = cluster_rank_loss(
                _score_tensor(output, weights.cluster_rank_score_field),
                _labels_for_mode(labels, excess_labels, weights.cluster_rank_label_mode),
                target_trade_dates,
                cluster_ids,
                min_size=weights.cluster_rank_min_size,
                member_topk_k=weights.cluster_rank_member_k,
                member_temperature=weights.cluster_rank_member_temperature,
                max_pairs_per_day=weights.cluster_rank_max_pairs_per_day,
                sample_weights=sample_weights,
            )
        if float(weights.in_cluster_rank) != 0.0:
            in_cluster_rank_loss_value = in_cluster_rank_loss(
                _score_tensor(output, weights.in_cluster_rank_score_field),
                _labels_for_mode(labels, excess_labels, weights.in_cluster_rank_label_mode),
                target_trade_dates,
                cluster_ids,
                topk_k=weights.in_cluster_rank_topk_k,
                max_pairs_per_cluster=weights.in_cluster_rank_max_pairs_per_cluster,
                min_size=weights.in_cluster_rank_min_size,
                max_clusters_per_day=weights.in_cluster_rank_max_clusters_per_day,
                sample_weights=sample_weights,
            )
    entropy = gate_entropy(output.gates)
    total = (
        weights.rank * rank_loss
        + weights.return_mse * return_loss
        + weights.secondary_return * secondary_return_loss
        + weights.direction_bce * direction_loss
        + weights.topk_return * topk_loss
        + aux_rank_total
        + aux_total
        + weights.combined_topk * combined_topk_loss
        + weights.consistency * consistency_loss
        + weights.context_score_topk * context_score_topk_loss
        + weights.trend_adjusted_topk * trend_adjusted_topk_loss
        + weights.cluster_topk * cluster_topk_loss
        + weights.cluster_rank * cluster_rank_loss_value
        + weights.in_cluster_rank * in_cluster_rank_loss_value
        - weights.gate_entropy * entropy
    )
    parts = {
        "loss": total.detach(),
        "rank_loss": rank_loss.detach(),
        "topk_loss": topk_loss.detach(),
        "return_loss": return_loss.detach(),
        "secondary_return_loss": secondary_return_loss.detach(),
        "direction_loss": direction_loss.detach(),
        "combined_topk_loss": combined_topk_loss.detach(),
        "consistency_loss": consistency_loss.detach(),
        "context_score_topk_loss": context_score_topk_loss.detach(),
        "trend_adjusted_topk_loss": trend_adjusted_topk_loss.detach(),
        "cluster_topk_loss": cluster_topk_loss.detach(),
        "cluster_rank_loss": cluster_rank_loss_value.detach(),
        "in_cluster_rank_loss": in_cluster_rank_loss_value.detach(),
        "gate_entropy": entropy.detach(),
    }
    parts.update(aux_rank_losses)
    parts.update(aux_losses)
    return total, parts


def soft_topk_return_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_trade_dates: Sequence[object],
    topk_k: int | None = None,
    mode: str = "soft_equal_topk",
    temperature: float = 0.1,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable long-only top-heavy objective grouped by target_trade_date.

    ``legacy_softmax`` concentrates weight with a full-universe softmax. The
    default ``soft_equal_topk`` is closer to the live TopN strategy: it builds a
    soft inclusion mask around the daily kth score and normalizes selected names
    toward equal weight.
    """
    device = scores.device
    total = torch.zeros((), dtype=scores.dtype, device=device)
    weight_total = torch.zeros((), dtype=scores.dtype, device=device)
    temp = max(float(temperature), 1e-6)
    mode = str(mode or "soft_equal_topk")
    date_keys = _date_keys(target_trade_dates)
    sample_weights = _prepare_sample_weights(sample_weights, scores)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor([i for i, value in enumerate(date_keys) if value == key], device=device)
        if index.numel() < 1:
            continue
        day_scores = scores[index]
        day_labels = labels[index]
        day_weights = sample_weights[index]
        finite = torch.isfinite(day_scores) & torch.isfinite(day_labels)
        day_scores = day_scores[finite]
        day_labels = day_labels[finite]
        day_weights = day_weights[finite]
        if day_scores.numel() < 1:
            continue
        portfolio_weights = _soft_portfolio_weights(day_scores, topk_k=topk_k, temperature=temp, mode=mode)
        day_weight = day_weights.mean().clamp_min(0.0)
        total = total - (portfolio_weights * day_labels).sum() * day_weight
        weight_total = weight_total + day_weight
    if float(weight_total.detach().cpu()) <= 0:
        return scores.sum() * 0.0
    return total / weight_total


def cluster_topk_return_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_trade_dates: Sequence[object],
    cluster_ids: torch.Tensor,
    *,
    topk_k: int = 6,
    mode: str = "soft_equal_topk",
    temperature: float = 0.1,
    min_size: int = 8,
    member_topk_k: int = 5,
    member_temperature: float = 0.1,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select strong same-day style clusters before selecting individual stocks."""
    device = scores.device
    total = torch.zeros((), dtype=scores.dtype, device=device)
    weight_total = torch.zeros((), dtype=scores.dtype, device=device)
    date_keys = _date_keys(target_trade_dates)
    sample_weights = _prepare_sample_weights(sample_weights, scores)
    min_size = max(int(min_size), 1)
    member_temp = max(float(member_temperature), 1e-6)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor([i for i, value in enumerate(date_keys) if value == key], device=device)
        if index.numel() < min_size:
            continue
        day_scores = scores[index]
        day_labels = labels[index]
        day_clusters = cluster_ids[index]
        day_weights = sample_weights[index]
        finite = (
            torch.isfinite(day_scores)
            & torch.isfinite(day_labels)
            & torch.isfinite(day_clusters)
            & day_clusters.ge(0)
        )
        day_scores = day_scores[finite]
        day_labels = day_labels[finite]
        day_clusters = day_clusters[finite].round().to(torch.long)
        day_weights = day_weights[finite]
        if day_scores.numel() < min_size:
            continue
        score_z = _zscore_tensor(day_scores)
        cluster_scores: list[torch.Tensor] = []
        cluster_labels: list[torch.Tensor] = []
        for cluster in torch.unique(day_clusters, sorted=True):
            members = day_clusters.eq(cluster)
            if int(members.sum().detach().cpu()) < min_size:
                continue
            member_scores = score_z[members]
            member_labels = day_labels[members]
            member_weights = _soft_portfolio_weights(
                member_scores,
                topk_k=min(int(member_topk_k or 0), int(member_scores.numel())),
                temperature=member_temp,
                mode="soft_equal_topk",
            )
            cluster_scores.append((member_weights * member_scores).sum())
            cluster_labels.append(member_labels.mean())
        if not cluster_scores:
            continue
        grouped_scores = torch.stack(cluster_scores)
        grouped_labels = torch.stack(cluster_labels)
        if grouped_scores.numel() < 1:
            continue
        portfolio_weights = _soft_portfolio_weights(
            grouped_scores,
            topk_k=min(int(topk_k or 0), int(grouped_scores.numel())),
            temperature=max(float(temperature), 1e-6),
            mode=str(mode or "soft_equal_topk"),
        )
        day_weight = day_weights.mean().clamp_min(0.0)
        total = total - (portfolio_weights * grouped_labels).sum() * day_weight
        weight_total = weight_total + day_weight
    if float(weight_total.detach().cpu()) <= 0:
        return scores.sum() * 0.0
    return total / weight_total


def cluster_rank_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_trade_dates: Sequence[object],
    cluster_ids: torch.Tensor,
    *,
    min_size: int = 8,
    member_topk_k: int = 5,
    member_temperature: float = 0.1,
    max_pairs_per_day: int = 512,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank dynamic clusters using the same daily pairwise surrogate as stocks."""
    device = scores.device
    total = torch.zeros((), dtype=scores.dtype, device=device)
    weight_total = torch.zeros((), dtype=scores.dtype, device=device)
    date_keys = _date_keys(target_trade_dates)
    sample_weights = _prepare_sample_weights(sample_weights, scores)
    min_size = max(int(min_size), 1)
    member_temp = max(float(member_temperature), 1e-6)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor([i for i, value in enumerate(date_keys) if value == key], device=device)
        if index.numel() < min_size:
            continue
        day_scores = scores[index]
        day_labels = labels[index]
        day_clusters = cluster_ids[index]
        day_weights = sample_weights[index]
        finite = (
            torch.isfinite(day_scores)
            & torch.isfinite(day_labels)
            & torch.isfinite(day_clusters)
            & day_clusters.ge(0)
        )
        day_scores = day_scores[finite]
        day_labels = day_labels[finite]
        day_clusters = day_clusters[finite].round().to(torch.long)
        day_weights = day_weights[finite]
        if day_scores.numel() < min_size:
            continue
        score_z = _zscore_tensor(day_scores)
        cluster_scores: list[torch.Tensor] = []
        cluster_labels: list[torch.Tensor] = []
        cluster_weights: list[torch.Tensor] = []
        for cluster in torch.unique(day_clusters, sorted=True):
            members = day_clusters.eq(cluster)
            if int(members.sum().detach().cpu()) < min_size:
                continue
            member_scores = score_z[members]
            member_labels = day_labels[members]
            member_weights = day_weights[members]
            soft_weights = _soft_portfolio_weights(
                member_scores,
                topk_k=min(int(member_topk_k or 0), int(member_scores.numel())),
                temperature=member_temp,
                mode="soft_equal_topk",
            )
            cluster_scores.append((soft_weights * member_scores).sum())
            cluster_labels.append(member_labels.mean())
            cluster_weights.append(member_weights.mean().clamp_min(0.0))
        if len(cluster_scores) < 2:
            continue
        grouped_scores = torch.stack(cluster_scores)
        grouped_labels = torch.stack(cluster_labels)
        grouped_weights = torch.stack(cluster_weights)
        if torch.unique(grouped_labels).numel() < 2:
            continue
        cluster_dates = [key] * int(grouped_scores.numel())
        loss = lambda_rankic_loss(
            grouped_scores,
            grouped_labels,
            cluster_dates,
            max_pairs_per_day=max(int(max_pairs_per_day), 1),
            sample_weights=grouped_weights,
        )
        day_weight = grouped_weights.mean().clamp_min(0.0)
        total = total + loss * day_weight
        weight_total = weight_total + day_weight
    if float(weight_total.detach().cpu()) <= 0:
        return scores.sum() * 0.0
    return total / weight_total


def in_cluster_rank_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_trade_dates: Sequence[object],
    cluster_ids: torch.Tensor,
    *,
    topk_k: int = 5,
    max_pairs_per_cluster: int = 512,
    min_size: int = 8,
    max_clusters_per_day: int = 32,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-focused ranking inside the dynamic clusters selected by context."""
    device = scores.device
    total = torch.zeros((), dtype=scores.dtype, device=device)
    weight_total = torch.zeros((), dtype=scores.dtype, device=device)
    date_keys = _date_keys(target_trade_dates)
    sample_weights = _prepare_sample_weights(sample_weights, scores)
    min_size = max(int(min_size), 2)
    max_clusters = max(int(max_clusters_per_day or 0), 0)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor([i for i, value in enumerate(date_keys) if value == key], device=device)
        if index.numel() < min_size:
            continue
        day_scores = scores[index]
        day_labels = labels[index]
        day_clusters = cluster_ids[index]
        day_weights = sample_weights[index]
        finite = (
            torch.isfinite(day_scores)
            & torch.isfinite(day_labels)
            & torch.isfinite(day_clusters)
            & day_clusters.ge(0)
        )
        day_scores = day_scores[finite]
        day_labels = day_labels[finite]
        day_clusters = day_clusters[finite].round().to(torch.long)
        day_weights = day_weights[finite]
        if day_scores.numel() < min_size:
            continue
        clusters, counts = torch.unique(day_clusters, sorted=True, return_counts=True)
        eligible = clusters[counts.ge(min_size)]
        eligible_counts = counts[counts.ge(min_size)]
        if eligible.numel() == 0:
            continue
        if max_clusters > 0 and eligible.numel() > max_clusters:
            order = torch.argsort(eligible_counts, descending=True)[:max_clusters]
            eligible = eligible[order]
        for cluster in eligible:
            members = day_clusters.eq(cluster)
            cluster_scores = day_scores[members]
            cluster_labels = day_labels[members]
            cluster_weights = day_weights[members]
            if cluster_scores.numel() < min_size or torch.unique(cluster_labels).numel() < 2:
                continue
            cluster_dates = [key] * int(cluster_scores.numel())
            loss = lambda_topk_rank_loss(
                cluster_scores,
                cluster_labels,
                cluster_dates,
                topk_k=max(int(topk_k or 0), 1),
                max_pairs_per_day=max(int(max_pairs_per_cluster), 1),
                sample_weights=cluster_weights,
            )
            cluster_weight = cluster_weights.mean().clamp_min(0.0)
            total = total + loss * cluster_weight
            weight_total = weight_total + cluster_weight
    if float(weight_total.detach().cpu()) <= 0:
        return scores.sum() * 0.0
    return total / weight_total


def topk_consistency_loss(
    scores: torch.Tensor,
    return_pred: torch.Tensor,
    direction_prob: torch.Tensor,
    target_trade_dates: Sequence[object],
    *,
    topk_k: int = 40,
    temperature: float = 0.1,
    return_weight: float = 0.7,
    direction_weight: float = 0.3,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align auxiliary heads only inside the deployable candidate pool."""
    device = scores.device
    total = torch.zeros((), dtype=scores.dtype, device=device)
    weight_total = torch.zeros((), dtype=scores.dtype, device=device)
    temp = max(float(temperature), 1e-6)
    date_keys = _date_keys(target_trade_dates)
    sample_weights = _prepare_sample_weights(sample_weights, scores)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor([i for i, value in enumerate(date_keys) if value == key], device=device)
        if index.numel() < 2:
            continue
        day_scores = scores[index]
        day_returns = return_pred[index]
        day_direction = direction_prob[index]
        day_weights = sample_weights[index]
        finite = torch.isfinite(day_scores) & torch.isfinite(day_returns) & torch.isfinite(day_direction)
        day_scores = day_scores[finite]
        day_returns = day_returns[finite]
        day_direction = day_direction[finite]
        day_weights = day_weights[finite]
        if day_scores.numel() < 2:
            continue
        pool_weights = _soft_portfolio_weights(day_scores, topk_k=topk_k, temperature=temp, mode="soft_equal_topk")
        score_z = _zscore_tensor(day_scores)
        return_z = _zscore_tensor(day_returns)
        direction_z = _zscore_tensor(torch.logit(day_direction.clamp(1e-6, 1.0 - 1e-6)))
        errors = (
            float(return_weight) * F.smooth_l1_loss(score_z, return_z.detach(), reduction="none")
            + float(direction_weight) * F.smooth_l1_loss(score_z, direction_z.detach(), reduction="none")
        )
        day_weight = day_weights.mean().clamp_min(0.0)
        total = total + (pool_weights.detach() * errors).sum() * day_weight
        weight_total = weight_total + day_weight
    if float(weight_total.detach().cpu()) <= 0:
        return scores.sum() * 0.0
    return total / weight_total


def strategy_window_return_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_trade_dates: Sequence[object],
    *,
    topk_k: int = 20,
    mode: str = "soft_equal_topk",
    temperature: float = 0.1,
    return_weight: float = 0.0,
    excess_weight: float = 1.0,
    downside_weight: float = 0.1,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Strategy-shaped objective for randomly sampled multi-day windows.

    The first version treats each day independently and optimizes the average
    soft TopN excess return over the sampled dates. It deliberately ignores
    path-dependent turnover/T+1 details so it can be used as a stable training
    signal without forcing a full differentiable backtest into the loop.
    """
    device = scores.device
    total_objective = torch.zeros((), dtype=scores.dtype, device=device)
    weight_total = torch.zeros((), dtype=scores.dtype, device=device)
    portfolio_total = torch.zeros((), dtype=scores.dtype, device=device)
    equal_total = torch.zeros((), dtype=scores.dtype, device=device)
    excess_total = torch.zeros((), dtype=scores.dtype, device=device)
    downside_total = torch.zeros((), dtype=scores.dtype, device=device)
    temp = max(float(temperature), 1e-6)
    mode = str(mode or "soft_equal_topk")
    date_keys = _date_keys(target_trade_dates)
    sample_weights = _prepare_sample_weights(sample_weights, scores)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor([i for i, value in enumerate(date_keys) if value == key], device=device)
        if index.numel() < 1:
            continue
        day_scores = scores[index]
        day_labels = labels[index]
        day_weights = sample_weights[index]
        finite = torch.isfinite(day_scores) & torch.isfinite(day_labels)
        day_scores = day_scores[finite]
        day_labels = day_labels[finite]
        day_weights = day_weights[finite]
        if day_scores.numel() < 1:
            continue
        portfolio_weights = _soft_portfolio_weights(day_scores, topk_k=topk_k, temperature=temp, mode=mode)
        portfolio_return = (portfolio_weights * day_labels).sum()
        equal_return = day_labels.detach().mean()
        excess_return = portfolio_return - equal_return
        downside = torch.relu(-excess_return)
        day_objective = (
            float(return_weight) * portfolio_return
            + float(excess_weight) * excess_return
            - float(downside_weight) * downside
        )
        day_weight = day_weights.mean().clamp_min(0.0)
        total_objective = total_objective + day_objective * day_weight
        portfolio_total = portfolio_total + portfolio_return.detach() * day_weight
        equal_total = equal_total + equal_return.detach() * day_weight
        excess_total = excess_total + excess_return.detach() * day_weight
        downside_total = downside_total + downside.detach() * day_weight
        weight_total = weight_total + day_weight
    if float(weight_total.detach().cpu()) <= 0:
        zero = scores.sum() * 0.0
        return zero, {
            "strategy_window_portfolio_return": zero.detach(),
            "strategy_window_equal_return": zero.detach(),
            "strategy_window_excess_return": zero.detach(),
            "strategy_window_downside": zero.detach(),
        }
    loss = -(total_objective / weight_total)
    parts = {
        "strategy_window_portfolio_return": portfolio_total / weight_total,
        "strategy_window_equal_return": equal_total / weight_total,
        "strategy_window_excess_return": excess_total / weight_total,
        "strategy_window_downside": downside_total / weight_total,
    }
    return loss, {key: value.detach() for key, value in parts.items()}


def strategy_window_return_loss_from_output(
    output: MSGCAOutput,
    labels: torch.Tensor,
    target_trade_dates: Sequence[object],
    *,
    score_field: str = "final_score",
    topk_k: int = 20,
    mode: str = "soft_equal_topk",
    temperature: float = 0.1,
    return_weight: float = 0.0,
    excess_weight: float = 1.0,
    downside_weight: float = 0.1,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return strategy_window_return_loss(
        _score_tensor(output, score_field),
        labels,
        target_trade_dates,
        topk_k=topk_k,
        mode=mode,
        temperature=temperature,
        return_weight=return_weight,
        excess_weight=excess_weight,
        downside_weight=downside_weight,
        sample_weights=sample_weights,
    )


def _soft_portfolio_weights(
    scores: torch.Tensor,
    *,
    topk_k: int | None,
    temperature: float,
    mode: str,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    if mode == "legacy_softmax":
        return torch.softmax(scores / temperature, dim=0)
    if mode != "soft_equal_topk":
        raise ValueError(f"Unsupported topk_return_mode: {mode}")
    k = int(topk_k or 0)
    if k <= 0 or k >= scores.numel():
        return torch.full_like(scores, 1.0 / float(scores.numel()))
    sorted_scores = torch.sort(scores.detach(), descending=True).values
    boundary = (sorted_scores[k - 1] + sorted_scores[k]) / 2.0
    inclusion = torch.sigmoid((scores - boundary) / temperature)
    denom = inclusion.sum().clamp_min(1e-12)
    return inclusion / denom


def _blend_labels(primary: torch.Tensor, secondary: torch.Tensor, secondary_weight: float) -> torch.Tensor:
    weight = min(max(float(secondary_weight), 0.0), 1.0)
    primary_finite = torch.isfinite(primary)
    secondary_finite = torch.isfinite(secondary)
    if weight <= 0:
        return primary
    if weight >= 1:
        return torch.where(secondary_finite, secondary, primary)
    blended = (1.0 - weight) * primary + weight * secondary
    blended = torch.where(primary_finite & secondary_finite, blended, primary)
    blended = torch.where(~primary_finite & secondary_finite, secondary, blended)
    return blended


def _score_tensor(output: MSGCAOutput, field: str) -> torch.Tensor:
    name = str(field or "y_score")
    if name == "y_score":
        return output.y_score
    if name == "return_pred":
        return output.return_pred
    if name == "direction_logit":
        return output.direction_logit
    if name == "direction_prob":
        return torch.sigmoid(output.direction_logit)
    if name == "final_score":
        return output.final_score if output.final_score is not None else output.y_score
    raise ValueError(f"Unsupported MSGCA score field: {field}")


def _combined_score_tensor(output: MSGCAOutput, target_trade_dates: Sequence[object], weights: LossWeights) -> torch.Tensor:
    final_score = _score_tensor(output, "final_score")
    return_pred = output.return_pred
    direction_logit = output.direction_logit
    y_score = output.y_score
    combined = torch.zeros_like(final_score)
    date_keys = _date_keys(target_trade_dates)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor(
            [i for i, value in enumerate(date_keys) if value == key],
            device=final_score.device,
        )
        if index.numel() == 0:
            continue
        combined[index] = (
            float(weights.combined_weight_final) * _zscore_tensor(final_score[index])
            + float(weights.combined_weight_return) * _zscore_tensor(return_pred[index])
            + float(weights.combined_weight_direction) * _zscore_tensor(direction_logit[index])
            + float(weights.combined_weight_y) * _zscore_tensor(y_score[index])
        )
    return combined


def _context_score_tensor(
    output: MSGCAOutput,
    target_trade_dates: Sequence[object],
    context: dict[str, torch.Tensor],
    variant: str,
) -> torch.Tensor:
    variant = str(variant or "context_s5").lower()
    if variant.startswith("exact_"):
        return _exact_context_score_tensor(output, target_trade_dates, context, variant)
    required_context = ["context_tr", "context_mf", "context_news", "context_oh", "context_br", "context_hp", "context_h"]
    if variant in {"context_theme_s2", "context_theme_s5", "theme_s2", "theme_s5"}:
        required_context.extend(["context_theme_strength", "context_theme_hp"])
    _require_context(context, tuple(required_context))
    final_score = _score_tensor(output, "final_score")
    y_score = output.y_score
    return_pred = output.return_pred
    direction_logit = output.direction_logit
    score = torch.zeros_like(final_score)
    date_keys = _date_keys(target_trade_dates)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor(
            [i for i, value in enumerate(date_keys) if value == key],
            device=final_score.device,
        )
        if index.numel() == 0:
            continue
        f = _zscore_tensor(final_score[index])
        y = _zscore_tensor(y_score[index])
        r = _zscore_tensor(return_pred[index])
        d = _zscore_tensor(direction_logit[index])
        h = _context_tensor(context, "context_h", final_score)[index].clamp(0.0, 1.0)
        d_eff = torch.clamp(d, min=0.0) + (1.0 - h) * torch.clamp(d, max=0.0)
        tr = _context_tensor(context, "context_tr", final_score)[index]
        mf = _context_tensor(context, "context_mf", final_score)[index]
        news = _context_tensor(context, "context_news", final_score)[index]
        oh = _context_tensor(context, "context_oh", final_score)[index]
        br = _context_tensor(context, "context_br", final_score)[index]
        hp = _context_tensor(context, "context_hp", final_score)[index]
        cap = _context_tensor(context, "context_cap", final_score)[index] if "context_cap" in context else torch.zeros_like(f)
        theme = _context_tensor(context, "context_theme_strength", final_score)[index] if "context_theme_strength" in context else torch.zeros_like(f)
        theme_hp = _context_tensor(context, "context_theme_hp", final_score)[index] if "context_theme_hp" in context else torch.zeros_like(f)
        if variant in {"context_a4", "a4"}:
            score[index] = 0.50 * f + 0.06 * y + 0.16 * r + 0.06 * d_eff + 0.07 * news + 0.07 * tr - 0.05 * oh - 0.08 * br + 0.03 * cap
        elif variant in {"context_a4_no_news", "a4_no_news"}:
            score[index] = 0.53 * f + 0.06 * y + 0.17 * r + 0.06 * d_eff + 0.08 * tr - 0.06 * oh - 0.09 * br + 0.03 * cap
        elif variant in {"context_s5", "context_s2", "context_s5_soft", "s5", "s2"}:
            score[index] = (
                0.42 * f
                + 0.08 * y
                + 0.18 * r
                + 0.04 * d_eff
                + 0.12 * tr
                + 0.15 * hp
                + 0.10 * mf
                + 0.06 * news
                - 0.10 * oh
                - 0.28 * br
                + 0.03 * cap
            )
        elif variant in {"context_s2_no_news", "s2_no_news"}:
            score[index] = (
                0.45 * f
                + 0.08 * y
                + 0.19 * r
                + 0.04 * d_eff
                + 0.13 * tr
                + 0.16 * hp
                + 0.11 * mf
                - 0.11 * oh
                - 0.30 * br
                + 0.03 * cap
            )
        elif variant in {"context_theme_s2", "context_theme_s5", "theme_s2", "theme_s5"}:
            score[index] = (
                0.36 * f
                + 0.07 * y
                + 0.17 * r
                + 0.04 * d_eff
                + 0.09 * tr
                + 0.16 * theme
                + 0.10 * hp
                + 0.07 * theme_hp
                + 0.09 * mf
                + 0.05 * news
                - 0.10 * oh
                - 0.25 * br
                + 0.03 * cap
            )
        else:
            raise ValueError(f"Unsupported strict context score variant: {variant}")
    return score


def _exact_context_score_tensor(
    output: MSGCAOutput,
    target_trade_dates: Sequence[object],
    context: dict[str, torch.Tensor],
    variant: str,
) -> torch.Tensor:
    _require_context(
        context,
        (
            "context_roc3",
            "context_roc5",
            "context_roc20",
            "context_roc60",
            "context_rsv20",
            "context_volume_ratio",
            "context_news_exact",
            "context_log_total_mv",
            "context_broken_ma",
        ),
    )
    final_score = _score_tensor(output, "final_score")
    y_score = output.y_score
    return_pred = output.return_pred
    direction_logit = output.direction_logit
    score = torch.zeros_like(final_score)
    date_keys = _date_keys(target_trade_dates)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor(
            [i for i, value in enumerate(date_keys) if value == key],
            device=final_score.device,
        )
        if index.numel() == 0:
            continue
        f = _zscore_tensor(final_score[index])
        y = _zscore_tensor(y_score[index])
        r = _zscore_tensor(return_pred[index])
        d = _zscore_tensor(direction_logit[index])
        roc3 = _zscore_tensor(_context_tensor(context, "context_roc3", final_score)[index])
        roc5 = _zscore_tensor(_context_tensor(context, "context_roc5", final_score)[index])
        roc20 = _zscore_tensor(_context_tensor(context, "context_roc20", final_score)[index])
        roc60 = _zscore_tensor(_context_tensor(context, "context_roc60", final_score)[index])
        rsv20 = _zscore_tensor(_context_tensor(context, "context_rsv20", final_score)[index])
        volume = _zscore_tensor(_context_tensor(context, "context_volume_ratio", final_score)[index])
        news = _zscore_tensor(_context_tensor(context, "context_news_exact", final_score)[index])
        cap = _zscore_tensor(_context_tensor(context, "context_log_total_mv", final_score)[index])
        broken = _context_tensor(context, "context_broken_ma", final_score)[index].clamp(0.0, 1.0)
        tr = 0.55 * roc20 + 0.35 * roc60 - 0.10 * rsv20
        oh = torch.sigmoid(0.80 * roc3 + 0.80 * roc5 + 0.50 * rsv20 + 0.30 * volume)
        br = torch.sigmoid(broken - 0.60 * roc20 - 0.40 * roc60)
        if variant in {"context_a4", "a4_exact"}:
            score[index] = 0.50 * f + 0.06 * y + 0.16 * r + 0.06 * d + 0.07 * news + 0.07 * tr - 0.05 * oh - 0.08 * br + 0.03 * cap
        elif variant in {"context_a4_no_news", "a4_exact_no_news"}:
            score[index] = 0.53 * f + 0.06 * y + 0.17 * r + 0.06 * d + 0.08 * tr - 0.06 * oh - 0.09 * br + 0.03 * cap
        elif variant in {"exact_s5_soft", "s5_exact_soft"}:
            score[index] = 0.42 * f + 0.08 * y + 0.18 * r + 0.04 * d + 0.12 * tr + 0.06 * news - 0.10 * oh - 0.28 * br + 0.03 * cap
        else:
            raise ValueError(f"Unsupported exact context score variant: {variant}")
    return score


def _trend_adjusted_labels(labels: torch.Tensor, context: dict[str, torch.Tensor], weights: LossWeights) -> torch.Tensor:
    _require_context(context, ("context_h", "context_oh", "context_br"))
    h = _context_tensor(context, "context_h", labels).clamp(0.0, 1.0)
    oh = _context_tensor(context, "context_oh", labels).clamp(0.0, 1.0)
    br = _context_tensor(context, "context_br", labels).clamp(0.0, 1.0)
    positive = torch.relu(labels)
    negative = torch.relu(-labels)
    return (
        labels
        + float(weights.trend_adjusted_positive_weight) * h * positive
        - float(weights.trend_overheat_negative_weight) * oh * negative
        - float(weights.trend_broken_negative_weight) * br * negative
    )


def _context_map(context: torch.Tensor | None, context_columns: Sequence[str] | None) -> dict[str, torch.Tensor]:
    if context is None or not context_columns:
        return {}
    columns = [str(column) for column in context_columns]
    if context.ndim != 2 or context.shape[1] != len(columns):
        raise ValueError(f"loss_context shape {tuple(context.shape)} does not match columns={len(columns)}")
    return {column: context[:, index] for index, column in enumerate(columns)}


def _context_tensor(context: dict[str, torch.Tensor], column: str, like: torch.Tensor) -> torch.Tensor:
    if column not in context:
        raise KeyError(f"Missing loss context column: {column}")
    values = context[column].to(device=like.device, dtype=like.dtype)
    return torch.where(torch.isfinite(values), values, torch.zeros_like(values))


def _require_context(context: dict[str, torch.Tensor], columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in context]
    if missing:
        raise KeyError(f"Missing loss context columns: {missing}")


def _aux_loss_name(aux: AuxTopKLoss) -> str:
    field = str(aux.score_field).replace("-", "_")
    return f"topk_aux_{field}_k{int(aux.k)}_loss"


def _aux_rank_loss_name(aux: AuxRankLoss) -> str:
    field = str(aux.score_field).replace("-", "_")
    suffix = f"_top{int(aux.topk_k)}" if int(aux.topk_k or 0) > 0 else ""
    return f"rank_aux_{field}{suffix}_loss"


def _labels_for_mode(labels: torch.Tensor, excess_labels: torch.Tensor, mode: str | None) -> torch.Tensor:
    normalized = str(mode or "raw").lower()
    if normalized in {"raw", "return", "open"}:
        return labels
    if normalized in {"excess", "market_excess", "market_excess_open"}:
        return excess_labels
    raise ValueError(f"Unsupported label mode: {mode}")


def _daily_excess(labels: torch.Tensor, target_trade_dates: Sequence[object]) -> torch.Tensor:
    date_keys = _date_keys(target_trade_dates)
    out = torch.full_like(labels, float("nan"))
    for key in sorted(set(date_keys)):
        index = torch.as_tensor([i for i, value in enumerate(date_keys) if value == key], device=labels.device)
        day = labels[index]
        finite = torch.isfinite(day)
        if not bool(finite.any()):
            continue
        mean = day[finite].mean()
        out[index] = day - mean
    return out


def _direction_targets_from_excess(
    excess_labels: torch.Tensor,
    target_trade_dates: Sequence[object],
    *,
    fixed_epsilon: float,
    std_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    date_keys = _date_keys(target_trade_dates)
    targets = torch.full_like(excess_labels, float("nan"))
    valid = torch.zeros_like(excess_labels, dtype=torch.bool)
    fixed = max(float(fixed_epsilon), 0.0)
    fraction = max(float(std_fraction), 0.0)
    for key in sorted(set(date_keys)):
        index = torch.as_tensor([i for i, value in enumerate(date_keys) if value == key], device=excess_labels.device)
        day = excess_labels[index]
        finite = torch.isfinite(day)
        if not bool(finite.any()):
            continue
        epsilon = fixed
        if fraction > 0:
            epsilon = max(epsilon, float(fraction) * float(day[finite].std(unbiased=False).detach().cpu()))
        targets[index] = day.gt(epsilon).to(day.dtype)
        valid[index] = finite & day.abs().gt(epsilon)
    return targets, valid


def _sample_pairs(labels: torch.Tensor, max_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = labels.numel()
    order = torch.argsort(labels)
    if n <= 1:
        empty = torch.empty(0, dtype=torch.long, device=labels.device)
        return empty, empty
    if n * (n - 1) // 2 <= max_pairs:
        left, right = torch.triu_indices(n, n, offset=1, device=labels.device)
        return left, right

    q = max(1, n // 5)
    bottom = order[:q]
    middle = order[max(0, n // 2 - q // 2) : min(n, n // 2 + q // 2)]
    top = order[-q:]
    candidates: list[tuple[torch.Tensor, torch.Tensor]] = []
    for left_group, right_group in ((top, bottom), (top, middle), (middle, bottom)):
        if left_group.numel() and right_group.numel():
            grid_l = left_group.repeat_interleave(right_group.numel())
            grid_r = right_group.repeat(left_group.numel())
            candidates.append((grid_l, grid_r))
    if not candidates:
        left, right = torch.triu_indices(n, n, offset=1, device=labels.device)
        return left[:max_pairs], right[:max_pairs]
    left = torch.cat([item[0] for item in candidates])
    right = torch.cat([item[1] for item in candidates])
    if left.numel() > max_pairs:
        perm = torch.randperm(left.numel(), device=labels.device)[:max_pairs]
        left = left[perm]
        right = right[perm]
    return left, right


def _rankic_delta_weights(labels: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    n = labels.numel()
    ranks = torch.argsort(torch.argsort(labels)).to(dtype=labels.dtype)
    denom = max(float(n - 1), 1.0)
    weights = (ranks[left] - ranks[right]).abs() / denom
    return weights.clamp_min(1.0 / denom)


def _date_keys(values: Sequence[object]) -> list[str]:
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in values]


def _prepare_sample_weights(sample_weights: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if sample_weights is None:
        return torch.ones_like(like)
    weights = sample_weights.to(device=like.device, dtype=like.dtype)
    if weights.shape != like.shape:
        raise ValueError(f"sample_weights shape {tuple(weights.shape)} does not match scores shape {tuple(like.shape)}")
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    return weights.clamp_min(0.0)


def _zscore_tensor(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return torch.zeros_like(values)
    clean = torch.where(finite, values, torch.zeros_like(values))
    selected = clean[finite]
    mean = selected.mean()
    std = selected.std(unbiased=False).clamp_min(1e-6)
    return torch.where(finite, (values - mean) / std, torch.zeros_like(values))


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    denom = weights.sum().clamp_min(1e-12)
    return (values * weights).sum() / denom


def direction_labels_from_returns(labels: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(labels, float("nan"))
    finite = torch.isfinite(labels)
    out[finite] = labels[finite].gt(0).to(labels.dtype)
    return out


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
