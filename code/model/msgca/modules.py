from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MSGCAOutput:
    y_score: torch.Tensor
    return_pred: torch.Tensor
    direction_logit: torch.Tensor
    gates: torch.Tensor
    g_price: torch.Tensor
    g_text: torch.Tensor
    g_fundamental: torch.Tensor
    final_score: torch.Tensor | None = None
    factor_group_weights: torch.Tensor | None = None


class MaskedRevIN(nn.Module):
    """Mask-aware RevIN over the time dimension."""

    def __init__(self, num_variables: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_variables, 1))
            self.bias = nn.Parameter(torch.zeros(num_variables, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=x.dtype)
        count = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (x * mask_f).sum(dim=-1, keepdim=True) / count
        centered = torch.where(mask, x - mean, torch.zeros_like(x))
        var = (centered.square() * mask_f).sum(dim=-1, keepdim=True) / count
        out = centered / torch.sqrt(var + self.eps)
        out = torch.where(mask, out, torch.zeros_like(out))
        if self.affine:
            out = out * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
            out = torch.where(mask, out, torch.zeros_like(out))
        return out


class PriceEncoder(nn.Module):
    """iTransformer-style variable-token encoder."""

    def __init__(
        self,
        num_variables: int,
        lookback: int,
        hidden_dim: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.revin = MaskedRevIN(num_variables)
        self.input_projection = nn.Linear(lookback, hidden_dim)
        self.variable_embedding = nn.Parameter(torch.zeros(num_variables, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid_tokens = mask.any(dim=-1)
        x_norm = self.revin(x, mask)
        tokens = self.input_projection(x_norm) + self.variable_embedding.unsqueeze(0)
        safe_key_padding = _safe_key_padding_mask(valid_tokens)
        tokens = self.encoder(tokens, src_key_padding_mask=safe_key_padding)
        tokens = self.norm(tokens)
        tokens = torch.where(valid_tokens.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return tokens, valid_tokens


class NumericTokenEncoder(nn.Module):
    """Project scalar feature columns into feature tokens."""

    def __init__(self, max_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.max_features = max_features
        self.value_projection = nn.Linear(1, hidden_dim)
        self.feature_embedding = nn.Parameter(torch.zeros(max_features, hidden_dim))
        self.missing_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, feature_count = x.shape
        if feature_count == 0:
            tokens = self.missing_token.expand(batch_size, 1, -1)
            return tokens, torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        if feature_count > self.max_features:
            raise ValueError(f"feature_count={feature_count} exceeds max_features={self.max_features}")
        token_mask = _normalize_feature_mask(modality_mask, feature_count, x.device)
        tokens = self.value_projection(x.unsqueeze(-1))
        tokens = tokens + self.feature_embedding[:feature_count].unsqueeze(0)
        tokens = self.dropout(self.norm(tokens))
        tokens = torch.where(token_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return tokens, token_mask


class FactorAwareEncoder(nn.Module):
    """Feature-token encoder with dynamic factor gates and group-token pooling."""

    def __init__(
        self,
        max_features: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        feature_count: int,
        group_ids: Sequence[int] | None = None,
        group_names: Sequence[str] | None = None,
        factor_layers: int = 2,
        group_layers: int = 1,
        group_prototypes: int = 1,
        use_factor_gate: bool = True,
        factor_gate_activation: str = "softmax",
    ) -> None:
        super().__init__()
        self.max_features = max(max_features, feature_count, 1)
        self.feature_count = int(feature_count)
        self.use_factor_gate = use_factor_gate
        self.factor_gate_activation = str(factor_gate_activation)
        if self.factor_gate_activation not in {"softmax", "sparsemax"}:
            raise ValueError("factor_gate_activation must be 'softmax' or 'sparsemax'")
        self.group_prototypes = max(int(group_prototypes), 1)
        ids = _prepare_group_ids(group_ids, self.feature_count)
        names = list(group_names or [])
        if ids:
            group_count = max(ids) + 1
            if len(names) < group_count:
                names = [*names, *[f"group_{idx}" for idx in range(len(names), group_count)]]
        else:
            group_count = 1
            names = ["missing"]
        self.group_names = names[:group_count]
        self.group_count = group_count
        self.register_buffer("group_ids", torch.as_tensor(ids or [0], dtype=torch.long), persistent=False)
        self.value_projection = nn.Linear(1, hidden_dim)
        self.feature_embedding = nn.Parameter(torch.zeros(self.max_features, hidden_dim))
        self.group_embedding = nn.Parameter(torch.zeros(group_count, hidden_dim))
        self.missing_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.token_mlp = nn.ModuleList(ResidualMLPEncoder(hidden_dim, dropout) for _ in range(max(factor_layers, 0)))
        self.factor_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.group_prototypes),
        )
        if group_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.group_encoder = nn.TransformerEncoder(layer, num_layers=group_layers)
        else:
            self.group_encoder = None
        self.group_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, feature_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, feature_count = x.shape
        if feature_count == 0:
            tokens = self.missing_token.expand(batch_size, 1, -1)
            group_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            group_weights = torch.zeros(batch_size, 1, dtype=x.dtype, device=x.device)
            return tokens, group_mask, group_weights
        if feature_count > self.max_features:
            raise ValueError(f"feature_count={feature_count} exceeds max_features={self.max_features}")
        if feature_count != self.feature_count:
            raise ValueError(f"feature_count={feature_count} does not match configured feature_count={self.feature_count}")

        valid = _normalize_feature_mask(feature_mask, feature_count, x.device)
        tokens = self.value_projection(x.unsqueeze(-1))
        tokens = tokens + self.feature_embedding[:feature_count].unsqueeze(0)
        group_ids = self.group_ids[:feature_count].to(device=x.device)
        tokens = tokens + self.group_embedding[group_ids].unsqueeze(0)
        tokens = self.dropout(self.input_norm(tokens))
        tokens = torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        for block in self.token_mlp:
            tokens = block(tokens)
            tokens = torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))

        feature_weights = self._feature_weights(tokens, valid)
        group_tokens, group_mask, group_weights = self._pool_groups(tokens, valid, feature_weights, group_ids)
        if self.group_encoder is not None:
            safe_mask = _safe_key_padding_mask(group_mask)
            group_tokens = self.group_encoder(group_tokens, src_key_padding_mask=safe_mask)
        group_tokens = self.group_norm(group_tokens)
        group_tokens = torch.where(group_mask.unsqueeze(-1), group_tokens, torch.zeros_like(group_tokens))
        return group_tokens, group_mask, group_weights

    def _feature_weights(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if not self.use_factor_gate:
            weights = valid.to(tokens.dtype).unsqueeze(-1)
            weights = weights.expand(-1, -1, self.group_prototypes)
            return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits = self.factor_gate(tokens)
        safe_valid = _ensure_any_available(valid)
        logits = logits.masked_fill(~safe_valid.unsqueeze(-1), torch.finfo(logits.dtype).min)
        if self.factor_gate_activation == "sparsemax":
            weights = sparsemax(logits, dim=1)
        else:
            weights = torch.softmax(logits, dim=1)
        return torch.where(valid.unsqueeze(-1), weights, torch.zeros_like(weights))

    def _pool_groups(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        feature_weights: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        weights_out: list[torch.Tensor] = []
        for group_id in range(self.group_count):
            in_group = group_ids.eq(group_id).view(1, -1)
            group_valid = valid & in_group
            group_mask = group_valid.any(dim=1)
            if self.use_factor_gate:
                weights = torch.where(group_valid.unsqueeze(-1), feature_weights, torch.zeros_like(feature_weights))
            else:
                weights = group_valid.to(tokens.dtype).unsqueeze(-1).expand(-1, -1, self.group_prototypes)
            denom = weights.sum(dim=1).clamp_min(1.0)
            group_proto_tokens = torch.einsum("bfh,bfp->bph", tokens, weights) / denom.unsqueeze(-1)
            pooled.append(group_proto_tokens)
            masks.extend([group_mask] * self.group_prototypes)
            weights_out.append(weights.sum(dim=1).mean(dim=1))
        group_tokens = torch.cat(pooled, dim=1)
        group_mask = torch.stack(masks, dim=1)
        group_weights = torch.stack(weights_out, dim=1)
        if not self.use_factor_gate:
            group_weights = group_weights / group_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return group_tokens, group_mask, group_weights


class ResidualMLPEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.net(tokens)


class SafeCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        key_value: torch.Tensor,
        key_value_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_kv, safe_kv_mask = _safe_attention_inputs(key_value, key_value_mask)
        attn_out, _ = self.attn(
            query,
            safe_kv,
            safe_kv,
            key_padding_mask=~safe_kv_mask,
            need_weights=False,
        )
        has_kv = key_value_mask.any(dim=1).view(-1, 1, 1)
        attn_out = torch.where(has_kv, attn_out, torch.zeros_like(attn_out))
        out = self.norm(query + self.dropout(attn_out))
        out = torch.where(query_mask.unsqueeze(-1), out, torch.zeros_like(out))
        return out


class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float, use_gate: bool = True, use_cross_attention: bool = True) -> None:
        super().__init__()
        self.use_gate = use_gate
        self.use_cross_attention = use_cross_attention
        self.price_text = SafeCrossAttention(hidden_dim, n_heads, dropout)
        self.price_fundamental = SafeCrossAttention(hidden_dim, n_heads, dropout)
        self.text_price = SafeCrossAttention(hidden_dim, n_heads, dropout)
        self.fundamental_price = SafeCrossAttention(hidden_dim, n_heads, dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        price_tokens: torch.Tensor,
        price_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        fundamental_tokens: torch.Tensor,
        fundamental_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_cross_attention:
            price_tokens = self.price_text(price_tokens, price_mask, text_tokens, text_mask)
            price_tokens = self.price_fundamental(price_tokens, price_mask, fundamental_tokens, fundamental_mask)
            text_tokens = self.text_price(text_tokens, text_mask, price_tokens, price_mask)
            fundamental_tokens = self.fundamental_price(fundamental_tokens, fundamental_mask, price_tokens, price_mask)

        pooled_price = masked_mean(price_tokens, price_mask)
        pooled_text = masked_mean(text_tokens, text_mask)
        pooled_fundamental = masked_mean(fundamental_tokens, fundamental_mask)
        available = torch.stack(
            [price_mask.any(dim=1), text_mask.any(dim=1), fundamental_mask.any(dim=1)],
            dim=1,
        )
        available = _ensure_any_available(available)
        context = torch.cat([pooled_price, pooled_text, pooled_fundamental, available.to(price_tokens.dtype)], dim=1)
        if self.use_gate:
            logits = self.gate(context)
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~available, mask_value)
            gates = torch.softmax(logits, dim=1)
        else:
            gates = available.to(price_tokens.dtype)
            gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused = (
            gates[:, 0:1] * pooled_price
            + gates[:, 1:2] * pooled_text
            + gates[:, 2:3] * pooled_fundamental
        )
        return fused, gates


class StrongFactorMLP(nn.Module):
    """Residual factor-only baseline using the same output heads as MSGCA."""

    def __init__(
        self,
        text_features: int,
        fundamental_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        depth: int = 3,
    ) -> None:
        super().__init__()
        input_dim = int(text_features) + int(fundamental_features)
        if input_dim <= 0:
            raise ValueError("StrongFactorMLP requires at least one factor feature")
        self.text_features = int(text_features)
        self.fundamental_features = int(fundamental_features)
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList(ResidualMLPEncoder(hidden_dim, dropout) for _ in range(max(depth, 1)))
        self.score_head = nn.Linear(hidden_dim, 1)
        self.return_head = nn.Linear(hidden_dim, 1)
        self.direction_head = nn.Linear(hidden_dim, 1)
        self.final_score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        fundamental_features: torch.Tensor,
        fundamental_mask: torch.Tensor,
    ) -> MSGCAOutput:
        text_mask = _normalize_feature_mask(text_mask, text_features.shape[1], text_features.device)
        fundamental_mask = _normalize_feature_mask(
            fundamental_mask,
            fundamental_features.shape[1],
            fundamental_features.device,
        )
        text = torch.where(text_mask, text_features, torch.zeros_like(text_features))
        fundamental = torch.where(fundamental_mask, fundamental_features, torch.zeros_like(fundamental_features))
        hidden = self.input(torch.cat([text, fundamental], dim=1))
        for block in self.blocks:
            hidden = block(hidden.unsqueeze(1)).squeeze(1)
        y_score = self.score_head(hidden).squeeze(-1)
        return_pred = self.return_head(hidden).squeeze(-1)
        direction_logit = self.direction_head(hidden).squeeze(-1)
        final_score = self.final_score_head(hidden).squeeze(-1)
        available = torch.stack(
            [
                torch.zeros(text_features.shape[0], dtype=torch.bool, device=text_features.device),
                text_mask.any(dim=1),
                fundamental_mask.any(dim=1),
            ],
            dim=1,
        )
        available = _ensure_any_available(available)
        gates = available.to(hidden.dtype)
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1.0)
        return MSGCAOutput(
            y_score=y_score,
            return_pred=return_pred,
            direction_logit=direction_logit,
            gates=gates,
            g_price=gates[:, 0],
            g_text=gates[:, 1],
            g_fundamental=gates[:, 2],
            final_score=final_score,
        )


class MSGCA(nn.Module):
    def __init__(
        self,
        price_variables: int,
        lookback: int,
        text_features: int,
        fundamental_features: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        price_layers: int = 2,
        enable_price: bool = True,
        enable_news: bool = True,
        enable_fundamental: bool = True,
        use_gate: bool = True,
        use_cross_attention: bool = True,
        max_text_features: int = 512,
        max_fundamental_features: int = 2048,
        factor_encoder: str = "simple",
        factor_layers: int = 2,
        factor_group_layers: int = 1,
        factor_group_prototypes: int = 1,
        use_factor_gate: bool = True,
        factor_gate_activation: str = "softmax",
        text_group_ids: Sequence[int] | None = None,
        text_group_names: Sequence[str] | None = None,
        fundamental_group_ids: Sequence[int] | None = None,
        fundamental_group_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if price_variables <= 0:
            raise ValueError("price_variables must be positive")
        self.enable_price = enable_price
        self.enable_news = enable_news
        self.enable_fundamental = enable_fundamental
        self.factor_encoder_type = factor_encoder
        self.price_encoder = PriceEncoder(price_variables, lookback, hidden_dim, n_heads, price_layers, dropout)
        if factor_encoder == "simple":
            self.text_encoder = NumericTokenEncoder(max(max_text_features, text_features, 1), hidden_dim, dropout)
            self.fundamental_encoder = NumericTokenEncoder(max(max_fundamental_features, fundamental_features, 1), hidden_dim, dropout)
            self.text_mlp = ResidualMLPEncoder(hidden_dim, dropout)
            self.fundamental_mlp = ResidualMLPEncoder(hidden_dim, dropout)
            self.factor_group_names: list[str] = []
        elif factor_encoder == "factor_aware":
            self.text_encoder = FactorAwareEncoder(
                max(max_text_features, text_features, 1),
                hidden_dim,
                n_heads,
                dropout,
                text_features,
                group_ids=text_group_ids,
                group_names=text_group_names,
                factor_layers=factor_layers,
                group_layers=factor_group_layers,
                group_prototypes=factor_group_prototypes,
                use_factor_gate=use_factor_gate,
                factor_gate_activation=factor_gate_activation,
            )
            self.fundamental_encoder = FactorAwareEncoder(
                max(max_fundamental_features, fundamental_features, 1),
                hidden_dim,
                n_heads,
                dropout,
                fundamental_features,
                group_ids=fundamental_group_ids,
                group_names=fundamental_group_names,
                factor_layers=factor_layers,
                group_layers=factor_group_layers,
                group_prototypes=factor_group_prototypes,
                use_factor_gate=use_factor_gate,
                factor_gate_activation=factor_gate_activation,
            )
            self.text_mlp = nn.Identity()
            self.fundamental_mlp = nn.Identity()
            self.factor_group_names = [
                *[f"text:{name}" for name in self.text_encoder.group_names],
                *[f"fundamental:{name}" for name in self.fundamental_encoder.group_names],
            ]
        else:
            raise ValueError(f"Unsupported factor_encoder: {factor_encoder}")
        self.hidden_dim = hidden_dim
        self.text_group_count = len(getattr(self.text_encoder, "group_names", [])) if factor_encoder == "factor_aware" else 0
        self.fundamental_group_count = (
            len(getattr(self.fundamental_encoder, "group_names", [])) if factor_encoder == "factor_aware" else 0
        )
        self.fusion = GatedCrossAttentionFusion(hidden_dim, n_heads, dropout, use_gate, use_cross_attention)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.return_head = nn.Linear(hidden_dim, 1)
        self.direction_head = nn.Linear(hidden_dim, 1)
        self.final_score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        price_window: torch.Tensor,
        price_mask: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        fundamental_features: torch.Tensor,
        fundamental_mask: torch.Tensor,
    ) -> MSGCAOutput:
        batch_size = price_window.shape[0]
        if self.enable_price:
            price_tokens, price_token_mask = self.price_encoder(price_window, price_mask)
        else:
            price_tokens, price_token_mask = self._empty_tokens(batch_size, price_window.device, price_window.dtype)

        text_group_weights = None
        fundamental_group_weights = None
        if not self.enable_news:
            text_tokens, text_token_mask = self._empty_tokens(batch_size, text_features.device, text_features.dtype)
            if self.factor_encoder_type == "factor_aware":
                text_group_weights = torch.zeros(batch_size, self.text_group_count, dtype=text_features.dtype, device=text_features.device)
        else:
            if self.factor_encoder_type == "factor_aware":
                text_tokens, text_token_mask, text_group_weights = self.text_encoder(text_features, text_mask)
            else:
                text_tokens, text_token_mask = self.text_encoder(text_features, text_mask)
            text_tokens = self.text_mlp(text_tokens)

        if not self.enable_fundamental:
            fundamental_tokens, fundamental_token_mask = self._empty_tokens(
                batch_size,
                fundamental_features.device,
                fundamental_features.dtype,
            )
            if self.factor_encoder_type == "factor_aware":
                fundamental_group_weights = torch.zeros(
                    batch_size,
                    self.fundamental_group_count,
                    dtype=fundamental_features.dtype,
                    device=fundamental_features.device,
                )
        else:
            if self.factor_encoder_type == "factor_aware":
                fundamental_tokens, fundamental_token_mask, fundamental_group_weights = self.fundamental_encoder(
                    fundamental_features,
                    fundamental_mask,
                )
            else:
                fundamental_tokens, fundamental_token_mask = self.fundamental_encoder(fundamental_features, fundamental_mask)
            fundamental_tokens = self.fundamental_mlp(fundamental_tokens)

        reference_dtype = (
            price_tokens.dtype
            if self.enable_price
            else text_tokens.dtype
            if self.enable_news
            else fundamental_tokens.dtype
            if self.enable_fundamental
            else price_tokens.dtype
        )
        price_tokens = price_tokens.to(dtype=reference_dtype)
        text_tokens = text_tokens.to(dtype=reference_dtype)
        fundamental_tokens = fundamental_tokens.to(dtype=reference_dtype)

        fused, gates = self.fusion(
            price_tokens,
            price_token_mask,
            text_tokens,
            text_token_mask,
            fundamental_tokens,
            fundamental_token_mask,
        )
        hidden = self.head(fused)
        y_score = self.score_head(hidden).squeeze(-1)
        return_pred = self.return_head(hidden).squeeze(-1)
        direction_logit = self.direction_head(hidden).squeeze(-1)
        final_score = self.final_score_head(hidden).squeeze(-1)
        factor_group_weights = None
        if text_group_weights is not None and fundamental_group_weights is not None:
            factor_group_weights = torch.cat(
                [
                    gates[:, 1:2] * text_group_weights,
                    gates[:, 2:3] * fundamental_group_weights,
                ],
                dim=1,
            )
        return MSGCAOutput(
            y_score=y_score,
            return_pred=return_pred,
            direction_logit=direction_logit,
            gates=gates,
            g_price=gates[:, 0],
            g_text=gates[:, 1],
            g_fundamental=gates[:, 2],
            final_score=final_score,
            factor_group_weights=factor_group_weights,
        )

    def _empty_tokens(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.zeros(batch_size, 1, self.hidden_dim, dtype=dtype, device=device)
        mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        return tokens, mask


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(tokens.dtype).unsqueeze(-1)
    count = mask_f.sum(dim=1).clamp_min(1.0)
    return (tokens * mask_f).sum(dim=1) / count


def gate_entropy(gates: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(gates * torch.log(gates.clamp_min(eps))).sum(dim=1).mean()


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation with exact zeros along `dim`."""
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_values = torch.sort(shifted, dim=dim, descending=True).values
    cumsum = sorted_values.cumsum(dim)
    dim_size = shifted.size(dim)
    view_shape = [1] * shifted.dim()
    view_shape[dim] = dim_size
    range_values = torch.arange(1, dim_size + 1, device=logits.device, dtype=logits.dtype).view(view_shape)
    support = 1 + range_values * sorted_values > cumsum
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumsum.gather(dim, support_size - 1) - 1) / support_size.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


def _normalize_feature_mask(mask: torch.Tensor, feature_count: int, device: torch.device) -> torch.Tensor:
    if feature_count == 0:
        return torch.zeros(mask.shape[0], 0, dtype=torch.bool, device=device)
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.dim() == 1:
        return mask.unsqueeze(1).expand(-1, feature_count)
    if mask.dim() == 2:
        if mask.shape[1] == feature_count:
            return mask
        if mask.shape[1] == 1:
            return mask.expand(-1, feature_count)
    raise ValueError(f"feature mask shape {tuple(mask.shape)} is incompatible with feature_count={feature_count}")


def _prepare_group_ids(group_ids: Sequence[int] | None, feature_count: int) -> list[int]:
    if feature_count <= 0:
        return []
    if group_ids is None or len(group_ids) == 0:
        return [0] * feature_count
    ids = [int(value) for value in group_ids]
    if len(ids) != feature_count:
        raise ValueError(f"group_ids length={len(ids)} does not match feature_count={feature_count}")
    if min(ids) < 0:
        raise ValueError("group_ids must be non-negative")
    return ids


def _safe_key_padding_mask(valid_tokens: torch.Tensor) -> torch.Tensor:
    safe_valid = _ensure_any_available(valid_tokens)
    return ~safe_valid


def _safe_attention_inputs(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    safe_mask = _ensure_any_available(mask)
    safe_tokens = tokens.clone()
    empty = ~mask.any(dim=1)
    if empty.any():
        safe_tokens[empty, 0, :] = 0.0
    return safe_tokens, safe_mask


def _ensure_any_available(mask: torch.Tensor) -> torch.Tensor:
    safe = mask.clone()
    empty = ~safe.any(dim=1)
    if empty.any():
        safe[empty, 0] = True
    return safe
