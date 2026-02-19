"""TiDAR: Think in Diffusion, Talk in Autoregression
Model definitions and loss computation for LIV backbone with dual AR + Diffusion.

Reference: arxiv.org/abs/2511.08923

─── Language Modelling (TiDARModel) ────────────────────────────────────────────
Training sequence layout: [clean_tokens | mask_tokens]  (doubled, length 2L)
  - Clean section: causal attention (standard AR next-token prediction)
  - Mask section: bidirectional attention to prefix + within block
  - Position IDs shared: mask pos_i == clean pos_i

Joint loss: L = 1/(1+α) * [α * L_AR + L_Diff]
  - L_AR:   cross-entropy on clean section (shifted by 1)
  - L_Diff: cross-entropy on mask section (predict original token)

─── Time Series Forecasting (TiDARTSModel) ──────────────────────────────────────
Training sequence layout: [lookback_tokens (L) | forecast_mask_tokens (H)]
  - Lookback section: L embedded timesteps, causal attention
  - Forecast section: H learnable [MASK] embeddings, attends to all lookback
    tokens + all other forecast positions (bidirectional)

Joint loss: L = 1/(1+α) * [α * L_AR + L_Diff]
  - L_AR:   MSE on clean (lookback) section — predict next timestep (shifted by 1)
  - L_Diff: MSE on mask (forecast) section — predict ground-truth future values

Inference modes:
  - Fast (draft-only):  use diff_head output directly as H-step forecast
  - Slow (AR-verified): step through ar_head autoregressively, refine drafts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from core.liv import (
    STARBackbone, SparsityType,
    TokenMixType, ChannelMixType,
)
from core.modeldef import RevIN


# =============================================================================
# Configuration (model-only — training config lives in src/train.py CLI)
# =============================================================================

@dataclass
class TiDARConfig:
    """TiDAR model configuration. No training hyperparams here."""
    vocab_size: int = 32000
    dim: int = 512
    num_heads: int = 8
    max_seq_len: int = 512

    # TiDAR-specific
    block_size: int = 8           # draft block length (used at inference)
    alpha: float = 1.0            # loss balance: higher = more weight on AR
    mask_token_id: int = -1       # auto-set to vocab_size

    # Default backbone (used when no genome/external backbone is provided)
    backbone_configs: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (1, 2, 3),   # SA-1: standard MHA
        (9, 1, 2),   # GMemless: SwiGLU FFN
        (5, 4, 1),   # Rec-1: Mamba-like SSM
        (9, 1, 2),   # GMemless: SwiGLU FFN
    ])

    def __post_init__(self):
        if self.mask_token_id == -1:
            self.mask_token_id = self.vocab_size


# =============================================================================
# TiDAR Model
# =============================================================================

class TiDARModel(nn.Module):
    """LIV backbone with TiDAR dual-mode (AR + Diffusion) training.

    Single forward pass processes doubled sequence [clean | mask] with a
    hybrid causal+bidirectional attention mask. Each LIV block auto-creates
    its own SparsityMask with correct fill value (−inf for softmax blocks,
    0 for others) via the sparsity_type parameter.

    Args:
        config:   TiDARConfig with model dimensions and alpha.
        backbone: Optional pre-built backbone (e.g. from GenomeModelBuilder).
                  If None, builds default STARBackbone from config.backbone_configs.
    """

    def __init__(self, config: TiDARConfig, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        # Embeddings: +1 slot for [MASK] token
        self.tok_emb = nn.Embedding(config.vocab_size + 1, config.dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)

        # LIV backbone — either external (genome-built) or default
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = STARBackbone(
                config.backbone_configs, config.dim,
                num_heads=config.num_heads,
                sparsity_type=SparsityType.TIDAR_HYBRID,
            )

        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor):
        """Forward with doubled sequence [clean | mask].

        Args:
            input_ids: [batch, seq_len] clean token IDs

        Returns:
            ar_logits:   [batch, seq_len - 1, vocab_size]
            diff_logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # ── Build doubled sequence: [clean_tokens | mask_tokens] ──
        mask_ids = torch.full(
            (batch, seq_len), self.config.mask_token_id,
            dtype=input_ids.dtype, device=device,
        )
        doubled = torch.cat([input_ids, mask_ids], dim=1)  # [B, 2L]

        # Position IDs: mask tokens mirror clean positions
        pos_ids = torch.arange(seq_len, device=device)
        pos_ids = pos_ids.repeat(2).unsqueeze(0)  # [1, 2L]

        # ── Embed ──
        x = self.tok_emb(doubled) + self.pos_emb(pos_ids)

        # ── LIV backbone with hybrid mask ──
        x = self.backbone(x, clean_len=seq_len)
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, 2L, vocab_size]

        # ── Split ──
        ar_logits = logits[:, :seq_len - 1]   # predict token i+1 from pos i
        diff_logits = logits[:, seq_len:]      # predict original from [MASK]

        return ar_logits, diff_logits


# =============================================================================
# Loss
# =============================================================================

def compute_tidar_loss(model: TiDARModel, input_ids: torch.Tensor):
    """Joint TiDAR loss: L = 1/(1+α) * [α * L_AR + L_Diff]

    Returns:
        (total_loss, ar_loss_scalar, diff_loss_scalar)
    """
    ar_logits, diff_logits = model(input_ids)
    alpha = model.config.alpha

    # AR: next-token prediction (shifted by 1)
    ar_targets = input_ids[:, 1:]
    ar_loss = F.cross_entropy(
        ar_logits.reshape(-1, ar_logits.size(-1)),
        ar_targets.reshape(-1),
    )

    # Diffusion: predict original token at each mask position
    diff_targets = input_ids
    diff_loss = F.cross_entropy(
        diff_logits.reshape(-1, diff_logits.size(-1)),
        diff_targets.reshape(-1),
    )

    total = (1.0 / (1.0 + alpha)) * (alpha * ar_loss + diff_loss)
    return total, ar_loss.item(), diff_loss.item()


# =============================================================================
# Time Series Config
# =============================================================================

@dataclass
class TiDARTSConfig:
    """TiDAR-TS model configuration. No training hyperparams here."""
    n_variates: int = 7           # C: number of input channels
    seq_len: int = 96             # L: lookback window length
    pred_len: int = 96            # H: forecast horizon length
    dim: int = 512                # D: model hidden dimension
    num_heads: int = 8

    # TiDAR-specific
    alpha: float = 1.0            # loss balance: higher = more weight on AR

    # Default backbone (used when no external backbone is provided)
    backbone_configs: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (1, 2, 3),   # SA-1: standard MHA
        (9, 1, 2),   # GMemless: SwiGLU FFN
        (5, 4, 1),   # Rec-1: Mamba-like SSM
        (9, 1, 2),   # GMemless: SwiGLU FFN
    ])


# =============================================================================
# TiDAR Time Series Model
# =============================================================================

class TiDARTSModel(nn.Module):
    """LIV backbone with TiDAR dual-mode (AR + Diffusion) training for time series.

    Sequence layout: [lookback (L) | forecast_mask (H)]
      - Lookback tokens: clean embedded timesteps, causal attention
      - Forecast tokens: learnable [MASK] embedding, bidirectional attention
        to all lookback tokens + all other forecast positions

    The TIDAR_HYBRID sparsity mask in core/liv.py handles this directly when
    called with clean_len=seq_len.

    Args:
        config:   TiDARTSConfig with dimensions and alpha.
        backbone: Optional pre-built backbone (e.g. from GenomeModelBuilder).
                  If None, builds default STARBackbone from config.backbone_configs.
    """

    def __init__(self, config: TiDARTSConfig, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        # RevIN: instance normalization over variates (same as STARLIVTSModel)
        self.revin = RevIN(config.n_variates)

        # Input projection: embed each timestep (C → D)
        self.input_proj = nn.Linear(config.n_variates, config.dim)

        # Learnable [MASK] embedding — broadcast over (B, H, D)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, config.dim))

        # Positional embedding for L + H total positions
        self.pos_emb = nn.Embedding(config.seq_len + config.pred_len, config.dim)

        # LIV backbone with hybrid causal+bidirectional mask
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = STARBackbone(
                config.backbone_configs, config.dim,
                num_heads=config.num_heads,
                sparsity_type=SparsityType.TIDAR_HYBRID,
            )

        self.norm = nn.LayerNorm(config.dim)

        # Output heads: D → C
        self.ar_head   = nn.Linear(config.dim, config.n_variates)  # lookback section
        self.diff_head = nn.Linear(config.dim, config.n_variates)  # forecast section

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_emb, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_backbone(self, x: torch.Tensor, clean_len: int) -> torch.Tensor:
        """Forward through backbone, passing clean_len for TIDAR_HYBRID sparsity.

        STARBackbone accepts (x, clean_len=...) directly.
        nn.Sequential (from GenomeModelBuilder) does not — iterate manually.
        """
        if isinstance(self.backbone, nn.Sequential):
            for block in self.backbone:
                x = block(x, clean_len=clean_len)
            return x
        return self.backbone(x, clean_len=clean_len)

    def _forward_core(self, x_norm: torch.Tensor):
        """Core forward on already-normalized input. Returns preds in normalized space.

        Args:
            x_norm: (B, L, C) RevIN-normalized lookback

        Returns:
            ar_pred:   (B, L-1, C) normalized next-step predictions
            diff_pred: (B, H, C)   normalized forecast (draft output)
        """
        B, L, C = x_norm.shape
        H = self.config.pred_len
        device = x_norm.device

        # ── Embed lookback: (B, L, D) ──
        clean_emb = self.input_proj(x_norm)

        # ── Forecast mask tokens: (B, H, D) ──
        mask_emb = self.mask_emb.expand(B, H, -1)

        # ── Concatenate into (B, L+H, D) and add positions ──
        combined = torch.cat([clean_emb, mask_emb], dim=1)
        pos_ids  = torch.arange(L + H, device=device).unsqueeze(0)  # (1, L+H)
        combined = combined + self.pos_emb(pos_ids)

        # ── LIV backbone with hybrid mask (clean_len=L) ──
        out = self._forward_backbone(combined, clean_len=L)
        out = self.norm(out)

        # ── Split and project ──
        clean_out = out[:, :L]    # (B, L, D)
        mask_out  = out[:, L:]    # (B, H, D)

        ar_pred   = self.ar_head(clean_out[:, :-1])  # (B, L-1, C)
        diff_pred = self.diff_head(mask_out)          # (B, H, C)

        return ar_pred, diff_pred

    def forward(self, x: torch.Tensor):
        """Forward with RevIN normalization + [lookback | forecast_mask] layout.

        Args:
            x: (B, L, C) lookback window (raw, unnormalized)

        Returns:
            ar_pred:   (B, L-1, C) — next-step predictions (normalized space)
            diff_pred: (B, H, C)   — forecast draft (normalized space)

        Note: outputs are in RevIN-normalized space. Use revin(out, mode='denorm')
        to recover raw-scale predictions (done automatically in forecast()).
        """
        x_norm = self.revin(x, mode='norm')  # normalizes x, stores _mean/_stdev
        return self._forward_core(x_norm)

    @torch.no_grad()
    def forecast(self, x: torch.Tensor, ar_steps: int = 0) -> torch.Tensor:
        """Inference: draft all H steps in raw scale, optionally refine with AR.

        Args:
            x:        (B, L, C) lookback window (raw, unnormalized)
            ar_steps: number of leading steps to refine autoregressively
                      (0 = pure draft / fast mode; H = full AR / slow mode)

        Returns:
            pred: (B, H, C) forecast in raw scale
        """
        # Draft: normalize input, run model, denorm output
        _, diff_pred_norm = self(x)                          # stores revin stats for x
        diff_pred = self.revin(diff_pred_norm, mode='denorm')  # (B, H, C) raw scale

        if ar_steps <= 0:
            return diff_pred

        # AR refinement: slide a window along the horizon in raw space.
        # Each self(window) call independently normalizes that window and
        # stores fresh revin stats, so the subsequent denorm is consistent.
        B, L, C = x.shape
        H = self.config.pred_len
        ar_steps = min(ar_steps, H)

        ctx = x.clone()                  # (B, L+t, C) grows in raw space
        refined = diff_pred.clone()      # (B, H, C) — updated in place

        for t in range(ar_steps):
            window = ctx[:, -L:]         # (B, L, C) raw window
            ar_pred_norm, _ = self(window)           # normalizes window, stores stats
            next_step_norm = ar_pred_norm[:, -1:]    # (B, 1, C) normalized
            next_step = self.revin(next_step_norm, mode='denorm')  # raw scale
            refined[:, t:t+1] = next_step
            ctx = torch.cat([ctx, next_step], dim=1)  # extend context in raw space

        return refined


# =============================================================================
# Time Series Loss
# =============================================================================

def compute_tidar_ts_loss(
    model: TiDARTSModel,
    x: torch.Tensor,
    y: torch.Tensor,
):
    """Joint TiDAR-TS loss: L = 1/(1+α) * [α * L_AR + L_Diff]

    Args:
        model: TiDARTSModel
        x:     (B, L, C) lookback window (raw scale)
        y:     (B, H, C) ground-truth forecast (raw scale)

    Returns:
        (total_loss, ar_loss_scalar, diff_loss_scalar)

    Loss is computed in raw scale (after RevIN denorm) so that MSE values are
    directly comparable to baseline numbers reported in the literature.
    """
    ar_pred_norm, diff_pred_norm = model(x)   # model.revin now holds stats for x
    alpha = model.config.alpha

    # Denorm predictions back to raw scale (uses stats stored by model(x))
    ar_pred   = model.revin(ar_pred_norm,   mode='denorm')  # (B, L-1, C)
    diff_pred = model.revin(diff_pred_norm, mode='denorm')  # (B, H, C)

    # AR: next-step prediction on lookback (predict x_{t+1} from x_t)
    ar_loss = F.mse_loss(ar_pred, x[:, 1:])

    # Diffusion: predict ground-truth forecast from mask tokens
    diff_loss = F.mse_loss(diff_pred, y)

    total = (1.0 / (1.0 + alpha)) * (alpha * ar_loss + diff_loss)
    return total, ar_loss.item(), diff_loss.item()


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    # ── Language model ──
    print("=== TiDAR Language Model ===")
    cfg = TiDARConfig(
        vocab_size=256, dim=128, num_heads=4, max_seq_len=64,
        backbone_configs=[(1, 2, 3), (9, 1, 2)],
    )
    model = TiDARModel(cfg)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randint(0, 256, (4, 32))
    loss, ar_l, diff_l = compute_tidar_loss(model, x)
    print(f"loss={loss.item():.4f}  ar={ar_l:.4f}  diff={diff_l:.4f}")
    print("OK")

    # ── Time series ──
    print()
    print("=== TiDAR Time Series ===")
    B, L, H, C, D = 4, 96, 48, 7, 128
    ts_cfg = TiDARTSConfig(
        n_variates=C, seq_len=L, pred_len=H, dim=D, num_heads=4,
        backbone_configs=[(1, 2, 3), (9, 1, 2)],
    )
    ts_model = TiDARTSModel(ts_cfg)
    print(f"Params: {sum(p.numel() for p in ts_model.parameters()):,}")

    x_ts = torch.randn(B, L, C)
    y_ts = torch.randn(B, H, C)
    ts_loss, ar_l, diff_l = compute_tidar_ts_loss(ts_model, x_ts, y_ts)
    print(f"loss={ts_loss.item():.4f}  ar={ar_l:.4f}  diff={diff_l:.4f}")

    # Inference — draft mode
    draft = ts_model.forecast(x_ts, ar_steps=0)
    assert draft.shape == (B, H, C), f"draft shape mismatch: {draft.shape}"
    print(f"Draft forecast: {tuple(draft.shape)}  OK")

    # Inference — partial AR refinement (first 4 steps)
    refined = ts_model.forecast(x_ts, ar_steps=4)
    assert refined.shape == (B, H, C), f"refined shape mismatch: {refined.shape}"
    print(f"AR-refined (4 steps): {tuple(refined.shape)}  OK")