"""TiDAR: Think in Diffusion, Talk in Autoregression
Model definitions and loss computation for LIV backbone with dual AR + Diffusion.

Reference: arxiv.org/abs/2511.08923

Training sequence layout: [clean_tokens | mask_tokens]  (doubled)
  - Clean section: causal attention (standard AR next-token prediction)
  - Mask section: bidirectional attention to prefix + within block
  - Position IDs shared: mask pos_i == clean pos_i

Joint loss: L = 1/(1+α) * [α * L_AR + L_Diff]
  - L_AR:   cross-entropy on clean section (shifted by 1)
  - L_Diff: cross-entropy on mask section (predict original token)
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
# Smoke test
# =============================================================================

if __name__ == "__main__":
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