"""TiDAR: Think in Diffusion, Talk in Autoregression
Training with LIV backbone using dual AR + Diffusion objectives.

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
import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple

from liv import (
    STARBackbone, SparsityType,
    TokenMixType, ChannelMixType,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TiDARConfig:
    # Model
    vocab_size: int = 32000
    dim: int = 512
    num_heads: int = 8
    max_seq_len: int = 512

    # TiDAR-specific
    block_size: int = 8           # draft block length (used at inference)
    alpha: float = 1.0            # loss balance: higher = more weight on AR
    mask_token_id: int = -1       # auto-set to vocab_size

    # Backbone: list of (featurizer_cls, token_mix_type, channel_mix_type)
    backbone_configs: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (1, 2, 3),   # SA-1: standard MHA
        (9, 1, 2),   # GMemless: SwiGLU FFN
        (5, 4, 1),   # Rec-1: Mamba-like SSM
        (9, 1, 2),   # GMemless: SwiGLU FFN
    ])

    # Training
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_steps: int = 500
    max_steps: int = 50000
    batch_size: int = 8
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    log_interval: int = 50
    save_interval: int = 5000
    save_dir: str = "./checkpoints"

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
    """

    def __init__(self, config: TiDARConfig):
        super().__init__()
        self.config = config

        # Embeddings: +1 slot for [MASK] token
        self.tok_emb = nn.Embedding(config.vocab_size + 1, config.dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)

        # LIV backbone — sparsity_type flows through **kwargs to each block
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
# LR Schedule
# =============================================================================

def get_lr(step: int, config: TiDARConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < config.warmup_steps:
        return config.lr * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps
    )
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


# =============================================================================
# Training Loop
# =============================================================================

def train(config: TiDARConfig, dataset):
    """Train TiDAR-LIV model.

    Args:
        config:  TiDARConfig
        dataset: iterable yielding [batch, seq_len] token-ID tensors
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TiDARModel(config).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"TiDAR-LIV | params: {param_count:,} | device: {device}")
    print(f"  alpha={config.alpha}, block_size={config.block_size}")
    print(f"  backbone: {config.backbone_configs}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    os.makedirs(config.save_dir, exist_ok=True)
    model.train()
    step = 0

    for batch in dataset:
        if step >= config.max_steps:
            break

        # LR schedule
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        input_ids = batch.to(device)
        loss, ar_loss, diff_loss = compute_tidar_loss(model, input_ids)

        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if step % config.log_interval == 0:
            print(
                f"step {step:6d} | loss {loss.item():.4f} | "
                f"ar {ar_loss:.4f} | diff {diff_loss:.4f} | lr {lr:.2e}"
            )

        if step > 0 and step % config.save_interval == 0:
            path = os.path.join(config.save_dir, f"tidar_step{step}.pt")
            torch.save(
                {"step": step, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "config": config},
                path,
            )
            print(f"  saved {path}")

        step += 1

    # Final save
    path = os.path.join(config.save_dir, "tidar_final.pt")
    torch.save(
        {"step": step, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "config": config},
        path,
    )
    print(f"Training done ({step} steps). Saved {path}")
    return model


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    cfg = TiDARConfig(
        vocab_size=256,
        dim=128,
        num_heads=4,
        max_seq_len=64,
        batch_size=4,
        max_steps=20,
        log_interval=5,
        backbone_configs=[
            (1, 2, 3),   # SA-1
            (9, 1, 2),   # GMemless
        ],
    )

    def dummy_data():
        for _ in range(cfg.max_steps):
            yield torch.randint(0, cfg.vocab_size, (cfg.batch_size, 32))

    train(cfg, dummy_data())