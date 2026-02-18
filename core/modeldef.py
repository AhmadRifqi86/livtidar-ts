"""Model definitions for STAR + TiDAR experiments.

All neural network model wrappers that sit on top of LIV backbones:
  - STARLanguageModel      — causal language model
  - STARVisionModel        — image classification
  - STARVisionDetector     — object detection
  - STARLIVTSModel         — time series forecasting (DMamba / S-Mamba / iTransformer)

Helper modules:
  - RMSNorm, RevIN, EMADecomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Norms
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ============================================================================
# Language Model
# ============================================================================

class STARLanguageModel(nn.Module):
    """Causal LM: embedding -> LIV backbone -> RMSNorm -> LM head (tied)."""

    def __init__(self, backbone, vocab_size, dim, tie_weights=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.backbone = backbone
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.backbone(self.embed(idx))
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Vision Classification
# ============================================================================

class STARVisionModel(nn.Module):
    """Vision model: patch_embed -> LIV backbone (causal=False) -> pool -> head."""

    def __init__(self, backbone, num_classes, dim,
                 img_size=32, patch_size=4, in_channels=3):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, dim) * 0.02
        )
        self.backbone = backbone
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, imgs, targets=None):
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.backbone(x)
        x = x.mean(dim=1)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss


# ============================================================================
# Object Detection
# ============================================================================

def _cxcywh_to_xyxy(boxes):
    """Convert (cx, cy, w, h) -> (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Binary focal loss for objectness."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * ((1 - p_t) ** gamma) * ce).mean()


def _box_giou_paired(pred, target):
    """GIoU for matched box pairs, both in cxcywh normalized [0,1]."""
    p = _cxcywh_to_xyxy(pred)
    t = _cxcywh_to_xyxy(target)
    inter_x1 = torch.max(p[:, 0], t[:, 0])
    inter_y1 = torch.max(p[:, 1], t[:, 1])
    inter_x2 = torch.min(p[:, 2], t[:, 2])
    inter_y2 = torch.min(p[:, 3], t[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_p = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    area_t = (t[:, 2] - t[:, 0]).clamp(min=0) * (t[:, 3] - t[:, 1]).clamp(min=0)
    union = area_p + area_t - inter
    iou = inter / (union + 1e-7)
    enc_x1 = torch.min(p[:, 0], t[:, 0])
    enc_y1 = torch.min(p[:, 1], t[:, 1])
    enc_x2 = torch.max(p[:, 2], t[:, 2])
    enc_y2 = torch.max(p[:, 3], t[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)
    return iou - (enc_area - union) / (enc_area + 1e-7)


class STARVisionDetector(nn.Module):
    """Object detection: patch_embed -> LIV backbone (causal=False) -> detection heads."""

    def __init__(self, backbone, num_classes, dim,
                 img_size=224, patch_size=16, in_channels=3):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, dim) * 0.02
        )
        self.backbone = backbone
        self.norm = RMSNorm(dim)

        self.cls_head = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, num_classes),
        )
        self.obj_head = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 4),
        )

        nn.init.normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, imgs, targets=None):
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.backbone(x)
        x = self.norm(x)

        cls_logits = self.cls_head(x)
        obj_logits = self.obj_head(x).squeeze(-1)
        bbox_pred = self.bbox_head(x).sigmoid()

        loss = None
        if targets is not None:
            loss = self._compute_loss(cls_logits, obj_logits, bbox_pred, targets)
        return (cls_logits, obj_logits, bbox_pred), loss

    def _compute_loss(self, cls_logits, obj_logits, bbox_pred, targets):
        B = cls_logits.size(0)
        device = cls_logits.device
        total_loss = torch.zeros(1, device=device, dtype=cls_logits.dtype)

        for b in range(B):
            boxes = targets[b]['boxes']
            labels = targets[b]['labels']
            obj_target = torch.zeros(self.num_patches, device=device)

            if len(boxes) > 0:
                cx, cy = boxes[:, 0], boxes[:, 1]
                gx = (cx * self.grid_size).long().clamp(0, self.grid_size - 1)
                gy = (cy * self.grid_size).long().clamp(0, self.grid_size - 1)
                patch_idx = gy * self.grid_size + gx

                uniq, inv = torch.unique(patch_idx, return_inverse=True)
                a_boxes = torch.zeros(len(uniq), 4, device=device)
                a_labels = torch.zeros(len(uniq), dtype=torch.long, device=device)
                for i in range(len(patch_idx)):
                    a_boxes[inv[i]] = boxes[i]
                    a_labels[inv[i]] = labels[i]

                obj_target[uniq] = 1.0

                pred_b = bbox_pred[b, uniq]
                l1 = F.l1_loss(pred_b, a_boxes, reduction='mean')
                giou = 1.0 - _box_giou_paired(pred_b, a_boxes).mean()

                cls_loss = F.cross_entropy(cls_logits[b, uniq], a_labels)

                total_loss = total_loss + cls_loss + l1 * 5.0 + giou * 2.0

            total_loss = total_loss + _sigmoid_focal_loss(obj_logits[b], obj_target)

        return total_loss / B


# ============================================================================
# Time Series Forecasting — Helper Modules
# ============================================================================

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022)."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        """x: (B, L, C)"""
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._stdev + self._mean
            return x


class EMADecomposition(nn.Module):
    """EMA-based trend-seasonal decomposition (DMamba-style).

    Learnable per-channel smoothing factor alpha decomposes input into
    trend (low-freq EMA) and seasonal (residual) components.
    """

    def __init__(self, n_variates):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((n_variates,), 0.5))

    def forward(self, x):
        """x: (B, L, C) → (seasonal, trend) each (B, L, C)"""
        alpha = torch.sigmoid(self.alpha)  # (C,) in [0,1]
        B, L, C = x.shape
        trend = torch.zeros_like(x)
        trend[:, 0] = x[:, 0]
        for t in range(1, L):
            trend[:, t] = alpha * trend[:, t - 1] + (1 - alpha) * x[:, t]
        seasonal = x - trend
        return seasonal, trend


# ============================================================================
# Time Series Forecasting — Main Model
# ============================================================================

class STARLIVTSModel(nn.Module):
    """Time series forecasting with LIV backbone.

    Three structural approaches from plan.md:

    DMamba (Approach A) — Decomposition + Dual Flow:
        (B,L,C) → RevIN → EMA decomp
          seasonal: embed(C→D) → backbone(L tokens, causal) → proj(D→C) → temporal(L→H)
          trend:    embed(C→D) → backbone_aux(L tokens, causal) → proj(D→C) → temporal(L→H)
          sum → RevIN denorm → (B,H,C)
        Genome: first half → seasonal backbone, second half → trend backbone
        Original DMamba: seasonal=Mamba, trend=MLP

    S-Mamba (Approach B) — Two-Stage Pipeline:
        (B,L,C) → transpose → (B,C,L) → embed(L→D) → (B,C,D)
          → backbone(C tokens, bidirectional variate correlation)
          → backbone_aux(C tokens, temporal processing)
          → proj(D→H) → transpose → (B,H,C)
        Genome: first half → variate backbone, second half → temporal backbone
        Original S-Mamba: variate=BiMamba, temporal=FFN

    iTransformer (Approach C) — Stacked Blocks:
        (B,L,C) → transpose → (B,C,L) → embed(L→D) → (B,C,D)
          → backbone(N blocks, each: cross-variate token_mix + per-variate channel_mix)
          → proj(D→H) → transpose → (B,H,C)
        Genome: all N layers → single backbone
        Original iTransformer: cross_variate=Attention, per_variate=FFN

    Args:
        backbone:      Primary LIV backbone
                       - dmamba: seasonal temporal processing (causal)
                       - smamba: variate correlation (bidirectional)
                       - itransformer: all stacked blocks (bidirectional)
        n_variates:    Number of input variates (C)
        seq_len:       Lookback window length (L)
        pred_len:      Forecast horizon length (H)
        dim:           Model hidden dimension (D)
        structure:     'dmamba', 'smamba', or 'itransformer'
        backbone_aux:  Secondary LIV backbone (optional)
                       - dmamba: trend temporal processing
                       - smamba: temporal dependencies processing
                       - itransformer: not used
    """

    def __init__(self, backbone, n_variates, seq_len, pred_len, dim,
                 structure='itransformer', backbone_aux=None):
        super().__init__()
        self.structure = structure
        self.n_variates = n_variates
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dim = dim

        if structure == 'dmamba':
            self.revin = RevIN(n_variates)
            self.decomp = EMADecomposition(n_variates)
            # Seasonal path: processes L temporal tokens (causal)
            self.seasonal_embed = nn.Linear(n_variates, dim)
            self.seasonal_backbone = backbone
            self.seasonal_proj = nn.Linear(dim, n_variates)
            self.seasonal_temporal = nn.Linear(seq_len, pred_len)
            # Trend path: separate backbone for different operators
            self.trend_embed = nn.Linear(n_variates, dim)
            self.trend_backbone = backbone_aux if backbone_aux is not None else backbone
            self.trend_proj = nn.Linear(dim, n_variates)
            self.trend_temporal = nn.Linear(seq_len, pred_len)

        elif structure == 'smamba':
            # Two-stage pipeline: variate correlation → temporal processing
            self.variate_embed = nn.Linear(seq_len, dim)
            self.backbone_variate = backbone        # variate correlation (bidirectional)
            self.backbone_temporal = backbone_aux if backbone_aux is not None else backbone
            self.variate_proj = nn.Linear(dim, pred_len)

        elif structure == 'itransformer':
            # Stacked blocks: cross-variate + per-variate in each block
            self.variate_embed = nn.Linear(seq_len, dim)
            self.backbone = backbone
            self.variate_proj = nn.Linear(dim, pred_len)

        else:
            raise ValueError(f"Unknown structure: {structure}")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:  x: (B, seq_len, C) lookback window
        Returns: pred: (B, pred_len, C) forecast
        """
        if self.structure == 'dmamba':
            return self._forward_dmamba(x)
        elif self.structure == 'smamba':
            return self._forward_smamba(x)
        else:
            return self._forward_itransformer(x)

    def _forward_dmamba(self, x):
        """Approach A: Decomposition + dual flow.

        seasonal path: (B,L,C) → embed(C→D) → backbone(L tokens) → proj(D→C) → temporal(L→H)
        trend path:    same with backbone_aux
        """
        # RevIN normalize
        x = self.revin(x, mode='norm')
        # EMA decompose
        seasonal, trend = self.decomp(x)

        # Seasonal: temporal processing via backbone
        s = self.seasonal_embed(seasonal)       # (B, L, D)
        s = self.seasonal_backbone(s)           # (B, L, D)
        s = self.seasonal_proj(s)               # (B, L, C)
        s = s.transpose(1, 2)                   # (B, C, L)
        s = self.seasonal_temporal(s)           # (B, C, H)
        s = s.transpose(1, 2)                   # (B, H, C)

        # Trend: temporal processing via backbone_aux
        t = self.trend_embed(trend)             # (B, L, D)
        t = self.trend_backbone(t)              # (B, L, D)
        t = self.trend_proj(t)                  # (B, L, C)
        t = t.transpose(1, 2)                   # (B, C, L)
        t = self.trend_temporal(t)              # (B, C, H)
        t = t.transpose(1, 2)                   # (B, H, C)

        # Aggregate + denormalize
        out = s + t
        out = self.revin(out, mode='denorm')
        return out

    def _forward_smamba(self, x):
        """Approach B: Two-stage variate correlation → temporal processing.

        (B,L,C) → (B,C,L) → embed(L→D) → variate_backbone(C tokens)
                                         → temporal_backbone(C tokens)
                                         → proj(D→H) → (B,H,C)
        """
        # Variate tokenization: each variate's L timepoints become a token
        x = x.transpose(1, 2)                   # (B, C, L)
        x = self.variate_embed(x)               # (B, C, D)

        # Stage 1: Variate correlation (bidirectional, cross-variate mixing)
        x = self.backbone_variate(x)            # (B, C, D)

        # Stage 2: Temporal processing (refines temporal info in D-dim)
        x = self.backbone_temporal(x)           # (B, C, D)

        # Project each variate's D-dim repr to pred_len
        x = self.variate_proj(x)                # (B, C, H)
        x = x.transpose(1, 2)                   # (B, H, C)
        return x

    def _forward_itransformer(self, x):
        """Approach C: Stacked blocks of cross-variate + per-variate.

        (B,L,C) → (B,C,L) → embed(L→D) → backbone(N blocks) → proj(D→H) → (B,H,C)
        Each backbone block: token_mix=cross-variate, channel_mix=per-variate
        """
        # Variate tokenization
        x = x.transpose(1, 2)                   # (B, C, L)
        x = self.variate_embed(x)               # (B, C, D)

        # N stacked blocks
        x = self.backbone(x)                    # (B, C, D)

        # Project each variate to pred_len
        x = self.variate_proj(x)                # (B, C, H)
        x = x.transpose(1, 2)                   # (B, H, C)
        return x