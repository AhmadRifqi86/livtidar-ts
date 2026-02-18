"""STAR Training: Evolution-guided architecture search with LIV operators.

Paper: STAR — Synthesis of Tailored Architectures (arXiv:2411.17800v1)

Training recipe from paper (Table A.1 / A.2):
  - AdamW: lr=8e-4, β1=0.9, β2=0.95, weight_decay=0.1
  - Cosine LR decay, 500-step warmup, grad_clip=1.0
  - Batch: 0.25M tokens/step, seq_len=4096
  - Evolution: 5000 steps/candidate, pop=16, gen=18
  - Post-evolution: top-8, 20000 steps (~5B tokens)

Usage:
    python train.py evolve --synthetic                  # quick test
    python train.py evolve --data_path tokens.pt        # real data
    python train.py train  --genome_path best.json ...  # train one genome
    python train.py both   --synthetic                  # evolve + train top-K
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nsga import (
    build_class_pool, random_genome, repair,
    Genome, LayerGene, Individual, FitnessResult,
    GenomeModelBuilder, FitnessEvaluator,
    count_params, estimate_kv_cache,
    non_dominated_sort, crowding_distance,
    tournament_select, crossover, mutate,
    DEFAULT_POP_SIZE, DEFAULT_GENERATIONS, DEFAULT_MUTATION_PROB,
    DEFAULT_ELITISM, DEFAULT_CROSSOVER_POINTS, DEFAULT_TOURNAMENT_K,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Language Model Wrapper
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


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


class STARVisionModel(nn.Module):
    """Vision model: patch_embed -> LIV backbone (causal=False) -> pool -> head.

    Build the backbone with causal=False:
        backbone = GenomeModelBuilder(pool, dim, causal=False).build(genome)
        model = STARVisionModel(backbone, num_classes=10, dim=64)
    """

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
        # [B, C, H, W] -> [B, num_patches, dim]
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.backbone(x)
        # Global average pool -> classify
        x = x.mean(dim=1)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss


# ============================================================================
# Object Detection: Utilities & Model
# ============================================================================

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]


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
    """Object detection: patch_embed -> LIV backbone (causal=False) -> detection heads.

    Each patch token predicts objectness, class, and bounding box (cx, cy, w, h).
    GT boxes are assigned to the patch containing their center.

    Build the backbone with causal=False:
        backbone = GenomeModelBuilder(pool, dim, causal=False).build(genome)
        model = STARVisionDetector(backbone, num_classes=20, dim=256)
    """

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
        B = imgs.size(0)
        # [B, C, H, W] -> [B, num_patches, dim]
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.backbone(x)
        x = self.norm(x)

        cls_logits = self.cls_head(x)                  # [B, P, num_classes]
        obj_logits = self.obj_head(x).squeeze(-1)      # [B, P]
        bbox_pred = self.bbox_head(x).sigmoid()         # [B, P, 4] in [0,1]

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
                # Assign each GT box to the patch containing its center
                cx, cy = boxes[:, 0], boxes[:, 1]
                gx = (cx * self.grid_size).long().clamp(0, self.grid_size - 1)
                gy = (cy * self.grid_size).long().clamp(0, self.grid_size - 1)
                patch_idx = gy * self.grid_size + gx

                # Deduplicate: keep last assignment per patch
                uniq, inv = torch.unique(patch_idx, return_inverse=True)
                a_boxes = torch.zeros(len(uniq), 4, device=device)
                a_labels = torch.zeros(len(uniq), dtype=torch.long, device=device)
                for i in range(len(patch_idx)):
                    a_boxes[inv[i]] = boxes[i]
                    a_labels[inv[i]] = labels[i]

                obj_target[uniq] = 1.0

                # Bbox loss: L1 + GIoU
                pred_b = bbox_pred[b, uniq]
                l1 = F.l1_loss(pred_b, a_boxes, reduction='mean')
                giou = 1.0 - _box_giou_paired(pred_b, a_boxes).mean()

                # Classification loss on positive patches
                cls_loss = F.cross_entropy(cls_logits[b, uniq], a_labels)

                total_loss = total_loss + cls_loss + l1 * 5.0 + giou * 2.0

            # Objectness focal loss on all patches
            total_loss = total_loss + _sigmoid_focal_loss(obj_logits[b], obj_target)

        return total_loss / B


# ============================================================================
# Cosine LR with Linear Warmup (Paper Table A.1)
# ============================================================================

def get_lr(step, warmup, total, max_lr, min_lr_ratio=0.1):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return max_lr * min_lr_ratio
    t = (step - warmup) / (total - warmup)
    return max_lr * (min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * t)))


# ============================================================================
# Data Loading
# ============================================================================

class TokenDataset:
    """Simple random-access token dataset for causal LM training."""

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    @classmethod
    def from_file(cls, path, seq_len):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            data = data.get("input_ids", data.get("tokens", next(iter(data.values()))))
        return cls(torch.as_tensor(data, dtype=torch.long).flatten(), seq_len)

    @classmethod
    def synthetic(cls, num_tokens, vocab_size, seq_len):
        return cls(torch.randint(0, vocab_size, (num_tokens,)), seq_len)

    def get_batch(self, batch_size, device="cpu"):
        hi = len(self.tokens) - self.seq_len - 1
        ix = torch.randint(0, hi, (batch_size,))
        x = torch.stack([self.tokens[i : i + self.seq_len] for i in ix])
        y = torch.stack([self.tokens[i + 1 : i + self.seq_len + 1] for i in ix])
        return x.to(device), y.to(device)


def load_data(args):
    """Load train/val token data from file, HuggingFace, or synthetic."""
    if args.synthetic:
        n = args.synthetic_tokens or 10_000_000
        train = TokenDataset.synthetic(n, args.vocab_size, args.seq_len)
        val = TokenDataset.synthetic(n // 10, args.vocab_size, args.seq_len)
        log.info(f"Synthetic data: {n:,} train / {n // 10:,} val tokens")
        return train, val

    if args.data_path:
        train = TokenDataset.from_file(args.data_path, args.seq_len)
        if args.val_path:
            val = TokenDataset.from_file(args.val_path, args.seq_len)
        else:
            sp = int(len(train.tokens) * 0.9)
            val = TokenDataset(train.tokens[sp:], args.seq_len)
            train = TokenDataset(train.tokens[:sp], args.seq_len)
        log.info(f"File data: {len(train.tokens):,} train / {len(val.tokens):,} val")
        return train, val

    if args.hf_dataset:
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
        except ImportError:
            raise RuntimeError("pip install datasets transformers")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        args.vocab_size = len(tokenizer)
        log.info(f"Loading {args.hf_dataset} ({args.tokenizer}, vocab={args.vocab_size})")

        ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
        target = args.hf_tokens or 50_000_000
        toks = []
        for ex in ds:
            text = ex.get("text", ex.get("content", ""))
            if text:
                toks.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            if len(toks) >= target:
                break

        all_t = torch.tensor(toks[:target], dtype=torch.long)
        sp = int(len(all_t) * 0.9)
        train = TokenDataset(all_t[:sp], args.seq_len)
        val = TokenDataset(all_t[sp:], args.seq_len)
        log.info(f"HF data: {sp:,} train / {len(all_t) - sp:,} val tokens")
        return train, val

    raise ValueError("Specify --data_path, --hf_dataset, or --synthetic")


# ============================================================================
# Detection Data Loading
# ============================================================================

class DetectionDataset:
    """Object detection dataset supporting VOC and COCO formats."""

    def __init__(self, items, img_size, augment=False):
        self.items = items  # list of (img_path, boxes[N,4], labels[N])
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def get_batch(self, batch_size, device="cpu"):
        from PIL import Image as PILImage

        indices = torch.randint(0, len(self.items), (batch_size,)).tolist()
        imgs, tgts = [], []
        for idx in indices:
            img_path, boxes, labels = self.items[idx]
            img = PILImage.open(img_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size), PILImage.BILINEAR)
            buf = bytearray(img.tobytes())
            img_t = torch.frombuffer(buf, dtype=torch.uint8).clone()
            img_t = img_t.reshape(self.img_size, self.img_size, 3)
            img_t = img_t.permute(2, 0, 1).float() / 255.0

            boxes_t = boxes.clone()
            labels_t = labels.clone()

            # Random horizontal flip
            if self.augment and random.random() > 0.5:
                img_t = img_t.flip(-1)
                boxes_t[:, 0] = 1.0 - boxes_t[:, 0]

            imgs.append(img_t)
            tgts.append({'boxes': boxes_t.to(device), 'labels': labels_t.to(device)})

        return torch.stack(imgs).to(device), tgts

    @classmethod
    def from_voc(cls, root, year='2007', split='trainval', img_size=224):
        """Load PASCAL VOC. Expects VOCdevkit/VOC{year}/ under root."""
        import xml.etree.ElementTree as ET

        voc_dir = os.path.join(root, 'VOCdevkit', f'VOC{year}')
        img_dir = os.path.join(voc_dir, 'JPEGImages')
        ann_dir = os.path.join(voc_dir, 'Annotations')
        split_file = os.path.join(voc_dir, 'ImageSets', 'Main', f'{split}.txt')

        with open(split_file) as f:
            image_ids = [line.strip() for line in f if line.strip()]

        items = []
        for img_id in image_ids:
            img_path = os.path.join(img_dir, f'{img_id}.jpg')
            ann_path = os.path.join(ann_dir, f'{img_id}.xml')
            tree = ET.parse(ann_path)
            root_el = tree.getroot()
            sz = root_el.find('size')
            w = float(sz.find('width').text)
            h = float(sz.find('height').text)
            if w < 1 or h < 1:
                continue

            boxes, labels = [], []
            for obj in root_el.findall('object'):
                diff = obj.find('difficult')
                if diff is not None and diff.text == '1':
                    continue
                name = obj.find('name').text
                if name not in VOC_CLASSES:
                    continue
                bb = obj.find('bndbox')
                x1 = float(bb.find('xmin').text) / w
                y1 = float(bb.find('ymin').text) / h
                x2 = float(bb.find('xmax').text) / w
                y2 = float(bb.find('ymax').text) / h
                boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                labels.append(VOC_CLASSES.index(name))

            if boxes:
                items.append((img_path,
                              torch.tensor(boxes, dtype=torch.float32),
                              torch.tensor(labels, dtype=torch.long)))

        log.info(f"VOC{year} {split}: {len(items)} images")
        return cls(items, img_size, augment=split in ('trainval', 'train'))

    @classmethod
    def from_coco(cls, img_dir, ann_file, img_size=224, max_images=None):
        """Load COCO. img_dir has images, ann_file is instances JSON."""
        with open(ann_file) as f:
            coco = json.load(f)

        cat_ids = sorted([c['id'] for c in coco['categories']])
        cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

        img_anns = {}
        for ann in coco['annotations']:
            img_anns.setdefault(ann['image_id'], []).append(ann)

        items = []
        for img_info in coco['images']:
            if max_images and len(items) >= max_images:
                break
            iid = img_info['id']
            if iid not in img_anns:
                continue
            img_path = os.path.join(img_dir, img_info['file_name'])
            w, h = img_info['width'], img_info['height']
            if w < 1 or h < 1:
                continue

            boxes, labels = [], []
            for ann in img_anns[iid]:
                if ann.get('iscrowd', 0):
                    continue
                ci = cat_to_idx.get(ann['category_id'])
                if ci is None:
                    continue
                bx, by, bw, bh = ann['bbox']
                boxes.append([(bx + bw / 2) / w, (by + bh / 2) / h, bw / w, bh / h])
                labels.append(ci)

            if boxes:
                items.append((img_path,
                              torch.tensor(boxes, dtype=torch.float32),
                              torch.tensor(labels, dtype=torch.long)))

        num_classes = len(cat_ids)
        log.info(f"COCO {os.path.basename(ann_file)}: {len(items)} images, "
                 f"{num_classes} classes")
        ds = cls(items, img_size, augment='train' in os.path.basename(ann_file))
        ds.num_classes = num_classes
        return ds


def load_detection_data(args):
    """Load detection train/val datasets."""
    ds = args.det_dataset
    if ds == 'voc':
        year = args.det_year
        train = DetectionDataset.from_voc(
            args.det_root, year, args.det_split, args.img_size)
        val = DetectionDataset.from_voc(
            args.det_root, year, args.det_val_split, args.img_size)
        args.num_classes = len(VOC_CLASSES)
    elif ds == 'coco':
        train = DetectionDataset.from_coco(
            args.det_root, args.det_ann, args.img_size, args.max_det_images)
        if args.det_val_ann:
            val = DetectionDataset.from_coco(
                args.det_val_root or args.det_root, args.det_val_ann,
                args.img_size, args.max_det_images)
        else:
            n = len(train.items)
            sp = int(n * 0.9)
            val_items = train.items[sp:]
            train.items = train.items[:sp]
            val = DetectionDataset(val_items, args.img_size, augment=False)
        if not hasattr(args, 'num_classes') or args.num_classes is None:
            args.num_classes = train.num_classes
    else:
        raise ValueError(f"Unknown detection dataset: {ds}")

    log.info(f"Detection data: {len(train)} train / {len(val)} val, "
             f"{args.num_classes} classes")
    return train, val


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_model(model, train_data, val_data, args, steps, prefix=""):
    """Train for `steps` steps. Returns validation perplexity."""
    device = args.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Gradient accumulation: total batch = batch_tokens / seq_len sequences
    total_bs = max(1, args.batch_tokens // args.seq_len)
    micro_bs = min(total_bs, args.micro_batch)
    grad_accum = max(1, total_bs // micro_bs)

    use_amp = args.amp and device != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_every = max(1, steps // 20)

    model.train()
    running = 0.0

    for step in range(steps):
        lr = get_lr(step, args.warmup_steps, steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(grad_accum):
            x, y = train_data.get_batch(micro_bs, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running += step_loss
        if (step + 1) % log_every == 0:
            log.info(f"{prefix}step {step+1}/{steps}  lr={lr:.2e}  loss={running/log_every:.4f}")
            running = 0.0

    ppl = evaluate_ppl(model, val_data, args)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl


@torch.no_grad()
def evaluate_ppl(model, val_data, args, n_batches=None):
    """Validation perplexity."""
    n_batches = n_batches or args.eval_batches
    device = next(model.parameters()).device
    micro_bs = min(max(1, args.batch_tokens // args.seq_len), args.micro_batch)

    model.eval()
    total_loss, total_tok = 0.0, 0
    for _ in range(n_batches):
        x, y = val_data.get_batch(micro_bs, device)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type != "cpu"
                                 if hasattr(device, "type") else args.amp and str(device) != "cpu"):
            _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tok += y.numel()

    model.train()
    avg = total_loss / max(total_tok, 1)
    return math.exp(min(avg, 20.0))


# ============================================================================
# Detection Training & Evaluation
# ============================================================================

def train_detector(model, train_data, val_data, args, steps, prefix=""):
    """Train detector for `steps` steps. Returns validation loss."""
    device = args.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
    )

    micro_bs = args.micro_batch
    total_bs = getattr(args, 'det_batch_size', 8)
    grad_accum = max(1, total_bs // micro_bs)
    use_amp = args.amp and device != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_every = max(1, steps // 20)

    model.train()
    running = 0.0

    for step in range(steps):
        lr = get_lr(step, args.warmup_steps, steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(grad_accum):
            imgs, targets = train_data.get_batch(micro_bs, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(imgs, targets)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running += step_loss
        if (step + 1) % log_every == 0:
            log.info(f"{prefix}step {step+1}/{steps}  lr={lr:.2e}  "
                     f"loss={running/log_every:.4f}")
            running = 0.0

    val_loss = evaluate_det_loss(model, val_data, args)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_loss


@torch.no_grad()
def evaluate_det_loss(model, val_data, args, n_batches=None):
    """Validation loss for detection."""
    n_batches = n_batches or args.eval_batches
    device = next(model.parameters()).device
    micro_bs = args.micro_batch

    model.eval()
    total_loss, count = 0.0, 0
    for _ in range(n_batches):
        imgs, targets = val_data.get_batch(micro_bs, device)
        with torch.amp.autocast("cuda", enabled=args.amp and str(device) != "cpu"):
            _, loss = model(imgs, targets)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


# ============================================================================
# Seed Genomes — Known-Good Architectures (Paper Section 4.1)
# ============================================================================

def seed_transformer_pp(n):
    """Transformer++: alternating SA-1 (attention) and GMemless (SwiGLU FFN)."""
    return Genome([LayerGene(1 if i % 2 == 0 else 9, i, 1, i, 0) for i in range(n)])


def seed_striped_mamba(n):
    """StripedMamba hybrid: [SA-1, Rec-1, GMemless] repeating."""
    pat = [1, 5, 9]
    return Genome([LayerGene(pat[i % 3], i, 1, i, 0) for i in range(n)])


def seed_hybrid_conv(n):
    """All-families hybrid: [SA-1, GConv-1, Rec-1, GMemless] repeating."""
    pat = [1, 7, 5, 9]
    return Genome([LayerGene(pat[i % 4], i, 1, i, 0) for i in range(n)])


# ============================================================================
# NSGA-II Evolution with Seed Support & Checkpointing
# ============================================================================

def evolve_with_seeds(args, quality_fn, seed_genomes=None, **evaluator_kwargs):
    """NSGA-II evolution loop with seed genomes, logging, and checkpoints.

    Uses all genetic operators from nsga.py; adds seed initialization and
    per-generation checkpointing.  Pass causal=False for detection tasks.
    """
    rng = random.Random(args.seed)
    class_pool = build_class_pool(getattr(args, "include_extended", False))
    evaluator = FitnessEvaluator(
        class_pool, args.dim, quality_fn, seq_len=args.seq_len,
        **evaluator_kwargs,
    )
    n_layers = args.num_layers
    pop_size = args.pop_size

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Initial population: seeds + random ---
    population = []
    if seed_genomes:
        for g in seed_genomes:
            repair(g, class_pool)
            log.info(f"Evaluating seed: {[l.liv_class for l in g.layers[:6]]}...")
            f = evaluator.evaluate(g)
            population.append(Individual(genome=g, fitness=f))
            log.info(f"  quality={f.quality:.2f}  params={f.param_count:,}")

    while len(population) < pop_size:
        g = random_genome(n_layers, class_pool, rng)
        repair(g, class_pool)
        log.info(f"Evaluating random genome {len(population)+1}/{pop_size}...")
        f = evaluator.evaluate(g)
        population.append(Individual(genome=g, fitness=f))
        log.info(f"  quality={f.quality:.2f}  params={f.param_count:,}")

    # --- Evolution loop (Paper: 18 generations, pop=16) ---
    for gen in range(args.generations):
        t0 = time.time()
        log.info(f"\n{'='*60}\nGeneration {gen+1}/{args.generations}\n{'='*60}")

        # 1. Non-dominated sort + crowding distance
        fronts = non_dominated_sort(population)
        for front in fronts:
            crowding_distance(population, front)
        population.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

        # Log Pareto front
        pf = [p for p in population if p.rank == 0]
        log.info(f"Pareto front: {len(pf)} individuals")
        for i, ind in enumerate(pf):
            log.info(f"  [{i}] q={ind.fitness.quality:.2f} "
                     f"p={ind.fitness.param_count:,} c={ind.fitness.kv_cache_size:,}")

        # Checkpoint
        _save_checkpoint(out_dir / f"gen_{gen+1:02d}.json", population, gen + 1)

        # 2. Create offspring
        offspring = []

        # Elitism (Paper: 2 elite individuals)
        for i in range(min(args.elitism, len(population))):
            offspring.append(Individual(
                genome=population[i].genome.copy(), fitness=population[i].fitness,
            ))

        # Crossover + mutation (Paper: 2-point crossover, 10% mutation)
        child_idx = len(offspring)
        while len(offspring) < pop_size:
            p1 = tournament_select(population, args.tournament_k, rng)
            p2 = tournament_select(population, args.tournament_k, rng)
            c1, c2 = crossover(p1.genome, p2.genome, args.crossover_points, rng)

            mutate(c1, args.mutation_prob, class_pool, rng)
            mutate(c2, args.mutation_prob, class_pool, rng)
            repair(c1, class_pool)
            repair(c2, class_pool)

            log.info(f"  Offspring {len(offspring)+1}/{pop_size}...")
            f1 = evaluator.evaluate(c1)
            offspring.append(Individual(genome=c1, fitness=f1))
            log.info(f"    q={f1.quality:.2f} p={f1.param_count:,}")

            if len(offspring) < pop_size:
                f2 = evaluator.evaluate(c2)
                offspring.append(Individual(genome=c2, fitness=f2))
                log.info(f"    q={f2.quality:.2f} p={f2.param_count:,}")

        population = offspring[:pop_size]
        log.info(f"Generation {gen+1} done ({time.time()-t0:.0f}s)")

    # Final sort
    fronts = non_dominated_sort(population)
    for front in fronts:
        crowding_distance(population, front)
    population.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

    _save_checkpoint(out_dir / "final.json", population, args.generations)
    return population, class_pool


def _save_checkpoint(path, population, generation):
    data = {
        "generation": generation,
        "individuals": [
            {
                "genome": ind.genome.flatten(),
                "fitness": {
                    "quality": ind.fitness.quality,
                    "param_count": ind.fitness.param_count,
                    "kv_cache_size": ind.fitness.kv_cache_size,
                } if ind.fitness else None,
                "rank": ind.rank,
            }
            for ind in population
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved: {path}")


def load_genome(path, num_layers):
    """Load genome from JSON checkpoint (takes best-quality individual)."""
    with open(path) as f:
        data = json.load(f)
    if "individuals" in data:
        best = min(data["individuals"], key=lambda x: x["fitness"]["quality"])
        flat = best["genome"]
    elif "genome" in data:
        flat = data["genome"]
    else:
        raise ValueError(f"Cannot parse genome from {path}")
    return Genome.from_flat(flat, num_layers)


# ============================================================================
# Post-Evolution: Train Top-K (Paper Section 4.2 — 5B tokens)
# ============================================================================

def post_evolution_train(population, class_pool, train_data, val_data, args):
    """Select top-K Pareto-optimal genomes and train to convergence."""
    candidates = sorted(
        [ind for ind in population if ind.rank == 0],
        key=lambda ind: ind.fitness.quality,
    )[:args.top_k]
    if not candidates:
        candidates = sorted(population, key=lambda ind: ind.fitness.quality)[:args.top_k]

    log.info(f"\n{'='*60}\nPost-evolution: training {len(candidates)} candidates\n{'='*60}")
    builder = GenomeModelBuilder(class_pool, args.dim)
    out_dir = Path(args.out_dir)
    results = []

    for i, ind in enumerate(candidates):
        log.info(f"\n--- Candidate {i+1}/{len(candidates)} ---")
        log.info(f"Evolution quality={ind.fitness.quality:.2f}  params={ind.fitness.param_count:,}")
        classes = [g.liv_class for g in ind.genome.layers]
        log.info(f"Layer classes: {classes}")

        backbone = builder.build(ind.genome)
        lm = STARLanguageModel(backbone, args.vocab_size, args.dim)
        log.info(f"Total params (with embed): {count_params(lm):,}")

        ppl = train_model(lm, train_data, val_data, args,
                          steps=args.full_train_steps, prefix=f"[C{i+1}] ")
        log.info(f"Candidate {i+1} final PPL: {ppl:.2f}")
        results.append((ind, ppl))

        torch.save({
            "genome": ind.genome.flatten(),
            "layer_classes": classes,
            "model_state_dict": lm.state_dict(),
            "ppl": ppl,
            "params": ind.fitness.param_count,
        }, out_dir / f"candidate_{i+1}.pt")

    results.sort(key=lambda x: x[1])
    log.info(f"\n{'='*60}\nFinal results (sorted by PPL):")
    for rank, (ind, ppl) in enumerate(results):
        log.info(f"  #{rank+1}  PPL={ppl:.2f}  params={ind.fitness.param_count:,}")
    log.info("=" * 60)
    return results


# ============================================================================
# Detection: NSGA Quality & Post-Evolution Training
# ============================================================================

def _make_detect_quality_fn(args, train_data, val_data):
    """Quality function for detection NSGA: wraps backbone in detector."""
    if getattr(args, "dry_run", False):
        return None
    evo_steps = getattr(args, "evolution_steps", 500)
    counter = [0]

    def quality_fn(backbone, genome):
        counter[0] += 1
        det = STARVisionDetector(
            backbone, args.num_classes, args.dim,
            img_size=args.img_size, patch_size=args.patch_size,
        )
        n = count_params(det)
        log.info(f"[DetEval #{counter[0]}] {n:,} params, {evo_steps} steps")
        val_loss = train_detector(det, train_data, val_data, args,
                                  steps=evo_steps, prefix=f"[DE{counter[0]}] ")
        log.info(f"[DetEval #{counter[0]}] val_loss={val_loss:.4f}")
        return val_loss

    return quality_fn


def post_det_train(population, class_pool, train_data, val_data, args):
    """Select top-K Pareto-optimal genomes and train detectors to convergence."""
    candidates = sorted(
        [ind for ind in population if ind.rank == 0],
        key=lambda ind: ind.fitness.quality,
    )[:args.top_k]
    if not candidates:
        candidates = sorted(population,
                            key=lambda ind: ind.fitness.quality)[:args.top_k]

    log.info(f"\n{'='*60}\nPost-evolution detection: "
             f"training {len(candidates)} candidates\n{'='*60}")
    builder = GenomeModelBuilder(class_pool, args.dim, causal=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, ind in enumerate(candidates):
        log.info(f"\n--- Candidate {i+1}/{len(candidates)} ---")
        classes = [g.liv_class for g in ind.genome.layers]
        log.info(f"Layer classes: {classes}")

        backbone = builder.build(ind.genome)
        det = STARVisionDetector(
            backbone, args.num_classes, args.dim,
            img_size=args.img_size, patch_size=args.patch_size,
        )
        log.info(f"Total params: {count_params(det):,}")

        val_loss = train_detector(det, train_data, val_data, args,
                                  steps=args.full_train_steps,
                                  prefix=f"[C{i+1}] ")
        log.info(f"Candidate {i+1} val_loss: {val_loss:.4f}")
        results.append((ind, val_loss))

        torch.save({
            'genome': ind.genome.flatten(),
            'layer_classes': classes,
            'model_state_dict': det.state_dict(),
            'val_loss': val_loss,
            'params': ind.fitness.param_count,
        }, out_dir / f"det_candidate_{i+1}.pt")

    results.sort(key=lambda x: x[1])
    log.info(f"\n{'='*60}\nFinal detection results (sorted by val_loss):")
    for rank, (ind, vl) in enumerate(results):
        log.info(f"  #{rank+1}  loss={vl:.4f}  params={ind.fitness.param_count:,}")
    log.info("=" * 60)
    return results


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="STAR Architecture Search + Training")
    sub = p.add_subparsers(dest="mode", required=True)

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    g_model = shared.add_argument_group("Model (Paper: 125M scale)")
    g_model.add_argument("--num_layers", type=int, default=24, help="LIV layers")
    g_model.add_argument("--dim", type=int, default=768, help="Model width")
    g_model.add_argument("--vocab_size", type=int, default=50304)
    g_model.add_argument("--seq_len", type=int, default=4096)

    g_train = shared.add_argument_group("Training (Paper Table A.1)")
    g_train.add_argument("--lr", type=float, default=8e-4)
    g_train.add_argument("--beta1", type=float, default=0.9)
    g_train.add_argument("--beta2", type=float, default=0.95)
    g_train.add_argument("--weight_decay", type=float, default=0.1)
    g_train.add_argument("--grad_clip", type=float, default=1.0)
    g_train.add_argument("--warmup_steps", type=int, default=500)
    g_train.add_argument("--batch_tokens", type=int, default=250_000,
                         help="Tokens per optimizer step (paper: 0.25M)")
    g_train.add_argument("--micro_batch", type=int, default=4,
                         help="Sequences per gradient accumulation micro-step")
    g_train.add_argument("--amp", action="store_true", help="FP16 mixed precision")
    g_train.add_argument("--eval_batches", type=int, default=50)

    g_data = shared.add_argument_group("Data")
    g_data.add_argument("--data_path", type=str, help="Pre-tokenized .pt file")
    g_data.add_argument("--val_path", type=str, help="Validation .pt file")
    g_data.add_argument("--hf_dataset", type=str, help="HuggingFace dataset name")
    g_data.add_argument("--hf_split", type=str, default="train")
    g_data.add_argument("--hf_tokens", type=int, default=50_000_000)
    g_data.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    g_data.add_argument("--synthetic", action="store_true", help="Random tokens")
    g_data.add_argument("--synthetic_tokens", type=int, default=None)

    g_sys = shared.add_argument_group("System")
    g_sys.add_argument("--device", type=str, default=None)
    g_sys.add_argument("--seed", type=int, default=42)
    g_sys.add_argument("--out_dir", type=str, default="star_output")

    # --- evolve ---
    ep = sub.add_parser("evolve", parents=[shared],
                        help="Run NSGA-II architecture search")
    ep.add_argument("--evolution_steps", type=int, default=5000,
                    help="Training steps per candidate (paper: 5000)")
    ep.add_argument("--pop_size", type=int, default=DEFAULT_POP_SIZE)
    ep.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    ep.add_argument("--mutation_prob", type=float, default=DEFAULT_MUTATION_PROB)
    ep.add_argument("--elitism", type=int, default=DEFAULT_ELITISM)
    ep.add_argument("--crossover_points", type=int, default=DEFAULT_CROSSOVER_POINTS)
    ep.add_argument("--tournament_k", type=int, default=DEFAULT_TOURNAMENT_K)
    ep.add_argument("--include_extended", action="store_true")
    ep.add_argument("--no_seeds", action="store_true")
    ep.add_argument("--dry_run", action="store_true",
                    help="Skip training; use param count as quality proxy")

    # --- train ---
    tp = sub.add_parser("train", parents=[shared],
                        help="Train a specific genome")
    tp.add_argument("--genome_path", type=str, required=True,
                    help="JSON file from evolution checkpoint")
    tp.add_argument("--train_steps", type=int, default=20_000)

    # --- both ---
    bp = sub.add_parser("both", parents=[shared],
                        help="Evolve then train top-K")
    bp.add_argument("--evolution_steps", type=int, default=5000)
    bp.add_argument("--full_train_steps", type=int, default=20_000)
    bp.add_argument("--top_k", type=int, default=8,
                    help="Post-evolution: train top-K candidates (paper: 8)")
    bp.add_argument("--pop_size", type=int, default=DEFAULT_POP_SIZE)
    bp.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    bp.add_argument("--mutation_prob", type=float, default=DEFAULT_MUTATION_PROB)
    bp.add_argument("--elitism", type=int, default=DEFAULT_ELITISM)
    bp.add_argument("--crossover_points", type=int, default=DEFAULT_CROSSOVER_POINTS)
    bp.add_argument("--tournament_k", type=int, default=DEFAULT_TOURNAMENT_K)
    bp.add_argument("--include_extended", action="store_true")
    bp.add_argument("--no_seeds", action="store_true")
    bp.add_argument("--dry_run", action="store_true")

    # --- Detection shared args ---
    det_shared = argparse.ArgumentParser(add_help=False)
    g_det = det_shared.add_argument_group("Detection")
    g_det.add_argument("--det_dataset", type=str, default="voc",
                       choices=["voc", "coco"], help="Detection dataset")
    g_det.add_argument("--det_root", type=str, default="./data",
                       help="Dataset root directory")
    g_det.add_argument("--det_year", type=str, default="2007",
                       help="VOC year (2007 or 2012)")
    g_det.add_argument("--det_split", type=str, default="trainval",
                       help="VOC training split")
    g_det.add_argument("--det_val_split", type=str, default="test",
                       help="VOC validation split")
    g_det.add_argument("--det_ann", type=str, default=None,
                       help="COCO train annotation JSON path")
    g_det.add_argument("--det_val_ann", type=str, default=None,
                       help="COCO val annotation JSON path")
    g_det.add_argument("--det_val_root", type=str, default=None,
                       help="COCO val images directory (if different from det_root)")
    g_det.add_argument("--max_det_images", type=int, default=None,
                       help="Max images to load (for quick tests)")
    g_det.add_argument("--img_size", type=int, default=224,
                       help="Input image size")
    g_det.add_argument("--patch_size", type=int, default=16,
                       help="Patch size for vision detector")
    g_det.add_argument("--num_classes", type=int, default=None,
                       help="Number of classes (auto-detected from dataset)")
    g_det.add_argument("--det_batch_size", type=int, default=8,
                       help="Total images per optimizer step")

    # --- detect-evolve ---
    dep = sub.add_parser("detect-evolve", parents=[shared, det_shared],
                         help="NSGA-II architecture search for object detection")
    dep.add_argument("--evolution_steps", type=int, default=500)
    dep.add_argument("--pop_size", type=int, default=DEFAULT_POP_SIZE)
    dep.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    dep.add_argument("--mutation_prob", type=float, default=DEFAULT_MUTATION_PROB)
    dep.add_argument("--elitism", type=int, default=DEFAULT_ELITISM)
    dep.add_argument("--crossover_points", type=int, default=DEFAULT_CROSSOVER_POINTS)
    dep.add_argument("--tournament_k", type=int, default=DEFAULT_TOURNAMENT_K)
    dep.add_argument("--include_extended", action="store_true")
    dep.add_argument("--no_seeds", action="store_true")
    dep.add_argument("--dry_run", action="store_true",
                     help="Skip training; use param count as quality proxy")

    # --- detect-train ---
    dtp = sub.add_parser("detect-train", parents=[shared, det_shared],
                         help="Train a specific genome for detection")
    dtp.add_argument("--genome_path", type=str, required=True,
                     help="JSON file from evolution checkpoint")
    dtp.add_argument("--train_steps", type=int, default=5000)

    # --- detect-both ---
    dbp = sub.add_parser("detect-both", parents=[shared, det_shared],
                         help="Evolve + train top-K for detection")
    dbp.add_argument("--evolution_steps", type=int, default=500)
    dbp.add_argument("--full_train_steps", type=int, default=5000)
    dbp.add_argument("--top_k", type=int, default=4,
                     help="Post-evolution: train top-K candidates")
    dbp.add_argument("--pop_size", type=int, default=DEFAULT_POP_SIZE)
    dbp.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    dbp.add_argument("--mutation_prob", type=float, default=DEFAULT_MUTATION_PROB)
    dbp.add_argument("--elitism", type=int, default=DEFAULT_ELITISM)
    dbp.add_argument("--crossover_points", type=int, default=DEFAULT_CROSSOVER_POINTS)
    dbp.add_argument("--tournament_k", type=int, default=DEFAULT_TOURNAMENT_K)
    dbp.add_argument("--include_extended", action="store_true")
    dbp.add_argument("--no_seeds", action="store_true")
    dbp.add_argument("--dry_run", action="store_true")

    return p


# ============================================================================
# Main Entry Points
# ============================================================================

def _get_seeds(args):
    if getattr(args, "no_seeds", False):
        return None
    return [
        seed_transformer_pp(args.num_layers),
        seed_striped_mamba(args.num_layers),
        seed_hybrid_conv(args.num_layers),
    ]


def _make_quality_fn(args, train_data, val_data):
    if getattr(args, "dry_run", False):
        return None
    evo_steps = getattr(args, "evolution_steps", 5000)
    counter = [0]

    def quality_fn(backbone, genome):
        counter[0] += 1
        lm = STARLanguageModel(backbone, args.vocab_size, args.dim)
        n = count_params(lm)
        log.info(f"[Eval #{counter[0]}] {n:,} params, {evo_steps} steps")
        ppl = train_model(lm, train_data, val_data, args,
                          steps=evo_steps, prefix=f"[E{counter[0]}] ")
        log.info(f"[Eval #{counter[0]}] PPL={ppl:.2f}")
        return ppl

    return quality_fn


def main():
    args = build_parser().parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {args.device}  Seed: {args.seed}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- Detection modes ---
    if args.mode.startswith("detect-"):
        # Set seq_len to num_patches for KV cache estimation
        args.seq_len = (args.img_size // args.patch_size) ** 2
        train_det, val_det = load_detection_data(args)

        if args.mode == "detect-evolve":
            det_qfn = _make_detect_quality_fn(args, train_det, val_det)
            population, pool = evolve_with_seeds(
                args, det_qfn, _get_seeds(args), causal=False,
            )
            pf = [p for p in population if p.rank == 0]
            log.info(f"\nDetection evolution complete. Pareto front: {len(pf)}")
            for i, ind in enumerate(pf):
                cls = [g.liv_class for g in ind.genome.layers]
                log.info(f"  [{i}] q={ind.fitness.quality:.4f} "
                         f"p={ind.fitness.param_count:,}  classes={cls}")

        elif args.mode == "detect-train":
            genome = load_genome(args.genome_path, args.num_layers)
            pool = build_class_pool()
            backbone = GenomeModelBuilder(pool, args.dim, causal=False).build(genome)
            det = STARVisionDetector(
                backbone, args.num_classes, args.dim,
                img_size=args.img_size, patch_size=args.patch_size,
            )
            log.info(f"Detection params: {count_params(det):,}")

            val_loss = train_detector(
                det, train_det, val_det, args, steps=args.train_steps)
            log.info(f"Final val_loss: {val_loss:.4f}")

            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save({
                'genome': genome.flatten(),
                'model_state_dict': det.state_dict(),
                'val_loss': val_loss,
            }, out / "trained_detector.pt")

        elif args.mode == "detect-both":
            det_qfn = _make_detect_quality_fn(args, train_det, val_det)
            population, pool = evolve_with_seeds(
                args, det_qfn, _get_seeds(args), causal=False,
            )
            post_det_train(population, pool, train_det, val_det, args)

        return

    # --- Language model modes ---
    train_data, val_data = load_data(args)

    if args.mode == "evolve":
        qfn = _make_quality_fn(args, train_data, val_data)
        population, pool = evolve_with_seeds(args, qfn, _get_seeds(args))

        pf = [p for p in population if p.rank == 0]
        log.info(f"\nEvolution complete. Pareto front: {len(pf)}")
        for i, ind in enumerate(pf):
            cls = [g.liv_class for g in ind.genome.layers]
            log.info(f"  [{i}] q={ind.fitness.quality:.2f} "
                     f"p={ind.fitness.param_count:,}  classes={cls}")

    elif args.mode == "train":
        genome = load_genome(args.genome_path, args.num_layers)
        pool = build_class_pool()
        backbone = GenomeModelBuilder(pool, args.dim).build(genome)
        lm = STARLanguageModel(backbone, args.vocab_size, args.dim)
        log.info(f"Params: {count_params(lm):,}")

        ppl = train_model(lm, train_data, val_data, args, steps=args.train_steps)
        log.info(f"Final PPL: {ppl:.2f}")

        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({
            "genome": genome.flatten(),
            "model_state_dict": lm.state_dict(),
            "ppl": ppl,
        }, out / "trained_model.pt")

    elif args.mode == "both":
        qfn = _make_quality_fn(args, train_data, val_data)
        population, pool = evolve_with_seeds(args, qfn, _get_seeds(args))
        post_evolution_train(population, pool, train_data, val_data, args)


if __name__ == "__main__":
    main()