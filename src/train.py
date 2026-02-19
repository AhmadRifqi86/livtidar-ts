"""STAR + TiDAR Unified Training: Evolution-guided architecture search with LIV operators.

Papers:
  - STAR — Synthesis of Tailored Architectures (arXiv:2411.17800v1)
  - TiDAR — Think in Diffusion, Act in Autoregression (arXiv:2511.08923)

Training recipe from STAR paper (Table A.1 / A.2):
  - AdamW: lr=8e-4, β1=0.9, β2=0.95, weight_decay=0.1
  - Cosine LR decay, 500-step warmup, grad_clip=1.0
  - Batch: 0.25M tokens/step, seq_len=4096
  - Evolution: 5000 steps/candidate, pop=16, gen=18
  - Post-evolution: top-8, 20000 steps (~5B tokens)

Modes:
  Language model:  evolve / train / both
  Detection:       detect-evolve / detect-train / detect-both
  TiDAR:           tidar-evolve / tidar-train / tidar-both
  Time series:     ts-evolve / ts-train / ts-both
  TiDAR-TS:        ts-tidar-evolve / ts-tidar-train / ts-tidar-both

Usage:
    python -m src.train evolve --synthetic
    python -m src.train train  --genome_path best.json ...
    python -m src.train both   --synthetic
    python -m src.train tidar-evolve --synthetic
    python -m src.train tidar-train  --genome_path best.json --synthetic
    python -m src.train tidar-both   --synthetic
    python -m src.train ts-train --ts_dataset ETTh1 --ts_structure itransformer
    python -m src.train ts-evolve --ts_dataset ETTh1 --ts_structure dmamba
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

from core.tidar import (
    TiDARConfig, TiDARModel, compute_tidar_loss,
    TiDARTSConfig, TiDARTSModel, compute_tidar_ts_loss,
)
from core.liv import SparsityType
from core.modeldef import (
    RMSNorm, STARLanguageModel, STARVisionModel, STARVisionDetector,
    RevIN, EMADecomposition, STARLIVTSModel,
)
from src.dataload import get_dataloader, get_dataset_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



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
        self.items = items
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

            if self.augment and random.random() > 0.5:
                img_t = img_t.flip(-1)
                boxes_t[:, 0] = 1.0 - boxes_t[:, 0]

            imgs.append(img_t)
            tgts.append({'boxes': boxes_t.to(device), 'labels': labels_t.to(device)})

        return torch.stack(imgs).to(device), tgts

    @classmethod
    def from_voc(cls, root, year='2007', split='trainval', img_size=224):
        """Load PASCAL VOC."""
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
        """Load COCO."""
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
# Training & Evaluation — Language Model
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
# Training & Evaluation — Detection
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
# Training & Evaluation — TiDAR
# ============================================================================

def train_tidar_model(model, train_data, val_data, args, steps, prefix=""):
    """Train TiDAR model for `steps` steps with AMP and grad accumulation.

    Uses compute_tidar_loss() from core.tidar for the joint AR+Diffusion loss.
    Returns validation TiDAR loss (lower is better).
    """
    device = args.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    total_bs = max(1, args.batch_tokens // args.seq_len)
    micro_bs = min(total_bs, args.micro_batch)
    grad_accum = max(1, total_bs // micro_bs)

    use_amp = args.amp and device != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_every = max(1, steps // 20)

    model.train()
    running_total, running_ar, running_diff = 0.0, 0.0, 0.0

    for step in range(steps):
        lr = get_lr(step, args.warmup_steps, steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        step_total, step_ar, step_diff = 0.0, 0.0, 0.0

        for _ in range(grad_accum):
            # TiDAR only needs input_ids (no separate targets)
            x, _ = train_data.get_batch(micro_bs, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, ar_loss, diff_loss = compute_tidar_loss(model, x)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            step_total += loss.item()
            step_ar += ar_loss / grad_accum
            step_diff += diff_loss / grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_total += step_total
        running_ar += step_ar
        running_diff += step_diff
        if (step + 1) % log_every == 0:
            n = log_every
            log.info(f"{prefix}step {step+1}/{steps}  lr={lr:.2e}  "
                     f"loss={running_total/n:.4f}  ar={running_ar/n:.4f}  "
                     f"diff={running_diff/n:.4f}")
            running_total, running_ar, running_diff = 0.0, 0.0, 0.0

    val_loss = evaluate_tidar_loss(model, val_data, args)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_loss


@torch.no_grad()
def evaluate_tidar_loss(model, val_data, args, n_batches=None):
    """Validation TiDAR loss (joint AR + Diffusion)."""
    n_batches = n_batches or args.eval_batches
    device = next(model.parameters()).device
    micro_bs = min(max(1, args.batch_tokens // args.seq_len), args.micro_batch)

    model.eval()
    total_loss, count = 0.0, 0
    for _ in range(n_batches):
        x, _ = val_data.get_batch(micro_bs, device)
        with torch.amp.autocast("cuda", enabled=args.amp and str(device) != "cpu"):
            loss, _, _ = compute_tidar_loss(model, x)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


def _build_tidar_from_backbone(backbone, args):
    """Wrap a genome-built backbone in a TiDARModel."""
    cfg = TiDARConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_heads=getattr(args, 'num_heads', 8),
        max_seq_len=args.seq_len,
        alpha=getattr(args, 'tidar_alpha', 1.0),
    )
    return TiDARModel(cfg, backbone=backbone)


def _build_ts_model(genome, class_pool, args):
    """Build STARLIVTSModel from a genome, splitting it for dual-backbone structures.

    Genome splitting strategy:
      dmamba:        first half → seasonal backbone (causal),
                     second half → trend backbone (causal)
      smamba:        first half → variate-correlation backbone (bidirectional),
                     second half → temporal-processing backbone (bidirectional)
      itransformer:  full genome → single backbone (bidirectional)
    """
    structure = args.ts_structure
    n = len(genome.layers)
    mid = n // 2

    if structure == 'dmamba':
        builder = GenomeModelBuilder(class_pool, args.dim, causal=True)
        g1, g2 = Genome(genome.layers[:mid]), Genome(genome.layers[mid:])
        repair(g1, class_pool)
        repair(g2, class_pool)
        backbone     = builder.build(g1)
        backbone_aux = builder.build(g2)
    elif structure == 'smamba':
        builder = GenomeModelBuilder(class_pool, args.dim, causal=False)
        g1, g2 = Genome(genome.layers[:mid]), Genome(genome.layers[mid:])
        repair(g1, class_pool)
        repair(g2, class_pool)
        backbone     = builder.build(g1)
        backbone_aux = builder.build(g2)
    else:  # itransformer
        builder = GenomeModelBuilder(class_pool, args.dim, causal=False)
        backbone     = builder.build(genome)
        backbone_aux = None

    return STARLIVTSModel(
        backbone, args.n_variates, args.ts_seq_len, args.ts_pred_len,
        args.dim, structure=structure, backbone_aux=backbone_aux,
    )


def _build_tidar_ts_model(genome, class_pool, args):
    """Build TiDARTSModel from a genome.

    Uses the full genome as a single backbone (no splitting).
    Each LIV block gets TIDAR_HYBRID sparsity so the lookback section
    attends causally and the forecast mask section attends bidirectionally.
    """
    builder = GenomeModelBuilder(
        class_pool, args.dim,
        causal=False,                              # sparsity mask handles causality
        sparsity_type=SparsityType.TIDAR_HYBRID,   # passed to every UnifiedLIV
    )
    backbone = builder.build(genome)

    cfg = TiDARTSConfig(
        n_variates=args.n_variates,
        seq_len=args.ts_seq_len,
        pred_len=args.ts_pred_len,
        dim=args.dim,
        num_heads=getattr(args, 'num_heads', 8),
        alpha=getattr(args, 'tidar_alpha', 1.0),
    )
    return TiDARTSModel(cfg, backbone=backbone)


# ============================================================================
# Training & Evaluation — Time Series
# ============================================================================

def train_ts_model(model, train_loader, val_loader, args, steps=None, prefix=""):
    """Train time series model with DataLoader. Returns validation MSE."""
    device = args.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
    )

    use_amp = args.amp and device != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = steps or getattr(args, 'train_steps', 10000)
    log_every = max(1, total_steps // 20)

    model.train()
    running = 0.0
    step = 0

    while step < total_steps:
        for batch_x, batch_y in train_loader:
            if step >= total_steps:
                break

            lr = get_lr(step, args.warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(batch_x)
                loss = F.mse_loss(pred, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            step += 1

            if step % log_every == 0:
                log.info(f"{prefix}step {step}/{total_steps}  lr={lr:.2e}  "
                         f"mse={running/log_every:.6f}")
                running = 0.0

    val_mse, val_mae = evaluate_ts(model, val_loader, args)
    log.info(f"{prefix}val MSE={val_mse:.6f}  MAE={val_mae:.6f}")
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_mse


@torch.no_grad()
def evaluate_ts(model, dataloader, args):
    """Evaluate time series model. Returns (MSE, MAE)."""
    device = next(model.parameters()).device
    model.eval()

    total_mse, total_mae, n = 0.0, 0.0, 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.amp.autocast("cuda", enabled=args.amp and str(device) != "cpu"):
            pred = model(batch_x)
        total_mse += F.mse_loss(pred, batch_y, reduction='sum').item()
        total_mae += F.l1_loss(pred, batch_y, reduction='sum').item()
        n += batch_y.numel()

    model.train()
    mse = total_mse / max(n, 1)
    mae = total_mae / max(n, 1)
    return mse, mae


# ============================================================================
# Training & Evaluation — TiDAR-TS
# ============================================================================

def train_tidar_ts_model(model, train_loader, val_loader, args, steps=None, prefix=""):
    """Train TiDARTSModel with joint AR + Diffusion loss. Returns validation MSE."""
    device = args.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
    )

    use_amp = args.amp and device != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = steps or getattr(args, 'train_steps', 10000)
    log_every = max(1, total_steps // 20)

    model.train()
    run_total = run_ar = run_diff = 0.0
    step = 0

    while step < total_steps:
        for batch_x, batch_y in train_loader:
            if step >= total_steps:
                break

            lr = get_lr(step, args.warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, ar_l, diff_l = compute_tidar_ts_loss(model, batch_x, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            run_total += loss.item()
            run_ar    += ar_l
            run_diff  += diff_l
            step += 1

            if step % log_every == 0:
                log.info(
                    f"{prefix}step {step}/{total_steps}  lr={lr:.2e}  "
                    f"loss={run_total/log_every:.4f}  "
                    f"ar={run_ar/log_every:.4f}  diff={run_diff/log_every:.4f}"
                )
                run_total = run_ar = run_diff = 0.0

    val_mse, val_mae = evaluate_tidar_ts(model, val_loader, args)
    log.info(f"{prefix}val MSE={val_mse:.6f}  MAE={val_mae:.6f}")
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_mse


@torch.no_grad()
def evaluate_tidar_ts(model, dataloader, args):
    """Evaluate TiDARTSModel using draft (diff_head) forecast. Returns (MSE, MAE)."""
    device = next(model.parameters()).device
    model.eval()

    total_mse, total_mae, n = 0.0, 0.0, 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.amp.autocast("cuda", enabled=args.amp and str(device) != "cpu"):
            pred = model.forecast(batch_x, ar_steps=0)   # fast draft mode
        total_mse += F.mse_loss(pred, batch_y, reduction='sum').item()
        total_mae += F.l1_loss(pred, batch_y, reduction='sum').item()
        n += batch_y.numel()

    model.train()
    return total_mse / max(n, 1), total_mae / max(n, 1)


def post_tidar_ts_train(population, class_pool, train_loader, val_loader,
                        test_loader, args):
    """Select top-K Pareto-optimal genomes and train TiDAR-TS models to convergence."""
    candidates = sorted(
        [ind for ind in population if ind.rank == 0],
        key=lambda ind: ind.fitness.quality,
    )[:args.top_k]
    if not candidates:
        candidates = sorted(population, key=lambda ind: ind.fitness.quality)[:args.top_k]

    log.info(f"\n{'='*60}\nPost-evolution TiDAR-TS: "
             f"training {len(candidates)} candidates\n{'='*60}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, ind in enumerate(candidates):
        log.info(f"\n--- Candidate {i+1}/{len(candidates)} ---")
        log.info(f"Evolution quality={ind.fitness.quality:.6f}  params={ind.fitness.param_count:,}")
        classes = [g.liv_class for g in ind.genome.layers]
        log.info(f"Layer classes: {classes}")

        ts_model = _build_tidar_ts_model(ind.genome, class_pool, args)
        log.info(f"Total params: {count_params(ts_model):,}")

        val_mse = train_tidar_ts_model(ts_model, train_loader, val_loader, args,
                                       steps=args.full_train_steps,
                                       prefix=f"[C{i+1}] ")
        log.info(f"Candidate {i+1} val MSE: {val_mse:.6f}")

        test_mse, test_mae = evaluate_tidar_ts(ts_model.to(args.device), test_loader, args)
        log.info(f"Candidate {i+1} test MSE: {test_mse:.6f}  MAE: {test_mae:.6f}")
        results.append((ind, val_mse, test_mse, test_mae))

        torch.save({
            "genome": ind.genome.flatten(),
            "layer_classes": classes,
            "model_state_dict": ts_model.state_dict(),
            "val_mse": val_mse,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "dataset": args.ts_dataset,
        }, out_dir / f"tidar_ts_candidate_{i+1}.pt")

    results.sort(key=lambda x: x[2])   # sort by test MSE
    log.info(f"\n{'='*60}\nFinal TiDAR-TS results (sorted by test MSE):")
    for rank, (ind, vl, tm, tma) in enumerate(results):
        log.info(f"  #{rank+1}  val={vl:.6f}  test_mse={tm:.6f}  "
                 f"test_mae={tma:.6f}  params={ind.fitness.param_count:,}")
    log.info("=" * 60)
    return results


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
    """NSGA-II evolution loop with seed genomes, logging, and checkpoints."""
    rng = random.Random(args.seed)
    class_pool = build_class_pool(getattr(args, "include_extended", False))
    evaluator = FitnessEvaluator(
        class_pool, args.dim, quality_fn, seq_len=args.seq_len,
        measure_latency=getattr(args, "measure_latency", False),
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

    # --- Evolution loop ---
    for gen in range(args.generations):
        t0 = time.time()
        log.info(f"\n{'='*60}\nGeneration {gen+1}/{args.generations}\n{'='*60}")

        fronts = non_dominated_sort(population)
        for front in fronts:
            crowding_distance(population, front)
        population.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

        pf = [p for p in population if p.rank == 0]
        log.info(f"Pareto front: {len(pf)} individuals")
        for i, ind in enumerate(pf):
            log.info(f"  [{i}] q={ind.fitness.quality:.2f} "
                     f"p={ind.fitness.param_count:,} c={ind.fitness.kv_cache_size:,}")

        _save_checkpoint(out_dir / f"gen_{gen+1:02d}.json", population, gen + 1)

        offspring = []
        for i in range(min(args.elitism, len(population))):
            offspring.append(Individual(
                genome=population[i].genome.copy(), fitness=population[i].fitness,
            ))

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
# Post-Evolution: Train Top-K
# ============================================================================

def post_evolution_train(population, class_pool, train_data, val_data, args):
    """Select top-K Pareto-optimal genomes and train LM to convergence."""
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


def post_tidar_train(population, class_pool, train_data, val_data, args):
    """Select top-K Pareto-optimal genomes and train TiDAR to convergence."""
    candidates = sorted(
        [ind for ind in population if ind.rank == 0],
        key=lambda ind: ind.fitness.quality,
    )[:args.top_k]
    if not candidates:
        candidates = sorted(population, key=lambda ind: ind.fitness.quality)[:args.top_k]

    log.info(f"\n{'='*60}\nPost-evolution TiDAR: "
             f"training {len(candidates)} candidates\n{'='*60}")
    builder = GenomeModelBuilder(class_pool, args.dim)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, ind in enumerate(candidates):
        log.info(f"\n--- Candidate {i+1}/{len(candidates)} ---")
        log.info(f"Evolution quality={ind.fitness.quality:.2f}  params={ind.fitness.param_count:,}")
        classes = [g.liv_class for g in ind.genome.layers]
        log.info(f"Layer classes: {classes}")

        backbone = builder.build(ind.genome)
        tidar = _build_tidar_from_backbone(backbone, args)
        log.info(f"Total params (with embed): {count_params(tidar):,}")

        val_loss = train_tidar_model(tidar, train_data, val_data, args,
                                     steps=args.full_train_steps,
                                     prefix=f"[C{i+1}] ")
        log.info(f"Candidate {i+1} final TiDAR loss: {val_loss:.4f}")
        results.append((ind, val_loss))

        torch.save({
            "genome": ind.genome.flatten(),
            "layer_classes": classes,
            "model_state_dict": tidar.state_dict(),
            "tidar_loss": val_loss,
            "params": ind.fitness.param_count,
        }, out_dir / f"tidar_candidate_{i+1}.pt")

    results.sort(key=lambda x: x[1])
    log.info(f"\n{'='*60}\nFinal TiDAR results (sorted by loss):")
    for rank, (ind, vl) in enumerate(results):
        log.info(f"  #{rank+1}  loss={vl:.4f}  params={ind.fitness.param_count:,}")
    log.info("=" * 60)
    return results


def post_ts_train(population, class_pool, train_loader, val_loader, test_loader, args):
    """Select top-K Pareto-optimal genomes and train TS models to convergence."""
    candidates = sorted(
        [ind for ind in population if ind.rank == 0],
        key=lambda ind: ind.fitness.quality,
    )[:args.top_k]
    if not candidates:
        candidates = sorted(population, key=lambda ind: ind.fitness.quality)[:args.top_k]

    log.info(f"\n{'='*60}\nPost-evolution TS: "
             f"training {len(candidates)} candidates\n{'='*60}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, ind in enumerate(candidates):
        log.info(f"\n--- Candidate {i+1}/{len(candidates)} ---")
        log.info(f"Evolution quality={ind.fitness.quality:.6f}  params={ind.fitness.param_count:,}")
        classes = [g.liv_class for g in ind.genome.layers]
        log.info(f"Layer classes: {classes}")

        ts_model = _build_ts_model(ind.genome, class_pool, args)
        log.info(f"Total params: {count_params(ts_model):,}")

        val_mse = train_ts_model(ts_model, train_loader, val_loader, args,
                                 steps=args.full_train_steps,
                                 prefix=f"[C{i+1}] ")
        log.info(f"Candidate {i+1} val MSE: {val_mse:.6f}")

        # Test evaluation
        test_mse, test_mae = evaluate_ts(ts_model.to(args.device), test_loader, args)
        log.info(f"Candidate {i+1} test MSE: {test_mse:.6f}  MAE: {test_mae:.6f}")
        results.append((ind, val_mse, test_mse, test_mae))

        torch.save({
            "genome": ind.genome.flatten(),
            "layer_classes": classes,
            "model_state_dict": ts_model.state_dict(),
            "val_mse": val_mse,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "params": ind.fitness.param_count,
            "structure": args.ts_structure,
            "dataset": args.ts_dataset,
        }, out_dir / f"ts_candidate_{i+1}.pt")

    results.sort(key=lambda x: x[1])
    log.info(f"\n{'='*60}\nFinal TS results (sorted by val MSE):")
    for rank, (ind, vmse, tmse, tmae) in enumerate(results):
        log.info(f"  #{rank+1}  val_MSE={vmse:.6f}  test_MSE={tmse:.6f}  "
                 f"test_MAE={tmae:.6f}  params={ind.fitness.param_count:,}")
    log.info("=" * 60)
    return results


# ============================================================================
# Quality Functions for NSGA-II
# ============================================================================

def _make_quality_fn(args, train_data, val_data):
    """Quality function for LM NSGA: wraps backbone in STARLanguageModel."""
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


def _make_tidar_quality_fn(args, train_data, val_data):
    """Quality function for TiDAR NSGA: wraps backbone in TiDARModel."""
    if getattr(args, "dry_run", False):
        return None
    evo_steps = getattr(args, "evolution_steps", 5000)
    counter = [0]

    def quality_fn(backbone, genome):
        counter[0] += 1
        tidar = _build_tidar_from_backbone(backbone, args)
        n = count_params(tidar)
        log.info(f"[TiDAR-Eval #{counter[0]}] {n:,} params, {evo_steps} steps")
        val_loss = train_tidar_model(tidar, train_data, val_data, args,
                                     steps=evo_steps, prefix=f"[TE{counter[0]}] ")
        log.info(f"[TiDAR-Eval #{counter[0]}] loss={val_loss:.4f}")
        return val_loss

    return quality_fn


def _make_ts_quality_fn(args, train_loader, val_loader, class_pool):
    """Quality function for TS NSGA: builds STARLIVTSModel via genome splitting."""
    if getattr(args, "dry_run", False):
        return None
    evo_steps = getattr(args, "evolution_steps", 500)
    counter = [0]

    def quality_fn(backbone, genome):
        # backbone arg is ignored; we rebuild from genome to support splitting
        counter[0] += 1
        ts_model = _build_ts_model(genome, class_pool, args)
        n = count_params(ts_model)
        log.info(f"[TS-Eval #{counter[0]}] {n:,} params, {evo_steps} steps")
        val_mse = train_ts_model(ts_model, train_loader, val_loader, args,
                                 steps=evo_steps, prefix=f"[TSE{counter[0]}] ")
        log.info(f"[TS-Eval #{counter[0]}] MSE={val_mse:.6f}")
        return val_mse

    return quality_fn


def _make_tidar_ts_quality_fn(args, train_loader, val_loader, class_pool):
    """Quality function for TiDAR-TS NSGA: builds TiDARTSModel from full genome."""
    if getattr(args, "dry_run", False):
        return None
    evo_steps = getattr(args, "evolution_steps", 500)
    counter = [0]

    def quality_fn(backbone, genome):
        # backbone arg is ignored; we rebuild from genome with TIDAR_HYBRID sparsity
        counter[0] += 1
        ts_model = _build_tidar_ts_model(genome, class_pool, args)
        n = count_params(ts_model)
        log.info(f"[TiDAR-TS-Eval #{counter[0]}] {n:,} params, {evo_steps} steps")
        val_mse = train_tidar_ts_model(ts_model, train_loader, val_loader, args,
                                       steps=evo_steps, prefix=f"[TTSE{counter[0]}] ")
        log.info(f"[TiDAR-TS-Eval #{counter[0]}] MSE={val_mse:.6f}")
        return val_mse

    return quality_fn


# ============================================================================
# CLI
# ============================================================================

def _add_shared_args():
    """Create shared argument parser for all modes."""
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

    return shared


def _add_evolution_args(parser):
    """Add evolution-specific arguments to a parser."""
    parser.add_argument("--evolution_steps", type=int, default=5000,
                        help="Training steps per candidate (paper: 5000)")
    parser.add_argument("--pop_size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--mutation_prob", type=float, default=DEFAULT_MUTATION_PROB)
    parser.add_argument("--elitism", type=int, default=DEFAULT_ELITISM)
    parser.add_argument("--crossover_points", type=int, default=DEFAULT_CROSSOVER_POINTS)
    parser.add_argument("--tournament_k", type=int, default=DEFAULT_TOURNAMENT_K)
    parser.add_argument("--include_extended", action="store_true",
                        help="Include Rec-3/Rec-4 (CfC) classes 18-21 in search")
    parser.add_argument("--measure_latency", action="store_true",
                        help="Measure backbone inference latency as 4th NSGA objective")
    parser.add_argument("--no_seeds", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip training; use param count as quality proxy")


def _add_detection_args():
    """Create detection-specific argument parser."""
    det_shared = argparse.ArgumentParser(add_help=False)
    g_det = det_shared.add_argument_group("Detection")
    g_det.add_argument("--det_dataset", type=str, default="voc",
                       choices=["voc", "coco"], help="Detection dataset")
    g_det.add_argument("--det_root", type=str, default="./data",
                       help="Dataset root directory")
    g_det.add_argument("--det_year", type=str, default="2007")
    g_det.add_argument("--det_split", type=str, default="trainval")
    g_det.add_argument("--det_val_split", type=str, default="test")
    g_det.add_argument("--det_ann", type=str, default=None)
    g_det.add_argument("--det_val_ann", type=str, default=None)
    g_det.add_argument("--det_val_root", type=str, default=None)
    g_det.add_argument("--max_det_images", type=int, default=None)
    g_det.add_argument("--img_size", type=int, default=224)
    g_det.add_argument("--patch_size", type=int, default=16)
    g_det.add_argument("--num_classes", type=int, default=None)
    g_det.add_argument("--det_batch_size", type=int, default=8)
    return det_shared


def _add_tidar_args():
    """Create TiDAR-specific argument parser."""
    tidar_shared = argparse.ArgumentParser(add_help=False)
    g_tidar = tidar_shared.add_argument_group("TiDAR")
    g_tidar.add_argument("--tidar_alpha", type=float, default=1.0,
                         help="Loss balance: L = 1/(1+α) * [α * L_AR + L_Diff]")
    g_tidar.add_argument("--num_heads", type=int, default=8,
                         help="Number of attention heads for TiDAR backbone")
    return tidar_shared


def _add_ts_args():
    """Create time series-specific argument parser."""
    ts_shared = argparse.ArgumentParser(add_help=False)
    g_ts = ts_shared.add_argument_group("Time Series")
    g_ts.add_argument("--ts_dataset", type=str, default="ETTh1",
                      help="Dataset name (ETTh1, ETTh2, ETTm1, ETTm2, "
                           "Electricity, Traffic, Exchange, Weather, ILI)")
    g_ts.add_argument("--ts_seq_len", type=int, default=96,
                      help="Lookback window length")
    g_ts.add_argument("--ts_pred_len", type=int, default=96,
                      help="Forecast horizon length")
    g_ts.add_argument("--ts_structure", type=str, default="itransformer",
                      choices=["dmamba", "smamba", "itransformer"],
                      help="Model structure approach")
    g_ts.add_argument("--ts_batch_size", type=int, default=32,
                      help="Batch size for time series training")
    g_ts.add_argument("--ts_num_workers", type=int, default=4,
                      help="DataLoader workers")
    return ts_shared


def build_parser():
    p = argparse.ArgumentParser(
        description="STAR + TiDAR: Architecture Search + Training")
    sub = p.add_subparsers(dest="mode", required=True)

    shared = _add_shared_args()
    det_shared = _add_detection_args()
    tidar_shared = _add_tidar_args()
    ts_shared = _add_ts_args()

    # --- LM: evolve ---
    ep = sub.add_parser("evolve", parents=[shared],
                        help="Run NSGA-II architecture search (LM)")
    _add_evolution_args(ep)

    # --- LM: train ---
    tp = sub.add_parser("train", parents=[shared],
                        help="Train a specific genome (LM)")
    tp.add_argument("--genome_path", type=str, required=True)
    tp.add_argument("--train_steps", type=int, default=20_000)

    # --- LM: both ---
    bp = sub.add_parser("both", parents=[shared],
                        help="Evolve then train top-K (LM)")
    _add_evolution_args(bp)
    bp.add_argument("--full_train_steps", type=int, default=20_000)
    bp.add_argument("--top_k", type=int, default=8)

    # --- Detection: detect-evolve ---
    dep = sub.add_parser("detect-evolve", parents=[shared, det_shared],
                         help="NSGA-II architecture search for detection")
    _add_evolution_args(dep)
    dep.set_defaults(evolution_steps=500)

    # --- Detection: detect-train ---
    dtp = sub.add_parser("detect-train", parents=[shared, det_shared],
                         help="Train a specific genome for detection")
    dtp.add_argument("--genome_path", type=str, required=True)
    dtp.add_argument("--train_steps", type=int, default=5000)

    # --- Detection: detect-both ---
    dbp = sub.add_parser("detect-both", parents=[shared, det_shared],
                         help="Evolve + train top-K for detection")
    _add_evolution_args(dbp)
    dbp.set_defaults(evolution_steps=500)
    dbp.add_argument("--full_train_steps", type=int, default=5000)
    dbp.add_argument("--top_k", type=int, default=4)

    # --- TiDAR: tidar-evolve ---
    tep = sub.add_parser("tidar-evolve", parents=[shared, tidar_shared],
                         help="NSGA-II architecture search with TiDAR objective")
    _add_evolution_args(tep)

    # --- TiDAR: tidar-train ---
    ttp = sub.add_parser("tidar-train", parents=[shared, tidar_shared],
                         help="Train a specific genome with TiDAR")
    ttp.add_argument("--genome_path", type=str, required=True)
    ttp.add_argument("--train_steps", type=int, default=20_000)

    # --- TiDAR: tidar-both ---
    tbp = sub.add_parser("tidar-both", parents=[shared, tidar_shared],
                         help="Evolve + train top-K with TiDAR")
    _add_evolution_args(tbp)
    tbp.add_argument("--full_train_steps", type=int, default=20_000)
    tbp.add_argument("--top_k", type=int, default=8)

    # --- TS: ts-evolve ---
    tsep = sub.add_parser("ts-evolve", parents=[shared, ts_shared],
                          help="NSGA-II architecture search for time series")
    _add_evolution_args(tsep)
    tsep.set_defaults(evolution_steps=500)

    # --- TS: ts-train ---
    tstp = sub.add_parser("ts-train", parents=[shared, ts_shared],
                          help="Train a specific genome for time series")
    tstp.add_argument("--genome_path", type=str, required=True)
    tstp.add_argument("--train_steps", type=int, default=10_000)
    tstp.add_argument("--include_extended", action="store_true",
                      help="Required if genome contains Rec-3/Rec-4 (classes 18-21)")

    # --- TS: ts-both ---
    tsbp = sub.add_parser("ts-both", parents=[shared, ts_shared],
                          help="Evolve + train top-K for time series")
    _add_evolution_args(tsbp)
    tsbp.set_defaults(evolution_steps=500)
    tsbp.add_argument("--full_train_steps", type=int, default=10_000)
    tsbp.add_argument("--top_k", type=int, default=8)

    # --- TiDAR-TS: ts-tidar-evolve ---
    ttep = sub.add_parser("ts-tidar-evolve", parents=[shared, ts_shared, tidar_shared],
                          help="NSGA-II architecture search for TiDAR-TS")
    _add_evolution_args(ttep)
    ttep.set_defaults(evolution_steps=500)

    # --- TiDAR-TS: ts-tidar-train ---
    tttp = sub.add_parser("ts-tidar-train", parents=[shared, ts_shared, tidar_shared],
                          help="Train a specific genome with TiDAR-TS objective")
    tttp.add_argument("--genome_path", type=str, required=True)
    tttp.add_argument("--train_steps", type=int, default=10_000)
    tttp.add_argument("--include_extended", action="store_true",
                      help="Required if genome contains Rec-3/Rec-4 (classes 18-21)")

    # --- TiDAR-TS: ts-tidar-both ---
    ttbp = sub.add_parser("ts-tidar-both", parents=[shared, ts_shared, tidar_shared],
                          help="Evolve + train top-K with TiDAR-TS objective")
    _add_evolution_args(ttbp)
    ttbp.set_defaults(evolution_steps=500)
    ttbp.add_argument("--full_train_steps", type=int, default=10_000)
    ttbp.add_argument("--top_k", type=int, default=8)

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


def main():
    args = build_parser().parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {args.device}  Seed: {args.seed}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── Detection modes ──
    if args.mode.startswith("detect-"):
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

    # ── TiDAR modes ──
    if args.mode.startswith("tidar-"):
        train_data, val_data = load_data(args)

        if args.mode == "tidar-evolve":
            qfn = _make_tidar_quality_fn(args, train_data, val_data)
            population, pool = evolve_with_seeds(args, qfn, _get_seeds(args))

            pf = [p for p in population if p.rank == 0]
            log.info(f"\nTiDAR evolution complete. Pareto front: {len(pf)}")
            for i, ind in enumerate(pf):
                cls = [g.liv_class for g in ind.genome.layers]
                log.info(f"  [{i}] q={ind.fitness.quality:.4f} "
                         f"p={ind.fitness.param_count:,}  classes={cls}")

        elif args.mode == "tidar-train":
            genome = load_genome(args.genome_path, args.num_layers)
            pool = build_class_pool()
            backbone = GenomeModelBuilder(pool, args.dim).build(genome)
            tidar = _build_tidar_from_backbone(backbone, args)
            log.info(f"TiDAR params: {count_params(tidar):,}")

            val_loss = train_tidar_model(
                tidar, train_data, val_data, args, steps=args.train_steps)
            log.info(f"Final TiDAR loss: {val_loss:.4f}")

            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save({
                "genome": genome.flatten(),
                "model_state_dict": tidar.state_dict(),
                "tidar_loss": val_loss,
            }, out / "trained_tidar.pt")

        elif args.mode == "tidar-both":
            qfn = _make_tidar_quality_fn(args, train_data, val_data)
            population, pool = evolve_with_seeds(args, qfn, _get_seeds(args))
            post_tidar_train(population, pool, train_data, val_data, args)

        return

    # ── Time series modes ──
    if args.mode.startswith("ts-"):
        # Load time series data
        info = get_dataset_info(args.ts_dataset)
        if not info.get("exists"):
            raise FileNotFoundError(
                f"Dataset '{args.ts_dataset}' not found. Run ./download_data.sh")
        args.n_variates = info["n_variates"]
        log.info(f"TS dataset: {args.ts_dataset}  variates={args.n_variates}  "
                 f"structure={args.ts_structure}  seq={args.ts_seq_len}  pred={args.ts_pred_len}")

        train_loader, val_loader, test_loader, ts_scaler = get_dataloader(
            dataset=args.ts_dataset,
            seq_len=args.ts_seq_len,
            pred_len=args.ts_pred_len,
            batch_size=args.ts_batch_size,
            num_workers=args.ts_num_workers,
        )

        # For inverted approaches (smamba/itransformer), backbone processes
        # variate tokens (C tokens) → causal=False.
        # For dmamba, backbone processes temporal sequence → causal=True.
        causal = (args.ts_structure == 'dmamba')

        # Set seq_len for KV cache estimation in NSGA
        # dmamba: backbone sees seq_len temporal tokens
        # smamba/itransformer: backbone sees n_variates tokens
        if causal:
            args.seq_len = args.ts_seq_len
        else:
            args.seq_len = args.n_variates

        pool = build_class_pool(getattr(args, "include_extended", False))

        if args.mode == "ts-evolve":
            qfn = _make_ts_quality_fn(args, train_loader, val_loader, pool)
            population, _ = evolve_with_seeds(
                args, qfn, _get_seeds(args), causal=causal)

            pf = [p for p in population if p.rank == 0]
            log.info(f"\nTS evolution complete. Pareto front: {len(pf)}")
            for i, ind in enumerate(pf):
                cls = [g.liv_class for g in ind.genome.layers]
                log.info(f"  [{i}] q={ind.fitness.quality:.6f} "
                         f"p={ind.fitness.param_count:,}  classes={cls}")

        elif args.mode == "ts-train":
            genome = load_genome(args.genome_path, args.num_layers)
            ts_model = _build_ts_model(genome, pool, args)
            log.info(f"TS params: {count_params(ts_model):,}")

            val_mse = train_ts_model(
                ts_model, train_loader, val_loader, args, steps=args.train_steps)

            # Test evaluation
            test_mse, test_mae = evaluate_ts(
                ts_model.to(args.device), test_loader, args)
            log.info(f"Final val MSE: {val_mse:.6f}")
            log.info(f"Final test MSE: {test_mse:.6f}  MAE: {test_mae:.6f}")

            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save({
                "genome": genome.flatten(),
                "model_state_dict": ts_model.state_dict(),
                "val_mse": val_mse,
                "test_mse": test_mse,
                "test_mae": test_mae,
                "structure": args.ts_structure,
                "dataset": args.ts_dataset,
            }, out / "trained_ts.pt")

        elif args.mode == "ts-both":
            qfn = _make_ts_quality_fn(args, train_loader, val_loader, pool)
            population, _ = evolve_with_seeds(
                args, qfn, _get_seeds(args), causal=causal)
            post_ts_train(population, pool, train_loader, val_loader,
                          test_loader, args)

        return

    # ── TiDAR-TS modes ──
    if args.mode.startswith("ts-tidar-"):
        info = get_dataset_info(args.ts_dataset)
        if not info.get("exists"):
            raise FileNotFoundError(
                f"Dataset '{args.ts_dataset}' not found. Run ./download_data.sh")
        args.n_variates = info["n_variates"]
        log.info(f"TiDAR-TS dataset: {args.ts_dataset}  variates={args.n_variates}  "
                 f"seq={args.ts_seq_len}  pred={args.ts_pred_len}  "
                 f"alpha={getattr(args, 'tidar_alpha', 1.0)}")

        train_loader, val_loader, test_loader, _ = get_dataloader(
            dataset=args.ts_dataset,
            seq_len=args.ts_seq_len,
            pred_len=args.ts_pred_len,
            batch_size=args.ts_batch_size,
            num_workers=args.ts_num_workers,
        )

        pool = build_class_pool(getattr(args, "include_extended", False))
        # Backbone processes L+H tokens total; use seq_len for KV-cache estimate
        args.seq_len = args.ts_seq_len + args.ts_pred_len

        if args.mode == "ts-tidar-evolve":
            qfn = _make_tidar_ts_quality_fn(args, train_loader, val_loader, pool)
            population, _ = evolve_with_seeds(
                args, qfn, _get_seeds(args), causal=False)

            pf = [p for p in population if p.rank == 0]
            log.info(f"\nTiDAR-TS evolution complete. Pareto front: {len(pf)}")
            for i, ind in enumerate(pf):
                cls = [g.liv_class for g in ind.genome.layers]
                log.info(f"  [{i}] q={ind.fitness.quality:.6f} "
                         f"p={ind.fitness.param_count:,}  classes={cls}")

        elif args.mode == "ts-tidar-train":
            genome = load_genome(args.genome_path, args.num_layers)
            ts_model = _build_tidar_ts_model(genome, pool, args)
            log.info(f"TiDAR-TS params: {count_params(ts_model):,}")

            val_mse = train_tidar_ts_model(
                ts_model, train_loader, val_loader, args, steps=args.train_steps)

            test_mse, test_mae = evaluate_tidar_ts(
                ts_model.to(args.device), test_loader, args)
            log.info(f"Final val MSE: {val_mse:.6f}")
            log.info(f"Final test MSE: {test_mse:.6f}  MAE: {test_mae:.6f}")

            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save({
                "genome": genome.flatten(),
                "model_state_dict": ts_model.state_dict(),
                "val_mse": val_mse,
                "test_mse": test_mse,
                "test_mae": test_mae,
                "dataset": args.ts_dataset,
                "alpha": getattr(args, 'tidar_alpha', 1.0),
            }, out / "trained_tidar_ts.pt")

        elif args.mode == "ts-tidar-both":
            qfn = _make_tidar_ts_quality_fn(args, train_loader, val_loader, pool)
            population, _ = evolve_with_seeds(
                args, qfn, _get_seeds(args), causal=False)
            post_tidar_ts_train(population, pool, train_loader, val_loader,
                                test_loader, args)

        return

    # ── Language model modes ──
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