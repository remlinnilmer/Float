#!/usr/bin/env python3
"""
cifar10_colab_a100_dualbench.py (BN-safe)

Key fix:
- If a model contains BatchNorm2d, validation uses the RAW model by default
  (EMA + BN running-stats mismatch can destroy accuracy and blow up CE loss).

Still:
- Floating first, ResNet18 second
- Same recipe, same epoch count
- Validation every epoch
- A100 speed knobs (bf16 autocast, TF32, cudnn benchmark)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

from FloatingBlock import FloatingConvBlockV1, FloatingBlockConfig


# -----------------------------
# Utils
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def has_batchnorm(model: nn.Module) -> bool:
    return any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


@torch.no_grad()
def update_ema_params(ema: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema.named_parameters())
    model_params = dict(model.named_parameters())
    for k, ev in ema_params.items():
        mv = model_params.get(k, None)
        if mv is None:
            continue
        ev.mul_(decay).add_(mv, alpha=1.0 - decay)

    ema_bufs = dict(ema.named_buffers())
    model_bufs = dict(model.named_buffers())
    for k, eb in ema_bufs.items():
        mb = model_bufs.get(k, None)
        if mb is None:
            continue
        eb.copy_(mb)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, v: float, n: int = 1) -> None:
        self.sum += float(v) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


# -----------------------------
# Mixup / CutMix (pickle-safe collate)
# -----------------------------

def _one_hot(labels: torch.Tensor, num_classes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y = torch.zeros(labels.size(0), num_classes, device=device, dtype=dtype)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    return y


def _rand_bbox(W: int, H: int, lam: float) -> Tuple[int, int, int, int]:
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


class MixupCutmixCollate:
    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float,
        cutmix_alpha: float,
        prob: float,
        label_smoothing: float,
    ) -> None:
        self.num_classes = int(num_classes)
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.prob = float(prob)
        self.label_smoothing = float(label_smoothing)

    def __call__(self, batch):
        images, labels = zip(*batch)
        x = torch.stack(images, dim=0)
        y = torch.tensor(labels, dtype=torch.long)

        do_mix = (self.prob > 0.0) and (self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0) and (random.random() < self.prob)
        if not do_mix:
            y_soft = _one_hot(y, self.num_classes, device=x.device, dtype=torch.float32)
            return self._ls(x, y_soft)

        use_cutmix = (self.cutmix_alpha > 0.0) and (self.mixup_alpha <= 0.0 or random.random() < 0.5)

        if use_cutmix:
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
            perm = torch.randperm(x.size(0))
            x_perm = x[perm]
            y_perm = y[perm]

            B, C, H, W = x.size()
            x1, y1, x2, y2 = _rand_bbox(W, H, lam)
            x[:, :, y1:y2, x1:x2] = x_perm[:, :, y1:y2, x1:x2]

            area = (x2 - x1) * (y2 - y1)
            lam = 1.0 - area / float(W * H)

            y_a = _one_hot(y, self.num_classes, device=x.device, dtype=torch.float32)
            y_b = _one_hot(y_perm, self.num_classes, device=x.device, dtype=torch.float32)
            y_soft = y_a * lam + y_b * (1.0 - lam)
            return self._ls(x, y_soft)

        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        perm = torch.randperm(x.size(0))
        x_perm = x[perm]
        y_perm = y[perm]

        x = x * lam + x_perm * (1.0 - lam)

        y_a = _one_hot(y, self.num_classes, device=x.device, dtype=torch.float32)
        y_b = _one_hot(y_perm, self.num_classes, device=x.device, dtype=torch.float32)
        y_soft = y_a * lam + y_b * (1.0 - lam)
        return self._ls(x, y_soft)

    def _ls(self, x: torch.Tensor, y_soft: torch.Tensor):
        if self.label_smoothing > 0.0:
            y_soft = y_soft * (1.0 - self.label_smoothing) + self.label_smoothing / float(self.num_classes)
        return x, y_soft


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        return -(target_probs * logp).sum(dim=1).mean()


# -----------------------------
# Models
# -----------------------------

def make_norm(norm: str, channels: int) -> nn.Module:
    n = norm.lower()
    if n == "bn":
        return nn.BatchNorm2d(channels)
    if n == "gn":
        g = 32
        while channels % g != 0 and g > 1:
            g //= 2
        return nn.GroupNorm(g, channels)
    if n == "rms":
        class _RMS(nn.Module):
            def __init__(self, eps: float = 1e-5) -> None:
                super().__init__()
                self.eps = eps
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                rms = torch.sqrt(x.pow(2).mean(dim=(2, 3), keepdim=True) + self.eps)
                return x / rms
        return _RMS()
    raise ValueError(f"Unsupported norm: {norm} (use bn|gn|rms)")


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, norm: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = make_norm(norm, out_ch)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = make_norm(norm, out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                make_norm(norm, out_ch),
            )
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act2(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, norm: str = "bn", widths=(64, 128, 256, 512), blocks=(2, 2, 2, 2)) -> None:
        super().__init__()
        w1, w2, w3, w4 = widths
        b1, b2, b3, b4 = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(3, w1, 3, stride=1, padding=1, bias=False),
            make_norm(norm, w1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(w1, w1, blocks=b1, stride=1, norm=norm)
        self.layer2 = self._make_layer(w1, w2, blocks=b2, stride=2, norm=norm)
        self.layer3 = self._make_layer(w2, w3, blocks=b3, stride=2, norm=norm)
        self.layer4 = self._make_layer(w3, w4, blocks=b4, stride=2, norm=norm)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w4, num_classes)
        self._init()

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int, norm: str) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride, norm=norm)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1, norm=norm))
        return nn.Sequential(*layers)

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class FloatingNetCIFAR(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        cfg: Optional[FloatingBlockConfig] = None,
        widths=(64, 128, 256, 512),
        blocks=(4, 4, 4, 4),
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else FloatingBlockConfig()

        w1, w2, w3, w4 = widths
        b1, b2, b3, b4 = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(3, w1, 3, stride=1, padding=1, bias=False),
            nn.SiLU(),
        )

        self.stage1 = self._make_stage(w1, w1, n=b1, stride=1)
        self.stage2 = self._make_stage(w1, w2, n=b2, stride=2)
        self.stage3 = self._make_stage(w2, w3, n=b3, stride=2)
        self.stage4 = self._make_stage(w3, w4, n=b4, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w4, num_classes)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def _make_stage(self, in_ch: int, out_ch: int, n: int, stride: int) -> nn.Sequential:
        layers = [FloatingConvBlockV1(in_ch, out_ch, 3, stride=stride, activation="silu", residual=True, cfg=self.cfg)]
        for _ in range(1, n):
            layers.append(FloatingConvBlockV1(out_ch, out_ch, 3, stride=1, activation="silu", residual=True, cfg=self.cfg))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# -----------------------------
# LR schedule
# -----------------------------

def cosine_warmup_lr(epoch: int, step: int, steps_per_epoch: int, base_lr: float, min_lr: float, warmup_epochs: int, total_epochs: int) -> float:
    t = epoch + step / max(1, steps_per_epoch)
    if warmup_epochs > 0 and t < warmup_epochs:
        return base_lr * (t / warmup_epochs)
    t2 = (t - warmup_epochs) / max(1e-8, (total_epochs - warmup_epochs))
    t2 = min(max(t2, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t2))


# -----------------------------
# Data
# -----------------------------

def build_transforms(randaugment_n: int, randaugment_m: int):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tfms = [
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if randaugment_n > 0:
        train_tfms.append(T.RandAugment(num_ops=randaugment_n, magnitude=randaugment_m))
    train_tfms.extend([T.ToTensor(), T.Normalize(mean, std)])
    test_tfms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose(train_tfms), test_tfms


# -----------------------------
# Train / Eval
# -----------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype: str) -> Tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    ce = nn.CrossEntropyLoss()

    use_amp = (device.type == "cuda")
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=dtype):
                logits = model(xb)
                loss = ce(logits, yb)
        else:
            logits = model(xb)
            loss = ce(logits, yb)

        acc = accuracy_top1(logits, yb)
        loss_meter.update(loss.item(), xb.size(0))
        acc_meter.update(acc, xb.size(0))

    return loss_meter.avg, acc_meter.avg


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        return -(target_probs * logp).sum(dim=1).mean()


def maybe_compile(model: nn.Module, do_compile: bool) -> nn.Module:
    if not do_compile:
        return model
    try:
        return torch.compile(model, mode="max-autotune")
    except TypeError:
        return torch.compile(model)


def train_one_epoch(
    raw_model: nn.Module,
    fwd_model: nn.Module,
    ema: Optional[nn.Module],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    scaler: Optional[torch.amp.GradScaler],
) -> Tuple[float, float]:
    raw_model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    soft_ce = SoftTargetCrossEntropy()

    steps_per_epoch = len(loader)
    use_amp = (device.type == "cuda") and (scaler is not None)
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    for step, (xb, y_soft) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        y_soft = y_soft.to(device, non_blocking=True)

        lr = cosine_warmup_lr(epoch, step, steps_per_epoch, args.lr, args.min_lr, args.warmup_epochs, args.epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=dtype):
                logits = fwd_model(xb)
                loss = soft_ce(logits, y_soft)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = fwd_model(xb)
            loss = soft_ce(logits, y_soft)
            loss.backward()
            optimizer.step()

        y_hard = y_soft.argmax(dim=1)
        acc = accuracy_top1(logits.detach(), y_hard)

        loss_meter.update(loss.item(), xb.size(0))
        acc_meter.update(acc, xb.size(0))

        if ema is not None:
            update_ema_params(ema, raw_model, decay=args.ema)

    return loss_meter.avg, acc_meter.avg


def run_experiment(
    name: str,
    build_fn,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[float, int]:
    raw_model = build_fn().to(device)
    fwd_model = maybe_compile(raw_model, args.compile)

    # EMA model mirrors raw_model (never compiled)
    use_ema = (not args.no_ema) and (0.0 < args.ema < 1.0)
    ema = None
    if use_ema:
        ema = build_fn().to(device)
        ema.load_state_dict(raw_model.state_dict())
        for p in ema.parameters():
            p.requires_grad_(False)
        ema.eval()

    # BN-safe: for models with BatchNorm, validate on RAW model by default
    bn_present = has_batchnorm(raw_model)
    if bn_present and not args.eval_ema_for_bn:
        eval_pref = "RAW (BN-safe)"
    else:
        eval_pref = "EMA" if ema is not None else "RAW"
    print(f"[{name}] Eval preference: {eval_pref}")

    optimizer = torch.optim.SGD(raw_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    best_epoch = -1

    eval_model = raw_model if (bn_present and not args.eval_ema_for_bn) else (ema if ema is not None else raw_model)
    vloss, vacc = evaluate(eval_model, test_loader, device, args.amp_dtype)
    print(f"[{name}] Init: val_loss={vloss:.4f} val_acc={vacc*100:.2f}%")

    t0 = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(raw_model, fwd_model, ema, train_loader, optimizer, device, epoch, args, scaler)

        eval_model = raw_model if (bn_present and not args.eval_ema_for_bn) else (ema if ema is not None else raw_model)
        vloss, vacc = evaluate(eval_model, test_loader, device, args.amp_dtype)

        if vacc > best_acc:
            best_acc = vacc
            best_epoch = epoch

        dt = time.time() - t0
        print(
            f"[{name}] Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc~={train_acc*100:.2f}% | "
            f"val_loss={vloss:.4f} val_acc={vacc*100:.2f}% | "
            f"best={best_acc*100:.2f}%@{best_epoch+1} | "
            f"elapsed={dt/60:.1f}m"
        )

    print(f"[{name}] Done. Best val_acc={best_acc*100:.2f}% at epoch {best_epoch+1}.")
    return best_acc, best_epoch + 1


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--auto-lr-scale", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--randaugment-n", type=int, default=2)
    parser.add_argument("--randaugment-m", type=int, default=9)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--ema", type=float, default=0.9999)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--compile", action="store_true")

    # BN-safe toggle: default False (do NOT eval EMA if BN present)
    parser.add_argument("--eval-ema-for-bn", action="store_true")

    parser.add_argument("--resnet-norm", type=str, default="bn", choices=["bn", "gn", "rms"])
    parser.add_argument("--f-blocks", type=str, default="4,4,4,4")
    parser.add_argument("--f-widths", type=str, default="64,128,256,512")

    parser.add_argument("--gate-groups", type=int, default=8)
    parser.add_argument("--gate-hidden", type=int, default=32)
    parser.add_argument("--gain-min", type=float, default=0.5)
    parser.add_argument("--gain-max", type=float, default=2.0)
    parser.add_argument("--bias-max", type=float, default=0.5)
    parser.add_argument("--ren-alpha", type=float, default=0.25)
    parser.add_argument("--learn-feature-scalers", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
        try:
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            try:
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    if args.auto_lr_scale:
        args.lr = args.lr * (args.batch_size / 128.0)

    train_tfm, test_tfm = build_transforms(args.randaugment_n, args.randaugment_m)

    train_set = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_tfm)
    test_set = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_tfm)

    collate = MixupCutmixCollate(
        num_classes=10,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        prob=args.mix_prob,
        label_smoothing=args.label_smoothing,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1024,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None,
    )

    f_blocks = tuple(int(x.strip()) for x in args.f_blocks.split(","))
    f_widths = tuple(int(x.strip()) for x in args.f_widths.split(","))
    if len(f_blocks) != 4 or len(f_widths) != 4:
        raise ValueError("--f-blocks and --f-widths must have 4 comma-separated ints")

    f_cfg = FloatingBlockConfig(
        gate_groups=args.gate_groups,
        gate_hidden=args.gate_hidden,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        bias_max=args.bias_max,
        norm="ren",
        ren_alpha=args.ren_alpha,
        add_tv=True,
        learn_feature_scalers=args.learn_feature_scalers,
        dropout_p=0.0,
        gain_soft_clip_coef=0.0,
    )

    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")
    print("Floating config:", asdict(f_cfg))
    print("Floating blocks:", f_blocks, "widths:", f_widths)

    def build_floating():
        return FloatingNetCIFAR(num_classes=10, cfg=f_cfg, widths=f_widths, blocks=f_blocks)

    def build_resnet():
        return ResNetCIFAR(num_classes=10, norm=args.resnet_norm, widths=(64, 128, 256, 512), blocks=(2, 2, 2, 2))

    run_experiment("FLOATING(depth-matched)", build_floating, train_loader, test_loader, device, args)
    run_experiment("RESNET18", build_resnet, train_loader, test_loader, device, args)


if __name__ == "__main__":
    main()
