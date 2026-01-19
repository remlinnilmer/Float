#!/usr/bin/env python3
"""
floating_block_v2.py

FloatingConvBlockV1: "nodes float on the batch wave" conv block with a concept-aligned norm.

Core idea:
- Treat normalized activations as the "wave field".
- Compute per-group *relative* wave descriptors from the current batch (no running-stat drift correction).
- Produce bounded gain/bias per group that modulates the wave without trying to "fix" it.

Key addition (v2):
- REN (Relative Energy Norm): a purpose-built "coordinate norm" for this concept.
  * Per-sample (batch-size tolerant / B=1 safe)
  * Per-group (matches gate grouping)
  * Non-centering (preserves bias/phase information)
  * Robust scale (mean-abs blended with RMS), with optional clamping
  * Optional detach of scale for stability (default True)

Enhancements (all optional via config):
- Distribution-aware features:
  * energy_entropy_g: entropy of normalized per-element energy within group (spread vs concentration)
  * excess_kurtosis_g: tail-heaviness via 4th moment
- Spatial awareness upgrades:
  * sobel_mag_g: mean Sobel gradient magnitude (directional edges) vs basic TV
- Wave smoothing:
  * avg_pool before feature extraction to reduce noise for tiny spatial dims / choppy batches
- Learnable per-feature scalers:
  * model learns which descriptors matter (lightweight, stable)
- DDP consistency:
  * optional all-reduce of final gain/bias so modulation is identical across ranks (esp. eval)

Dependency: torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Distributed helpers
# ----------------------------

def _dist_is_ready() -> bool:
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


@torch.no_grad()
def _ddp_allreduce_mean_(x: torch.Tensor) -> torch.Tensor:
    """
    In-place mean all-reduce on any tensor.
    """
    if not _dist_is_ready():
        return x
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    x.div_(float(torch.distributed.get_world_size()))
    return x


# ----------------------------
# Config
# ----------------------------

@dataclass
class FloatingBlockConfig:
    # Grouping for gate (grouped channels share one gain/bias)
    gate_groups: int = 8
    gate_hidden: int = 32

    # Bounded modulation (prevents runaway resonance)
    gain_min: float = 0.5
    gain_max: float = 2.0
    bias_max: float = 0.5

    # Normalization choice
    # - "ren": Relative Energy Norm (recommended for this concept)
    # - "rms": classic RMSNorm-like per-channel over spatial dims
    # - "gn": GroupNorm
    # - "bn": BatchNorm2d
    norm: Literal["ren", "rms", "gn", "bn"] = "ren"

    # Shared eps
    eps: float = 1e-5

    # BN only
    bn_momentum: float = 0.1

    # GN only
    gn_groups: int = 8  # for norm="gn" only

    # REN only (purpose-built coordinate norm)
    # scale = (1-ren_alpha) * mean_abs + ren_alpha * rms
    ren_alpha: float = 0.25
    # Clamp scale to avoid pathological tiny/huge scalings (0 disables a side)
    ren_scale_min: float = 0.0
    ren_scale_max: float = 0.0
    # Detach scale from gradients for stability (recommended)
    ren_detach_scale: bool = True

    # Eval behavior (BN only)
    # - "running": deterministic (BN uses running stats)
    # - "batch": use current batch wave even at eval (truer to "floating", less deterministic)
    eval_wave: Literal["running", "batch"] = "running"

    # DDP sync behavior
    # If True, all-reduce features before MLP (same "sea state" everywhere).
    sync_gate_feats: bool = False
    # If True, all-reduce final gain/bias after MLP (ensures identical modulation).
    sync_gate_outputs: bool = False

    # Optional distribution-aware features
    add_energy_entropy: bool = False   # entropy of normalized energy within group
    add_kurtosis: bool = False         # excess kurtosis via 4th moment

    # Spatial features
    add_tv: bool = True               # basic roughness (finite differences)
    add_sobel: bool = False           # Sobel gradient magnitude (directional, a bit more compute)

    # Wave smoothing before feature extraction
    wave_smooth_ks: int = 0           # 0 disables; else avg_pool kernel size (odd recommended)

    # Learnable per-feature scalers (lightweight)
    learn_feature_scalers: bool = False

    # Regularization (lightweight, optional)
    # Small penalty to discourage gains saturating at bounds; does NOT push to identity.
    gain_soft_clip_coef: float = 0.0

    # Dropout
    dropout_p: float = 0.0


# ----------------------------
# Norm
# ----------------------------

class _RMSNorm2d(nn.Module):
    """
    Simple RMS normalization for 2D conv features, per-channel over spatial dims.
    No running stats, affine=False.
    """
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        rms = torch.sqrt(x.pow(2).mean(dim=(2, 3), keepdim=True) + self.eps)
        return x / rms


class _RelativeEnergyNorm2d(nn.Module):
    """
    Relative Energy Norm (REN): a stateless "coordinate norm" designed for the floating-wave concept.

    - Per-sample (batch-size tolerant; B=1 stable)
    - Per-gate-group (matches gate grouping)
    - Non-centering (preserves phase/bias)
    - Robust scale via mean-abs blended with RMS
    - Optional scale clamping
    - Optional scale detach for stability
    """
    def __init__(
        self,
        channels: int,
        gate_groups: int,
        eps: float = 1e-5,
        alpha: float = 0.25,
        scale_min: float = 0.0,
        scale_max: float = 0.0,
        detach_scale: bool = True,
    ):
        super().__init__()
        C = int(channels)
        G = int(gate_groups)
        if G <= 0:
            raise ValueError("gate_groups must be > 0")
        if C % G != 0:
            raise ValueError("channels must be divisible by gate_groups")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in [0, 1]")

        self.channels = C
        self.gate_groups = G
        self.channels_per_group = C // G

        self.eps = float(eps)
        self.alpha = float(alpha)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.detach_scale = bool(detach_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B,G,Cg,H,W]
        B, C, H, W = x.shape
        xg = x.view(B, self.gate_groups, self.channels_per_group, H, W)

        # Robust per-sample, per-group scale
        # mean_abs: [B,G,1,1,1]
        mean_abs = xg.abs().mean(dim=(2, 3, 4), keepdim=True)
        # rms: [B,G,1,1,1]
        rms = torch.sqrt(xg.pow(2).mean(dim=(2, 3, 4), keepdim=True) + self.eps)

        scale = (1.0 - self.alpha) * mean_abs + self.alpha * rms
        scale = scale + self.eps

        # Optional clamp (0 disables a side)
        if self.scale_min > 0.0:
            scale = scale.clamp_min(self.scale_min)
        if self.scale_max > 0.0:
            scale = scale.clamp_max(self.scale_max)

        if self.detach_scale:
            scale = scale.detach()

        yg = xg / scale
        return yg.view(B, C, H, W)


# ----------------------------
# Gate
# ----------------------------

class FloatingGateGrouped(nn.Module):
    """
    Per-group gate producing (gain_g, bias_g) from batch wave descriptors.

    Features are computed from the *normalized* activation field (the wave), so they are scale-free.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        gain_min: float,
        gain_max: float,
        bias_max: float,
        learn_feature_scalers: bool,
        eps: float,
    ):
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be > 0")
        if hidden <= 0:
            raise ValueError("hidden must be > 0")
        if gain_max <= gain_min:
            raise ValueError("gain_max must be > gain_min")
        if bias_max < 0:
            raise ValueError("bias_max must be >= 0")

        self.in_dim = int(in_dim)
        self.learn_feature_scalers = bool(learn_feature_scalers)
        self.eps = float(eps)

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

        # Bounds
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.bias_max = float(bias_max)

        # Optional learnable per-feature scaling (positive)
        # scale = exp(log_scale), init 0 => scale 1
        if self.learn_feature_scalers:
            self.feat_log_scale = nn.Parameter(torch.zeros(self.in_dim))
        else:
            self.register_parameter("feat_log_scale", None)

        # Initialize close to identity
        with torch.no_grad():
            last: nn.Linear = self.mlp[-1]  # type: ignore[assignment]
            last.weight.zero_()
            last.bias.zero_()

            gmin, gmax = self.gain_min, self.gain_max
            target = float(min(max(1.0, gmin + 1e-6), gmax - 1e-6))
            s = (target - gmin) / (gmax - gmin)
            s = float(min(max(s, 1e-6), 1.0 - 1e-6))
            r = float(torch.log(torch.tensor(s / (1.0 - s))))
            last.bias[0] = r  # raw_gain
            last.bias[1] = 0.0  # raw_bias

    def forward(self, feats_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feats_g: [G,D]
        returns:
          gain_g: [G] in [gain_min, gain_max]
          bias_g: [G] in [-bias_max, bias_max]
        """
        if self.learn_feature_scalers and self.feat_log_scale is not None:
            scale = torch.exp(self.feat_log_scale).to(device=feats_g.device, dtype=feats_g.dtype)
            feats_g = feats_g * scale

        raw = self.mlp(feats_g)  # [G,2]
        raw_gain, raw_bias = raw[:, 0], raw[:, 1]

        gmin, gmax = self.gain_min, self.gain_max
        gain = gmin + (gmax - gmin) * torch.sigmoid(raw_gain)
        bias = self.bias_max * torch.tanh(raw_bias)
        return gain, bias


# ----------------------------
# Block
# ----------------------------

class FloatingConvBlockV1(nn.Module):
    """
    Conv -> Norm -> FloatingGate -> Act -> (optional residual)

    This block is intentionally *not* a controller:
    - No drift-to-running-stat correction.
    - No target energy homeostasis.
    - No EMA hysteresis blending.

    It "floats" by reading the current batch wave and modulating in-bounds.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups_conv: int = 1,
        bias: bool = False,
        activation: str = "silu",
        residual: bool = True,
        cfg: Optional[FloatingBlockConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg if cfg is not None else FloatingBlockConfig()

        if padding is None:
            padding = (kernel_size // 2) * dilation

        G = int(self.cfg.gate_groups)
        if G <= 0:
            raise ValueError("cfg.gate_groups must be > 0")
        if out_channels % G != 0:
            raise ValueError("out_channels must be divisible by cfg.gate_groups")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.gate_groups = G
        self.channels_per_group = out_channels // G

        self.residual = bool(residual)
        self.dropout_p = float(self.cfg.dropout_p)
        self.eps = float(self.cfg.eps)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups_conv,
            bias=bias,
        )

        # Norm layer
        norm_kind = self.cfg.norm.lower()
        self.norm_kind = norm_kind
        if norm_kind == "bn":
            self.norm = nn.BatchNorm2d(
                out_channels,
                eps=self.cfg.eps,
                momentum=self.cfg.bn_momentum,
                affine=False,
                track_running_stats=True,
            )
        elif norm_kind == "gn":
            if self.cfg.gn_groups <= 0:
                raise ValueError("cfg.gn_groups must be > 0")
            self.norm = nn.GroupNorm(
                num_groups=int(self.cfg.gn_groups),
                num_channels=out_channels,
                eps=self.cfg.eps,
                affine=False,
            )
        elif norm_kind == "rms":
            self.norm = _RMSNorm2d(out_channels, eps=self.cfg.eps)
        elif norm_kind == "ren":
            self.norm = _RelativeEnergyNorm2d(
                channels=out_channels,
                gate_groups=self.gate_groups,
                eps=self.cfg.eps,
                alpha=self.cfg.ren_alpha,
                scale_min=self.cfg.ren_scale_min,
                scale_max=self.cfg.ren_scale_max,
                detach_scale=self.cfg.ren_detach_scale,
            )
        else:
            raise ValueError("cfg.norm must be one of: 'ren','rms','gn','bn'")

        # Choose feature dimension based on enabled metrics
        # Base features (always):
        #   m1, e1, (m2-1), m3
        # plus optional: tv, sobel, energy_entropy, excess_kurtosis
        self.add_tv = bool(self.cfg.add_tv)
        self.add_sobel = bool(self.cfg.add_sobel)
        self.add_energy_entropy = bool(self.cfg.add_energy_entropy)
        self.add_kurtosis = bool(self.cfg.add_kurtosis)

        feat_names = ["m1", "e1", "m2c", "m3"]
        if self.add_tv:
            feat_names.append("tv")
        if self.add_sobel:
            feat_names.append("sobel")
        if self.add_energy_entropy:
            feat_names.append("energy_entropy")
        if self.add_kurtosis:
            feat_names.append("excess_kurtosis")
        self.feat_dim = len(feat_names)

        self.gate = FloatingGateGrouped(
            in_dim=self.feat_dim,
            hidden=int(self.cfg.gate_hidden),
            gain_min=float(self.cfg.gain_min),
            gain_max=float(self.cfg.gain_max),
            bias_max=float(self.cfg.bias_max),
            learn_feature_scalers=bool(self.cfg.learn_feature_scalers),
            eps=self.cfg.eps,
        )

        # Activation
        act = activation.lower()
        if act == "silu":
            self.act = nn.SiLU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=False)
        elif act == "gelu":
            self.act = nn.GELU()
        elif act in ("identity", "none", ""):
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Residual projection if needed
        self.proj = nn.Identity()
        if self.residual and (in_channels != out_channels or stride != 1):
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        # Optional freeze-to-static
        self.register_buffer("frozen_static", torch.tensor(False), persistent=True)
        self.register_buffer("static_gamma_c", torch.ones(out_channels), persistent=True)
        self.register_buffer("static_beta_c", torch.zeros(out_channels), persistent=True)

        # Diagnostics
        self.register_buffer("last_gain_g", torch.ones(G), persistent=False)
        self.register_buffer("last_bias_g", torch.zeros(G), persistent=False)
        self._last_aux = torch.tensor(0.0)

        # Sobel kernels (registered buffers, constructed lazily on first use)
        self.register_buffer("_sobel_kx", torch.empty(0), persistent=False)
        self.register_buffer("_sobel_ky", torch.empty(0), persistent=False)

    def aux_loss(self) -> torch.Tensor:
        return self._last_aux

    @torch.no_grad()
    def freeze_to_static(self) -> None:
        """
        Capture the *current* last gate into static per-channel gamma/beta and bypass gate compute.
        Useful for deployment consistency/speed.
        """
        gamma_c = self._expand_group(self.last_gain_g).detach()
        beta_c = self._expand_group(self.last_bias_g).detach()
        self.static_gamma_c.copy_(gamma_c)
        self.static_beta_c.copy_(beta_c)
        self.frozen_static.fill_(True)

    @torch.no_grad()
    def unfreeze(self) -> None:
        self.frozen_static.fill_(False)

    def _group_view(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B,C,H,W] -> [B,G,Cg,H,W]
        B, C, H, W = y.shape
        return y.view(B, self.gate_groups, self.channels_per_group, H, W)

    def _expand_group(self, v_g: torch.Tensor) -> torch.Tensor:
        # [G] -> [C]
        return v_g.view(self.gate_groups, 1).expand(self.gate_groups, self.channels_per_group).reshape(-1)

    def _maybe_smooth_wave(self, y_wave: torch.Tensor) -> torch.Tensor:
        ks = int(self.cfg.wave_smooth_ks)
        if ks <= 1:
            return y_wave
        pad = ks // 2
        # avg_pool keeps "floating" feel: it reduces noise without imposing targets
        return F.avg_pool2d(y_wave, kernel_size=ks, stride=1, padding=pad)

    def _ensure_sobel_kernels(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._sobel_kx.numel() != 0 and self._sobel_kx.device == device and self._sobel_kx.dtype == dtype:
            return
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            device=device, dtype=dtype
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]],
            device=device, dtype=dtype
        ).view(1, 1, 3, 3)
        self._sobel_kx = kx
        self._sobel_ky = ky

    def _wave_feats_fp32(self, y_wave: torch.Tensor) -> torch.Tensor:
        """
        Compute per-group wave descriptors in fp32 for AMP safety.
        y_wave: [B,C,H,W] (normalized wave field)
        returns feats_g: [G,D] fp32
        """
        eps = self.eps
        y = self._maybe_smooth_wave(y_wave)
        yg = self._group_view(y).float()  # [B,G,Cg,H,W]
        B, G, Cg, H, W = yg.shape

        # Base moments (scale-free in normalized wave)
        m1 = yg.mean(dim=(0, 2, 3, 4))                               # [G]
        e1 = yg.abs().mean(dim=(0, 2, 3, 4))                         # [G]
        m2 = (yg * yg).mean(dim=(0, 2, 3, 4))                        # [G]
        m3 = (yg * yg * yg).mean(dim=(0, 2, 3, 4))                   # [G]
        m2c = m2 - 1.0                                               # center around ~0

        feats_list = [m1, e1, m2c, m3]

        # TV roughness (cheap)
        if self.add_tv:
            if W > 1:
                dx = (yg[..., :, 1:] - yg[..., :, :-1]).abs().mean(dim=(0, 2, 3, 4))
            else:
                dx = m1.abs() * 0.0
            if H > 1:
                dy = (yg[..., 1:, :] - yg[..., :-1, :]).abs().mean(dim=(0, 2, 3, 4))
            else:
                dy = m1.abs() * 0.0
            tv = dx + dy
            feats_list.append(tv)

        # Sobel magnitude (more directional than TV; modest compute)
        if self.add_sobel:
            yy = y.float()
            self._ensure_sobel_kernels(device=yy.device, dtype=yy.dtype)
            C = yy.size(1)
            kx = self._sobel_kx.expand(C, 1, 3, 3)
            ky = self._sobel_ky.expand(C, 1, 3, 3)
            gx = F.conv2d(yy, kx, bias=None, stride=1, padding=1, groups=C)
            gy = F.conv2d(yy, ky, bias=None, stride=1, padding=1, groups=C)
            mag = torch.sqrt(gx * gx + gy * gy + eps)                # [B,C,H,W]
            mag_g = self._group_view(mag).mean(dim=(0, 2, 3, 4))      # [G]
            feats_list.append(mag_g)

        # Energy entropy: entropy of normalized per-element energy within group
        if self.add_energy_entropy:
            e = (yg * yg).reshape(B, G, Cg * H * W) + eps            # [B,G,N]
            e_sum = e.sum(dim=0)                                     # [G,N]
            p = e_sum / (e_sum.sum(dim=1, keepdim=True) + eps)       # [G,N]
            ent = -(p * torch.log(p + eps)).sum(dim=1)               # [G]
            ent = ent / torch.log(torch.tensor(float(p.size(1)), device=ent.device, dtype=ent.dtype))
            feats_list.append(ent)

        # Excess kurtosis (tail heaviness)
        if self.add_kurtosis:
            m4 = (yg.pow(4)).mean(dim=(0, 2, 3, 4))                  # [G]
            kurt = m4 / (m2 * m2 + eps) - 3.0                        # [G]
            feats_list.append(kurt)

        feats = torch.stack(feats_list, dim=1)                       # [G,D]

        # Optional DDP sync so every rank sees the same sea state
        if self.cfg.sync_gate_feats:
            _ddp_allreduce_mean_(feats)

        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = feats.clamp(min=-10.0, max=10.0)
        return feats

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Conv
        y = self.conv(x)

        # Norm -> wave field
        if self.norm_kind == "bn":
            if (not self.training) and (self.cfg.eval_wave == "batch"):
                y_wave = F.batch_norm(
                    y,
                    running_mean=None,
                    running_var=None,
                    weight=None,
                    bias=None,
                    training=True,
                    momentum=0.0,
                    eps=self.cfg.eps,
                )
            else:
                y_wave = self.norm(y)
        else:
            y_wave = self.norm(y)

        # Frozen path (deployment)
        if bool(self.frozen_static.item()):
            gamma = self.static_gamma_c.to(device=y_wave.device, dtype=y_wave.dtype).view(1, -1, 1, 1)
            beta = self.static_beta_c.to(device=y_wave.device, dtype=y_wave.dtype).view(1, -1, 1, 1)
            y_mod = y_wave * gamma + beta
            if self.dropout_p > 0.0:
                y_mod = F.dropout(y_mod, p=self.dropout_p, training=self.training)
            out_local = self.act(y_mod)
            out = out_local + self.proj(x) if self.residual else out_local
            self._last_aux = out.new_zeros(())
            if return_aux:
                return out, self._last_aux
            return out

        # Floating gate: read the wave, don’t correct it
        feats_g = self._wave_feats_fp32(y_wave)                        # [G,D] fp32
        gain_g, bias_g = self.gate(feats_g.to(y_wave.dtype))           # [G] in wave dtype

        # Optional DDP: all-reduce final modulation so ranks apply identical gain/bias
        if self.cfg.sync_gate_outputs:
            gg = gain_g.detach().clone().float()
            bb = bias_g.detach().clone().float()
            _ddp_allreduce_mean_(gg)
            _ddp_allreduce_mean_(bb)
            gain_g = gg.to(dtype=y_wave.dtype, device=y_wave.device)
            bias_g = bb.to(dtype=y_wave.dtype, device=y_wave.device)

        # Record diagnostics
        with torch.no_grad():
            self.last_gain_g.copy_(gain_g.detach().float())
            self.last_bias_g.copy_(bias_g.detach().float())

        # Expand and modulate
        gain_c = self._expand_group(gain_g).view(1, -1, 1, 1)
        bias_c = self._expand_group(bias_g).view(1, -1, 1, 1)
        y_mod = y_wave * gain_c + bias_c

        # Optional: soft clip penalty to discourage bound saturation (not identity pressure)
        aux = y_mod.new_zeros(())
        if self.cfg.gain_soft_clip_coef > 0.0:
            g = gain_g
            eps = 1e-6
            dist_to_min = (g - float(self.cfg.gain_min)).clamp_min(eps)
            dist_to_max = (float(self.cfg.gain_max) - g).clamp_min(eps)
            sat = (1.0 / dist_to_min + 1.0 / dist_to_max).mean()
            aux = aux + float(self.cfg.gain_soft_clip_coef) * sat

        if self.dropout_p > 0.0:
            y_mod = F.dropout(y_mod, p=self.dropout_p, training=self.training)

        out_local = self.act(y_mod)
        out = out_local + self.proj(x) if self.residual else out_local

        self._last_aux = aux
        if return_aux:
            return out, aux
        return out


# ----------------------------
# Quick sanity
# ----------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = FloatingBlockConfig(
        gate_groups=8,
        gate_hidden=32,
        gain_min=0.5,
        gain_max=2.0,
        bias_max=0.5,
        norm="ren",
        eps=1e-5,
        ren_alpha=0.25,
        ren_scale_min=0.0,
        ren_scale_max=0.0,
        ren_detach_scale=True,
        eval_wave="running",
        sync_gate_feats=False,
        sync_gate_outputs=False,
        add_energy_entropy=True,
        add_kurtosis=True,
        add_tv=True,
        add_sobel=False,
        wave_smooth_ks=3,
        learn_feature_scalers=True,
        gain_soft_clip_coef=0.0,
        dropout_p=0.0,
    )

    block = FloatingConvBlockV1(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=1,
        activation="silu",
        residual=True,
        cfg=cfg,
    ).to(device)

    block.train()
    opt = torch.optim.Adam(block.parameters(), lr=1e-3)

    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        xb = torch.randn(8, 3, 32, 32, device=device)
        yb, aux = block(xb, return_aux=True)
        loss = yb.mean() + aux
        loss.backward()
        opt.step()

    block.eval()
    with torch.no_grad():
        ye = block(torch.randn(1, 3, 32, 32, device=device))
        assert ye.shape == (1, 32, 32, 32)

    block.freeze_to_static()
    with torch.no_grad():
        yf = block(torch.randn(1, 3, 32, 32, device=device))
        assert yf.shape == (1, 32, 32, 32)

    print(
        "OK",
        "feat_dim=", block.feat_dim,
        "gain_g_mean=", float(block.last_gain_g.mean().cpu()),
        "bias_g_mean=", float(block.last_bias_g.mean().cpu()),
        "frozen=", bool(block.frozen_static.item()),
        "norm=", cfg.norm,
    )
