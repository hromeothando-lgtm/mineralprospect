"""
FractalProspect-Net v4
======================
Dual-branch patch-based CNN for mineral prospectivity mapping.

Branch 1 — Spectral (15×15×35):
    Multi-scale conv (3×3, 5×5, 7×7) → concat → ResBlocks → CBAM → GAP → 256-d

Branch 2 — Stochastic (15×15×24):
    6 LSI maps (Theorem 1) + 18 fractal maps (Theorem 2)
    Conv → ResBlocks → CBAM → GAP → 128-d

Fusion:
    Cross-branch attention gate → concat 384-d → MLP → MC Dropout → sigmoid

References:
    Cheng 2007 (Theorem 1 — LSI), Aguilar-Ayala 2024 (Theorem 2 — fractal)
    Wang & Zuo 2022 (cross-branch attention), Lin et al 2017 (Focal Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import (
    N_SPECTRAL, N_STOCHASTIC,
    SPECTRAL_BRANCH_DIM, STOCHASTIC_BRANCH_DIM,
    FUSION_DIM, MLP_DIMS, DROPOUT_RATE, PATCH_SIZE
)


# ── Building Blocks ───────────────────────────────────────────────

class ResBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN + skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention (CBAM channel branch)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        r = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """CBAM spatial attention branch."""
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module — channel then spatial."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


# ── Branch 1: Multi-Scale Spectral CNN ───────────────────────────

class SpectralBranch(nn.Module):
    """
    Input: (B, 35, 15, 15)
    Three parallel conv streams at 3×3, 5×5, 7×7 → concat 192ch
    → 3 ResBlocks → CBAM → GAP → 256-d vector
    """
    def __init__(self, in_channels=N_SPECTRAL, out_dim=SPECTRAL_BRANCH_DIM):
        super().__init__()
        self.stream3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stream5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stream7 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.bn_fuse = nn.BatchNorm2d(192)
        self.res_blocks = nn.Sequential(
            ResBlock(192), ResBlock(192), ResBlock(192))
        self.cbam = CBAM(192)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(192, out_dim)

    def forward(self, x):
        s3 = self.stream3(x)
        s5 = self.stream5(x)
        s7 = self.stream7(x)
        f  = F.relu(self.bn_fuse(torch.cat([s3, s5, s7], dim=1)), inplace=True)
        f  = self.res_blocks(f)
        f  = self.cbam(f)
        f  = self.gap(f).flatten(1)
        return self.proj(f)


# ── Branch 2: Stochastic CNN ──────────────────────────────────────

class StochasticBranch(nn.Module):
    """
    Input: (B, 24, 15, 15) — 6 LSI + 18 fractal maps
    Conv → 2 ResBlocks → CBAM → GAP → 128-d vector
    """
    def __init__(self, in_channels=N_STOCHASTIC, out_dim=STOCHASTIC_BRANCH_DIM):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.res_blocks = nn.Sequential(ResBlock(64), ResBlock(64))
        self.cbam = CBAM(64)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        f = self.entry(x)
        f = self.res_blocks(f)
        f = self.cbam(f)
        f = self.gap(f).flatten(1)
        return self.proj(f)


# ── Cross-Branch Attention Gate ───────────────────────────────────

class CrossBranchGate(nn.Module):
    """
    Learns scalar weights w1, w2 for spectral and stochastic branches.
    Gate is differentiable — can zero-weight a noisy branch automatically.
    Based on joint singularity-based weighting (Wang & Zuo 2022).
    """
    def __init__(self, spectral_dim=SPECTRAL_BRANCH_DIM,
                 stochastic_dim=STOCHASTIC_BRANCH_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(spectral_dim + stochastic_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, f_spec, f_stoch):
        concat = torch.cat([f_spec, f_stoch], dim=1)
        weights = self.gate(concat)          # (B, 2)
        w1 = weights[:, 0:1]                # spectral weight
        w2 = weights[:, 1:2]                # stochastic weight
        fused = torch.cat([
            f_spec  * w1,
            f_stoch * w2
        ], dim=1)                            # (B, 384)
        return fused, weights


# ── FractalProspect-Net v4 ────────────────────────────────────────

class FractalProspectNet(nn.Module):
    """
    Full dual-branch architecture.

    Forward returns:
        logit      : (B, 1) raw sigmoid logit
        uncertainty: None during training; (B,) variance over MC passes at inference
        gate_weights: (B, 2) branch attention weights — interpretable
    """
    def __init__(self,
                 patch_size=PATCH_SIZE,
                 dropout_rate=DROPOUT_RATE,
                 spectral_dim=SPECTRAL_BRANCH_DIM,
                 stochastic_dim=STOCHASTIC_BRANCH_DIM,
                 mlp_dims=MLP_DIMS):
        super().__init__()
        self.spectral_branch   = SpectralBranch(N_SPECTRAL, spectral_dim)
        self.stochastic_branch = StochasticBranch(N_STOCHASTIC, stochastic_dim)
        self.attention_gate    = CrossBranchGate(spectral_dim, stochastic_dim)

        # Fusion MLP: 384 → 256 → 128 → 64 → 1
        dims = [spectral_dim + stochastic_dim] + mlp_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=dropout_rate))
        self.fusion_mlp = nn.Sequential(*layers)

    def forward(self, x_spec, x_stoch):
        f_spec  = self.spectral_branch(x_spec)
        f_stoch = self.stochastic_branch(x_stoch)
        fused, gate_weights = self.attention_gate(f_spec, f_stoch)
        logit = self.fusion_mlp(fused)
        return logit, gate_weights

    def mc_dropout_uncertainty(self, x_spec, x_stoch, T=50):
        """
        MC Dropout epistemic uncertainty.
        Keeps dropout active during T forward passes.
        Returns: mean prediction, variance (uncertainty).
        """
        self.train()  # activates dropout
        with torch.no_grad():
            preds = torch.stack([
                torch.sigmoid(self.forward(x_spec, x_stoch)[0])
                for _ in range(T)
            ], dim=0)
        self.eval()
        return preds.mean(0), preds.var(0)
