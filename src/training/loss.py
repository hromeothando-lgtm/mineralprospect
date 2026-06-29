"""
Physics-Informed Weighted Focal Loss
=====================================
Combines:
1. Weighted Focal Loss (Lin et al 2017) — handles 1:1 class imbalance
   and hard positives simultaneously.
2. Physics-Informed Singularity Regularisation — penalises predictions
   that violate Theorem 1: patches where LSI alpha < 2 (positive
   singularity zones) should never be predicted as Low prospectivity.
   This injects the geological prior directly into the training objective
   and reduces the effective label requirement.

L_total = FL(p_t) + beta * L_singularity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, beta=0.1,
                 singularity_band_idx=0):
        """
        gamma : focal modulation — down-weights easy negatives dynamically
        alpha : class weight for positive class (None = compute from batch)
        beta  : weight of singularity regularisation term (Optuna HPO)
        singularity_band_idx : index of mean LSI band in stochastic patch
                               (Band 0 of x_stoch = first LSI map)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta  = beta
        self.sing_idx = singularity_band_idx

    def focal_loss(self, logits, targets, alpha):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        fl  = alpha * (1 - p_t) ** self.gamma * bce
        return fl.mean()

    def singularity_loss(self, logits, x_stoch):
        """
        Penalise predictions of Low (p < 0.5) on positive singularity patches.
        Positive singularity: mean LSI value < 2 across the patch.
        This encodes Theorem 1 as a hard constraint on the loss surface.
        """
        lsi_mean = x_stoch[:, self.sing_idx, :, :].mean(dim=[1, 2])
        pos_sing_mask = (lsi_mean < 2.0).float()
        probs = torch.sigmoid(logits.squeeze(1))
        # Penalise low predictions in positive singularity zones
        penalty = pos_sing_mask * F.relu(0.5 - probs)
        return penalty.mean()

    def forward(self, logits, targets, x_stoch):
        targets = targets.float().unsqueeze(1) \
            if targets.dim() == 1 else targets.float()

        if self.alpha is None:
            pos_rate = targets.mean().clamp(0.01, 0.99)
            alpha = 1.0 - pos_rate
        else:
            alpha = torch.tensor(self.alpha, device=logits.device)

        fl   = self.focal_loss(logits, targets, alpha)
        sing = self.singularity_loss(logits, x_stoch)
        return fl + self.beta * sing, fl.item(), sing.item()
