"""Loss functions for ATAC-seq profile and count prediction.

Adapted from ../multiome/src/training/loss.py. Simplified for per-cell-type
ChromBPNet training.

Uses count likelihoods (not MSE) for statistically faithful modeling:
  - Multinomial NLL for profile shape
  - Poisson NLL for total counts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def multinomial_nll_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    total_counts: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multinomial NLL loss for profile prediction.

    Models the profile as a multinomial distribution over base positions.

    Args:
        logits: Predicted profile logits (batch, length).
        targets: Target profiles (batch, length).
        total_counts: Total counts per sample (batch,).
        eps: Numerical stability constant.

    Returns:
        Scalar loss.
    """
    # Normalize targets to probabilities
    target_probs = targets / (targets.sum(dim=-1, keepdim=True) + eps)

    # Log softmax of predictions
    log_probs = F.log_softmax(logits, dim=-1)

    # Multinomial NLL: -sum(counts * target_probs * log_probs)
    nll = -torch.sum(total_counts.unsqueeze(-1) * target_probs * log_probs, dim=-1)

    return nll.mean()


def poisson_nll(
    log_rate: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Poisson NLL loss for count prediction.

    Args:
        log_rate: Predicted log(rate) (batch,) or (batch, n).
        target: Target counts (same shape).

    Returns:
        Per-element NLL.
    """
    log_rate = torch.clamp(log_rate, min=-20, max=20)
    mu = torch.exp(log_rate)
    nll = mu - target * log_rate + torch.lgamma(target + 1)
    return nll


class ChromBPNetLoss(nn.Module):
    """Combined loss for ChromBPNet: profile shape + count magnitude.

    Loss = profile_weight * multinomial_NLL + count_weight * poisson_NLL

    Args:
        profile_weight: Weight for multinomial profile loss.
        count_weight: Weight for Poisson count loss.
    """

    def __init__(self, profile_weight: float = 1.0, count_weight: float = 0.5):
        super().__init__()
        self.profile_weight = profile_weight
        self.count_weight = count_weight

    def forward(
        self,
        pred_profile: torch.Tensor,
        pred_count: torch.Tensor,
        target_profile: torch.Tensor,
        target_count: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred_profile: Profile logits (batch, length).
            pred_count: Predicted log(count) (batch,) or (batch, 1).
            target_profile: Target profile (batch, length).
            target_count: Target count (batch,) or (batch, 1).

        Returns:
            Dict with 'loss', 'profile_loss', 'count_loss'.
        """
        pred_count = pred_count.squeeze(-1)
        target_count = target_count.squeeze(-1)

        profile_loss = multinomial_nll_loss(
            pred_profile, target_profile, target_count,
        )

        count_loss = poisson_nll(pred_count, target_count).mean()

        total = self.profile_weight * profile_loss + self.count_weight * count_loss

        return {
            "loss": total,
            "profile_loss": profile_loss.detach(),
            "count_loss": count_loss.detach(),
        }


class MultiCellChromBPNetLoss(nn.Module):
    """Loss for multi-cell-type ChromBPNet (Stage 3).

    Sums per-cell-type losses and adds an optional differential penalty
    to prevent cell-type collapse.

    Args:
        profile_weight: Weight for profile loss.
        count_weight: Weight for count loss.
        diff_weight: Weight for differential loss term.
        min_variance: Minimum cross-cell-type variance before penalizing.
    """

    def __init__(
        self,
        profile_weight: float = 1.0,
        count_weight: float = 0.5,
        diff_weight: float = 0.1,
        min_variance: float = 0.1,
    ):
        super().__init__()
        self.profile_weight = profile_weight
        self.count_weight = count_weight
        self.diff_weight = diff_weight
        self.min_variance = min_variance

    def forward(
        self,
        pred_profile: torch.Tensor,
        pred_count: torch.Tensor,
        target_profile: torch.Tensor,
        target_count: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-cell loss.

        Args:
            pred_profile: (batch, n_cell_types, length) profile logits.
            pred_count: (batch, n_cell_types) predicted log counts.
            target_profile: (batch, n_cell_types, length) target profiles.
            target_count: (batch, n_cell_types) target counts.

        Returns:
            Dict with loss components.
        """
        n_cell_types = pred_profile.size(1)

        # Per-cell-type profile and count losses
        profile_losses = []
        count_losses = []

        for ct in range(n_cell_types):
            p_loss = multinomial_nll_loss(
                pred_profile[:, ct, :],
                target_profile[:, ct, :],
                target_count[:, ct],
            )
            profile_losses.append(p_loss)

            c_loss = poisson_nll(pred_count[:, ct], target_count[:, ct]).mean()
            count_losses.append(c_loss)

        profile_loss = torch.stack(profile_losses).mean()
        count_loss = torch.stack(count_losses).mean()

        total = self.profile_weight * profile_loss + self.count_weight * count_loss

        # Differential loss: penalize when predictions are too similar across cell types
        diff_loss = torch.tensor(0.0, device=pred_profile.device)
        if self.diff_weight > 0 and n_cell_types > 1:
            # Variance across cell types at each position
            cell_variance = pred_profile.var(dim=1)  # (batch, length)
            diff_loss = F.relu(self.min_variance - cell_variance).mean()
            total = total + self.diff_weight * diff_loss

        return {
            "loss": total,
            "profile_loss": profile_loss.detach(),
            "count_loss": count_loss.detach(),
            "diff_loss": diff_loss.detach(),
        }
