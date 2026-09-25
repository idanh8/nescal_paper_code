import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.secondary_losses import SECONDARY_LOSS_REGISTRY

class FocalLoss(nn.Module):
    """
    Standard Focal Loss for classification.
    Adapted from Mukhoti et al. (2020) "Calibrating Deep Neural Networks using Focal Loss"
    """
    def __init__(self, gamma: float = 3.0, reduction: str = 'mean'):
        super().__init__()
        assert gamma >= 0, "gamma must be non-negative"
        assert reduction in ('mean', 'sum', 'none')
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:
            input = input.permute(0, 2, 3, 1).reshape(-1, input.size(1))
            target = target.reshape(-1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':   return loss.mean()
        elif self.reduction == 'sum':  return loss.sum()
        else:                          return loss


_FLSD53_THRESHOLDS = {0.2: 5.0, 0.5: 3.0}

class FocalLossAdaptive(nn.Module):
    def __init__(
        self,
        gamma_high: float = 2.0,
        gamma_thresholds: dict = None,
        reduction: str = 'mean'
    ):
        """
        Adaptive Focal Loss with sample-dependent gamma schedule (FLSD-53).
        Adapted from Mukhoti et al. (2020) "Calibrating Deep Neural Networks using Focal Loss"
        gamma_high:        gamma used when pt >= highest threshold (easy examples)
        gamma_thresholds:  dict mapping pt thresholds to gammas for hard examples,
                           e.g. {0.2: 5.0, 0.5: 3.0}
        """
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.gamma_high = gamma_high
        self.reduction = reduction

        thresholds = gamma_thresholds or _FLSD53_THRESHOLDS
        sorted_items = sorted(thresholds.items())
        self.register_buffer(
            'boundaries', torch.tensor([k for k, _ in sorted_items])
        )
        self.register_buffer(
            'gamma_values', torch.tensor([v for _, v in sorted_items])
        )

    def _get_gamma(self, pt: torch.Tensor) -> torch.Tensor:

        indices = torch.bucketize(pt, self.boundaries.to(pt.device), right=False)

        gamma = torch.where(
            indices < len(self.gamma_values),
            self.gamma_values.to(pt.device)[indices.clamp(max=len(self.gamma_values) - 1)],
            torch.full_like(pt, self.gamma_high)
        )
        return gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:
            input = input.permute(0, 2, 3, 1).reshape(-1, input.size(1))
            target = target.reshape(-1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        gamma = self._get_gamma(pt)
        loss = -((1 - pt) ** gamma) * logpt

        if self.reduction == 'mean':   return loss.mean()
        elif self.reduction == 'sum':  return loss.sum()
        else:                          return loss

class BrierScoreLoss(nn.Module):
    """
    Brier Score Loss for classification.
    Adapted from Mukhoti et al. (2020) "Calibrating Deep Neural Networks using Focal Loss"
    """
    def __init__(self):
        super(BrierScoreLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:

            input = input.permute(0, 2, 3, 1).reshape(-1, input.size(1))
            target = target.reshape(-1)

        probs = F.softmax(input, dim=1)
        target_one_hot = F.one_hot(target, num_classes=input.size(1)).float()

        squared_diff = (target_one_hot - probs) ** 2
        loss = squared_diff.sum(dim=1).mean()
        return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, alpha: float = 0.1):
        """
        Adapted from Hebbalaguppe et. al (2022) "Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration"
        alpha: smoothing factor in [0, 1).
               alpha=0 reduces to standard cross-entropy.
        """
        super().__init__()
        assert 0.0 <= alpha < 1.0, "alpha must be in [0, 1)"
        self.alpha = alpha
        self.confidence = 1.0 - alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:
            input = input.permute(0, 2, 3, 1).reshape(-1, input.size(1))
            target = target.reshape(-1)

        num_classes = input.size(1)
        log_probs = input.log_softmax(dim=1)

        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.alpha / (num_classes - 1))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-smooth_dist * log_probs, dim=1))

class PrimarySecondaryLoss(nn.Module):

    def __init__(self, primary_loss: nn.Module, secondary_loss: nn.Module, lambda_reg: float = None):
        super().__init__()
        self.primary_loss_fn = primary_loss
        self.secondary_loss_fn = secondary_loss


        if lambda_reg is None:
            lambda_reg = getattr(secondary_loss, 'default_lambda_reg', 1.0)
        self.lambda_reg = lambda_reg


        self.requires_global_probs = getattr(secondary_loss, 'requires_global_probs', False)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, node_indices=None, **kwargs) -> tuple:
        primary_loss   = self.primary_loss_fn(logits, labels)
        secondary_loss = self.secondary_loss_fn(logits, labels, node_indices=node_indices, **kwargs)
        total_loss = primary_loss + self.lambda_reg * secondary_loss
        return total_loss, primary_loss, secondary_loss


    def update_global_probs(self, probs: torch.Tensor, indices=None):
        """Forwarded to secondary if it supports global prob updates (e.g. NCL)."""
        if hasattr(self.secondary_loss_fn, 'update_global_probs'):
            self.secondary_loss_fn.update_global_probs(probs, indices)

    def precompute_matrices(self, features: torch.Tensor, **kwargs):
        """Forwarded to secondary if it supports precomputation (e.g. NCL)."""
        if hasattr(self.secondary_loss_fn, 'precompute_matrices'):
            self.secondary_loss_fn.precompute_matrices(features, **kwargs)


class _StandaloneLoss(nn.Module):
    requires_global_probs = False

    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> tuple:
        loss = self.loss_fn(logits, labels)
        zero = torch.tensor(0.0, device=logits.device)
        return loss, loss, zero


def get_primary_loss(primary_loss_name: str, loss_params: dict) -> nn.Module:
    secondary = loss_params.get('secondary_loss', None)

    if primary_loss_name == 'XE':
        return nn.CrossEntropyLoss()

    elif primary_loss_name == 'FL':
            gamma = loss_params.get('gamma', 2.0)
            fl_variant = loss_params.get('fl_variant', 'adaptive' if secondary is None else 'fixed')
            if fl_variant == 'adaptive':
                return FocalLossAdaptive(gamma_high=gamma)
            elif fl_variant == 'fixed':
                return FocalLoss(gamma=gamma)
            else:
                raise ValueError(f"Unknown fl_variant '{fl_variant}'; must be 'fixed' or 'adaptive'.")

    elif primary_loss_name == 'BL':
        return BrierScoreLoss()

    elif primary_loss_name == 'LS':
        alpha = loss_params.get('alpha', 0.1)
        return LabelSmoothingLoss(alpha=alpha)

    else:
        raise ValueError(f"Unknown primary loss: '{primary_loss_name}'")


def get_secondary_loss(secondary_loss_name: str, secondary_params: dict, lambda_reg: float = None) -> nn.Module:
    if secondary_loss_name == 'ncl':
        from losses.nescal import PreNeighborhoodConsistencyLoss
        return PreNeighborhoodConsistencyLoss(lambda_reg=lambda_reg, **secondary_params)

    if secondary_loss_name not in SECONDARY_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown secondary loss: '{secondary_loss_name}'. "
            f"Valid options: {list(SECONDARY_LOSS_REGISTRY.keys()) + ['ncl']}"
        )

    cls = SECONDARY_LOSS_REGISTRY[secondary_loss_name]
    return cls(**secondary_params)


def get_loss(primary_loss_name: str, secondary_loss_name: str, loss_params: dict) -> nn.Module:
    secondary_params = loss_params.get('secondary_params', {})
    lambda_reg       = loss_params.get('lambda_reg', None)

    if secondary_loss_name is None:
        primary = get_primary_loss(primary_loss_name, loss_params)
        return _StandaloneLoss(primary)

    else:
        primary   = get_primary_loss(primary_loss_name, loss_params)
        secondary = get_secondary_loss(secondary_loss_name, secondary_params, lambda_reg=lambda_reg)
        return PrimarySecondaryLoss(primary, secondary, lambda_reg=lambda_reg)
