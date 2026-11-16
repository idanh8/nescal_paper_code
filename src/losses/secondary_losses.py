import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5


class SoftAvUCLoss(nn.Module):
    """
    Soft AvUC loss (equation 15 in https://arxiv.org/pdf/2108.00106.pdf).
    Adapted from the original TensorFlow implementation by the authors.
    """
    default_lambda_reg = 1.0

    def __init__(self, temp: float = 1.0, theta: float = 0.5):
        super().__init__()
        assert temp > 0,          "temp must be positive"
        assert 0 < theta < 1,     "theta must be in (0, 1)"
        self.temp  = temp
        self.theta = theta

    def _soft_uncertainty(self, e: torch.Tensor) -> torch.Tensor:
        e = e.clamp(EPS, 1 - EPS)
        return torch.sigmoid(
            (1 / self.temp) * torch.log(
                e * (1 - self.theta) / ((1 - e) * self.theta)
            )
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)


        accuracies = probs.argmax(dim=1).eq(labels).float()


        C = probs.size(1)
        entmax = math.log(C)
        log_safe_probs = (1 - EPS) * probs + (EPS / C)
        entropies = -(log_safe_probs * log_safe_probs.log()).sum(dim=1)


        qus = self._soft_uncertainty(entropies / entmax)
        qcs = 1 - qus


        correct   = accuracies > 0.5
        incorrect = ~correct


        tanh_ent = torch.tanh(entropies)

        nac = (qcs * (1 - tanh_ent))[correct].sum()
        nau = (qus * tanh_ent)[correct].sum()
        nic = (qcs * (1 - tanh_ent))[incorrect].sum()
        niu = (qus * tanh_ent)[incorrect].sum()

        avuc_loss = torch.log(
            torch.tensor(1.0, device=logits.device) +
            (nau + nic) / torch.clamp(nac + niu, min=EPS)
        )

        return avuc_loss


class MMCELoss(nn.Module):
    """
    MMCE_w loss from "Trainable Calibration Measures For Neural Networks
    From Kernel Mean Embeddings" (Kumar et al., 2018).
    Adapted from the original TensorFlow implementation by the authors.
    """
    default_lambda_reg = 4.0

    def __init__(self):
        super().__init__()

    def _laplacian_kernel(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -torch.abs(matrix[..., 0] - matrix[..., 1]) / (2 * 0.2)
        )

    def _get_pairs(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> tuple:
        def outer_pairs(a, b):
            a_exp = a.unsqueeze(1).expand(len(a), len(b))
            b_exp = b.unsqueeze(0).expand(len(a), len(b))
            return torch.stack([a_exp, b_exp], dim=2)
        return (
            outer_pairs(tensor1, tensor1),
            outer_pairs(tensor2, tensor2),
            outer_pairs(tensor1, tensor2),
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        predicted_probs, predicted_labels = probs.max(dim=1)

        correct_mask = predicted_labels.eq(labels).float()
        n_correct    = correct_mask.sum()
        n_incorrect  = (1 - correct_mask).sum()

        cond_k       = (n_correct   > 0).float()
        cond_k_p     = (n_incorrect > 0).float()
        both_present = cond_k * cond_k_p

        if both_present == 0:
            return torch.tensor(0.0, device=logits.device)

        correct_prob   = predicted_probs[correct_mask.bool()]
        incorrect_prob = predicted_probs[(1 - correct_mask).bool()]

        correct_pairs, incorrect_pairs, correct_incorrect_pairs = \
            self._get_pairs(correct_prob, incorrect_prob)

        correct_kernel           = self._laplacian_kernel(correct_pairs)
        incorrect_kernel         = self._laplacian_kernel(incorrect_pairs)
        correct_incorrect_kernel = self._laplacian_kernel(correct_incorrect_pairs)

        w_correct   = 1.0 - correct_prob
        w_incorrect = incorrect_prob

        correct_correct_vals = torch.mean(
            correct_kernel * w_correct.unsqueeze(1) * w_correct.unsqueeze(0)
        )
        incorrect_incorrect_vals = torch.mean(
            incorrect_kernel * w_incorrect.unsqueeze(1) * w_incorrect.unsqueeze(0)
        )
        correct_incorrect_vals = torch.mean(
            correct_incorrect_kernel * w_correct.unsqueeze(1) * w_incorrect.unsqueeze(0)
        )

        m = n_correct
        n = n_incorrect

        mmd_error = (
              (1.0 / (m * m + 1e-5)) * correct_correct_vals
            + (1.0 / (n * n + 1e-5)) * incorrect_incorrect_vals
            - (2.0 / (m * n + 1e-5)) * correct_incorrect_vals
        )

        return both_present.detach() * torch.sqrt(torch.clamp(mmd_error, min=1e-10))


class MDCALoss(nn.Module):
    """
    Multi-class Difference in Confidence and Accuracy (MDCA) loss from
    "A Stitch in Time Saves Nine" (Hebbalaguppe et al., 2022). Code adapted from authors
    """
    default_lambda_reg = 1.0

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        C     = probs.shape[1]

        avg_conf  = probs.mean(dim=0)
        avg_count = F.one_hot(labels, num_classes=C).float().mean(dim=0)

        return torch.abs(avg_conf - avg_count).mean()


class MbLSLoss(nn.Module):
    default_lambda_reg = 0.1

    def __init__(self, margin: float = 10.0):
        super().__init__()
        assert margin > 0, "margin must be positive"
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:


        if logits.dim() > 2:
            logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
        max_logits = logits.max(dim=1).values.unsqueeze(1).expand_as(logits)
        diff = max_logits - logits
        return F.relu(diff - self.margin).mean()

from losses.ncr import NCRLoss, NeighborVariantLoss, RankMixupLoss

SECONDARY_LOSS_REGISTRY = {
    's-avuc': SoftAvUCLoss,
    'mmce':   MMCELoss,
    'mdca':   MDCALoss,
    'mbls':   MbLSLoss,
    'ncr': NCRLoss,
    'nbr_variant': NeighborVariantLoss,
    'rank_mixup': RankMixupLoss,
}
