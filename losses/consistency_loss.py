"""Manuscript-aligned batch-adaptive Focal losses and dual-gradient KL.

Counts are obtained solely from each training mini-batch. For each class,
Laplace smoothing adds one to its count, followed by normalized inverse
frequency weighting. The binary branch folds four-class labels into FTA/FTC.
"""
import torch
from torch import nn
import torch.nn.functional as F


def batch_inverse_frequency(targets: torch.Tensor, num_classes: int, smoothing: float = 1.0):
    """Normalized inverse-frequency weights for the current batch.

    smoothing=1 corresponds to the Laplace adjustment described in the
    supplementary manuscript. The binary two-class case is also equivalent
    to complementary smoothed-class proportions.
    """
    if smoothing <= 0:
        raise ValueError('smoothing must be positive for absent classes')
    with torch.no_grad():
        counts = torch.bincount(targets.detach().long(), minlength=num_classes)
        counts = counts.to(dtype=torch.float32, device=targets.device) + smoothing
        inverse = counts.reciprocal()
        return inverse / inverse.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets, alpha):
        logp = F.log_softmax(logits, dim=1)
        log_pt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        return (-(1-pt).pow(self.gamma) * log_pt * alpha[targets]).mean()


class ConsistencyLoss(nn.Module):
    """Binary focal + four-class focal + lambda * KL(folded four-class || binary).

    No stop-gradient: KL gradients flow into both prediction heads.
    """
    def __init__(self, lambda_consistency=0.5, gamma=2., smoothing=1.0):
        super().__init__()
        self.lambda_c = lambda_consistency
        self.smoothing = smoothing
        self.focal = FocalLoss(gamma=gamma)

    def forward(self, binary_logits, four_class_logits, labels_four_class):
        binary_targets = (labels_four_class != 0).long()
        alpha_binary = batch_inverse_frequency(binary_targets, 2, self.smoothing)
        alpha_four_class = batch_inverse_frequency(labels_four_class, 4, self.smoothing)
        loss_binary = self.focal(binary_logits, binary_targets, alpha_binary)
        loss_four_class = self.focal(four_class_logits, labels_four_class, alpha_four_class)
        four_class_prob = F.softmax(four_class_logits, dim=1)
        folded = torch.stack([four_class_prob[:, 0], four_class_prob[:, 1:].sum(dim=1)], dim=1)
        loss_kl = F.kl_div(F.log_softmax(binary_logits, dim=1), folded, reduction='batchmean')
        return (loss_binary + loss_four_class + self.lambda_c * loss_kl,
                loss_binary.detach(), loss_four_class.detach(), loss_kl.detach())
