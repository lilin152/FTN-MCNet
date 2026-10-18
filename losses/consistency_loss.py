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

    def forward(self, root_logits, leaf_logits, labels4):
        root_targets = (labels4 != 0).long()
        alpha_root = batch_inverse_frequency(root_targets, 2, self.smoothing)
        alpha_leaf = batch_inverse_frequency(labels4, 4, self.smoothing)
        lr = self.focal(root_logits, root_targets, alpha_root)
        ll = self.focal(leaf_logits, labels4, alpha_leaf)
        leaf_prob = F.softmax(leaf_logits, dim=1)
        folded = torch.stack([leaf_prob[:, 0], leaf_prob[:, 1:].sum(dim=1)], dim=1)
        lc = F.kl_div(F.log_softmax(root_logits, dim=1), folded, reduction='batchmean')
        return lr + ll + self.lambda_c * lc, lr.detach(), ll.detach(), lc.detach()
