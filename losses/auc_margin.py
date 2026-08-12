import torch
import torch.nn as nn


class AUCMarginLoss(nn.Module):
    """
    AUC Margin Loss v2 from LibAUC (Yuan et al. 2021, arXiv:2012.03173).

    Multi-label extension matching MultiLabelAUCMLoss v2 from LibAUC.
    Removes class prior p from the formulation (factored out as constant),
    using conditional means (E[·|y=1], E[·|y=0]) directly.

    Per-label loss:
      E[(h - a)^2 | y=1] + E[(h - b)^2 | y=0]
      + 2*alpha*(m + E[h|y=0] - E[h|y=1]) - alpha^2

    Learnable parameters (a_k, b_k, alpha_k) per label:
      - a_k: tracks mean prediction on positive samples
      - b_k: tracks mean prediction on negative samples
      - alpha_k: dual variable (maximized, constrained >= 0)
    """

    def __init__(self, num_classes: int, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.num_classes = num_classes

        # Learnable primal-dual parameters (per class)
        self.a = nn.Parameter(torch.zeros(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))
        self.alpha = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(logits)  # (B, K)
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()

        num_pos = pos_mask.sum(dim=0)  # (K,)
        num_neg = neg_mask.sum(dim=0)  # (K,)

        # Mask for labels that have both positives and negatives in this batch
        valid = (num_pos > 0) & (num_neg > 0)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Safe denominators (clamp for division, but only use valid labels)
        num_pos_safe = num_pos.clamp(min=1)
        num_neg_safe = num_neg.clamp(min=1)

        # E[(h - a)^2 | y=1]
        term1 = ((preds - self.a) ** 2 * pos_mask).sum(dim=0) / num_pos_safe

        # E[(h - b)^2 | y=0]
        term2 = ((preds - self.b) ** 2 * neg_mask).sum(dim=0) / num_neg_safe

        # 2*alpha*(m + E[h|y=0] - E[h|y=1])
        mean_pos = (preds * pos_mask).sum(dim=0) / num_pos_safe
        mean_neg = (preds * neg_mask).sum(dim=0) / num_neg_safe
        term3 = 2 * self.alpha * (self.margin + mean_neg - mean_pos)

        # -alpha^2
        term4 = -(self.alpha ** 2)

        per_label = term1 + term2 + term3 + term4
        return per_label[valid].mean()
