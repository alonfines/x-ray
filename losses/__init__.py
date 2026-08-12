from .auc_margin import AUCMarginLoss
from .bce import get_bce_loss, calculate_pos_weights


def get_loss_function(loss_type: str = 'bce', weighted: bool = False,
                      pos_weight=None, num_classes: int = None,
                      margin: float = 1.0):
    if loss_type == 'auc_margin':
        return AUCMarginLoss(num_classes=num_classes, margin=margin)
    return get_bce_loss(weighted=weighted, pos_weight=pos_weight)
