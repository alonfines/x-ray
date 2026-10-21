from .bce import get_bce_loss, calculate_pos_weights


def get_loss_function(loss_type: str = 'bce', weighted: bool = False,
                      pos_weight=None):
    return get_bce_loss(weighted=weighted, pos_weight=pos_weight)
