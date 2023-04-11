import torch.nn as nn

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    'smooth_l1': nn.SmoothL1Loss,
}


def get_loss_func(loss_name):
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]