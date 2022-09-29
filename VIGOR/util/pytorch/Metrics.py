from util.Types import *
import torch
import numpy as np
from util.pytorch.UtilityFunctions import detach
from util import Defaults as d


def get_metrics(predictions: torch.Tensor, targets: torch.Tensor, loss_function: Callable,
                weights: torch.Tensor = None) -> dict:
    """
    Option to calculate additional metrics based on the predictions and targets.
    For example, a discrimination task can use accuracy as an additional metric,
    while a regression may be interested in a Mean Squared Error.
    Args:
        predictions: The network predictions
        targets: The ground truth targets/labels corresponding to these predictions
        loss_function: Loss function for the training. Used to decide on the metric
        weights: (optional) Sample-wise weights

    Returns: A dictionary containing the metrics. May be empty

    """
    if loss_function.__name__ == "binary_cross_entropy_with_logits":
        return {d.ACCURACY: accuracy(predictions, targets, weights, absolute=False)}
    elif loss_function.__name__ in ["mse_loss", "drex_loss"]:
        return {d.ROOT_MEAN_SQUARED_ERROR: root_mean_squared_error(predictions=predictions,
                                                                   targets=targets, weights=weights),
                d.MEAN_ABSOLUTE_ERROR: mean_absolute_error(predictions=predictions, targets=targets, weights=weights)}
    else:
        return {}


def root_mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if weights is None:
        return detach(torch.sqrt(torch.mean((predictions - targets) ** 2)))
    else:
        return detach(torch.sqrt(torch.mean(weights * ((predictions - targets) ** 2))))


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if weights is None:
        return detach(torch.mean(torch.abs(predictions - targets)))
    else:
        return detach(torch.mean(torch.abs(predictions - targets) * weights))


def accuracy(predictions: torch.Tensor, targets: torch.Tensor,
             weights: Optional[torch.Tensor] = None, absolute: bool = False, from_logits: bool = True) -> np.array:
    """
    Calculates the (possibly weighted) accuracy of the model predictions with the respective ground truth targets.
    Args:
        predictions: Model predictions
        targets: Ground truth targets/labels
        weights: Element-wise weights.
        absolute: If true, returns the total amount of accurate predictions rather than the percentage
        from_logits: Whether the predictions are in the form of logits (i.e., in (-infty, infty), or sigmoid activations
        (in (0, 1)).

    Returns: The (weighted) accuracy, i.e., the percentage of correct predictions

    """
    if from_logits:
        threshold = 0
    else:
        threshold = 0.5
    if weights is not None:
        _accuracy = (sum((predictions[targets == 1] >= threshold) * weights[targets == 1]) + sum(
            (predictions[targets == 0] < threshold) * weights[targets == 0]))
    else:
        _accuracy = (sum(predictions[targets == 1] >= threshold) + sum(predictions[targets == 0] < threshold))
    _accuracy = detach(_accuracy)
    if absolute:
        return _accuracy
    else:
        return _accuracy / len(predictions)
