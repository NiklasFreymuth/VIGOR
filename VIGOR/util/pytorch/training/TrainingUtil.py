import numpy as np
import torch
from sklearn.model_selection import train_test_split as split
from torch.utils.data import DataLoader

from util import Defaults as d
from util.pytorch.DictionaryDataset import DictionaryDataset
from util.pytorch.Metrics import get_metrics
from util.pytorch.UtilityFunctions import detach
from util.Types import *


def format_history(history: List[dict]) -> dict:
    """
    Formats a dict of epoch-wise metrics into a coherent history object. Turns a list of dicts with identical keys
    into a single dict with lists of values of these keys
    Args:
        history: A dictionary [epoch: {metric: [epoch_value]}]

    Returns: A dictionary {metric: [epoch_value]}

    """
    formatted_history = {k: np.array([history[i][k] for i in range(len(history))]) for k in history[0].keys()}
    return formatted_history


def compute_loss_and_metrics(batch_size: int, loss: torch.Tensor, output_dict: dict, scalars: dict,
                             targets: torch.Tensor, weights: torch.Tensor,
                             loss_function: Callable,
                             stepwise_loss: bool) -> Tuple[torch.Tensor, dict]:
    """
    Adds the discriminator loss and metrics building on this to the loss function and the recording
    Args:
        batch_size: The size of the current minibatch
        loss: The current loss
        output_dict: Dictionary containing the outputs of the network.
        scalars: Dictionary containing every scalar to be recorded
        targets: Target values for the discriminator
        weights: Weights of these values
        loss_function: The discriminator loss function to use
        stepwise_loss: Whether to apply the loss for every step of a time_sequence (True) or for an aggregation
          of the steps (False).

    Returns: A tuple (loss, scalars)

    """
    if targets is not None:
        predictions = output_dict[d.PREDICTIONS].squeeze()  # logits for classification, mse for regression
        if batch_size == 1:
            predictions = predictions.reshape((1,) + predictions.shape)
        metrics = get_metrics(predictions=predictions, targets=targets, loss_function=loss_function, weights=weights)

        if stepwise_loss:  # compute the loss for every step and aggregate over it afterwards
            assert loss_function.__name__ == "binary_cross_entropy_with_logits", "Stepwise loss only implemented for BCE"
            stepwise_predictions = output_dict.get(d.STEPWISE_LOGITS).squeeze()
            if batch_size == 1:
                stepwise_predictions = stepwise_predictions.reshape((1,) + stepwise_predictions.shape)
            stepwise_targets = targets[..., None].repeat(1, stepwise_predictions.shape[1])
            if weights is not None:
                stepwise_weights = weights[..., None].repeat(1, stepwise_predictions.shape[1])
            else:
                stepwise_weights = None
            scalars, prediction_loss = _compute_loss(loss_function=loss_function, predictions=stepwise_predictions,
                                                     scalars=scalars, targets=stepwise_targets,
                                                     weights=stepwise_weights)
        else:

            scalars, prediction_loss = _compute_loss(loss_function=loss_function, predictions=predictions,
                                                     scalars=scalars, targets=targets, weights=weights)
        scalars = {**scalars, **metrics}

        loss += prediction_loss
    return loss, scalars


def _compute_loss(loss_function: Callable, predictions: torch.Tensor, scalars: Dict,
                  targets: torch.Tensor, weights: torch.Tensor) -> Tuple[Dict, torch.Tensor]:
    if loss_function.__name__ == "binary_cross_entropy_with_logits":
        prediction_loss = loss_function(predictions, targets, weights)
        scalars[d.BINARY_CROSS_ENTROPY] = prediction_loss.item()
    elif loss_function.__name__ == "mse_loss":
        prediction_loss = loss_function(predictions, targets)
        scalars[d.MEAN_SQUARED_ERROR] = prediction_loss.item()
    elif loss_function.__name__ == "drex_loss":
        prediction_loss = loss_function(predictions, targets)
        scalars[d.COMPARISON_BASED_LOSS] = prediction_loss.item()
    else:
        raise ValueError("Unknown loss function {}".format(loss_function.__name__))
    return scalars, prediction_loss


def dataloader_from_dictionary_data(data: Dict[str, np.array], batch_size: int,
                                    validation_split: float = 0) -> Tuple[DataLoader, Optional[DataLoader]]:
    """

    Args:
        data: A dictionary over samples, targets/labels and potentially weights and other information needed for the
          training
        batch_size: Size of each minibatch. Is given to the dataloader to determine how to partition the data each
          epoch
        validation_split: Part of the data that is to be split into a designated validation set. This data will not
          be shuffled before splitting

    Returns: A tuple (train_data_loader, validation_data_loader), where the latter is None if validation_split=0

    """
    if validation_split > 0:
        train_data = {}
        validation_data = {}
        for key, value in data.items():
            train_data[key], validation_data[key] = split(value, test_size=validation_split, shuffle=False)
            # shuffling here would mix the expert demonstrations of previous iterations into the validation set,
            # so we instead shuffle the learner data earlier and independent of the expert one
        validation_data = {k: torch.tensor(v.astype(np.float32)) for k, v in validation_data.items()}
        validation_dataset = DictionaryDataset(**validation_data)
        validation_data_loader = DataLoader(validation_dataset, batch_size=2 * batch_size, shuffle=False)
        # can have a larger batch_size since we do not track gradients
    else:  # no validation set
        validation_data_loader = None
        train_data = data
    train_data = {k: torch.tensor(v.astype(np.float32)) for k, v in train_data.items()}
    train_dataset = DictionaryDataset(**train_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_data_loader, validation_data_loader
