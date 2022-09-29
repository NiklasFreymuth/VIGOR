from torch import nn as nn
import torch
from torch.nn import utils
from util.pytorch.modules.UtilityModules import SaveBatchnorm1d, GhostBatchnorm, View, \
    add_activation_and_regularization_layers
from util.pytorch.UtilityFunctions import get_feedforward_layout
import numpy as np
from util.Types import *


def _build_feedforward_module(in_features: Union[tuple, int], feedforward_config: dict,
                              regularization_config: dict,
                              out_features: Optional[int] = None,
                              use_bias_in_last_layer: bool = True) -> Tuple[nn.ModuleList, np.array]:
    """
    Builds the discriminator (sub)network. This part of the network accepts some latent space as the input and
    outputs a classification
    Args:
        in_features: Number of input features
        feedforward_config: Dictionary containing the specification for the feedforward network. Includes num_layers,
        layer_size, batch_norm and dropout
        regularization_config: Dict containing information about batchnorm , spectral norm and dropout
        out_features: Number of output dimensions. If None are provided, the last hiden layer will be used as the output

    Returns: A nn.ModuleList representing the discriminator module

    """
    discriminator_modules = nn.ModuleList()
    if isinstance(in_features, (int, np.int32, np.int64)):  # can use in_features directly
        pass
    elif isinstance(in_features, tuple):
        if len(in_features) == 1:  # only one feature dimension
            in_features = in_features[0]
        else:  # more than one feature dimension. Need to flatten first
            in_features: int = int(np.prod(in_features))
            discriminator_modules.append(
                View(default_shape=(in_features,), custom_repr="Flattening feedforward input to 1d."))
    else:
        raise ValueError("Unknown type for 'in_features' parameter in Feedforward.py: '{}'".format(type(in_features)))

    if "network_shape" in feedforward_config:  # new layout
        network_layout = get_feedforward_layout(**feedforward_config)
    else:  # old layout
        num_layers = feedforward_config.get("num_layers")
        layer_size = feedforward_config.get("layer_size")
        network_layout = get_feedforward_layout(max_neurons_per_layer=layer_size, num_layers=num_layers)

    spectral_norm: bool = regularization_config.get("spectral_norm")
    previous_shape = in_features
    for current_layer_size in network_layout:
        # add "main" layer
        if spectral_norm:
            discriminator_modules.append(utils.spectral_norm(nn.Linear(in_features=previous_shape,
                                                                       out_features=current_layer_size)))
        else:
            discriminator_modules.append(nn.Linear(in_features=previous_shape,
                                                   out_features=current_layer_size))

        discriminator_modules = add_activation_and_regularization_layers(torch_module_list=discriminator_modules,
                                                                         in_features=current_layer_size,
                                                                         regularization_config=regularization_config)

        previous_shape = current_layer_size

    if out_features is not None:
        discriminator_modules.append(nn.Linear(previous_shape, out_features, bias=use_bias_in_last_layer))

    return discriminator_modules, network_layout


class Feedforward(nn.Module):
    """
    Feedforward module. Gets some input x and computes an output f(x).
    """

    def __init__(self, in_features: Union[tuple, int], feedforward_config: dict, regularization_config: dict,
                 out_features: Optional[int] = None, use_bias_in_last_layer: bool = True):
        """
        Initializes a new encoder module that consists of multiple different layers that together encode some
        input into a latent space
        Args:
            in_features: The input shape for the feedforward network
            out_features: The output dimension for the feedforward network
            feedforward_config: Dict containing information about what kind of feedforward network to build
            regularization_config: Dict containing information about batchnorm , spectral norm and dropout
        """
        super().__init__()
        self.feedforward_layers, self.network_layout = _build_feedforward_module(in_features=in_features,
                                                                                 feedforward_config=feedforward_config,
                                                                                 regularization_config=regularization_config,
                                                                                 out_features=out_features,
                                                                                 use_bias_in_last_layer=use_bias_in_last_layer)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass for the given input tensor
        Args:
            tensor: Some input tensor x

        Returns: The processed tensor f(x)

        """
        for feedforward_layer in self.feedforward_layers:
            tensor = feedforward_layer(tensor)
        return tensor
