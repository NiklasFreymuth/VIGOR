import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Variable, grad
from torch.nn import functional as F


def get_feedforward_layout(max_neurons_per_layer: int, num_layers: int, network_shape: str = "block") -> np.ndarray:
    """
    Creates a list of neurons per layer to create a feedforward network with
    Args:
        max_neurons_per_layer: Maximum number of neurons for a single layer
        num_layers: Number of layers of the network.
        network_shape: Shape of the network. May be
            "=" or "block" for a network where every layer has the same size (=max_neurons_per_layer)
                E.g., {num_layers=3, max_neurons_per_layer=32} will result in [32, 32, 32]
            ">" or "contracting" for a network that gets exponentially smaller.
                E.g., {num_layers=3, max_neurons_per_layer=32} will result in [32, 16, 8]
            "<" or "expanding" for a network that gets exponentially bigger.
                E.g., {num_layers=3, max_neurons_per_layer=32} will result in [8, 16, 32]
            "><" or "hourglass" for a network with a bottleneck in the middle.
                E.g., {num_layers=3, max_neurons_per_layer=32} will result in [32, 16, 32]
            "<>" or "rhombus" for a network that expands towards the middle and then contracts again
                E.g., {num_layers=3, max_neurons_per_layer=32} will result in [16, 32, 16]
            Overall, the network_shape heavily influences the total number of neurons as well as weights between them.
            We can thus order parameters by network_shape as block>=hourglass>=rhombus>=contracting/expanding, where
            the first two equalities only hold for networks with less than 3 layers, and the last equality holds for
            exactly 1 layer (in which all shapes are equal).
            In case of an even number of layers, "rhombus" and "hourglass" will repeat the "middle" layer once. I.e.,
            {num_layers=4, max_neurons_per_layer=32, network_shape="rhombus"} will result in an array [16, 32, 32, 16]

    Returns: A 1d numpy array of length num_layers. Each entry specifies the number of neurons to use in that layer.
        The maximum number of neurons will always be equal to max_neurons_per_layer. Depending on the shape, the other
        layers will have max_neurons_per_layer/2^i neurons, where i depends on the number of "shrinking" layers between
        the current layer and one of maximum size. In other words, the smaller layers shrink exponentially.
    """
    assert isinstance(max_neurons_per_layer, int), \
        "Need to have an integer number of maximum neurons. Got '{}' of type '{}'".format(max_neurons_per_layer,
                                                                                          type(max_neurons_per_layer))
    if num_layers == 0:
        return np.array([])  # empty list, i.e., no network
    if network_shape in ["=", "==", "block"]:
        return np.repeat(max_neurons_per_layer, num_layers)
    elif network_shape in [">", "contracting"]:
        return np.array([int(np.maximum(1, max_neurons_per_layer // (2 ** distance)))
                         for distance in range(num_layers)])
    elif network_shape in ["<", "expanding"]:
        return np.array([int(np.maximum(1, max_neurons_per_layer // (2 ** current_layer)))
                         for current_layer in reversed(range(num_layers))])
    elif network_shape in ["><", "hourglass"]:
        return np.array([int(np.maximum(1, max_neurons_per_layer // (2 ** distance)))
                         for distance in list(range((num_layers + 1) // 2)) + list(reversed(range(num_layers // 2)))])
    elif network_shape in ["<>", "rhombus"]:
        # we want a way to produce a list like [n, n-1, ..., 1, 0, 1, ..., n] that works for both num_layers = 2n and
        # num_layers=2n+1.
        return np.array([int(np.maximum(1, max_neurons_per_layer // (2 ** distance)))
                         for distance in
                         list(reversed(range((num_layers + 1) // 2))) +
                         list(range((num_layers + 1) // 2))[num_layers % 2:]])
    else:
        raise ValueError("Unknown network_shape '{}'. Eligible shapes are = ('block'), > ('contracting'), "
                         "< ('expanding'), >< ('hourglass'), <> ('rhombus')".format(network_shape))


def get_convolutional_output_size(encoder_config: dict, input_shape: tuple, network_type="dreamer"):
    """
    Computes the size of the output of a convolutional layer (or set of layers) with the given parameters and input
    shape
    Args:
        encoder_config: Dictionary containing information about the encoder
        input_shape:

    Returns:

    """
    num_layers = encoder_config.get("num_layers")
    stride = encoder_config.get("stride")
    kernel_size = encoder_config.get("kernel_size")
    padding = get_padding(kernel_size)
    base_depth = encoder_config.get("base_depth")

    if network_type == "dreamer":
        num_channels = get_dreamer_channel_depths(base_depth=base_depth, num_layers=num_layers)[-1]
    else:
        raise NotImplementedError("Unknown Convolutional Network type '{}'".format(network_type))

    input_shape = np.array(input_shape)
    for i in range(num_layers):
        input_shape = np.floor((input_shape - kernel_size + 2 * padding) / stride + 1)
    return int(num_channels * (np.prod(input_shape)))


def get_dreamer_channel_depths(base_depth: int, num_layers: int):
    return [int(base_depth * (2 ** i)) for i in range(num_layers)]


def get_padding(kernel_size: int) -> int:
    return (kernel_size - 1) // 2


def detach(tensor: torch.Tensor) -> np.array:
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


def reparameterize(mean: Tensor, log_variance: Tensor) -> Tensor:
    """
    Adapted from https://github.com/AntixK/PyTorch-VAE/blob/master/models/base.py
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mean: (Tensor) Mean of the latent Gaussian [B x D]
    :param log_variance: (Tensor) Log of the Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    standard_deviation = torch.exp(0.5 * log_variance)
    epsilon = torch.randn_like(standard_deviation)
    return epsilon * standard_deviation + mean


def get_timestamps(tensor: torch.Tensor, shaped: bool = False) -> torch.Tensor:
    """
    Calculates timestamps for time-series data based on a given encoding pattern
    Args:
        tensor: Input tensor of shape (#batches, #steps, *features)
        shaped: Whether to return a tensor of size (#batches, #timesteps, encoding_size) (True), or a tensor of size
            (#batches*#timesteps, encoding_size) (False)

    Returns: A tensor of shape (#batches, #timesteps, encoding_size) if shaped=True,
      a tensor of shape (#batches*#timesteps, encoding_size) else

    """
    num_batches = tensor.shape[0]
    num_timesteps = tensor.shape[1]

    timestamps = torch.linspace(start=0, end=1, steps=num_timesteps)
    if shaped:
        timestamps = timestamps.repeat(num_batches, 1).unsqueeze(dim=2)
    else:
        timestamps = timestamps.repeat(num_batches).unsqueeze(dim=1)
    return timestamps
