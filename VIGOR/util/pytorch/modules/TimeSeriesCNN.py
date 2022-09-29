import torch
from torch import nn
from torch.nn import utils
from util.Types import *
from util.pytorch.modules.UtilityModules import add_activation_and_regularization_layers


def _build_convolutional_layers(in_features: int, convolutional_config: dict,
                                regularization_config: dict, out_features: Optional[int] = None) -> nn.ModuleList:
    """
    Builds the encoder layers of the Network
    Args:
        in_features: The input dimension of theencoder network
        convolutional_config: Dict containing information about what kind of CNN to build. Note that we take
        the symmetric filter size, meaning that a filter size of 3 corresponds to 2 elements on the left and
        2 on the right of the current element.
        regularization_config: Dict containing information about batchnorm , spectral norm and dropout
        out_features: The output dimension per timestep

    Returns: A ModuleList that represents an encoder network used to process image-based inputs into
    some latent representation

    """
    num_layers = convolutional_config.get("num_layers")
    num_channels = convolutional_config.get("num_channels")
    kernel_size = convolutional_config.get("kernel_size")
    padding_str = convolutional_config.get("padding")

    if padding_str == "zero":
        padding = ((kernel_size+1)//2) - 1
    elif padding_str in [None, False]:
        padding = 0
    else:
        raise ValueError(f"Unknown padding value '{padding_str}'")
    conv_1d_layers = [nn.Conv1d(in_channels=in_features if current_layer == 0 else num_channels,
                                out_channels=num_channels,
                                kernel_size=kernel_size,
                                padding=padding)
                      for current_layer in range(num_layers)]
    last_layer = nn.Conv1d(in_channels=num_channels,
                           out_channels=1,
                           kernel_size=kernel_size,
                           padding=padding)

    spectral_norm = regularization_config.get("spectral_norm")

    convolutional_layers = nn.ModuleList()
    for current_layer in range(num_layers):
        if spectral_norm:
            convolutional_layer = utils.spectral_norm(conv_1d_layers[current_layer])
        else:
            convolutional_layer = conv_1d_layers[current_layer]
        convolutional_layers.append(convolutional_layer)

        convolutional_layers = add_activation_and_regularization_layers(torch_module_list=convolutional_layers,
                                                                        in_features=num_channels,
                                                                        regularization_config=regularization_config)

    if out_features is not None:
        convolutional_layers.append(last_layer)
    return convolutional_layers


class TimeSeriesCNN(nn.Module):
    """
    ImageEncoder Module. This module in turn consists of multiple submodules, which are the individual layers
    """

    def __init__(self, in_features: int, convolutional_config: dict, regularization_config: dict,
                 out_features: Optional[int] = None):
        """
        Initializes a new encoder module that consists of multiple different layers that together encode some
        input into a latent space
        Args:
            in_features: Input dimension per timestep
            convolutional_config: Dict containing information about what kind of CNN to build
            regularization_config: Dict containing information about batchnorm, spectral norm and dropout
            out_features: The output dimension per timestep
        """
        super().__init__()
        self.convolutional_layers = _build_convolutional_layers(in_features=in_features,
                                                                convolutional_config=convolutional_config,
                                                                regularization_config=regularization_config,
                                                                out_features=out_features)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes an element-wise convolution over the inputs
        Args:
            tensor: Some input tensor x

        Returns: An element-wise convoluted computation f_conv(x)

        """
        for convolutional_layer in self.convolutional_layers:
            tensor = convolutional_layer(tensor)
        return tensor
