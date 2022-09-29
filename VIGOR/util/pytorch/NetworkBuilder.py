import torch
import torch.nn.functional as F
from util.pytorch.architectures.Network import Network
from util.pytorch.architectures.Feedforward import Feedforward
from util.pytorch.architectures.LSTMDiscriminator import LSTMDiscriminator
from util.pytorch.architectures.SharedSequenceDiscriminator import SharedSequenceDiscriminator
from util.Types import *


def get_network(input_shape, network_config: dict, out_features, network_type: str) -> Network:
    feedforward_config = network_config.get("feedforward")
    regularization_config = network_config.get("regularization")
    use_gpu = network_config.get("use_gpu")
    if network_type in ["time_series", "sequence"]:  # each datapoint is a sequence over timesteps
        time_series_config = network_config.get("time_series")
        architecture = time_series_config.get("architecture")
        if architecture == "lstm":  # use an lstm architecture
            lstm_config = time_series_config.get("lstm")
            network = LSTMDiscriminator(input_shape=input_shape, feedforward_config=feedforward_config,
                                        lstm_config=lstm_config, regularization_config=regularization_config,
                                        out_features=out_features, use_gpu=use_gpu)
        elif architecture == "shared_mlp":  # a stepwise multi layer perceptron that is aggregated
            shared_mlp_config = time_series_config.get("shared_mlp")
            network = SharedSequenceDiscriminator(input_shape=input_shape, feedforward_config=feedforward_config,
                                                  regularization_config=regularization_config,
                                                  shared_mlp_config=shared_mlp_config, out_features=out_features,
                                                  use_gpu=use_gpu)
        elif architecture == "1d_cnn":
            convolutional_config = time_series_config.get("1d_cnn")
            from util.pytorch.architectures.ConvolutionalSequenceDiscriminator import ConvolutionalSequenceDiscriminator
            network = ConvolutionalSequenceDiscriminator(input_shape=input_shape,
                                                         convolutional_config=convolutional_config,
                                                         regularization_config=regularization_config,
                                                         out_features=out_features, use_gpu=use_gpu)
        else:
            raise ValueError("Unknown time-series network architecture '{}'".format(architecture))
    elif network_type in ["default", "vanilla"]:
        network = Feedforward(input_shape=input_shape, feedforward_config=feedforward_config,
                              regularization_config=regularization_config, out_features=out_features, use_gpu=use_gpu)
    else:
        raise NotImplementedError("Unknown network_type '{}'".format(network_type))
    if network_config.get("use_gpu"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = network.to(device=device)
    return network


def weighted_mse_loss(input_tensor: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(weight * (input_tensor - target) ** 2)


def drex_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    # since we shuffle the data every batch, we can just compare each element to its predecessor
    offset_target = torch.cat((target[1:], target[:1]), dim=0)
    binarized_target = target < offset_target
    offset_input = torch.cat((input[1:], input[:1]), dim=0)
    input_pairs = torch.stack((input, offset_input), dim=-1)

    comparison_loss = F.cross_entropy(input=input_pairs, target=binarized_target.to(torch.int64))

    comparison_loss = comparison_loss / 2
    return comparison_loss


class NetworkBuilder:
    """
    Wrapper for building a pytorch network and a corresponding loss function and optimizer
    """

    def __init__(self, input_shape, network_config: dict, network_type: str, output_shape=1, loss: str = "bce"):
        """
        Initializes the network builder
        Args:
            input_shape: Shape of the input to the neural network
            network_config: Configuration details, including the network topology, regularization methods
                and how to train it
            output_shape: (optional) Desired output shape of the neural network. Defaults to 1
            loss: (optional) Loss function used. Defaults to "bce", which uses binary_cross_entropy_with_logits
        """
        self.network = get_network(input_shape, network_config, output_shape, network_type=network_type)
        if loss == "bce":
            self.loss_function = F.binary_cross_entropy_with_logits
        elif loss == "mse":
            self.loss_function = F.mse_loss
        elif loss == "drex":  # preference-based loss!
            self.loss_function = drex_loss
        else:
            raise NotImplementedError("Loss '{}' currently not implemented".format(loss))

    def __call__(self) -> Tuple[Network, LossFunction]:
        """
        Wrapper for retrieving the relevant parts of the builder, i.e. the network and what to train it with
        Returns:

        """
        return self.network, self.loss_function
