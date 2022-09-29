import torch
from util import Defaults as d
import numpy as np
from util.Types import *
from util.pytorch.architectures.Network import Network

from util.pytorch.modules.TimeSeriesCNN import TimeSeriesCNN
from util.pytorch.UtilityFunctions import get_timestamps

class ConvolutionalSequenceDiscriminator(Network):
    """
    The ImageDiscriminator has the same purpose and usage as the discriminator, the difference being
    that the inputs must be encoded with some encoder subnetwork and then fed through a latent space beforehand
    """

    def __init__(self, input_shape: tuple, convolutional_config: dict, regularization_config: Dict[Key, Any],
                 out_features: int = 1, use_gpu: bool = False):
        """
        Creates a 1d CNN for convolutions over time-series data. Convolves over timesteps and outputs a scalar reward
        for each step that can then be aggregated. Uses a timestamp per step.
        Args:
            input_shape: Dimensionality/Shape of the input data
            convolutional_config: Describes the number of layers, channels, stride etc. for the convolutional network
            out_features: Dimensionality/Shape of the output discriminator. Used for image-based observations
            use_gpu: Whether to keep this network (and all related tensors) on the gpu.
        """
        super().__init__(input_shape=input_shape,
                         out_features=out_features,
                         use_gpu=use_gpu)
        stepwise_dimension = int(np.prod(input_shape[1:]))

        num_timestamp_features = 1   # encode timestamp as normalized value in [0,1]

        stepwise_features = stepwise_dimension + num_timestamp_features

        self._convolution = TimeSeriesCNN(in_features=stepwise_features,
                                          convolutional_config=convolutional_config,
                                          regularization_config=regularization_config,
                                          out_features=out_features)


        stepwise_aggregation_method = convolutional_config.get("stepwise_aggregation_method")
        if stepwise_aggregation_method == "mean":
            self._aggregation_method = torch.mean
        elif stepwise_aggregation_method == "sum":
            self._aggregation_method = torch.sum
        else:
            raise ValueError(f"Unknown aggregation method '{stepwise_aggregation_method}'")

        self._kwargs["convolutional_config"] = convolutional_config

    def forward(self, tensor: torch.Tensor, as_dict: bool = False) -> Union[dict, torch.Tensor]:
        """
        Performs a forward pass through the network using the given input tensor
        Args:
            tensor: Input tensor to be fed through the network
            as_dict: Whether to return the resulting tensors as a dict or simply as is. The latter is more convenient
            to use, the dict version is more flexible
        Returns:

        """
        tensor = self._to_potential_gpu(tensor=tensor)
        timestamps = get_timestamps(tensor=tensor, shaped=True)
        timestamps = self._to_potential_gpu(timestamps)

        # append timestamps
        tensor = torch.cat(tensors=(tensor, timestamps), dim=-1)

        tensor = tensor.transpose(2, 1)  # put the timestamps in the back of the tensor

        stepwise_discriminator_logits = self._convolution(tensor=tensor)
        stepwise_discriminator_logits = stepwise_discriminator_logits.transpose(2, 1)
        # stepwise_discriminator_logits = stepwise_discriminator_logits.squeeze(dim=1)

        # aggregate using mean to get 1 logit per prediction
        discriminator_logits = self._aggregation_method(stepwise_discriminator_logits, dim=1)
        if as_dict:
            return {d.PREDICTIONS: discriminator_logits,
                    d.STEPWISE_LOGITS: stepwise_discriminator_logits}
        else:
            return discriminator_logits
