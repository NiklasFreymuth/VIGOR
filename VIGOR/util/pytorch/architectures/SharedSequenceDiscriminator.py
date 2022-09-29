import torch
from torch import nn
from util.Types import *

from util.pytorch.UtilityFunctions import get_timestamps
from util.pytorch.architectures.Feedforward import Feedforward
from util.pytorch.modules.UtilityModules import View
import numpy as np
import util.Defaults as d


class SharedSequenceDiscriminator(Feedforward):
    """
    Creates a simple discriminator for markovian sequences. For a given sequence, every element is fed through a
    regular discriminator. The resulting logits are then aggregated using mean aggregation and used for a common output.
    Since this is precisely the same thing the regular Feedforward class does (except for the aggregation), we pretty
    much only need to deal with another tensor dimension (the sequence steps) in the forward and tell the architecture
    to aggregate over that.
    The easiest way to do this is to reshape this additional dimension into the batchsize and undo that afterwards
    """

    def __init__(self, input_shape: tuple, feedforward_config: Dict[Key, Any], regularization_config: dict,
                 shared_mlp_config: Dict[Key, Any], out_features: int = 1, use_gpu: bool = False):
        include_next_step = shared_mlp_config.get("include_next_step")

        sequence_length = input_shape[0]
        stepwise_dimension = int(np.prod(input_shape[1:]))
        if include_next_step:
            stepwise_dimension = 2 * stepwise_dimension  # include features of next step
            sequence_length = sequence_length - 1

        num_timestamp_features = 1  # encode timestamp as normalized value in [0,1]
        super().__init__(input_shape=input_shape,
                         latent_dimension=stepwise_dimension + num_timestamp_features,
                         feedforward_config=feedforward_config,
                         regularization_config=regularization_config,
                         out_features=out_features, use_gpu=use_gpu)

        self._shaping_modules = nn.ModuleDict(
            {"unshape": View(default_shape=stepwise_dimension,
                             custom_repr="Flattening over time_steps to stepwise observations."),
             "reshape": View(default_shape=(sequence_length, 1),
                             custom_repr="Restoring time_steps to full trajectories.")})
        self._kwargs["shared_mlp_config"] = shared_mlp_config
        self._include_next_step = include_next_step

    def forward(self, tensor: torch.Tensor, as_dict: bool = False) -> Union[dict, torch.Tensor]:
        """
        Performs a forward pass through the sequenceDiscriminator network using the given input tensor
        Args:
            tensor: Input tensor to be fed through the network. Has shape (batch_size, sequence_length, num_features)
            as_dict: Whether to return the resulting tensors as a dict or simply as is. The latter is more convenient
            to use, the dict version is more flexible
        Returns: Either a tensor with the logits or a dictionary with the predictions.
        In this case, the dict just contains the discriminator logits

        """
        tensor = self._to_potential_gpu(tensor=tensor)
        if self._include_next_step:
            # with this, we effectively reduce the number of steps by 1, because we would need to pad otherwise.
            next_steps = tensor[:, 1:, ...]
            tensor = torch.cat(tensors=(tensor[:, :-1, ...], next_steps), dim=-1)  # include next steps as features

        timestamps = get_timestamps(tensor=tensor)
        timestamps = self._to_potential_gpu(timestamps)

        tensor: torch.Tensor = self._shaping_modules["unshape"](tensor)
        # while hacky, this whole shaping thing is much faster than looping over timesteps.

        # append timestamps
        tensor = torch.cat(tensors=(tensor, timestamps), dim=1)

        stepwise_discriminator_logits = self._feedforward(tensor=tensor)
        stepwise_discriminator_logits = self._shaping_modules["reshape"](stepwise_discriminator_logits)

        # aggregate using mean to get 1 logit per prediction
        discriminator_logits = torch.mean(stepwise_discriminator_logits, dim=1)
        if as_dict:
            return {d.PREDICTIONS: discriminator_logits,
                    d.STEPWISE_LOGITS: stepwise_discriminator_logits}
        else:
            return discriminator_logits
