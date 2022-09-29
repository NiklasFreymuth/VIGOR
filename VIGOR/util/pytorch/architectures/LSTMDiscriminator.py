import torch
from torch import nn
from util.Types import *
from util.pytorch.architectures.Network import Network
from util.pytorch.modules.LSTM import LSTM
from util.pytorch.modules.UtilityModules import View
from util.pytorch.modules.Feedforward import Feedforward as FeedforwardNetwork
import util.Defaults as d
import numpy as np
import copy


class LSTMDiscriminator(Network):
    """
    Creates a discriminator for time-series data using an LSTM backbone. We use the self._feedforward-network as the
    "decoder" part of the LSTM, i.e., we feed the data through the LSTM and then compute self._feedforward(outputs)
    for all outputs to get a final representation.
    Depending on lstm_config.get("aggregate_timesteps"), the output is then either the final representation for
    the last timestep, or the mean of the final representations of all timesteps. In either case, we compute and log
    all timesteps.
    To get a richer initial representation into the LSTM, we optionally also include an encoder (another feedforward
    network) which is specified in the lstm_config.
    """

    def __init__(self, input_shape: tuple, feedforward_config: dict, lstm_config: dict, regularization_config: dict,
                 out_features: int = 1, use_gpu: bool = False):
        super().__init__(input_shape=input_shape,
                         out_features=out_features, use_gpu=use_gpu)
        self._kwargs = {**self._kwargs,
                        "lstm_config": lstm_config,
                        "regularization_config": regularization_config,
                        "feedforward_config": feedforward_config
                        }

        self._aggregate_timesteps = lstm_config.get("aggregate_timesteps")
        # whether to aggregate over outputs for all timesteps (true) or only use last output (false)

        lstm_dim = lstm_config.get("hidden_dim")

        self._timestamps = lstm_config.get("timestamps", False)
        sequence_length = input_shape[0]
        self._shaping_modules = nn.ModuleDict()  # to process all lstm in-&outputs with the same network
        self._shaping_modules["output_unshape"] = View(default_shape=lstm_dim, custom_repr="Flattening over time_steps")
        self._shaping_modules["output_reshape"] = View(default_shape=(sequence_length, 1),
                                                       custom_repr="Restoring shapes for each time_step")

        stepwise_input_size = np.prod(input_shape[1:])

        subnetworks = self._get_subnetworks(lstm_dim=lstm_dim,
                                            out_features=out_features,
                                            feedforward_config=copy.deepcopy(feedforward_config),
                                            regularization_config=regularization_config,
                                            stepwise_input_size=stepwise_input_size,
                                            lstm_config=lstm_config,
                                            sequence_length=sequence_length
                                            )
        self._encoder, self._lstm, self._feedforward = subnetworks

    def _get_subnetworks(self, lstm_dim: int, out_features, feedforward_config: Dict[Key, Any],
                         regularization_config: Dict[Key, Any], stepwise_input_size: int, lstm_config: Dict[Key, Any],
                         sequence_length: int) -> List:
        if feedforward_config.get("max_neurons_per_layer", None) == "tied":
            feedforward_config["max_neurons_per_layer"] = lstm_config.get("hidden_dim")

        decoder = FeedforwardNetwork(in_features=lstm_dim, out_features=out_features,
                                     feedforward_config=feedforward_config,
                                     regularization_config=regularization_config)
        lstm_input_dim, encoder = self._get_encoder(stepwise_input_size=stepwise_input_size,
                                                    lstm_config=lstm_config,
                                                    regularization_config=regularization_config,
                                                    sequence_length=sequence_length)

        self._lstm_input_dim = lstm_input_dim
        if self._timestamps in ["lstm", "both"]:
            lstm_hidden_dim = lstm_input_dim + 1
        else:
            lstm_hidden_dim = lstm_input_dim
        lstm = LSTM(input_shape=lstm_hidden_dim,
                    lstm_config=lstm_config,
                    regularization_config=regularization_config)
        return [encoder, lstm, decoder]

    def _get_encoder(self, stepwise_input_size: int, lstm_config: Dict[Key, Any], regularization_config: Dict[Key, Any],
                     sequence_length: int) -> tuple:
        if lstm_config.get("use_encoder") and lstm_config.get("encoder").get("num_layers") > 0:
            encoder_config = copy.deepcopy(lstm_config.get("encoder"))
            if encoder_config.get("max_neurons_per_layer", None) == "tied":
                encoder_config["max_neurons_per_layer"] = lstm_config.get("hidden_dim")

            if self._timestamps in ["encoder", "both", True]:
                encoder_in_features = stepwise_input_size + 1  # add timestamps as additional features to encoder
            else:
                encoder_in_features = stepwise_input_size
            encoder = FeedforwardNetwork(in_features=encoder_in_features,
                                         feedforward_config=encoder_config,
                                         regularization_config=regularization_config)
            lstm_input_dim = encoder.network_layout[-1]
            self._shaping_modules["input_unshape"] = View(default_shape=stepwise_input_size,
                                                          custom_repr="Flattening over time_steps for stepwise preprocessing")
            self._shaping_modules["input_reshape"] = View(default_shape=(sequence_length, lstm_input_dim),
                                                          custom_repr="Restoring time_steps to full trajectories.")
        else:
            encoder = None
            lstm_input_dim = stepwise_input_size
        return lstm_input_dim, encoder

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

        stepwise_discriminator_logits = self.get_stepwise_logits(encoder=self._encoder, lstm=self._lstm,
                                                                 decoder=self._feedforward, tensor=tensor)

        if self._aggregate_timesteps:
            discriminator_logits = torch.mean(stepwise_discriminator_logits, dim=1)  # turn into 1 logit per prediction
        else:
            discriminator_logits = stepwise_discriminator_logits[:, -1, :]  # only consider last logit
        if as_dict:
            return {d.PREDICTIONS: discriminator_logits,
                    d.STEPWISE_LOGITS: stepwise_discriminator_logits}
        else:
            return discriminator_logits

    def get_stepwise_logits(self, encoder: FeedforwardNetwork, lstm: LSTM, decoder: FeedforwardNetwork,
                            tensor: torch.Tensor) -> torch.Tensor:
        num_timesteps = tensor.shape[1]
        num_batches = tensor.shape[0]

        if self._timestamps in ["encoder", "both", True]:
            timestamps = torch.linspace(start=0, end=1, steps=num_timesteps)
            timestamps = self._to_potential_gpu(timestamps)
            timestamps = timestamps.repeat(num_batches).unsqueeze(dim=1)
            tensor = self._shaping_modules["input_unshape"](tensor)

            # append timestamps
            tensor = torch.cat(tensors=(tensor, timestamps), dim=1)

        else:
            tensor = self._shaping_modules["input_unshape"](tensor)

        tensor = encoder(tensor=tensor)
        tensor = self._shaping_modules["input_reshape"](tensor=tensor,
                                                        shape=(num_timesteps, self._lstm_input_dim))

        if self._timestamps in ["lstm", "both"]:
            timestamps = torch.linspace(start=0, end=1, steps=num_timesteps)
            timestamps = self._to_potential_gpu(timestamps)
            timestamps = timestamps.repeat(num_batches, 1).unsqueeze(dim=-1)
            tensor = torch.cat(tensors=(tensor, timestamps), dim=-1)
            lstm_output = lstm(tensor)
        else:
            lstm_output = lstm(tensor)
        # get stepwise logits. These are always needed for logging purposes, though we only use them in the
        # algorithm for self._aggregate_timesteps = True
        unshaped_output = self._shaping_modules["output_unshape"](lstm_output)
        stepwise_discriminator_logits = decoder(tensor=unshaped_output)
        stepwise_discriminator_logits = self._shaping_modules["output_reshape"](stepwise_discriminator_logits,
                                                                                shape=(num_timesteps, 1))
        return stepwise_discriminator_logits
