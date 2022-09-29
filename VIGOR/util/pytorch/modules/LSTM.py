import torch
import numpy as np
from torch import nn
from util.pytorch.modules.UtilityModules import View
from util.Types import *


class LSTM(nn.Module):
    """
    Simple LSTM wrapper module. This module keeps a single nn.LSTM submodule that it uses to compute its forward()
    """

    def __init__(self, input_shape: Union[int, tuple], lstm_config: dict, regularization_config: dict):
        """
        Initializes a new encoder module that consists of multiple different layers that together encode some
        input into a latent space
        Args:
            input_shape: The input shape for each LSTM step
            lstm_config: Dict containing parameters for the lstm
            regularization_config: Dict containing information about batchnorm , spectral norm and dropout
        """
        super().__init__()
        hidden_dim = lstm_config.get("hidden_dim")
        num_layers = lstm_config.get("num_layers")
        bidirectional = lstm_config.get("is_bidirectional")
        dropout = regularization_config.get("dropout", 0) if num_layers > 1 else 0

        input_size = int(np.prod(input_shape))
        if isinstance(input_shape, tuple) and len(input_shape) > 1:
            # more than one feature dimension. Need to flatten first
            self.flatten = View(default_shape=(input_size,),
                                custom_repr="Flattening individual timestpes feedforward input to 1d.")
        else:
            self.flatten = None

        self.bidirectional: bool = bidirectional
        self.hidden_dim: int = hidden_dim

        if dropout > 0:
            self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers,
                                      dropout=dropout, bidirectional=bidirectional, batch_first=True)
        else:
            self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_dim,
                                      num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forwards the given input tensor using this LSTM cell
        Args:
            tensor: Some input tensor x. Includes timesteps, batch_size and features per step

        Returns: The forwarded tensor f_lstm(x) as a tensor of shape (batch_size, sequence_length, hidden_dim)
        """
        if self.flatten is not None:
            tensor = self.flatten(tensor)
        lstm_outputs, _ = self.lstm_layer(tensor)
        if self.bidirectional:
            # shape=(batch, seq_len, num_directions*hidden_size), so we need to mean up corresponding outputs
            lstm_outputs = (lstm_outputs[:, :, :self.hidden_dim] +
                            torch.flip(lstm_outputs[:, :, self.hidden_dim:], dims=(1,)))/2
            # add up "forward" and "backward" direction in the right order
        return lstm_outputs
