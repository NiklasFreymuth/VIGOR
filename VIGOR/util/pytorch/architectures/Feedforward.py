import torch
from util.pytorch.architectures.Network import Network
import util.Defaults as d
from util.Types import *

from util.pytorch.modules.Feedforward import Feedforward as Feedforward_module

class Feedforward(Network):
    def __init__(self, input_shape: Union[int, tuple], feedforward_config: Dict[Key, Any],
                 regularization_config: Dict[Key, Any], out_features: int = 1,
                 latent_dimension: int = None, use_gpu: bool = False):
        """
        Creates a discriminator based on the given specifications
        Args:
            input_shape: Dimensionality/Shape of the input data
            feedforward_config: Dictionary containing the specification for feedforward network. Includes max_neurons,
            num_layers and a general shape
            regularization_config: Dict containing information about batchnorm , spectral norm and dropout
            out_features: Dimensionality of the output
            discriminator. Used for image-based observations
            latent_dimension: (Optional) Size of the latent space that this discriminator acts upon. If not specified,
            this will be set to input_shape
            use_gpu: Whether to keep this network (and all related tensors) on the gpu.
        """
        super().__init__(input_shape=input_shape, out_features=out_features,
                         use_gpu=use_gpu)

        # append "new" information to kwargs
        self._kwargs = {**self._kwargs,
                        "feedforward_config": feedforward_config,
                        "regularization_config": regularization_config}
        in_features = latent_dimension if latent_dimension is not None else input_shape

        self._feedforward = Feedforward_module(in_features=in_features, out_features=out_features,
                                               feedforward_config=feedforward_config,
                                               regularization_config=regularization_config)

    def forward(self, tensor: torch.Tensor, as_dict: bool = False) -> Union[dict, torch.Tensor]:
        """
        Performs a forward pass through the discriminator network using the given input tensor
        Args:
            tensor: Input tensor to be fed through the network. Has shape (batch_size, num_features)
            as_dict: Whether to return the resulting tensors as a dict or simply as is. The latter is more convenient
            to use, the dict version is more flexible
        Returns: Either a tensor with the logits or a dictionary with the predictions.
        In this case, the dict just contains the discriminator logits

        """
        tensor = self._to_potential_gpu(tensor=tensor)
        discriminator_logits = self._feedforward(tensor=tensor)
        if as_dict:
            return {d.PREDICTIONS: discriminator_logits}
        else:
            return discriminator_logits

    @property
    def feedforward_network(self) -> Feedforward_module:
        return self._feedforward
