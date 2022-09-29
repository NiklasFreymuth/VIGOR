import torch
from torch import nn
from util.pytorch.UtilityFunctions import reparameterize
from util.Types import *


class View(nn.Module):
    """
    Custom Layer to change the shape of an incoming tensor.
    """

    def __init__(self, default_shape: Union[tuple, int], custom_repr: str = None):
        """
        Utility layer to reshape an input into the given shape. Auto-converts for different batch_sizes
        Args:
            default_shape: The shape (without the batch size) to convert to
            custom_repr: Custom representation string of this layer.
            Shown when printing the layer or a network containing it
        """
        import numpy as np
        super(View, self).__init__()
        if isinstance(default_shape, (int, np.int32, np.int64)):
            self._default_shape = (default_shape,)
        elif isinstance(default_shape, tuple):
            self._default_shape = default_shape
        else:
            raise ValueError("Unknown type for 'shape' parameter of View module: {}".format(default_shape))
        self._custom_repr = custom_repr

    def forward(self, tensor: torch.Tensor, shape: Optional[Iterable] = None) -> torch.Tensor:
        """
        Shape the given input tensor with the provided shape, or with the default_shape if None is provided
        Args:
            tensor: Input tensor of arbitrary shape
            shape: The shape to fit the input into

        Returns: Same tensor, but shaped according to the shape parameter (if provided) or self._default_shape)

        """
        tensor = tensor.contiguous()  # to have everything nicely arranged in memory, which is a requisite for .view()
        if shape is None:
            return tensor.view((-1, *self._default_shape))  # -1 to deal with different batch sizes
        else:
            return tensor.view((-1, *shape))

    def extra_repr(self) -> str:
        """
        To be printed when using print(self) on either this class or some module implementing it
        Returns: A string containing information about the parameters of this module
        """
        if self._custom_repr is not None:
            return "{} default_shape={}".format(self._custom_repr, self._default_shape)
        else:
            return "default_shape={}".format(self._default_shape)


class SaveBatchnorm1d(nn.BatchNorm1d):
    """
    Custom Layer to deal with Batchnorms for 1-sample inputs
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size = tensor.shape[0]
        if batch_size == 1 or len(tensor.shape) == 1:
            return tensor
        else:
            return super().forward(input=tensor)


class GhostBatchnorm(nn.Module):
    """
    Implements Ghost Batch Normalization using either 1d or 2d batch normalization.
    Ghost Batch Normalization essentially consists of multiple small batch normalization instances
    that act on different subsets of the input data, thus increasing the noise of the calculated means and stds
    Code adapted from https://medium.com/deeplearningmadeeasy/ghost-batchnorm-explained-e0fa9d651e03
    """

    def __init__(self, num_features: int, num_ghost_batches: int = 8, batchnorm_dimension: int = 1, *args, **kwargs):
        """
        Initializes Ghost Batch Normalization for the given number of parallel batchnorms
        Args:
            num_features: Number of input features/dimension of the layer
            num_ghost_batches: Number of ghost batches to run in parallel. Defaults to 8
            *args: Additional arguments for the batchnorm layer
            **kwargs: Additional keyword arguments for the batchnorm layer
        """
        assert batchnorm_dimension == 1 or batchnorm_dimension == 2, "Must be 1d or 2d batchnorm"
        super().__init__()
        self.num_ghost_batches = num_ghost_batches
        if batchnorm_dimension == 1:
            self.batchnorm = SaveBatchnorm1d(num_features, *args, **kwargs)
        else:  # batchnorm_dimension == 2:
            self.batchnorm = nn.BatchNorm2d(num_features, *args, **kwargs)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ghost_batches = tensor.chunk(self.num_ghost_batches, 0)
        ghost_batches = [self.batchnorm(ghost_batch) for ghost_batch in ghost_batches]
        return torch.cat(ghost_batches, dim=0)

    def extra_repr(self) -> str:
        """
        To be printed when using print(self) on either this class or some module implementing it
        Returns: A string containing information about the parameters of this module
        """
        return 'num_features={}, num_ghost_batches={}'.format(self.batchnorm.num_features, self.num_ghost_batches)


def add_activation_and_regularization_layers(torch_module_list: nn.ModuleList, in_features: int,
                                             regularization_config: Dict[Key, Any]) -> nn.ModuleList:
    """
    Takes an existing torch module list and potentially appends activation functions, dropout and normalization
    methods
    Args:
        torch_module_list:
        in_features:
        regularization_config:

    Returns: The same moduleList but with appended activation and regularization layers

    """
    dropout = regularization_config.get("dropout")
    batch_norm = regularization_config.get("batch_norm")
    layer_norm = regularization_config.get("layer_norm")
    activation_function: str = regularization_config.get("activation_function").lower()

    # add activation function
    if activation_function == "leakyrelu":
        torch_module_list.append(nn.LeakyReLU())
    elif activation_function == "swish":
        torch_module_list.append(nn.SiLU())
    else:
        raise ValueError("Unknown activation function '{}'".format(activation_function))

    # add normalization method
    if batch_norm:
        if isinstance(batch_norm, str) and batch_norm.lower() == "ghost":
            torch_module_list.append(GhostBatchnorm(num_features=in_features))
        else:
            torch_module_list.append(SaveBatchnorm1d(num_features=in_features))
    elif layer_norm:
        torch_module_list.append(nn.LayerNorm(normalized_shape=in_features))

    # add dropout
    if dropout:
        torch_module_list.append(nn.Dropout(p=dropout))
    return torch_module_list
