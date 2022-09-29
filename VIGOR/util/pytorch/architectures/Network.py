"""
Basic PyTorch Network. Defines some often-needed utility functions. Can be instantiated by child classes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.pytorch import Metrics
from util.pytorch.UtilityFunctions import detach


class Network(nn.Module):

    def __init__(self, input_shape: tuple, out_features=1, use_gpu: bool = False):
        """
        Basic Network initialization
        Args:
            input_shape: Dimensionality/Shape of the input data
            out_features: Output dimension of the network
            use_gpu: Whether to keep this network (and all related tensors) on the gpu.
        """
        super().__init__()
        self._kwargs = {"input_shape": input_shape,
                        "out_features": out_features,
                        "type": type(self),
                        "use_gpu": use_gpu}
        self._input_shape = input_shape
        if use_gpu:
            self._gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._gpu_device = None

    @property
    def input_shape(self):
        return self._input_shape

    def load(self, load_path):
        """
        Loads a state_dict saved at the specified path
        Args:
            load_path:

        Returns:

        """
        if not load_path.endswith(".pt"):
            load_path = load_path + ".pt"
        self.eval()  # needs to be here to have dropout etc. consistent
        self.load_state_dict(torch.load(load_path))

    def _to_potential_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor is not None, "Must provide tensor, given {}".format(tensor)
        if self._gpu_device and not tensor.is_cuda:
            return tensor.to(device=self._gpu_device)
        else:
            return tensor  # do nothing

    def evaluate(self, samples, targets, weights=None, metric="acc"):
        """
        Evaluates the given input samples and their targets with respect to the given metric.
        Casts inputs to tensors if necessary, and returns numpy objects
        Args:
            samples: Input samples. Something that can be cast to a tensor and fits the shape of the network
            targets: targets. Must also be able to be cast to a tensor and fit the output shape of the network
            weights: (optional) sample weights. Can be used for weighted metrics.
            metric: The metric to evaluate based on

        Returns:

        """
        with torch.no_grad():
            if not type(samples) == torch.Tensor:
                samples = torch.Tensor(samples)
            samples = self._to_potential_gpu(tensor=samples)
            if not type(targets) == torch.Tensor:
                targets = torch.Tensor(targets)
            targets = self._to_potential_gpu(tensor=targets)
            self.eval()
            predictions = self(samples).squeeze()
            if metric == "acc":
                return Metrics.accuracy(predictions=predictions, targets=targets, weights=weights)
            elif metric == "bce":  # binary_cross_entropy
                evaluation = detach(F.binary_cross_entropy_with_logits(predictions, targets))
            else:
                raise NotImplementedError("Unknown metric '{}'".format(metric))
        return evaluation

    @property
    def kwargs(self):
        return self._kwargs
