from util.Types import *

import numpy as np

from util.Functions import logsumexp


def normalize_samples(samples: np.array, normalization_type: str = "bound", value_range: Union[list, tuple] = (-1, 1),
                      return_dict: bool = True) -> Union[Tuple[np.array, dict], np.array]:
    """
    Normalizes the given samples to either a fixed bound or according to a normal distribution. Optionally returns
    the parameters used for the normalization to allow to undo it. The normalization is based on the last axis of
    data.
    :param samples: The samples to normalize
    :param normalization_type: "bound" for hard bounds, "normal" for a zmuc normal distribution.
    :param value_range: The range that the values can take in each dimension. Only used for normalization_type=="bound"
    :return: Either the samples normalized according to the normalization type, or both the samples and a dict of
        the parameters that were used for the normalization
    """
    if normalization_type is None:
        if return_dict:
            return samples, None
        else:
            return samples
    elif normalization_type == "bound":  # [-1, 1]^d
        if samples.ndim >= 2:  # normalize last dimension
            axis = tuple(np.arange(samples.ndim-1))
        else:
            axis = 0
        min_value = np.array(np.min(samples, axis=axis))  # dimension-wise minimum
        max_value = np.array(np.max(samples, axis=axis))
        if np.all(min_value == max_value):
            # same values everywhere
            return np.full(samples.shape, value_range[1]), {"all_equal": True}
        normalization_parameters = {"min": min_value, "max": max_value}
    elif normalization_type in ["distribution", "zmuc", "normal"]:  # N(0,I) normalization
        if samples.ndim >= 2:  # normalize last dimension
            axis = tuple(np.arange(samples.ndim-1))
        else:
            axis = 0
        mean = np.mean(samples, axis=axis)
        std = np.std(samples, axis=axis)
        normalization_parameters = {"mean": mean, "std": std}
    else:
        raise NotImplementedError("Unknown normalization normalization_type '{}'".format(normalization_type))
    samples = normalize_from_dict(samples=samples, normalization_parameters=normalization_parameters,
                                  value_range=value_range)
    if return_dict:
        return samples, normalization_parameters
    else:
        return samples


def normalize_from_dict(samples: np.array, normalization_parameters: Optional[Dict[str, Any]], value_range=(-1, 1)):
    """
    Normalizes a set of samples according to a given dictionary.
    :param samples: A numpy array of samples
    :param normalization_parameters: Dictionary containing either mean/std or min/max of the data to be normalized
    :param value_range: The range that the values can take in each dimension
    :return: Normalized samples
    """
    if samples is None:
        return None
    if normalization_parameters is None:
        return samples
    elif "mean" in normalization_parameters:
        normalized_samples = (samples - normalization_parameters["mean"]) / normalization_parameters["std"]
    elif "min" in normalization_parameters:
        min_value = normalization_parameters["min"]
        max_value = normalization_parameters["max"]
        normalized_samples = ((value_range[1] - value_range[0]) * ((samples - min_value) / (max_value - min_value))) + \
                             value_range[0]
    else:
        raise NotImplementedError("Unknown normalization_parameters '{}'".format(normalization_parameters))
    return normalized_samples


def denormalize(samples, normalization_parameters: Optional[Dict[str, Any]]):
    """
    de-normalizes the given samples using the provided normalization_parameters.
    Automatically looks for both bounded normalization (in [a,b]^d)
    and data that is normalized via a normal distribution
    :param samples: The normalized samples assumed to roughly in [-1,1]^d
    :param normalization_parameters: A dictionary {min: min_value, max:max_value}
    :return: The denormalized samples
    """
    if normalization_parameters is None:
        return samples
    if isinstance(samples, list):
        samples = np.array(samples)
    if "all_equal" in normalization_parameters:
        return samples
    if "mean" in normalization_parameters:
        denormalized_samples = (samples * normalization_parameters["std"]) + normalization_parameters["mean"]
    elif "min" in normalization_parameters:
        min_value = normalization_parameters["min"]
        max_value = normalization_parameters["max"]
        denormalized_samples = (max_value - min_value) * ((samples + 1) / 2) + min_value
    else:
        raise NotImplementedError("Unknown normalization_parameters '{}'".format(normalization_parameters))
    return denormalized_samples


def log_normalize(samples: np.ndarray) -> np.ndarray:
    """
    Normalizes the given samples in logspace, i.e. returns an offset of the samples such that
    sum(exp(samples))==1
    Args:
        samples: The samples to log_normalize

    Returns: Log-normalized samples

    """
    return samples - logsumexp(samples)


def normalize_function(function: Callable, normalization_parameters: dict):
    """
    Normalizes the given function by denormalizing all inputs wrt to the given normalization parameters. This effectively
    normalizes the function by reversing the normalization of the input
    Args:
        function: Function to normalize. Must take some input x
        normalization_parameters: Dict containing details about normalization

    Returns: The function that can deal with normalized inputs

    """
    return lambda x: function(denormalize(x, normalization_parameters=normalization_parameters))
