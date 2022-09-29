import collections
from util.Types import *

import numpy as np
from scipy.special import expit
import copy
from functools import partial, update_wrapper


def merge_dictionaries(a: dict, b: dict) -> dict:
    """
    does a deep merge of the given dictionaries
    Args:
        a:
        b:

    Returns: A new dictionary that is a merge version of the two provided ones

    """

    def _merge_dictionaries(dict1, dict2):
        for key in dict2:
            if key in dict1:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    _merge_dictionaries(dict1[key], dict2[key])
                elif dict1[key] == dict2[key]:
                    pass  # same leaf value
                else:
                    dict1[key] = dict2[key]
            else:
                dict1[key] = dict2[key]
        return dict1

    return _merge_dictionaries(copy.deepcopy(a), copy.deepcopy(b))


def sigmoid(x):
    """
    Simple wrapper function for the scipy sigmoid implementation
    Args:
        x:

    Returns:
        sigma(x)=1/1+e^-x
    """
    return expit(x)


def inverse_sigmoid(x):
    """
    sigma^-1(x) = log(x/(1-x))
    Args:
        x:

    Returns: sigma^-1(x) = log(x/(1-x))

    """
    return np.log(x / (1 - x))


def logsumexp(samples: np.ndarray, axis=0) -> np.ndarray:
    """
    Uses the identity
    np.log(np.sum_i(np.exp(sample_i))) = np.log(np.sum_i(np.exp(sample_i-maximum)))+maximum
    to calculate a sum of e.g. densities without numerical loss
    Normally there would be some loss here because one would need to exponentiate the log densities, sum them, and
    then apply the log again.
    :param samples: Samples to perform logsumexp on
    :return: The
    """
    assert len(samples) > 0, "Must have at least 1 sample to logsumexp it. Given {}".format(samples)
    maximum_log_density = np.max(samples, axis=axis)
    offset_term = samples - maximum_log_density
    exp_term = np.exp(offset_term)
    sum_term = np.sum(exp_term, axis=axis)
    log_term = np.log(sum_term)
    results = log_term + maximum_log_density
    return results


def joint_bootstrap(*args):
    """
    Jointly bootstrap np arrays with the same length and possibly different dimensions
    Args:
        args: A number of arrays

    Returns:
        A bootstrapped version of the arrays, i.e., n random samples with replacement for arrays of length n

    """
    first_array = args[0]
    assert all(len(first_array) == len(other_array) for other_array in args), "All arrays must have same length"
    num_samples = len(first_array)
    choices = np.random.choice(num_samples, num_samples)
    return (array[choices] for array in args)


def joint_shuffle(*args):
    """
    Jointly shuffles all given numpy arrays with the same length and possibly different dimensions
    Args:
        args: A number of arrays

    Returns:
        A random permutation of all arrays. E.g.
        ([a2, a1, a4, a3], [b2, b1, b4, b3], [c2, c1, c4, c3], ...)
    """
    first_array = args[0]
    assert all(len(first_array) == len(other_array) for other_array in args), "All arrays must have same length"
    permutation = np.random.permutation(len(first_array))
    return (array[permutation] for array in args)


def joint_sort(a, b, *args, reverse=False):
    """
    Jointly sorts np arrays a and b with same length and possibly different dimensions
    Args:
        a: Some sortable 1d-array
        b: Some other array
        reverse: Whether to flip an array or not
        args: Additional arrays to be sorted alongside a and b. Will be treated like b

    Returns:
        The sorted array a and the array b sorted in the same way

    """
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)

    if reverse:
        positions = np.squeeze(-a).argsort()
    else:
        positions = np.squeeze(a).argsort()

    if args:
        additional_arrays = []
        for arr in args:
            if isinstance(arr, list):
                arr = np.array(arr)
            additional_arrays.append(arr[positions])
        return (a[positions], b[positions], *additional_arrays)
    else:
        return a[positions], b[positions]


def save_concatenate(*args):
    """
    Concatenates the given arrays along axis 0, ignoring arrays that are None or have length 0
    Args:

    Returns: The concatenated array. a if b is None and vice versa

    """
    non_nones = [array for array in args if array is not None and len(array)]
    return np.concatenate(non_nones, axis=0)


def get_from_nested_dict(dictionary: Dict[Any, Any], list_of_keys: List[Any],
                         raise_error: bool = False,
                         default_return: Optional[Any] = None) -> Any:
    """
    Utility function to traverse through a nested dictionary. For a given dict d and a list of keys [k1, k2, k3], this
    will return d.get(k1).get(k2).get(k3) if it exists, and default_return otherwise
    Args:
        dictionary: The dictionary to search through
        list_of_keys: List of keys in the order to traverse the dictionary by
        raise_error: Raises an error if any of the keys is missing in its respective subdict. If false, returns the
        default_return instead
        default_return: The thing to return if the dictionary does not contain the next key at some level

    Returns:

    """
    current_dict = dictionary
    for key in list_of_keys:
        if isinstance(current_dict, dict):  # still traversing dictionaries
            if key in current_dict:
                current_dict = current_dict.get(key)
            else:
                return default_return
        elif current_dict is None:  # key of sub-dictionary not found
            if raise_error:
                raise ValueError("Dict '{}' does not contain list_of_keys '{}'".format(dictionary, list_of_keys))
            else:
                return default_return
    return current_dict  # bottom level reached


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def format_title(title: str, linebreaks: bool = True) -> str:
    if linebreaks:
        title = title.replace(" ", "\n").replace("/", "\n")
    return title.replace("_", " ").title()


def get_first_satisfying_id(list_of_items: list, condition: Callable, default=None) -> int:
    """
    Returns the first id that satisfies the given condition in the list and the default value if there is None
    Args:
        list_of_items:
        condition:
        default:

    Returns:

    """
    try:
        return next(idx for idx, element in enumerate(list_of_items) if condition(element))
    except StopIteration:
        return default


def symlog_spacing(data, num_positions, epsilon=1e-2):
    """
    Returns num_positions values that evenly partition the data with respect to its symmetric-logarithmic representation
    Args:
        data:
        num_positions:

    Returns:

    """

    if isinstance(data, list):
        data = np.array(data)
    data_max = np.max(data)
    data_min = np.min(data)
    min_sign = np.sign(np.min(data))
    max_sign = np.sign(data_max)
    if data_min == data_max:
        return np.array([data_min - epsilon, data_min, data_min + epsilon])

    elif min_sign == max_sign:
        if min_sign == 1:  # only positives
            return np.logspace(np.log10(data_min * (1 - epsilon)), np.log10(data_max * (1 + epsilon)),
                               num=num_positions)
        else:  # only negatives
            symlog_spacing = -(
                np.logspace(np.log10(np.min(-data) * (1 - epsilon)), np.log10(np.max(-data) * (1 + epsilon)),
                            num=num_positions))
            return symlog_spacing[::-1]
    else:  # both positive and negative values
        if num_positions == 3:
            return np.array([data_min, 0, data_max])
        log_max = np.log10(data_max + epsilon)
        log_min = np.log10(-(data_min - epsilon))

        num_positive_positions = num_positions // 2
        num_negative_positions = num_positions - num_positive_positions - 1
        positive_spacing = np.logspace(np.log10(epsilon / 10), log_max, num=num_positive_positions + 1)[1:]
        # add an additional draw and cutoff the first value to not oversample around 0
        negative_spacing = np.logspace(np.log10(epsilon / 10), log_min, num=num_negative_positions + 1)[1:]
        negative_spacing = -negative_spacing[::-1]
        spacing = np.concatenate((negative_spacing, [0], positive_spacing), axis=0)
        return spacing


def interweave(arr1: np.array, arr2: np.array):
    """
    Interweave two n-dimensional arrays on their first dimension. All other dimensions must have the same shape
    :param arr1: An array, e.g. [1,2,3]     or [1,2]
    :param arr2: Another array, e.g. [4,5,6]    or [3,4,5,6]
    :return: The interweaved array, e.g. [1,4,2,5,3,6]      or [1,3,4,2,5,6]
    """
    if len(arr1) == 0:
        return arr2
    if len(arr2) == 0:
        return arr1

    assert arr1.ndim == arr2.ndim, "Arrays must have same dimensions. Shapes given are {} and {}".format(arr1.shape,
                                                                                                         arr2.shape)
    assert arr1.shape[1:] == arr2.shape[1:], "Arrays must have same size everywhere but in the first dimension. " \
                                             "Shapes given are {} and {}".format(arr1.shape, arr2.shape)
    total_length = arr1.shape[0] + arr2.shape[0]
    interwoven_array = np.empty((total_length, *arr1.shape[1:]))
    # create a new array with equal shape for all other dimensions

    if arr1.shape == arr2.shape:
        interwoven_array[0::2] = arr1
        interwoven_array[1::2] = arr2

    else:  # unbalanced train/test data
        ratios = [x * arr1.shape[0] / total_length for x in range(total_length)]
        ratios = np.floor(ratios)
        diff = np.diff(ratios, append=ratios[-1])  # finds out where the floors change, i.e. a new index begins
        # the append is important because otherwise the last index may fail
        arr1_indizes = np.where(np.r_[1, diff[:-1]])[0]

        arr2_indizes = np.ones(interwoven_array.shape[0], bool)
        arr2_indizes[arr1_indizes] = False  # implemented as a boolean mask

        interwoven_array[arr1_indizes, ...] = arr1
        interwoven_array[arr2_indizes, ...] = arr2
    return interwoven_array


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def first_positive(array: np.ndarray, axis: int, invalid_value: int = -1):
    """
    Find the first non-zero entry along axis, or return invalid_value otherwise
    Args:
        array:
        axis:
        invalid_value:

    Returns:

    """
    positive_mask = array > 0
    return np.where(positive_mask.any(axis=axis), positive_mask.argmax(axis=axis), invalid_value)


def prefix_keys(dictionary: Dict[str, Any], prefix: Union[str, List[str]], separator: str = "/") -> Dict[str, Any]:
    """
    Utility function to add a prefix or a sorted list of prefix with a given separator to all keys of a dictionary.
    Does not consider nested dictionaries

    Example:
        prefix_keys(dictionary = {"a": 123, "b": 456}, prefix="X", separator = ".") --> {"X.a": 123, "X.b": 456}
        prefix_keys(dictionary = {"a": 123, "b": 456}, prefix=["X", "YZ"]) --> {"X/YZ/a": 123, "X/YZ/b": 456}
    Args:
        dictionary:
        prefix: The prefix to add before each key. May be a list, in which case the prefixes are added in this order
        separator: The separator to use between prefix and actual key. For a list of prefixes, the separator is added
          after every key

    Returns: The prefixed dictionary

    """
    if isinstance(prefix, str):
        prefix = [prefix]
    prefix = separator.join(prefix + [""])
    return {prefix + k: v for k, v in dictionary.items()}


def to_contiguous_float_array(two_dim_array: np.array) -> np.array:
    return np.ascontiguousarray([np.array(x, dtype=np.float64) for x in two_dim_array])


def get_2d_rototranslation_matrix(context: np.array, inverse: bool = False) -> np.ndarray:
    """

    Args:
        context: An array of shape 3 with entries (x_offset, y_offset, angle) with
            angle:
            x_offset: How far to move/translate in x direction
            y_offset: How far to move/translate in y direction
        inverse: If True, creates the inverse rototranslation matrix

    Returns: A 3x3 matrix describing a rototranslation in 3d

    """
    x_offset, y_offset, angle = context
    angle_in_rad = angle / 180 * np.pi
    rototranslation_matrix = np.array([
        [np.cos(angle_in_rad), -np.sin(angle_in_rad), x_offset],
        [np.sin(angle_in_rad), np.cos(angle_in_rad), y_offset],
        [0, 0, 1]])
    if inverse:
        rototranslation_matrix = np.linalg.inv(rototranslation_matrix)
    return rototranslation_matrix


def apply_rototranslation(pointwise_array: np.array, rototranslation_matrix: np.array) -> np.array:
    """
    Applies the given rototranslation to the set of points

    Args:
        pointwise_array: Array of points of shape (..., 2)
        rototranslation_matrix: 2d rototranslationmatrix of shape (3,3)

    Returns: A rotatranslated array of shape (..., 2)

    """
    extended_array = np.ones((*pointwise_array.shape[:-1], 3))
    extended_array[..., :2] = pointwise_array
    rotated_pointwise_array = np.einsum("...i, ji->...j", extended_array, rototranslation_matrix)
    return rotated_pointwise_array[..., :2]

def apply_3drotation(pointwise_array: np.array, rotation_matrix: np.array) -> np.array:
    """
    Applies the given 3d rotation matrix to the set of points

    Args:
        pointwise_array: Array of points of shape (..., 3)
        rototranslation_matrix: 2d rototranslationmatrix of shape (3,3)

    Returns: A rotatranslated array of shape (..., 2)

    """
    rotated_pointwise_array = np.einsum("...i, ...ji->...j", pointwise_array, rotation_matrix)
    return rotated_pointwise_array


def append_timestamps(features: np.array) -> np.array:
    """
    Appends normalized timestamps (in [0,1]) to each sample of the given array of features
    Args:
        features: Array of shape (..., #timesteps, :)

    Returns:

    """
    num_rollouts = features.shape[:-2]
    num_timesteps = features.shape[-2]
    timestamps = np.linspace(start=0, stop=1, num=num_timesteps)
    timestamps = np.tile(timestamps, (*num_rollouts, 1))
    features = np.concatenate((features,
                               timestamps[..., None]),
                              axis=-1)
    return features

def unflatten_dict(dictionary, delimiter: str = "."):
    result_dict = {}
    for key, value in dictionary.items():
        parts = key.split(delimiter)
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result_dict

