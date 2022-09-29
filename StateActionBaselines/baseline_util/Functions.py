from baseline_util.Types import *


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
            current_dict = current_dict.get(key, None)
        elif current_dict is None:  # key of sub-dictionary not found
            if raise_error:
                raise ValueError("Dict '{}' does not contain list_of_keys '{}'".format(dictionary, list_of_keys))
            else:
                return default_return
    return current_dict  # bottom level reached


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


def plot_box_2d(box_corners: np.array, **kwargs):
    import matplotlib.pyplot as plt
    plt.plot([box_corners[0, 0],
              box_corners[1, 0],
              box_corners[3, 0],
              box_corners[2, 0],
              box_corners[0, 0]],
             [box_corners[0, 1],
              box_corners[1, 1],
              box_corners[3, 1],
              box_corners[2, 1],
              box_corners[0, 1]],
             **kwargs)
