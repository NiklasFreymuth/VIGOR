import numpy as np

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

angles = np.load("angles.npz")["arr_0"]
geometric = np.load("geometric.npz")["arr_0"]
samples = np.load("samples.npz")["arr_0"]


shuffled_angles = []
shuffled_geometric = []
shuffled_samples = []

for _angles, _geometric, _samples in zip(angles, geometric, samples):
    _shuffled_angles, _shuffled_geometric, _shuffled_samples = joint_shuffle(_angles, _geometric, _samples)
    shuffled_angles.append(_shuffled_angles)
    shuffled_geometric.append(_shuffled_geometric)
    shuffled_samples.append(_shuffled_samples)
shuffled_angles = np.array(shuffled_angles)
shuffled_geometric = np.array(shuffled_geometric)
shuffled_samples = np.array(shuffled_samples)

np.savez_compressed("samples.npz", shuffled_samples)
np.savez_compressed("angles.npz", shuffled_angles)
np.savez_compressed("geometric.npz", shuffled_geometric)

