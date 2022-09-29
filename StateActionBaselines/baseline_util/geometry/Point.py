import numpy as np
import matplotlib.pyplot as plt


class Point:
    """
    Object-oriented representation of a 2d point
    """

    def __init__(self, x_coordinate: float, y_coordinate: float):
        """
        Args:
            x_coordinate: x position of the point
            y_coordinate: y position of the point
        """
        self._position = np.array([x_coordinate, y_coordinate])

    def get_distances(self, samples: np.array) -> np.array:
        """
        Gets the shortest distance of every point in points to this one
        Args:
            samples: An array of shape (..., 2) of points

        Returns: An array of shape (...) where each entry is the closest distance of the respective point to this point

        """
        assert samples.shape[-1] == 2, "Need to have x and y coordinate in last dimension, " \
                                      f"provided array of shape '{samples.shape}'"
        return np.linalg.norm(samples - self._position, axis=-1)

    def distance_to(self, other_point) -> float:
        return np.linalg.norm(self._position - other_point.position)

    def plot(self, position: int = None, **kwargs):
        if "zorder" not in kwargs:
            kwargs["zorder"] = 100
        if position is not None:
            kwargs.setdefault("markersize", 12)
            kwargs.setdefault("markerfacecolor", kwargs.get("color", kwargs.get("c", "k")))
            kwargs.setdefault("markeredgewidth", 0.7)
            kwargs.setdefault("markeredgecolor", "w")
            kwargs.setdefault("marker", "$" + str(position) + "$")

        if "marker" not in kwargs:
            kwargs["marker"] = "x"
        plt.plot(self._position[0], self._position[1], **kwargs)

    @property
    def position(self) -> np.array:
        return self._position

    def __repr__(self):
        return "Point=(X: {}, Y: {})".format(self._position[0], self._position[1])

    def __str__(self):
        return self.__repr__()
