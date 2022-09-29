import numpy as np
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x: float, y: float):
        """

        Args:
            x: x position of the point
            y: y position of the point
        """
        self.position = np.array([x, y])

    def get_distances(self, points: np.array) -> np.array:
        """
        Gets the shortest distance of every point in points to this one
        Args:
            points: An array of shape (..., 2) of points

        Returns: An array of shape (...) where each entry is the closest distance of the respective point to this point

        """
        return np.linalg.norm(points - self.position, axis=-1)

    def distance_to(self, other_point) -> float:
        return np.linalg.norm(self.position - other_point.position)

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
        plt.plot(self.position[0], self.position[1], **kwargs)

    def __str__(self):
        return "X: {}, Y: {}".format(self.position[0], self.position[1])
