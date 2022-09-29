"""
A simple object-oriented representation of a circle
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class Circle:
    def __init__(self, radius: float, x_coordinate: float, y_coordinate: float):
        self._radius = radius
        self._center = np.array([x_coordinate, y_coordinate])
        self._circle_patch = None

    def closest_distance(self, samples: np.array) -> np.array:
        """
        Calculate the closest distance of the provided samples to the circle. The distance is negative if the sample
        is inside the circle, and positive otherwise
        Args:
            samples: An array of arbitrary shape as long as the last dimension is over x and y coordinates

        Returns: The closest position for every sample

        """
        assert samples.shape[
                   -1] == 2, "Need to have x and y coordinate in last dimension, provided array of shape {}".format(
            samples.shape)
        distances_to_center = np.linalg.norm(samples - self._center, axis=-1)
        distances_to_circle = distances_to_center - self._radius
        return distances_to_circle

    def collides_with(self, other_circle):
        return self.closest_distance(other_circle.center) - other_circle.radius < 0

    def plot(self, fill: bool = True, fillcolor: str= "g", position: Optional[int]=None, **kwargs):
        if not "zorder" in kwargs:
            kwargs["zorder"] = 100
        if self._circle_patch is None:
            self._circle_patch = plt.Circle(self.center, self.radius, alpha=0.5, **kwargs)
            plt.gca().add_patch(self._circle_patch)
        self._circle_patch.set_fill(fill)
        if fill:
            self._circle_patch.set_color(fillcolor)

        if position is not None:  # add position as label to circle
            plt.plot(self.center[0], self.center[1], markersize=12,
                     markerfacecolor="k", markeredgecolor="w",
                     markeredgewidth=0.7, marker="$" + str(position) + "$")
        return self._circle_patch

    def positions_from_angles(self, angles: np.ndarray) -> np.ndarray:
        x_positions = self._center[0] + self._radius * np.cos(angles)
        y_positions = self._center[1] + self._radius * np.sin(angles)
        return np.vstack((x_positions, y_positions)).T

    @property
    def parameters(self) -> np.array:
        return np.array([self._radius, *self._center])

    @property
    def center(self) -> np.array:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    def __repr__(self):
        return f"Circle(r={self._radius}, pos=({self._center[0]},{self._center[1]}))"

    def __str__(self):
        return self.__repr__()
