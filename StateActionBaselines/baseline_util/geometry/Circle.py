"""
A simple object-oriented representation of a circle
"""
from baseline_util.Types import *
import numpy as np
import matplotlib.pyplot as plt
from baseline_util.geometry.Point import Point


class Circle(Point):
    def __init__(self, radius: float, x_coordinate: float, y_coordinate: float):
        self._radius = radius
        super().__init__(x_coordinate=x_coordinate, y_coordinate=y_coordinate)

    def get_distances(self, samples: np.array) -> np.array:
        """
        Calculate the closest distance of the provided samples to the circle. The distance is negative if the sample
        is inside the circle, and positive otherwise
        Args:
            samples: An array of arbitrary shape as long as the last dimension is over x and y coordinates

        Returns: The closest position for every sample

        """
        distances_to_center = super().get_distances(samples=samples)
        distances_to_circle = distances_to_center - self._radius
        return distances_to_circle

    def distance_to(self, other_circle):
        """
        Calculates the signed distances to another circle. >0 if the circles do not touch, ==0 if they are tangent,
        and <0 if they intersect.
        :param other_circle: Another circle.
        :return:
        """
        return self.get_distances(other_circle.center) - other_circle.radius

    def intersect(self, other_circle):
        return self.distance_to(other_circle) < 0

    def plot(self, fill: bool = True, position: Optional[int]=None, **kwargs):
        if not "zorder" in kwargs:
            kwargs["zorder"] = 100
        if fill:
            plt.gca().add_patch(plt.Circle(self.position, self.radius, fill=True, alpha=0.5, **kwargs))
        else:
            plt.gca().add_patch(plt.Circle(self.position, self.radius, fill=False, **kwargs))
        if position is not None:  # add position as label to circle
            plt.plot(self.position[0], self.position[1], markersize=12,
                     markerfacecolor="k", markeredgecolor="w",
                     markeredgewidth=0.7, marker="$" + str(position) + "$")

    def positions_from_angles(self, angles: np.ndarray) -> np.ndarray:
        x_positions = self.position[0] + self._radius * np.cos(angles)
        y_positions = self.position[1] + self._radius * np.sin(angles)
        return np.vstack((x_positions, y_positions)).T

    @property
    def parameters(self) -> np.array:
        return np.array([self._radius, *self.position])

    @property
    def radius(self) -> float:
        return self._radius

    def __repr__(self):
        return f"Circle(r={self._radius}, pos=({self.position[0]},{self.position[1]}))"
