"""
A simple object-oriented representation of a circle
"""
import sys
import os
import pathlib

cwd = pathlib.Path(os.getcwd())
sys.path.append(str(cwd.parent.parent))

from util.Types import *
from util.geometry.LineSegment import LineSegment
import numpy as np
import matplotlib.pyplot as plt


def point_lists_to_line_segments(points: np.array) -> np.array:
    """
    Turns an array of 2d points of into an array of segments that these positions span along the 2nd axis
    Args:
        points: Array of shape (..., :, 2), where the second-to-last axis is the one being transformed over

    Returns: An array of shape (..., :-1, 4)

    """
    return np.concatenate((points[..., :-1, :], points[..., 1:, :]), axis=-1)


class Circle:
    def __init__(self, radius: float, x_coordinate: float, y_coordinate: float):
        self._radius = radius
        self._center = np.array([x_coordinate, y_coordinate])

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

    def get_distances(self, samples: np.array) -> np.array:
        """
        Calculate the closest distance of the provided samples to the circle. The distance is negative if the sample
        is inside the circle, and positive otherwise
        Args:
            samples: An array of arbitrary shape as long as the last dimension is over x and y coordinates

        Returns: The closest position for every sample

        """
        return self.closest_distance(samples=samples)

    def collides_with(self, other_circle):
        return self.closest_distance(other_circle.center) - other_circle.radius < 0

    def distance_to_line_segments(self, line_segments: Union[List[LineSegment], np.array]) -> np.array:
        """
        Calculates the (signed) distance of every line segment to the circle. The distance is negative iff the
        line segment is inside the circle's radius at its closest point, and positive otherwise
        Args:
            line_segments: Either a list of line segments, or a numpy array of shape (..., 4), where the last
                dimension is the (x0, y0, x1, y1) coordinates defining the line segment

        Returns: An array of shape len(List)/(...) that contains the shortest signed distance of every segment to the
        circle

        """
        if isinstance(line_segments, np.ndarray):
            # vectorized and fast version. Adapted from http://paulbourke.net/geometry/pointlineplane/
            segment_distances = line_segments[..., :2] - line_segments[..., 2:]
            tangent_positions = np.sum((self._center - line_segments[..., :2]) * (-segment_distances), axis=-1)
            segment_lengths = np.linalg.norm(segment_distances, axis=-1)

            # the normalized tangent position is in [0,1] if the projection to the line segment is directly possible
            normalized_tangent_positions = tangent_positions / segment_lengths ** 2

            # it gets clipped to [0,1] otherwise, i.e., clips projections to the boundary of the line segment
            normalized_tangent_positions[normalized_tangent_positions > 1] = 1  # clip too big values
            normalized_tangent_positions[normalized_tangent_positions < 0] = 0  # clip too small values

            # the tangent points are some convex combination of the start and end of the line segments
            tangent_points = line_segments[..., :2] - normalized_tangent_positions[..., None] * segment_distances
            projection_vectors = self._center[None, :] - tangent_points
            distances = np.linalg.norm(projection_vectors, axis=-1)
            distances = distances - self._radius  # to get the distance to the circle rather than its center

            return distances
        elif isinstance(line_segments, list) and isinstance(line_segments[0], LineSegment):
            # slow but nicely object oriented version for line segment objects
            return np.array([line_segment.get_distances(self._center) - self._radius for line_segment in line_segments])
        else:
            raise ValueError("Unknown type for line_segments: '{}'".format(type(line_segments)))

    def plot(self, fill: bool = True, position: Optional[int]=None, paper_version: bool = False, **kwargs):
        if paper_version:
            from util.colors.SimpleColors import SimpleColors
            colors = SimpleColors()



            if position == 0:
                plt.gca().add_patch(plt.Circle(self.center, self.radius, fill=True, color=colors(9), alpha=0.5, zorder=0))
            else:
                plt.gca().add_patch(plt.Circle(self.center, self.radius, fill=True, color=colors(5), alpha=0.5, zorder=0))
            plt.gca().add_patch(plt.Circle(self.center, self.radius, fill=False, **kwargs))

        else:
            if not "zorder" in kwargs:
                kwargs["zorder"] = 100
            if fill:
                plt.gca().add_patch(plt.Circle(self.center, self.radius, fill=True, alpha=0.5, **kwargs))
            plt.gca().add_patch(plt.Circle(self.center, self.radius, fill=False, **kwargs))
            if position is not None:  # add position as label to circle
                plt.plot(self.center[0], self.center[1], markersize=12,
                         markerfacecolor="k", markeredgecolor="w",
                         markeredgewidth=0.7, marker="$" + str(position) + "$")

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


if __name__ == '__main__':
    np.random.seed(10)
    c = Circle(1, 1, 0)

    sample_points = 10 * np.random.random(4000000).reshape(2, -1, 2, 5, 2) - 5
    _sample_segments = point_lists_to_line_segments(points=sample_points)

    c.plot()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    _distances = c.distance_to_line_segments(_sample_segments)

    _distances = _distances.reshape(-1)
    _sample_segments = _sample_segments.reshape(-1, 4)

    # # plot segments
    # for i, (distance, segment) in enumerate(zip(_distances, _sample_segments)):
    #     plt.plot(segment[0::2], segment[1::2], "-", color=colors[i%10], label=distance)
    #
    # plt.legend()
    # plt.axis("scaled")
    # plt.grid()
    # plt.show()
