import numpy as np
import matplotlib.pyplot as plt


class LineSegment:
    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        """

        Args:
            x0: x position of the first endpoint of the segment
            x1: x position of the second endpoint of the segment
            y0: y position of the first endpoint of the segment
            y1: y position of the second endpoint of the segment
        """
        self.p0 = np.array([x0, y0])
        self.p1 = np.array([x1, y1])
        self._distance_vector = self.p1 - self.p0
        self._total_length = np.linalg.norm(self._distance_vector)

    @property
    def total_length(self) -> np.array:
        return self._total_length

    @property
    def get_parameters(self) -> np.array:
        return np.concatenate((self.p0, self.p1))

    def get_distances(self, points: np.array) -> np.array:
        """
        Gets the shortest distance of every point in points to this line segment
        Args:
            points: An array of shape (..., 2) of points

        Returns: An array of shape (...) where each entry is the closest distance of the respective point to the line
        segment

        """
        normalized_tangent_positions = self.get_normalized_tangent_positions(points=points, clip=True)

        tangent_points = self.p0 + normalized_tangent_positions[..., None] * self._distance_vector
        projection_vectors = points - tangent_points

        distances = np.linalg.norm(projection_vectors, axis=-1)
        return distances

    def get_normalized_tangent_positions(self, points: np.array, clip: bool = True) -> np.array:
        """
        Gets a value u that corresponds to the positions p* = p0+u*(p1-p0) that is the tangent point for the given
        input wrt. the line segment. Note that for u > 1 or u < 0, this means that the tangent point lies outside the
        segment.
        Args:
            points: An array of shape (..., 2) of points
            clip: Whether to clip the tangent positions to [0, 1] or not. Geometrically, clipping them corresponds
                to the tangent point being set to the start/endpoint of the line segment, while not clipping can result
                in tangent points lying outside of the segment

        Returns: An array of shape (...) where each entry is the factor u mentioned above. This may be interpreted
        as the position of the tangent point along the line segment, where 0 is one end and 1 is the other

        """
        tangent_positions = np.sum((points - self.p0) * self._distance_vector, axis=-1)

        # the normalized tangent position is in [0,1] iff the projection to the line segment is directly possible
        normalized_tangent_positions = tangent_positions / self._total_length ** 2

        if clip:
            # project to the boundary of the line segment
            normalized_tangent_positions[normalized_tangent_positions > 1] = 1  # clip too big values
            normalized_tangent_positions[normalized_tangent_positions < 0] = 0  # clip too small values
        return normalized_tangent_positions

    def plot(self, **kwargs):
        plt.plot(np.array([self.p0[0], self.p1[0]]), np.array([self.p0[1], self.p1[1]]), **kwargs)
