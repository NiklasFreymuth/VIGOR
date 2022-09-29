import numpy as np
import matplotlib.pyplot as plt
from util.colors.AbstractColors import AbstractColors
from util.Types import *


class SmartColors(AbstractColors):
    """
    Provides colors for plotting
    """

    def __init__(self, initial_position: float = 0.1):
        """
        Creates a flexible amount of pairwise different colors for visualization.
        This is done by greedily assigning colors to ids in a way that each color is persistent
        and as far away from the others as possible.
        Args:
            initial_position: just a different starting position for the first color. In [0,1]
            This is the number of newly added colors to "wait for" before re-introducing a previously deleted color
        """

        super().__init__()
        self._initial_position = initial_position
        self._colors = None
        self._color_tuples = {}

    def __call__(self, color_id):
        """
        get the i-th color of the color_cycle. Will give the 0th color for i = len(self._colors)
        :param color_id: the color to be returned
        :return: A pyplot color
        """
        if color_id not in self._color_tuples.keys():
            self._add_color(new_id=color_id)
        return self._color_tuples.get(color_id)[1]

    @property
    def colors(self):
        return {k: v[1] for k, v in self._color_tuples.items()}

    def assign_colors(self, ids: List[int]) -> None:
        """
        Assigns "optimal" colors for the given list of ids under two constraints:
        - the colors of ids that already existed on the previous call must stay the same
        - newly added ids are assigned a color that has the maximum distance to already existing ones
        Args:
            ids: A list of ids

        Returns:

        """
        deleted_ids = [x for x in self._color_tuples.keys() if x not in ids]
        for deleted_id in deleted_ids:
            del self._color_tuples[deleted_id]
        new_ids = [x for x in ids if x not in self._color_tuples.keys()]
        for new_id in new_ids:
            self._add_color(new_id=new_id)

    def _add_color(self, new_id: int):
        """
        Adds a new color to self._colors based on the already existing ones based on the best distance of this
        color to all existing ones
        Args:
            new_id: New Id to add a color for

        Returns:

        """
        assert new_id not in self._color_tuples.keys(), "Id already exists"
        if not self._color_tuples:  # no entries yet
            new_position = self._initial_position
        else:  # at least 1 entry
            current_positions = np.array(sorted([x[0] for x in self._color_tuples.values()]))
            current_positions = np.concatenate((current_positions, [1 + current_positions[0]]), axis=0)  # create loop
            distances = np.diff(current_positions)
            max_distance = int(np.argmax(distances))
            new_position = ((current_positions[max_distance] + current_positions[
                (max_distance + 1) % len(current_positions)]) / 2) % 1

        self._color_tuples[new_id] = (new_position, plt.cm.hsv(new_position))
