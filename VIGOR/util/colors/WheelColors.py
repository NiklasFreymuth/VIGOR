import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from util.colors.AbstractColors import AbstractColors
from util.Types import *


class WheelColors(AbstractColors):
    """
    Provides num_colors equidistant colors for plotting
    """

    def __init__(self, num_colors: int):
        """
        Creates the colors used for plotting. The colors used here are the traditional 10 matplotlib colors.
        This may not be enough for some applications
        """
        super().__init__()
        self.color_map = plt.cm.hsv
        self.num_colors = num_colors
        self._colors = [self.color_map((x + 0.5) / num_colors) for x in np.arange(0, num_colors)]
        # prepare colobar
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', self._colors, self.color_map.N)
        self._scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_colors))
        self._bounds = np.arange(num_colors + 1)

    def __call__(self, color_id: int) -> Tuple[float]:
        """
        get the i-th color of the color_wheel. Will raise an exception for invalid color ids
        :param color_id: the color to be returned
        :return: A pyplot color as an RGBA tuple
        """
        assert 0 <= color_id < self.num_colors, "Color_id must be in [0,{}), given {}".format(self.num_colors,
                                                                                              color_id)
        return self._colors[color_id]

    def as_list(self) -> List[Tuple[float]]:
        return self._colors

    def draw_colorbar(self, label: str = "Colorbar", horizontal: bool = False) -> matplotlib.pyplot.colorbar:
        if self.num_colors > 1:
            ticks = self._bounds[:-1]
            squishing_factor = int(len(ticks) // 30) + 1
            ticks = ticks[::squishing_factor]
            colorbar = plt.colorbar(self._scalar_mappable, spacing='proportional', ticks=ticks, boundaries=self._bounds,
                                    orientation="horizontal" if horizontal else "vertical")
            colorbar.set_label(label, rotation=None if horizontal else 90)
