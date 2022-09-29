import matplotlib.pyplot as plt
from util.colors.AbstractColors import AbstractColors


class SimpleColors(AbstractColors):
    """
    Provides colors for plotting
    """

    def __init__(self):
        """
        Creates the colors used for plotting. The colors used here are the traditional 10 matplotlib colors.
        This may not be enough for some applications
        """
        super().__init__()
        self._colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def __call__(self, color_id):
        """
        get the i-th color of the color_cycle. Will give the 0th color for i = len(self._colors)
        :param color_id: the color to be returned
        :return: A pyplot color
        """
        return self._colors[color_id % len(self._colors)]

    def draw_colorbar(self, label: str = "Colorbar", horizontal: bool = False) -> plt.colorbar:
        import numpy as np
        import matplotlib.colors as plt_colors
        num_colors = 10
        # prepare colobar
        cmap = plt_colors.LinearSegmentedColormap.from_list(name='Custom cmap', colors=self._colors, N=256)
        scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_colors))
        bounds = np.arange(num_colors+1)
        ticks = bounds[:-1]
        colorbar = plt.colorbar(scalar_mappable, ticks=ticks, boundaries=bounds,
                                orientation="horizontal" if horizontal else "vertical")
        colorbar.set_label(label, rotation=None if horizontal else 90)
