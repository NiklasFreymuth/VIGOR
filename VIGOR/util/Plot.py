import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from matplotlib.patches import Rectangle

from util.Functions import symlog_spacing


def draw_subplot(num_subplots: int, position: int, title: str, data: np.ndarray, symlog=False,
                 shared_axis: plt.Axes = None, num_y_ticks=5, color=None, color2=None, linestyle=None,
                 linestyle2=None, xlabel="Iteration", xlabel_minor=None, x_major_formatter=None,
                 x_major_locator=plt.MaxNLocator(integer=True),
                 x_minor_formatter=ticker.NullFormatter(), x_minor_locator=plt.NullLocator(), figure_size=None):
    """
    A subplot in the overall plot displaying one of the evaluation metrics
    Args:
        position: Position of the current subplot
        title: Title of this subplot
        data: Data used for the plot. Must be a numpy array.
        If the array is 1d, the data will be plotted straightforwardly. If the array is 2d, each dimension will be
        plotted separately in the same plot
        symlog: Whether to use a logarithmic scale for the y-axis or not. Defaults to false
        shared_axis: If provided, the plot will share the x-axis this axis and return the same axis
        num_y_ticks: Number of ticks to draw on the y-axis. The ticks will be evenly spaced
        color: The color to use for the main graph/line
        color2: The color to use for the second graph/line
        linestyle: Style used for the first line
        linestyle2: Style used for the second line
        xlabel: Label of the x axis
        xlabel_minor: Label of the x axis for minor ticks
        x_major_formatter: matplotlib.ticker.Formatter() that can be used to customize the format of the major x ticks
        x_major_locator: matplotlib.pyplot.Locator () that can be used to customize the locations of the major x ticks
        x_minor_formatter: matplotlib.ticker.Formatter() that can be used to customize the format of the minor x ticks
        x_minor_locator: matplotlib.pyplot.Locator () that can be used to customize the locations of the minor x ticks

    Returns:
        The subplot axis is shared_axis is None, and shared_axis otherwise
    """
    position = position + 1  # matplotlib offset
    ax: plt.Axes = plt.subplot(num_subplots, 1, position, sharex=shared_axis)
    ax.grid(b=True, which="major", color="lightgray", linestyle="-")

    # aligning and formatting x axis
    if x_major_formatter is not None:
        # fixing xticks with FixedLocator but also using MaxNLocator to avoid cramped x-labels
        import matplotlib.ticker as mticker
        label_format = '{:,.0f}'
        ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels([label_format.format(x) for x in ticks_loc])
        ax.xaxis.set_major_formatter(x_major_formatter)

    transparent_color = (1, 1, 1, 0)

    if shared_axis is None:  # the first plot
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", which="major", bottom=False, top=True)
        if xlabel_minor is not None:
            ax.tick_params(axis="x", which="minor", bottom=True, top=True, labelsize=0, direction="in",
                           labelcolor=transparent_color, zorder=100)

    elif position < num_subplots:  # in-between plots have no labels but major and minor ticks pointing inwards
        ax.tick_params(axis="x", direction="in", which="both", bottom=True, top=True, pad=-15,
                       labelcolor=transparent_color, labelsize=0, zorder=100)
    else:  # last plot
        ax.xaxis.set_minor_formatter(x_minor_formatter)
        ax.xaxis.set_minor_locator(x_minor_locator)
        if xlabel_minor is not None:
            ax.tick_params(axis="x", which="minor", labelsize=8, rotation=60)
            ax.tick_params(axis="x", which="major", bottom=True, labelsize=0, top=True, direction="in",
                           labelcolor="white")
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel(xlabel_minor)
        else:
            ax.set_xlabel(xlabel)

    # y axis spacing
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(labelright=True)
    if symlog:  # rescale appropriate metrics for nicer vis
        plt.yscale("symlog", linthresh=0.1)
        spacing = symlog_spacing(data, num_y_ticks)
        plt.yticks(spacing)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(num_y_ticks))

    # setting label
    h = plt.ylabel(title)
    h.set_rotation(0)
    if figure_size is None:
        ax.yaxis.set_label_coords(-0.4, 0.3)  # displacement of the metric name labels to their plots
    else:
        ax.yaxis.set_label_coords((figure_size[0]/90)-0.30, 0.3)

    # actually plot
    if np.ndim(data) == 1:
        plt.plot(np.arange(0, len(data)), np.array(data), color=color, linestyle=linestyle)
        plt.xlim(0, len(data))
    elif np.ndim(data) == 2:
        plt.plot(np.arange(0, len(data)), data[:, 0], color=color, linestyle=linestyle)
        if data.shape[1] > 1:
            plt.plot(np.arange(0, len(data)), data[:, 1], color=color2, linestyle=linestyle2)
        plt.xlim(0, data.shape[0])
    if shared_axis is not None:
        return shared_axis, ax
    else:
        return ax, ax


def draw_2d_covariance(mean, covmatrix, chisquare_val=2.4477, return_raw=False, denormalize_fn=None, *args, **kwargs):
    (largest_eigval, smallest_eigval), eigvec = np.linalg.eig(covmatrix)
    phi = -np.arctan2(eigvec[0, 1], eigvec[0, 0])

    a = chisquare_val * np.sqrt(largest_eigval)
    b = chisquare_val * np.sqrt(smallest_eigval)

    ellipse_x_r = a * np.cos(np.linspace(0, 2 * np.pi))
    ellipse_y_r = b * np.sin(np.linspace(0, 2 * np.pi))

    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    r_ellipse = np.array([ellipse_x_r, ellipse_y_r]).T @ R
    if return_raw:
        return mean[0] + r_ellipse[:, 0], mean[1] + r_ellipse[:, 1]
    else:
        x = mean[0] + r_ellipse[:, 0]
        y = mean[1] + r_ellipse[:, 1]
        if denormalize_fn is not None:
            x = denormalize_fn(x, 0)
            y = denormalize_fn(y, 1)
        return plt.plot(x, y, *args, **kwargs)


def set_labels():
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")


def plot_origin():
    plt.gca().add_patch(Rectangle(xy=(-0.2, -0.2), width=0.4, height=0.4, facecolor="black", alpha=1, zorder=0))


