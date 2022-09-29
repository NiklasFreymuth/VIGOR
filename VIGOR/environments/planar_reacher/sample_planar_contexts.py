import sys
import os
import pathlib

cwd = pathlib.Path(os.getcwd())
sys.path.append(str(cwd.parent.parent.parent))

from util.geometry.Circle import Circle
from util.geometry.Point import Point
from util.colors.WheelColors import WheelColors
import numpy as np
import matplotlib.pyplot as plt


def get_mean_positions(num_targets: int, circle: Circle,  num_offsets: int = 0, initial_angle: float = 1.5 * np.pi):
    initial_angle = initial_angle - (2 * np.pi / (num_targets+num_offsets))
    point_angles = (-(np.cumsum(np.repeat(2 * np.pi / (num_targets+num_offsets), num_targets)) + initial_angle)) % (2 * np.pi)
    positions = circle.positions_from_angles(point_angles)
    return positions


def plot_contexts(circle: Circle, visualized_context_points: np.ndarray, title: str):
    colors = WheelColors(num_colors=len(visualized_context_points))
    for context_id, context in enumerate(visualized_context_points):
        for point_position, point in enumerate(context):
            Point(x=point[0], y=point[1]).plot(c=colors(context_id), position=point_position)

    circle.plot(fill=False)
    Circle(radius=5, x_coordinate=0, y_coordinate=0).plot(fill=True)
    colors.draw_colorbar(label="Context id")
    plt.axis("scaled")
    plt.xlim(-4, 6)
    plt.ylim(-5, 5)
    plt.grid()

    plt.title(title)


def main(num_targets: int = 4, data_source: str = "vips",
         circle: Circle = Circle(radius=2.5, x_coordinate=0.5, y_coordinate=0),
         plot: bool = True, save: bool = True, show: bool = False,
         noise: float = 0.5, num_contexts: int = 200, visualized_contexts: int = 50, num_offsets: int = 0):
    """
    Draw contexts for the updated multi_point_reacher environments.
    A context currently consists of n points that are drawn from one center point each using some Gaussian noise \sigma.
    The center points are equidistant points on a circle with position (x,y) and radius r.
    """
    mean_positions = get_mean_positions(num_targets, circle, num_offsets=num_offsets)
    context_points = np.random.normal(loc=mean_positions, scale=noise, size=(num_contexts, *mean_positions.shape))

    if save:
        save_folder = data_source

        if not os.path.exists(save_folder):
            from pathlib import Path
            path = Path(save_folder)
            path.mkdir(parents=True)
        print("Savefolder: {}".format(save_folder))

        np.save(os.path.join(save_folder, "contexts"), context_points.reshape(num_contexts, -1))
    else:
        save_folder = ""
    if plot:
        title = f"{visualized_contexts}/{num_contexts} contexts of {num_targets} targets from {circle} with noise {noise}"
        plot_contexts(circle=circle, visualized_context_points=context_points[:visualized_contexts], title=title)

        if save:
            plt.savefig(os.path.join(save_folder, "contexts.pdf"), format="pdf",
                        dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)

        if show:
            plt.show()
        plt.clf()


if __name__ == '__main__':
    np.random.seed(0)
    for num_targets in [2, 3, 4]:
        main(num_targets=num_targets, show=False, num_offsets=0)
