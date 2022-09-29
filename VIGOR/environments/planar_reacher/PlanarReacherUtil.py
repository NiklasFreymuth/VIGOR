from typing import Union, List

import numpy as np
from matplotlib import pyplot as plt, patheffects as pe

from util.geometry.Circle import Circle
from util.geometry.Point import Point


def get_target_distance_from_positions(positions_over_time: np.array,
                                       target_points: Union[List[Point], List[Circle]]):
    all_distances = np.array([target_point.get_distances(positions_over_time[..., -1, :])
                              for target_point in target_points])
    all_distances = np.maximum(all_distances, 0)  # force distances to be non-negative
    # shape (#targets, #samples, #steps)
    min_distances = np.min(all_distances, axis=-1)  # shape (#targets, #samples)
    min_distances[-1] = all_distances[-1, ..., -1]  # last target needs to be hit at last step
    distances = np.sum(min_distances, axis=0)  # shape (#samples, )
    return distances


def plot_planar_trajectories(actual_plotted_steps: int, current_color, rollout_joint_positions: np.array,
                             step_plot_frequency: int, trace_only: bool = False, paper_version: bool = False,
                             as_samples: bool = False, show_joints_in_paper_ready: bool = False) -> None:
    if paper_version:
        if as_samples:
            plt.plot(rollout_joint_positions[:, -1, 0], rollout_joint_positions[:, -1, 1], "--",
                     markerfacecolor="k", color=current_color, alpha=0.2)
        else:
            if show_joints_in_paper_ready:
                for current_step, step_joints in enumerate(rollout_joint_positions):  # [0,14,29]]):
                    plt.plot(step_joints[:, 0], step_joints[:, 1], 'o-', markerfacecolor='k',
                             alpha=(current_step + 5) / 50, color=current_color)
            plt.plot(rollout_joint_positions[:, -1, 0], rollout_joint_positions[:, -1, 1], "--",
                     zorder=10000 if show_joints_in_paper_ready else 100,
                     markerfacecolor="k", color=current_color, alpha=1,
                     path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
            plt.plot(rollout_joint_positions[29, -1, 0], rollout_joint_positions[29, -1, 1], 'o', markerfacecolor='k',
                     alpha=0.75,
                     color=current_color)
    elif trace_only:
        plt.plot(rollout_joint_positions[:, -1, 0], rollout_joint_positions[:, -1, 1], "o-",
                 markerfacecolor="k", color=current_color, alpha=0.5)
    else:
        for current_step, step_joints in enumerate(rollout_joint_positions[::step_plot_frequency]):
            plt.plot(step_joints[:, 0], step_joints[:, 1], 'o-', markerfacecolor='k',
                     alpha=current_step / (2 * actual_plotted_steps), color=current_color)
        if step_plot_frequency > 1:
            # also always plot the last step
            plt.plot(rollout_joint_positions[-1, :, 0], rollout_joint_positions[-1, :, 1], "o-",
                     markerfacecolor="k", color=current_color)

        plt.plot(rollout_joint_positions[:, -1, 0], rollout_joint_positions[:, -1, 1], "--",
                 color=current_color, path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
