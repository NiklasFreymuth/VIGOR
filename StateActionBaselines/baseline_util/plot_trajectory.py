import numpy as np
import matplotlib.pyplot as plt
from baseline_util.Types import *


def plot_time_series_trajectory(num_plotted_steps: int,
                                joint_positions: np.array,
                                colors=None,
                                step_plot_frequency: int = 1, trace_only: bool = False) -> None:
    """

    Args:
        num_plotted_steps: Number of steps to plot
        joint_positions: Array of shape [#steps, #links+1, 2], where the last dimension is the x,y position of the joint
        colors: A single color, or a list of colors per step
        step_plot_frequency:
        trace_only:

    Returns:

    """
    if trace_only:
        plt.plot(joint_positions[:, -1, 0], joint_positions[:, -1, 1], "o-",
                 markerfacecolor="k", color=colors, alpha=0.5)
    else:
        for current_step, step_joints in enumerate(joint_positions[::step_plot_frequency]):
            color = colors[current_step] if isinstance(colors, list) else colors
            plt.plot(step_joints[:, 0], step_joints[:, 1], 'o-', markerfacecolor='k',
                     alpha=(current_step+num_plotted_steps) / (2.5 * num_plotted_steps), color=color)
        if step_plot_frequency > 1:
            # also always plot the last step
            color = colors[-1] if isinstance(colors, list) else colors
            plt.plot(joint_positions[-1, :, 0], joint_positions[-1, :, 1], "o-",
                     markerfacecolor="k", color=colors)

        import matplotlib.patheffects as pe
        plt.plot(joint_positions[:, -1, 0], joint_positions[:, -1, 1], "--",
                 color="grey", alpha=0.8, path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])


def plot_3d_trajectories(goal_positions: np.array, trajectory: np.array,
                         radius: float) -> None:
    """

    Args:
        goal_positions:
        trajectory: An array of shape (#rollouts, #steps, (x,y,z)) of positions over time
        radius: Radius of the individual targets

    Returns:

    """
    ax = plt.gca()
    for goal_index, goal_position in enumerate(goal_positions):
        ax.scatter(goal_position[0], goal_position[1], goal_position[2],
                   label=f"Goal {goal_index}",
                   s=radius * 10000, alpha=0.2)

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.4)
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
               color=plt.gca().lines[-1].get_color())
    ax.set_xlabel('X')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel('Y')
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlabel('Z')
    ax.set_zlim(0, 1)
    ax.view_init(azim=-60, elev=30)
    plt.legend(loc="upper left", ncol=1, fontsize=7)


def plot_target_distances(cartesian_positions: np.array, goal_positions: np.array,
                          radius: float) -> None:
    """
    Plots a simplified distance-based view of the 3d teleoperation reacher task for a set of trajectories.
    For each trajectory, the distance to the first target is plotted until this target is left.
    Afterwards, the distance to the second target is used.
    Args:
        cartesian_positions: A batch of cartesian rollouts of shape (#samples, #timesteps, 3)
        goal_positions: Goal positions of shape (#goals, 3)
        radius:

    Returns:

    """
    distances = np.linalg.norm(cartesian_positions[:, None, :] - goal_positions[ None, :, :], axis=-1)
    # shape (#samples, #timesteps, #goal_positions)
    # distances = np.maximum(0, distances-radius)
    plt.plot([0, len(distances)], [radius, radius], color="k", label="Target Radius")
    plt.plot(distances[:, 0], label="1st Target")
    plt.plot(distances[:, 1], label="2nd Target")

    legend = plt.legend(loc="upper right", ncol=1, fontsize=7)
    plt.xlabel("Timestep")
    plt.ylabel("Distance to current target")
    plt.grid()
    legend.set_zorder(10000)  # put the legend on top
