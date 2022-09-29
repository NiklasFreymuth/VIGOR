from typing import Optional, List

import numpy as np
from matplotlib import pyplot as plt


def last_satisfying_positions(cartesian_positions: np.array, goal_position: np.array, radius: float) -> np.array:
    """

    Args:
        cartesian_positions: An array (#samples, #timesteps, 3) of (x,y,z) positions of the robot trajectory over time
        goal_position: (x,y,z) goal position to reach
        radius: Radius/size of the goal sphere

    Returns: The indices of the timesteps where the goal was first left after being reached, as an array of shape
      (#samples, )

    """
    distances = np.linalg.norm(cartesian_positions - goal_position[None, None, :],
                               axis=-1)
    satisfying_distances = (distances <= radius).astype(int)
    last_satisfying_position_index = np.argmin(satisfying_distances - np.roll(satisfying_distances, 1), axis=1)
    return last_satisfying_position_index.astype(int)


def first_satisfying_positions(cartesian_positions: np.array, goal_position: np.array, radius: float) -> np.array:
    """

    Args:
        cartesian_positions: An array (#samples, #timesteps, 3) of (x,y,z) positions of the robot trajectory over time
        goal_position: (x,y,z) goal position to reach
        radius: Radius/size of the goal sphere

    Returns: The indices of the timesteps where the goal was first reached, as an array of shape
      (#samples, ). If a goal is never reached, 0 is returned instead

    """
    distances = np.linalg.norm(cartesian_positions - goal_position[None, None, :],
                               axis=-1)
    satisfying_distances = (distances <= radius).astype(int)
    last_satisfying_position_index = np.argmax(satisfying_distances - np.roll(satisfying_distances, 1), axis=1)
    return last_satisfying_position_index.astype(int)


def plot_target_distances(cartesian_positions: np.array, goal_positions: np.array,
                          radius: float, labels: Optional[List[str]] = None,
                          merge_target_projections: bool = False, draw_dashes: bool = True) -> None:
    """
    Plots a simplified distance-based view of the 3d teleoperation reacher task for a set of trajectories.
    For each trajectory, the distance to the first target is plotted until this target is left.
    Afterwards, the distance to the second target is used.
    Args:
        cartesian_positions: A batch of cartesian rollouts of shape (#samples, #timesteps, 3)
        goal_positions: Goal positions of shape (#goals, 3)
        radius:
        merge_target_projections: Whether to show the distances for each target over time, or only for the currently
          active target
        draw_dashes: Whether to draw dashes or not

    Returns:

    """
    from util.colors.SmartColors import SmartColors
    colors = SmartColors()

    first_target_indices = last_satisfying_positions(cartesian_positions=cartesian_positions,
                                                     goal_position=goal_positions[0],
                                                     radius=radius)

    distances = np.linalg.norm(cartesian_positions[:, :, None, :] - goal_positions[None, None, :, :], axis=-1)
    # shape (#samples, #timesteps, #goal_positions)
    # distances = np.maximum(0, distances-radius)
    plt.plot([0, len(distances[0])], [radius, radius], color="k")
    component_colors = [plt.Line2D([0], [0], color="k"),
                        plt.Line2D([0], [0], color="grey"),
                        plt.Line2D([0], [0], linestyle="--" if draw_dashes else None,
                                   color="grey")]
    if merge_target_projections:
        component_colors.append(plt.Line2D([0], [0], linestyle=":", color="grey"))

    for index, (first_target_index, current_sample) in enumerate(zip(first_target_indices, distances)):
        current_color = colors(index)
        if merge_target_projections:
            if first_target_index == 0:
                # first goal not reached
                plt.plot(current_sample[:, 0], color=current_color)
            else:
                plt.plot(np.arange(first_target_index),
                         current_sample[:first_target_index, 0],
                         color=current_color)
                plt.plot([first_target_index - 1, first_target_index],
                         [current_sample[first_target_index - 1, 0], current_sample[first_target_index, 1]],
                         ":", color=current_color)
                plt.plot(np.arange(first_target_index, len(current_sample)),
                         current_sample[first_target_index:, 1], "--",
                         color=current_color)
        else:
            plt.plot(current_sample[:, 0],
                     color=current_color)
            if draw_dashes:
                plt.plot(current_sample[:, 1], "--",
                         color=current_color)
            else:
                plt.plot(current_sample[:, 1],
                         color=current_color)

        # plotting a nice legend
        component_colors.append(plt.Line2D([0], [0], color=current_color))
        # to prevent the changing alpha from carrying over to the legend

    component_legends = ["Target Radius", "1st Target", "2nd Target"]
    if merge_target_projections:
        component_legends.append("Target Reached")
    if labels is not None:
        assert len(labels) == len(cartesian_positions), "Need to provide one label per sample"
        component_legends.extend(labels)
    else:
        component_legends.extend([f"Sample #{index + 1}" for index in range(len(cartesian_positions))])

    legend = plt.legend(component_colors, component_legends,
                        loc="upper right", ncol=1, fontsize=7)
    plt.xlabel("Timestep")
    plt.ylabel("Distance to current target")
    plt.grid()
    legend.set_zorder(10000)  # put the legend on top


def plot_3d_trajectories(goal_positions: np.array, positions_over_time: np.array,
                         radius: float, labels: Optional[List[str]] = None) -> None:
    """

    Args:
        goal_positions: Shape (#goals, 3)
        positions_over_time: An array of shape (#rollouts, #steps, (x,y,z)) of positions over time
        radius: Radius of the individual targets
        labels: (Optional) list of labels per rollout. Must have length (#rollouts) if provided

    Returns:

    """
    ax = plt.gca()
    for goal_index, goal_position in enumerate(goal_positions):
        ax.scatter(goal_position[0], goal_position[1], goal_position[2], label=f"Goal {goal_index}",
                   s=radius * 10000, alpha=0.2)
    for trajectory_index, trajectory in enumerate(positions_over_time):
        if isinstance(labels, list):
            label = labels[trajectory_index]
        else:
            label = None
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.4)
        ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   color=plt.gca().lines[-1].get_color(), label=label)
    ax.set_xlabel('X')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel('Y')
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlabel('Z')
    ax.set_zlim(0, 1)
    ax.view_init(azim=-60, elev=30)
    plt.legend(loc="upper left", ncol=1, fontsize=7)


def is_working_trajectory(cartesian_positions: np.array, goal_positions: np.array, radius: float,
                          include_last_step_distance: bool = False) -> bool:
    if cartesian_positions is None:
        return False
    for goal_position in goal_positions:
        distances = np.linalg.norm(cartesian_positions - goal_position[None, :], axis=-1)
        if np.all(distances >= radius):
            return False

    if include_last_step_distance:
        last_distance = np.linalg.norm(cartesian_positions[-1] - goal_positions[-1])
        if last_distance >= radius:
            return False
    return True


def closest_to_target(cartesian_positions: np.array, goal_position: np.array) -> int:
    distances = np.linalg.norm(cartesian_positions - goal_position[None, :], axis=-1)
    return int(np.argmin(distances))


def last_satisfying_position(cartesian_positions: np.array, goal_position: np.array, radius: float) -> int:
    """

    Args:
        cartesian_positions: An array (#timesteps, 3) of (x,y,z) positions of the robot trajectory over time
        goal_position: (x,y,z) goal position to reach
        radius: Radius/size of the goal sphere

    Returns: The index of the timestep where the goal was first left after being reached

    """
    distances = np.linalg.norm(cartesian_positions - goal_position[None, :], axis=-1)
    satisfying_distances = (distances <= radius).astype(int)
    last_satisfying_position_index = np.argmin(satisfying_distances[1:] - np.roll(satisfying_distances, 1)[1:])
    if last_satisfying_position_index == 0:
        # demonstration never leaves second target
        first_satisfying_position_index = np.argmax(satisfying_distances - np.roll(satisfying_distances, 1))
        last_satisfying_position_index = np.min((len(satisfying_distances),
                                                 first_satisfying_position_index + len(satisfying_distances) // 30))
        # last_timestep_index = np.argmax(np.max(np.abs(np.diff(cartesian_positions, axis=0)), axis=1) > 1.0e-7)
    return int(last_satisfying_position_index)
