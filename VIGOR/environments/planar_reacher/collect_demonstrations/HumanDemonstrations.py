import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from ProgressBar import ProgressBar
from InteractivePlanarRobot import InteractivePlanarRobot
from XboxController import XboxController


def _plot_time_series_trajectories(actual_plotted_steps: int, current_color,
                                   rollout_joint_positions: np.array,
                                   step_plot_frequency: int, trace_only: bool = False) -> None:
    if trace_only:
        plt.plot(rollout_joint_positions[:, -1, 0], rollout_joint_positions[:, -1, 1], "ro-",
                 markerfacecolor="k", color=current_color, alpha=0.5)
    else:
        for current_step, step_joints in enumerate(rollout_joint_positions[::step_plot_frequency]):
            plt.plot(step_joints[:, 0], step_joints[:, 1], 'ro-', markerfacecolor='k',
                     alpha=current_step / (2 * actual_plotted_steps), color=current_color)
        if step_plot_frequency > 1:
            # also always plot the last step
            plt.plot(rollout_joint_positions[-1, :, 0], rollout_joint_positions[-1, :, 1], "ro-",
                     markerfacecolor="k", color=current_color)

        import matplotlib.patheffects as pe
        plt.plot(rollout_joint_positions[:, -1, 0], rollout_joint_positions[:, -1, 1], "--",
                 color=current_color, path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])


def record_trajectories(contexts: np.array, dynamic: str, gamepad: XboxController, num_contexts: int,
                        num_steps: int) -> Dict[int, List[np.array]]:
    trajectories = {context_id: [] for context_id in range(num_contexts)}
    stop_recording = False
    context_id = 0
    num_links = 5
    ax, fig = plt.subplots()

    while not stop_recording:
        progress_bar = ProgressBar(num_iterations=num_steps, verbose=False)

        context_id = context_id % num_contexts
        env = InteractivePlanarRobot(context=contexts[context_id], dynamic=dynamic,
                                     delta_time=0.1, include_rotation=True, num_links=num_links,
                                     current_rollout=len(trajectories.get(context_id)) + 1, current_context=context_id,
                                     common_suplot=(ax, fig)
                                     # initial_target=[0.5 * num_links, 0.5 * num_links, 0]
                                     )
        for i in range(num_steps):

            # read and format gamepad input

            gamepad_input = gamepad.read()
            progress_bar(context_id=context_id,
                         # x_pos_left=gamepad_input[0],
                         # y_pos_left=gamepad_input[1],
                         # x_pos_right=gamepad_input[6],
                         # y_pos_right=gamepad_input[7],
                         # a=gamepad_input[2],
                         # b=gamepad_input[3],
                         # x=gamepad_input[4],
                         y=gamepad_input[5])
            action = np.array((*gamepad_input[:2], np.arcsin(gamepad_input[6])))
            done = gamepad_input[2]
            reset = gamepad_input[3]
            prev = gamepad_input[4]
            stop = gamepad_input[5]

            env.step(action=action)
            env.render()
            if done:
                trajectories[context_id].append(env.history)
                context_id += 1
                break
            elif reset:
                break
            elif prev:
                context_id -= 1
                break
            elif stop:
                return trajectories


def subsample_trajectories(trajectories: Dict[int, List[np.array]], num_subsampling_steps: int) -> Dict[int, np.array]:
    """
    Subsample the trajectories
    Args:
        trajectories: A dictionary of lists of trajectories, i.e., {context_id, [trajectory]}
        num_subsampling_steps: Number of steps to get out in the end.

    Returns:

    """
    subsampled_trajectories = {context_id: np.array([np.array(trajectory)[np.linspace(0,
                                                                                      len(trajectory) - 1,
                                                                                      num_subsampling_steps).astype(int)]
                                                     for trajectory in trajectory_list if len(trajectory) > 1])
                               for context_id, trajectory_list in trajectories.items()}
    return subsampled_trajectories


def save_trajectories(num_contexts: int, trajectories: Dict[int, np.array]):
    """
    Save new trajectories by appending them to existing ones. This does not include any quality checks of the
    trajectories, and assumes matching context ids
    Args:
        num_contexts:
        trajectories:

    Returns:

    """
    from pathlib import Path
    import pickle
    trajectory_file_name = "trajectories.pkl"
    if Path(trajectory_file_name).is_file():
        with open(trajectory_file_name, "rb") as trajectory_file:
            old_trajectories = pickle.load(trajectory_file)
    else:
        old_trajectories = {context_id: [] for context_id in range(num_contexts)}
    all_trajectories = {}
    for context_id, old_trajectory_list in old_trajectories.items():
        new_trajectory_list = trajectories.get(context_id)
        if len(new_trajectory_list) > 0:
            if len(old_trajectory_list) == 0:  # add current data as first data for this context
                all_trajectories[context_id] = new_trajectory_list
            else:  # expert data for this context already exists
                all_trajectories[context_id] = np.concatenate((old_trajectory_list, new_trajectory_list), axis=0)
        else:
            all_trajectories[context_id] = old_trajectory_list

        print(f"  Context: {context_id} has {len(all_trajectories[context_id]):02d} entries")
    with open(trajectory_file_name, "wb") as trajectory_file:
        pickle.dump(all_trajectories, trajectory_file, -1)


def main(num_steps: int = 5000, dynamic: str = "acceleration", num_contexts=30):
    gamepad = XboxController()
    contexts = np.load("contexts.npy")[:num_contexts]

    print("Recording Trajectories")
    trajectories = record_trajectories(contexts=contexts, dynamic=dynamic, gamepad=gamepad,
                                       num_contexts=num_contexts, num_steps=num_steps)
    print("Subsampling Trajectories")
    trajectories = subsample_trajectories(trajectories, num_subsampling_steps=30)

    print("Saving Trajectories")
    save_trajectories(num_contexts=num_contexts, trajectories=trajectories)


if __name__ == '__main__':
    main(dynamic="velocity", num_contexts=12)
