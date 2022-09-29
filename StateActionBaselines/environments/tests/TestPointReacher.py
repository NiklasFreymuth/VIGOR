import numpy as np
from environments.get_environment import get_environment
from src.train import get_contexts
from src.collect_demonstrations import get_expert_trajectories
import matplotlib.pyplot as plt


def main():
    np.random.seed(0)
    config = {"num_train_contexts": 3,
              "rollouts_per_context": 1,
              "observation_type": "reward_like",
              "task": "planar_reacher",
              "include_target_encoding": False,
              "include_timesteps": False,
              "num_framestacks": 1}

    train_context_ids, test_context_ids = get_contexts(config=config)
    environment = get_environment(config=config, context_ids=train_context_ids)
    trajectories = get_expert_trajectories(config=config, context_ids=train_context_ids, shuffle_demonstrations=False)

    ground_truth_observations = np.load("data/planar_reacher/human_promp/geometric.npz")["arr_0"]
    ground_truth_observations = ground_truth_observations[train_context_ids, :1]
    ground_truth_observations = ground_truth_observations.reshape(-1, *ground_truth_observations.shape[-2:])

    ground_truth_actions = np.load("data/planar_reacher/human_promp/angles.npz")["arr_0"]
    ground_truth_actions = ground_truth_actions[train_context_ids, :1]
    ground_truth_actions = ground_truth_actions.reshape(-1, *ground_truth_actions.shape[-2:])

    fig = plt.figure(figsize=(32, 18))
    plot_title = "Sanity Check"
    fig.suptitle(plot_title)

    for position, trajectory in enumerate(trajectories):
        observations = trajectory.obs

        ground_truth_observation = ground_truth_observations[position]
        print("All close:", np.allclose(ground_truth_observation[:, :2], observations[1:, :2]))

        print("Ground Truth: \n", ground_truth_observation[:5, 2:4])
        print("Environment: \n", observations[:5, 2:4])  # offset since we start with an "empty"
        print("\n\n")

    for context_position, context_id in enumerate(train_context_ids):
        _ = environment.reset()
        environment.envs[0].set_context_position(context_position=context_position)  # go over all contexts
        done = False
        actions = trajectories[context_position].acts
        step = 0
        while not done:
            action = actions[step]
            observations, rewards, done, info = environment.step(actions=[action])
            step = step + 1

        info_dict = info[0]
        info_dict.pop("terminal_observation", None)
        print("Info dict:", info_dict)

        fig.add_subplot(int((len(train_context_ids) + 2) // 3), 3, context_position + 1)

        plt.title(f"Context ID: {context_id}")
        environment.render()

    plt.tight_layout()
    plt.savefig(f"{plot_title.replace(' ', '_').lower()}.pdf", format="pdf",
                dpi=200, transparent=False, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
