from imitation.data.types import TrajectoryWithRew
from baseline_util.Types import *
from environments.get_environment import get_environment
from baseline_util.Keys import PANDA_REACHER, PLANAR_REACHER, BOX_PUSHER, ONLINE_BOX_PUSHER

"""
Utility function to create usable/compatible imitation learning data from given expert demonstrations 
"""


def get_expert_trajectories(config: ConfigDict, context_ids: np.array,
                            shuffle_demonstrations: bool = True) -> List[TrajectoryWithRew]:
    actions = actions_from_demonstrations(config=config,
                                          context_ids=context_ids, 
                                          shuffle_demonstrations=shuffle_demonstrations)

    processed_trajectories = collect_demonstrations_for_contexts(actions=actions,
                                                                 config=config,
                                                                 context_ids=context_ids)
    return processed_trajectories


def actions_from_demonstrations(config: ConfigDict, context_ids: List[int], shuffle_demonstrations: bool) -> np.array:
    """
    Return the environment actions (i.e., joint velocities over time) from the expert

    Args:
        config: Dictionary of task configuration. Contains
            rollouts_per_context: Number of rollouts to use per context
            task: The task to choose
        context_ids: List of contexts to consider

    Returns: An array of shape (#contexts, #rollouts, #steps, #links)

    """

    rollouts_per_context = config.get("rollouts_per_context")
    task: str = config.get("task")

    if task == PLANAR_REACHER:
        modality: str = "angles"
        num_links: int = 5
        contextual_angles = np.load(f"data/{task}/{modality}.npz")["arr_0"]
        # shape (#contexts, #samples, #steps, {#angles+contexts, #angles})

        absolute_actions = contextual_angles[..., :num_links]
        start_position = np.zeros(num_links)
    elif task == PANDA_REACHER:
        absolute_actions = np.load(f"data/{task}/joint.npz")["arr_0"]
        start_position = np.load(f"data/{task}/start_position.npy")
    elif task in [BOX_PUSHER, ONLINE_BOX_PUSHER]:
        absolute_actions = np.load(f"data/{BOX_PUSHER}/joint.npz")["arr_0"]
        start_position = np.array([0.45, 0])
        # start_position = np.array([0, 0])
    else:
        raise ValueError(f"Unknown task '{task}'")

    start_positions = start_position[None, None, None, :]
    start_positions = np.repeat(start_positions, repeats=absolute_actions.shape[0], axis=0)
    start_positions = np.repeat(start_positions, repeats=absolute_actions.shape[1], axis=1)

    zero_prefix_absolute_actions = np.concatenate((start_positions, absolute_actions), axis=2)
    actions = np.diff(zero_prefix_absolute_actions, axis=2)
    # start positions may be different from 0 in case we have some non-zero resting position of the robot
    # take diff to get velocities rather than actuations over time
    # shape (#contexts, #samples, #steps, action_dimension)

    if shuffle_demonstrations:
        [np.random.shuffle(sample) for sample in actions]
    actions = actions[context_ids, :rollouts_per_context]
    return actions


def collect_demonstrations_for_contexts(actions: np.array,
                                        config: ConfigDict,
                                        context_ids: np.array) -> List[TrajectoryWithRew]:
    environment = get_environment(config=config, context_ids=context_ids)

    trajectories = []

    for context_position, context in enumerate(actions):
        for rollout_id, rollout_actions in enumerate(context):

            initial_observation = environment.reset()[0]
            environment.envs[0].set_context_position(context_position=context_position)
            observations = [initial_observation]
            rewards = []

            done = False
            current_step = 0
            while not done:
                current_action = rollout_actions[current_step]

                observation, reward, done, additional_information = environment.step(np.array([current_action]))

                if done:
                    observations.append(additional_information[0]["terminal_observation"])
                else:
                    observations.append(observation[0])
                rewards.append(reward[0])
                current_step = current_step + 1
            observations = np.array(observations)
            rewards = np.array(rewards)

            trajectories.append(TrajectoryWithRew(obs=observations,
                                                  acts=rollout_actions,
                                                  infos=None,
                                                  terminal=True,
                                                  rews=rewards))

    return trajectories
