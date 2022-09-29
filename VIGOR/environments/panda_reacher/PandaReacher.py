import util.Defaults as d

from environments.AbstractPandaRobot import AbstractPandaRobot, get_geometric_features
from util.Types import *
import matplotlib.pyplot as plt
import numpy as np
from algorithms.distributions.GMM import GMM
from recording.AdditionalPlot import AdditionalPlot
from environments.panda_reacher.PandaReacherUtil import plot_target_distances, \
    plot_3d_trajectories


def _get_minimum_target_distances(distances_over_time: np.array) -> np.array:
    best_goalwise_distances = np.empty(shape=(distances_over_time.shape[0], distances_over_time.shape[2]))
    best_goalwise_distances[:, 0] = np.min(distances_over_time[..., 0], axis=1)
    best_goalwise_distances[:, 1] = distances_over_time[:, -1, 1]  # last step for second target
    # shape (#samples, #goal_positions)

    minimum_target_distances = np.sum(best_goalwise_distances, axis=-1)
    # shape (#samples, )
    return minimum_target_distances


class PandaReacher(AbstractPandaRobot):
    """
    Models the task of reaching a series of target spheres in 3d space using a Franka Emika Panda robot.
    """

    def __init__(self, config: ConfigDict,
                 contexts: Dict[int, np.array],
                 parameter_dict: ConfigDict):
        """
        """
        super().__init__(config=config, contexts=contexts, parameter_dict=parameter_dict)
        self.radius = parameter_dict.get("radius")

    def uld(self, joints_over_time: np.array, context_id: int,
            return_as_dict: bool = False, **kwargs) -> Union[ValueDict, np.array]:
        """
        uld is for "Unnormalized log density", which the reward can be interpreted as in a variational inference setting
        Takes an array of trajectories in the form of angles over time. Computes the reward for the Panda
        Reacher for each trajectory.
        The reward is composed of different parts that are added together
        Args:
            joints_over_time: An array of shape (..., self._total_steps, #joints)
            context_id: Id of the context for this batch of trajectories
            return_as_dict: Whether to return the reward as a scalar per sample, or as a dict of scalars per
            sample, where each entry corresponds to one part of the reward

        Returns: An array of shape (...) that contains a scalar reward for each trajectory. For single trajectories,
            an array of shape (1,) will be returned, i.e., a list with a single element.
            If return_as_dict, a dictionary of the decomposition of the reward will be returned instead

        """
        if joints_over_time.ndim == 2:
            joints_over_time = joints_over_time[None, ...]
        target_distance_penalty = self._get_target_distance_penalty(joints_over_time,
                                                                    context_id=context_id)

        acceleration_penalty, velocity_penalty = self._get_smoothness_penalties(joints_over_time=joints_over_time)

        total_reward = -(target_distance_penalty + velocity_penalty + acceleration_penalty)
        if return_as_dict:
            minimum_target_distances = self.get_minimum_target_distances(joints_over_time=joints_over_time,
                                                                         context_id=context_id)

            return_dict = {d.TARGET_DISTANCE_PENALTY: target_distance_penalty,
                           d.VELOCITY_PENALTY: velocity_penalty,
                           d.ACCELERATION_PENALTY: acceleration_penalty,
                           d.TOTAL_REWARD: total_reward,
                           d.SUCCESS: (minimum_target_distances.get("distance_to_boundaries") == 0).astype(np.float32),
                           **minimum_target_distances}
            return return_dict
        else:
            return total_reward

    def _get_target_distance_penalty(self, joints_over_time: np.array, context_id: int) -> np.array:
        positions_over_time = self.forward_kinematic(joints_over_time)

        goal_positions = self.contexts[context_id].reshape(-1, 3)

        distances_over_time = np.linalg.norm(positions_over_time[:, :, None, :] -
                                             goal_positions[None, None, :, :], axis=-1)
        distances_over_time = np.maximum(0, distances_over_time - self.radius)
        # shape (#samples, #timesteps, #goal_positions)
        default_value = 1
        valid_distances = np.copy(distances_over_time)
        target_state_mask = np.empty(distances_over_time.shape)
        target_state_mask.fill(default_value)
        satisfying_distances = (distances_over_time == 0).astype(int)
        satisfying_step_mask = satisfying_distances - np.roll(satisfying_distances, 1, axis=1)
        first_satisfying_step_indices = np.argmin(satisfying_step_mask, axis=1)
        for goal_position_index, goal_position in enumerate(goal_positions[:-1]):
            # for each goal position, mask all timesteps for subsequent goals until that current goal is reached
            # if a goal is not reached, its satisfying step is 0. In this case we mask all subsequent steps
            for index, satisfying_step in enumerate(first_satisfying_step_indices[..., goal_position_index]):
                if satisfying_step == 0:
                    valid_distances[index, :, goal_position_index + 1:] = default_value
                else:
                    valid_distances[index, :satisfying_step, goal_position_index + 1:] = default_value
        valid_distances[:, :-1, -1] = default_value  # only allow the last step for the last target
        best_goalwise_distances = np.min(valid_distances, axis=1)  # shape (#samples, #goal_positions)
        target_distance_penalty = np.sum(best_goalwise_distances, axis=-1)  # shape (#samples, )
        return target_distance_penalty

    ###########################
    # movement primitive part #
    ###########################
    def geometric_features(self, joints_over_time: np.ndarray, contexts: np.array) -> np.ndarray:
        contexts = contexts.reshape(*contexts.shape[:-1], -1, 3)  # "unflatten" individual targets
        features = get_geometric_features(target_positions=contexts,
                                          joints_over_time=joints_over_time,
                                          positions_over_time=self.forward_kinematic(joints_over_time))
        return features

    #########################
    # visualization utility #
    #########################

    def plot(self, policy: GMM, context_id: int, **kwargs):
        context = self.contexts[context_id]
        contexts = np.repeat(context[None, :], repeats=len(policy.means), axis=0)
        positions_over_time = self.vectorized_mp_features(policy.means,
                                                          contexts=contexts,
                                                          feature_representation="cartesian")
        # shape (#rollouts, #timesteps, 3)

        plot_3d_trajectories(goal_positions=self.contexts[context_id].reshape(-1, 3),
                             positions_over_time=positions_over_time,
                             radius=self.radius, labels=[f"{np.round(x * 100, decimals=2)}" for x in policy.weights])

    def plot_samples(self, policy_samples: np.array, context_id: int, **kwargs):
        context = self.contexts[context_id]
        contexts = np.repeat(context[None, :], repeats=len(policy_samples), axis=0)
        positions_over_time = self.vectorized_mp_features(policy_samples,
                                                          contexts=contexts,
                                                          feature_representation="cartesian")
        # shape (#samples, #timesteps, 3)

        plot_3d_trajectories(goal_positions=context.reshape(-1, 3),
                             positions_over_time=positions_over_time,
                             radius=self.radius)

    ####################
    # Additional plots #
    ####################

    def policy_sample_visualization(self, policy: GMM, context_id: int):
        samples = policy.sample(5)
        self.plot_samples(policy_samples=samples, context_id=context_id)

    def policy_sample_target_projection(self, policy: GMM, context_id: int):
        samples = policy.sample(5)
        self._plot_sample_target_projection(policy_samples=samples, context_id=context_id)

    def policy_mean_target_projection(self, policy: GMM, context_id: int):
        context = self.contexts[context_id]
        contexts = np.repeat(context[None, :], repeats=len(policy.means), axis=0)
        positions_over_time = self.vectorized_mp_features(policy.means, contexts=contexts,
                                                          feature_representation="cartesian")
        # shape (#samples, #timesteps, 3)

        plot_target_distances(cartesian_positions=positions_over_time,
                              goal_positions=self.contexts[context_id].reshape(-1, 3),
                              radius=self.radius,
                              labels=[f"{np.round(x * 100, decimals=2)}" for x in policy.weights],
                              merge_target_projections=False)
        self._set_projection_plot_limits()

    def _plot_sample_target_projection(self, policy_samples: np.array,
                                       context_id: int,
                                       merge_target_projections: bool = False,
                                       draw_dashes: bool = True):
        context = self.contexts[context_id]
        contexts = np.repeat(context[None, :], repeats=len(policy_samples), axis=0)
        positions_over_time = self.vectorized_mp_features(policy_samples, contexts=contexts,
                                                          feature_representation="cartesian")
        # shape (#samples, #timesteps, 3)

        plot_target_distances(cartesian_positions=positions_over_time,
                              goal_positions=self.contexts[context_id].reshape(-1, 3),
                              radius=self.radius,
                              merge_target_projections=merge_target_projections,
                              draw_dashes=draw_dashes)
        self._set_projection_plot_limits()

    def _set_projection_plot_limits(self):
        axes = plt.gca()
        axes.set_xlim([0, int(1 / self.dt)])
        axes.set_ylim([0, 1])
        return axes

    def get_additional_plots(self) -> List[AdditionalPlot]:
        """
        Collect all functions that should be used to draw additional plots. These functions take as argument the
        current policy and return the title of the plot that they draw

        Returns: A list of functions that can be called to draw plots. These functions take the current policy as
          argument, and draw in the current matplotlib figure. They return the title of the plot.

        """
        plots = super().get_additional_plots()
        plots.extend([
            AdditionalPlot(function=self.policy_sample_visualization,
                           is_policy_based=True,
                           uses_context_id=True,
                           uses_iteration_wise_figures=True,
                           is_expensive=True,
                           projection="3d"),
            AdditionalPlot(function=self.policy_mean_target_projection,
                           is_policy_based=True,
                           uses_context_id=True,
                           uses_iteration_wise_figures=True,
                           is_expensive=True),
            AdditionalPlot(function=self.policy_sample_target_projection,
                           is_policy_based=True,
                           uses_context_id=True,
                           uses_iteration_wise_figures=True,
                           is_expensive=True)

        ])
        return plots

    ####################
    # metric recording #
    ####################

    def get_minimum_target_distances(self, joints_over_time: np.array, context_id: np.array) -> ValueDict:
        """

        Args:
            joints_over_time: shape (#samples, #timesteps, #joints)
            context_id:

        Returns:

        """
        goal_positions = self.contexts[context_id].reshape(-1, 3)
        positions_over_time = self.forward_kinematic(joints_over_time)
        center_distances_over_time = np.linalg.norm(positions_over_time[:, :, None, :] -
                                                    goal_positions[None, None, :, :], axis=-1)
        minimum_center_distance = _get_minimum_target_distances(distances_over_time=center_distances_over_time)

        boundary_distances_over_time = np.maximum(0, center_distances_over_time - self.radius)
        minimum_boundary_distance = _get_minimum_target_distances(distances_over_time=boundary_distances_over_time)
        target_distances = {"distance_to_centers": minimum_center_distance,
                            "distance_to_boundaries": minimum_boundary_distance,
                            }
        return target_distances

    def get_metrics(self, policy: GMM, context_id: int) -> ValueDict:
        return super().get_metrics(policy=policy, context_id=context_id)
