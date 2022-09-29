from environments.planar_reacher.PlanarReacherUtil import get_target_distance_from_positions
from util.geometry.Point import Point
import util.Defaults as d

from environments.AbstractEnvironment import AbstractEnvironment
from util.Types import *
from util.geometry.Circle import Circle
import matplotlib.pyplot as plt
import numpy as np
from algorithms.distributions.GMM import GMM
from util.Plot import set_labels, plot_origin
from environments.planar_reacher.PlanarReacherUtil import plot_planar_trajectories
from util.colors.SimpleColors import SimpleColors
from util.colors.WheelColors import WheelColors
from mp_lib import det_promp
from recording.AdditionalPlot import AdditionalPlot
from environments.planar_reacher.planar_geometric_features import get_features_from_angles

"""
Planar Point Reaching tasks where the goal is to reach multiple points in a fixed order
"""


class PlanarReacher(AbstractEnvironment):

    def __init__(self, config: ConfigDict,
                 contexts: Dict[int, np.array],
                 parameter_dict: Dict[str, Any],
                 ):
        """
        Initializes the PlanarReacher, a task where a planar robot has to reach a series of points from a constant
        starting position in a given order
        Args:
            config: Configuration dictionary of the experiment
            contexts: A dictionary of context ids and the corresponding target positions
            parameter_dict: Dictionary containing additional parameters for the PlanarReacher. Contains
                acceleration_penalty_lambda: The weight of the acceleration penalty
                velocity_penalty_lambda: The weight of the velocity penalty
                num_links: Number of links of the robot arm
                velocity_std: Standard deviation of the Gaussian penalizing the angle smoothness
                    for every timestep
                acceleration_std: Standard deviation of the Gaussian penalizing the acceleration, which is realized
                 via np.diff(np.diff(angles_over_time))
                target_std: Standard deviation of the Gaussian penalizing the "target", i.e.,
                    whether the end-effector moves from one end of the line segment to the other
                obstacle_collision_penalty: The penalty for touching any of the circles during for a timestep
                variant: Either "fixed_steps" for fixed timesteps at which target positions need to be reached,
                or "variable_steps" for a general ordering of the targets but no predetermined timesteps
                target_distance_threshold: Only used for "variable_steps" variant. Minimum euclidean distance to a
                target for it to be considered to have been reached.
                reward_scaling_alpha: Scales the returned reward/unnormalized log density by the given factor.
                total_steps: Total number of timesteps
        """
        super().__init__(config=config, contexts=contexts)
        self._radius = parameter_dict.get("radius")
        self._acceleration_penalty_lambda = parameter_dict.get("acceleration_penalty_lambda")
        self._velocity_penalty_lambda = parameter_dict.get("velocity_penalty_lambda")
        self._total_steps = parameter_dict.get("total_steps")
        self._num_links = parameter_dict.get("num_links")

        self.dt = 0.01
        self.time_limit = self.dt * self._total_steps  # for 30 steps, the environment is limited to [0,0.3]
        self.start_pos = np.zeros(self._num_links)

        # promp
        num_basis_functions = 5
        self.num_parameters = num_basis_functions * self._num_links
        self.weights_scale = 0.5

        self.mp = det_promp.DeterministicProMP(n_basis=num_basis_functions,
                                               n_dof=self._num_links,
                                               width=0.005,
                                               off=0.01,
                                               zero_start=True,
                                               zero_goal=False)

        # plotting
        self._simple_colors = SimpleColors()
        self._desired_plotted_steps = 50

        target_points = {context_id: [Point(contexts[context_id][2 * pos], contexts[context_id][2 * pos + 1])
                                      for pos in range(parameter_dict.get("num_targets"))]
                         for context_id in contexts}
        self._target_points = target_points  # mapping from context id to target points
        self._target_circles: Dict[int, List[Circle]] = {context_id: [Circle(radius=self._radius,
                                                                             x_coordinate=point.position[0],
                                                                             y_coordinate=point.position[1])
                                                                      for point in context]
                                                         for context_id, context in target_points.items()}

    def reward(self, samples: np.array, context_id: int,
               return_as_dict: bool = False, include_collision: bool = True, **kwargs):
        assert context_id in self.contexts.keys(), "Context id not in contexts"
        contexts = self.contexts[context_id]
        for shape in samples.shape[:-1]:
            contexts = np.repeat(contexts[None, :], shape, axis=0)

        angles_over_time = self.vectorized_mp_features(params=samples,
                                                       contexts=contexts,
                                                       feature_representation="angles",
                                                       append_contexts=False)
        return self.uld(angles_over_time=angles_over_time,
                        context_id=context_id,
                        return_as_dict=return_as_dict,
                        include_collision=include_collision, **kwargs)

    def uld(self, angles_over_time: np.array, context_id: int, return_as_dict: bool = False,
            include_collision: bool = True, **kwargs) -> Union[ValueDict, np.array]:
        """
        uld is for "Unnormalized log density", which the reward can be interpreted as in a variational inference setting
        Takes an array of trajectories in the form of angles over time. Computes the reward for the line tracer
        for each trajectory. The reward consists of different Gaussian penalties, each of which corresponds to one
        attribute of the solution that we want to have.
        Args:
            angles_over_time: An array of shape (..., self._total_steps, self._num_links)
            context_id: The context id of the trajectories
            return_as_dict: (Optional) Whether to return the reward as a scalar per sample, or as a dict of scalars per
            sample, where each entry corresponds to one part of the reward
            include_collision: Whether to include the collision penalty in the reward or not.
                "False" effectively ignores collisions

        Returns: An array of shape (...) that contains a scalar reward for each trajectory

        """
        angle_accelerations = np.diff(np.diff(angles_over_time, axis=-2), axis=-2)
        angle_accelerations = np.mean(angle_accelerations ** 2, axis=(-2, -1)) * self._acceleration_penalty_lambda

        angle_velocities = np.diff(angles_over_time, axis=-2)
        angle_velocities = np.mean(angle_velocities ** 2, axis=(-2, -1)) * self._velocity_penalty_lambda

        distance_to_centers, distance_to_boundaries = self.get_target_distances(joints_over_time=angles_over_time,
                                                                                context_id=context_id, )
        total_reward = -(distance_to_boundaries + angle_velocities + angle_accelerations)
        if return_as_dict:  # build and return dictionary
            return_dict = {d.TOTAL_REWARD: total_reward,
                           d.VELOCITY_PENALTY: angle_velocities,
                           d.ACCELERATION_PENALTY: angle_accelerations,
                           d.DISTANCE_TO_CENTER: distance_to_centers,
                           d.DISTANCE_TO_BOUNDARY: distance_to_boundaries,
                           d.SUCCESS: (distance_to_boundaries == 0).astype(np.float32),
                           }
            return return_dict
        else:
            return total_reward

    ###########################
    # movement primitive part #
    ###########################

    def vectorized_mp_features(self, params: np.array, contexts: np.array,
                               feature_representation: str,
                               append_contexts: bool = False) -> np.array:
        """
        Computes the rollouts for the MP in a vectorized fashion and then parses them into features observed by the
        environment
        Args:
            params: Array of parameters for the MP. Has shape (..., #promp_dimension)
            contexts: Array of contexts to assign to each parameter.
                Must have shape (..., #context_dimension) or shape (#context_dimension,) if only 1 context is used
            feature_representation: The type of features to return.
            append_contexts: Whether to append the contexts to trajectory features or not. For some representations
                (e.g. angles), the features do not depend on the context in any way. In these cases, the context needs
                to be appended during training to give the learner the ability to discriminate between the different
                contexts.

        Returns: An array of observed features for every dmp parameter. The dimensionality/size of these observations
        depends on the environment

        """
        assert contexts.shape[:-1] == params.shape[:-1], f"Contexts and params must have the same shape, " \
                                                         f"given '{contexts.shape}' and '{params.shape}'"
        contexts = np.atleast_2d(contexts)
        params = np.atleast_2d(params)
        trajectories = np.apply_along_axis(self.mp_rollout, axis=-1, arr=params)
        all_trajectory_features = get_features_from_angles(angles_over_time=trajectories,
                                                           target_points=contexts,
                                                           feature_representation=feature_representation,
                                                           append_contexts=append_contexts)
        return all_trajectory_features

    def mp_rollout(self, params: np.array) -> np.array:
        """
        Performs a single rollout using the DMP with the given parameters
        Args:
            params: Input parameters for the DMP

        Returns: A trajectory as angles over time

        """
        params = params.reshape(-1, self.mp.n_dof) * self.weights_scale
        self.mp.set_weights(self.time_limit, params)
        _, trajectory, _, _ = self.mp.compute_trajectory(frequency=1 / self.dt, scale=1.)

        if self.mp.zero_start:
            trajectory += self.start_pos[None, :]
        return trajectory

    #########################
    # visualization utility #
    #########################

    def plot(self, policy: GMM, context_id: int, **kwargs):
        self._plot_trajectories(gmm=policy, context_id=context_id)
        self._plot_frame(context_id=context_id)

    def _plot_context(self, context_id: int, paper_version: bool = False):
        assert context_id in self._target_circles.keys(), "Context id not in target circles"
        for position, context_object in enumerate(self._target_circles[context_id]):
            context_object.plot(position=position, paper_version=paper_version)

    def _plot_frame(self, context_id: int, paper_version: bool = False):
        self._plot_context(context_id=context_id, paper_version=paper_version)
        plot_origin()
        set_labels()
        self._set_limits()

    def _plot_trajectories(self, gmm: GMM, context_id: int):
        component_colors = []
        context = self.contexts[context_id]
        contexts = np.repeat(context[None, :], gmm.num_components, axis=0)
        joint_positions = self.vectorized_mp_features(params=gmm.means,
                                                      contexts=contexts,
                                                      feature_representation="plot",
                                                      append_contexts=False)
        num_samples = len(joint_positions)
        step_plot_frequency = np.maximum(1, int((self._total_steps * num_samples) / self._desired_plotted_steps))
        actual_plotted_steps = self._total_steps / step_plot_frequency

        if len(joint_positions) == 1:  # only 1 component
            wheel_colors = WheelColors(num_colors=int(actual_plotted_steps))
            for current_step, step_joints in enumerate(joint_positions[0][::step_plot_frequency]):
                plt.plot(step_joints[:, 0], step_joints[:, 1], 'o-', markerfacecolor='k',
                         color=wheel_colors(current_step), alpha=0.65)
            wheel_colors.draw_colorbar(label="Step")
        else:
            for i, rollout_joint_positions in enumerate(joint_positions):
                current_color = self._colors(color_id=gmm.component_ids[i])
                plot_planar_trajectories(actual_plotted_steps, current_color, rollout_joint_positions,
                                         step_plot_frequency)

                component_colors.append(plt.Line2D([0], [0], color=current_color))
                # to prevent the changing alpha from carrying over to the legend

            legend = plt.legend(component_colors, [np.round(x * 100, decimals=2) for x in gmm.weights],
                                loc="upper center", ncol=3, fontsize=9)
            legend.set_zorder(10000)  # put the legend on top

    def _set_limits(self):
        axes = plt.gca()
        axes.set_aspect(aspect="equal")
        axes.set_xlim([-self._num_links, self._num_links * 1.1])
        axes.set_ylim([-self._num_links, self._num_links])
        return axes

    def plot_samples_per_component(self, *, policy: GMM,
                                   context_id: int,
                                   num_samples_per_component: int = 20,
                                   trace_only: bool = False, paper_version: bool = False):
        """
        Plot the given samples in point reacher task space
        Args:
            trace_only: Whether to plot the full samples (False), or only the end-effector trace (True)

        Returns:

        """
        samples_per_component = policy.sample_per_component(num_samples_per_component=num_samples_per_component)
        for component_idx, policy_samples in enumerate(samples_per_component):
            current_color = self._simple_colors(component_idx)
            joint_positions = self.vectorized_mp_features(params=policy_samples,
                                                          contexts=self.contexts[context_id],
                                                          feature_representation="plot",
                                                          append_contexts=False)

            num_samples = len(policy_samples)
            step_plot_frequency = np.maximum(1, int((self._total_steps * num_samples) / (self._desired_plotted_steps)))
            actual_plotted_steps = self._total_steps / step_plot_frequency

            # plot given samples
            for position, rollout_joint_positions in enumerate(joint_positions):
                plot_planar_trajectories(actual_plotted_steps, current_color,
                                         rollout_joint_positions=rollout_joint_positions,
                                         step_plot_frequency=step_plot_frequency, trace_only=trace_only,
                                         paper_version=paper_version, as_samples=True)

    def plot_distance_projections(self, samples: np.array, context_id: int):
        import matplotlib.pyplot as plt
        from util.colors.SimpleColors import SimpleColors
        colors = SimpleColors()
        features = self.vectorized_mp_features(params=samples,
                                               contexts=self.contexts[context_id],
                                               feature_representation="geometric",
                                               append_contexts=False)
        distances = features[..., :2]

        for index, current_sample in enumerate(distances):
            current_color = colors(index)

            plt.plot(current_sample[:, 0], "--",
                     color=current_color)
            plt.plot(current_sample[:, 1],
                     color=current_color)

        plt.plot([0, len(distances[0])], [self._radius, self._radius], color="k")
        axes = plt.gca()
        axes.set_xlim([0, 30])
        axes.set_ylim([0, 8])
        plt.xlabel("Timestep")
        plt.ylabel("Distance To Current Center")

    ####################
    # Additional plots #
    ####################

    def policy_sample_visualization(self, policy: GMM, context_id: int):
        policy_samples = policy.sample(5)
        self.plot_samples(policy_samples=policy_samples, context_id=context_id, trace_only=False)

    def plot_samples(self, *, policy_samples: np.array, context_id: int,
                     trace_only: bool = False, paper_version: bool = False,
                     show_joints_in_paper_ready: bool = True):
        """
        Plot the given samples in point reacher task space
        Args:
            policy_samples: Samples to plot
            context_id: ID of the context to use for the samples
            trace_only: Whether to plot the full samples (False), or only the end-effector trace (True)
            paper_version: Whether to plot the samples in the paper ready format
            show_joints_in_paper_ready: Whether to show the joints in the paper ready format

        Returns:

        """
        context = self.contexts[context_id]
        contexts = np.repeat(context[None, :], repeats=len(policy_samples), axis=0)
        joint_positions = self.vectorized_mp_features(params=policy_samples,
                                                      contexts=contexts,
                                                      feature_representation="plot",
                                                      append_contexts=False)

        num_samples = len(policy_samples)
        step_plot_frequency = np.maximum(1, int((self._total_steps * num_samples) / (self._desired_plotted_steps)))
        actual_plotted_steps = self._total_steps / step_plot_frequency

        # plot given samples
        self._plot_frame(paper_version=paper_version, context_id=context_id)
        for position, rollout_joint_positions in enumerate(joint_positions):
            current_color = self._simple_colors(position)

            plot_planar_trajectories(actual_plotted_steps, current_color,
                                     rollout_joint_positions=rollout_joint_positions,
                                     step_plot_frequency=step_plot_frequency, trace_only=trace_only,
                                     paper_version=paper_version, show_joints_in_paper_ready=show_joints_in_paper_ready)

    def end_effector_trace(self, policy: GMM, context_id: int):
        policy_samples = policy.sample(5)
        self.plot_samples(policy_samples=policy_samples, context_id=context_id,
                          trace_only=True)

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
                           is_expensive=True),
            AdditionalPlot(function=self.end_effector_trace,
                           is_policy_based=True,
                           uses_context_id=True,
                           uses_iteration_wise_figures=True,
                           is_expensive=True),
        ])
        return plots

    ####################
    # metric recording #
    ####################
    def get_target_distances(self, joints_over_time: np.array, context_id: int) -> Tuple[np.array, np.array]:
        """
        Calculates the average distance of samples from the policy to the target. If a mirrored target exists, the
         minimum of these distances is used. Evaluates at the last timestep, as this is when the targets need to be
         reached
        Args:
            joints_over_time: Sample trajectories to evaluate
            context_id: Context id of the samples

        Returns: Scalar values that represent the sum of best distances to the centers and boudnaries of the targets

        """
        from environments.planar_reacher.PlanarRobotForwardKinematic import forward_kinematic
        positions_over_time = forward_kinematic(joint_angles=joints_over_time, flatten_over_positions=False)
        center_distances = get_target_distance_from_positions(positions_over_time=positions_over_time,
                                                              target_points=self._target_points[context_id])
        boundary_distances = get_target_distance_from_positions(positions_over_time=positions_over_time,
                                                                target_points=self._target_circles[context_id])
        return center_distances, boundary_distances

    def get_metrics(self, policy: GMM, context_id: int) -> ValueDict:
        environment_metrics = super().get_metrics(policy=policy, context_id=context_id)

        metrics = {**environment_metrics,
                   "negative_policy_elbo": self.negative_policy_elbo(policy=policy, context_id=context_id),
                   "equiweighted_negative_policy_elbo": self.negative_policy_elbo(policy=policy.get_equiweighted_gmm(),
                                                                                  context_id=context_id)}
        return metrics
