import gym
from baseline_util.Types import *
from baseline_util.geometry.Point import Point
from baseline_util.geometry.Circle import Circle
from scipy.stats import multivariate_normal
from scipy.stats import norm
from baseline_util.forward_kinematic import forward_kinematic
import baseline_util.Keys as k
from gym import spaces
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from baseline_util.plot_trajectory import plot_time_series_trajectory
from baseline_util.colors.WheelColors import WheelColors


def get_observation_dimension(observation_configuration_parameters: ConfigDict, num_targets: int, num_links: int):
    observation_type = observation_configuration_parameters.get("observation_type")
    include_timestep = observation_configuration_parameters.get("include_timestep")
    include_target_encoding = observation_configuration_parameters.get("include_target_encoding")

    if observation_type == "reward_like":
        observation_dimension = num_targets + 2
    elif observation_type == "linkwise":
        observation_dimension = (num_targets + 2) * num_links
    elif observation_type == "gym":
        observation_dimension = 5 * num_links + 4 * num_targets
    else:
        raise ValueError(f"Unknown observation_type '{observation_type}'")

    if include_timestep:
        observation_dimension += 1

    if include_target_encoding:
        observation_dimension += num_targets
    return observation_dimension


class PlanarReacher(gym.Env):
    def __init__(self, environment_parameters: ConfigDict):
        """
        Initializes a pointreaching task, where a planar robot has to approximately reach a series of points from a 
        constant starting position in a given order
        Args:
            environment_parameters: Dictionary of parameters. Contains
                num_links: Number of links of the robot arm
                target_points: A list of points to reach
                radius: A radius for all points
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
                dt: Delay between consecutive timesteps. Relevant for the velocity and acceleration penalties
                total_steps: Total number of timesteps
        """
        # configuring the general environment
        self.num_links = environment_parameters.get("num_links")
        self.total_steps = environment_parameters.get("total_steps")
        self.contexts = environment_parameters.get("contexts")
        self.radius = environment_parameters.get("radius")
        if self.radius is None:  # use points
            self.radius = 0  # for reward computation
        assert self.radius >= 0, f"may not have negative radius, given '{self.radius}'"

        # designing the reward function
        target_std = environment_parameters.get("target_std")
        velocity_std = environment_parameters.get("velocity_std")
        acceleration_std = environment_parameters.get("acceleration_std")
        self.reward_scaling_alpha = environment_parameters.get("reward_scaling_alpha")

        self.target_gaussian = multivariate_normal(mean=np.zeros(1), cov=np.eye(1) * target_std)
        self.velocity_smoothness_gaussian = norm(loc=0, scale=velocity_std) if velocity_std is not None else None
        self.acceleration_gaussian = norm(loc=0, scale=acceleration_std) if acceleration_std is not None else None

        # internal observation machine
        target_distance_threshold = max(1, self.radius)
        distance_gaussian = multivariate_normal(cov=np.eye(2) * target_std)
        # log densities for positions that are {sqrt(2)*target_distance_threshold, sqrt(2)*10}
        # units away from a target point in each direction, i.e., that have an l1 distance of at least
        # {target_distance_threshold, 10} to it.
        self.threshold_log_density = distance_gaussian.logpdf([target_distance_threshold])
        self.masked_log_density = distance_gaussian.logpdf([10])

        self.previous_action = np.zeros(self.num_links)

        # action and observation spaces
        self.observation_configuration = environment_parameters.get("observation_configuration")
        # how the observation is computed.
        # The observation is a geometric representation of the task,
        # with different options corresponding to different sets
        # of represented features. Options are
        # "reward_like" for a distance to each target + velocity and acceleration smoothness per step
        # may also include a normalized timestep and a one-hot encoding of eligible targets
        # "linkwise" for the reward_like representation but for each link
        # "gym" for a representation that is similar to that of the gym PlanarReacher environment, only for more
        #   angles/links and more targets

        observation_dimension = get_observation_dimension(
            observation_configuration_parameters=self.observation_configuration,
            num_targets=2,
            num_links=self.num_links)
        self.action_space = spaces.Box(
            low=-2 * np.pi, high=2 * np.pi, shape=(self.num_links,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dimension,), dtype=np.float32
        )

        # Initializing variables
        self.joint_angles = None  # keep history of positions for easy plotting and evaluating
        self.current_step = None
        self.target_points = None
        self.target_circles = None
        self.reached_targets = None
        self.hit_targets = None
        self.num_resets = 0

        self.previous_joint_angles = None
        self.previous_active_targets = None
        self.previous_targets = None

        self.evaluate_reward = True

    def reset(self, context_id: Optional[int] = None, **kwargs) -> np.array:
        """
        Reset the environment by choosing a new context and resetting its step, state and collected actions/observations

        Returns:

        """
        if self.current_step == 0:
            # prevent sb3 from resetting our environment if we do not want it to
            return self.get_observation(action=np.zeros(self.num_links))

        # keep history from before last reset for rendering purposes
        self.previous_joint_angles = self.joint_angles
        self.previous_active_targets = self.hit_targets
        self.previous_targets = self.target_circles if self.target_circles is not None else self.target_points

        # get current context
        if context_id is not None:
            context = self.contexts[context_id]
        else:  # cycle through contexts
            context = self.contexts[self.num_resets % len(self.contexts)]

        # set this context
        self.set_target_points(context)

        # reset state, step and saved actions
        self.joint_angles = np.zeros(shape=(self.total_steps, self.num_links))
        self.current_step = 0

        self.num_resets = self.num_resets + 1

        self.reached_targets = np.zeros(len(self.target_points), dtype=bool)  # targets that have been reached
        self.hit_targets = np.zeros(shape=(self.total_steps + 1, len(self.target_points)))  #
        # targets that have been reached and then left again
        self.update_active_targets()

        self.previous_action = np.zeros(self.num_links)

        return self.get_observation(action=np.zeros(self.num_links))

    def set_target_points(self, context):
        self.target_points: List[Point] = [Point(x_coordinate=context[2 * i], y_coordinate=context[2 * i + 1])
                                           for i in range(len(context) // 2)]  # convert np array to points
        if self.radius > 0:  # circles rather than points
            self.target_circles: List[Circle] = [Circle(self.radius, *target_point.position)
                                                 for target_point in self.target_points]

    def update_active_targets(self):
        """
        Updates one-hot encoding of active targets based on end-effector proximity and its previous state.
        Effectively encodes task progress. Multiple targets may be active at the same time.

        A target is active iff
        * the previous target has been reached and
        * it has not been reached OR has been reached but never left since then. A target counts as "left" if
          the end-effector was sufficiently close to it at a previous step, but then moved out of a threshold range
          afterwards.
        Returns: An array of 0s and 1s where an entry is 1 if the corresponding target is active at the current step

        """
        distance_mask = self.distance_to_targets <= self.radius  #
        # self.target_gaussian.logpdf(self.distance_to_targets) > self.threshold_log_density
        # potentially active targets

        # targets that have been reached at some point at or before the current timestep
        self.reached_targets = np.logical_or(distance_mask, self.reached_targets)
        last_reached_target = np.max(np.where(self.reached_targets), initial=-1)

        active_targets = np.zeros(len(self.target_points))
        if last_reached_target >= 0:
            active_targets[last_reached_target] = 1
        self.hit_targets[self.current_step] = active_targets

    def step(self, action: np.ndarray) -> Tuple[np.array, float, bool, dict]:
        """
        Takes a step in the PlanarReacher environment. 
        This steps corresponds to an angular velocity for each of the reacher's joints.
        :param action: Velocity to apply to the angles
        :return: A tuple (observation, reward, done, additional_info), where
            * observation depends on the selected observation_type. This is described in detail in self.get_observation()
            * reward is 0 for all steps but the last, where it corresponds to a weighted sum of the log densities for
              target distance, smoothness and acceleration
            * done signals whether or not the environment is finished. It is 0 for all steps but the last.
            * additional_info is a dictionary containing optional information. It currently is empty except for the
              last step, where it contains
                * a decomposition of the reward,
                * the best distance to all targets
        """
        current_angles = self.current_joint_angles
        next_angles = current_angles + action
        self.joint_angles[self.current_step] = next_angles

        self.current_step += 1

        self.update_state(action=action)
        observation = self.get_observation(action=action)
        self.previous_action = action

        if self.is_done:
            reward_dict = self.get_reward_dict()
            reward = reward_dict.get(k.TOTAL_REWARD)
            additional_info = reward_dict
            distance_to_target_center, distance_to_target_boundary = self.get_minimum_target_distances()
            additional_info[k.DISTANCE_TO_CENTERS] = distance_to_target_center
            additional_info[k.DISTANCE_TO_BOUNDARIES] = distance_to_target_boundary
            additional_info[k.SUCCESS] = float((additional_info.get(k.DISTANCE_TO_BOUNDARIES) == 0))
            additional_info["convergence_rate"] = additional_info.get(k.DISTANCE_TO_BOUNDARIES) < 0.1
            angle_velocities, angle_accelerations = self.get_smoothness_factors()
            additional_info["angle_velocities"] = angle_velocities
            additional_info["angle_accelerations"] = angle_accelerations
        else:
            reward = 0
            additional_info = {}
        return observation, reward, self.is_done, additional_info

    def get_smoothness_factors(self) -> Tuple[float, float]:
        assert self.is_done, "Can only get best target distance at end of each trajectory/episode"
        angles_over_time = self.joint_angles
        # acceleration of the angles over time
        angle_accelerations = np.diff(np.diff(angles_over_time, axis=-2), axis=-2) ** 2

        # smoothness of the angle velocities over time
        angle_velocities = np.diff(angles_over_time, axis=-2) ** 2
        # shape (#samples, #timesteps-{1,2}, #joints)

        angle_velocities = float(np.mean(angle_velocities))
        angle_accelerations = float(np.mean(angle_accelerations))
        # mean over all to get a singular velocity/acceleration scalar

        return angle_velocities, angle_accelerations

    def get_minimum_target_distances(self) -> Tuple[float, float]:
        assert self.is_done, "Can only get best target distance at end of each trajectory/episode"
        angles_over_time = self.joint_angles
        positions_over_time = forward_kinematic(joint_angles=angles_over_time, flatten_over_positions=False)
        # shape: (#steps, #joints, (x,y))

        # distance to circles, i.e., should be able to be 0

        all_distances = np.array([target_point.get_distances(positions_over_time[..., -1, :])
                                  for target_point in self.target_points])
        all_distances = np.maximum(all_distances, 0)
        # shape (#targets, #samples, #steps)
        min_distances = np.min(all_distances, axis=-1)  # shape (#targets, #samples)
        min_distances[-1] = all_distances[-1, ..., -1]  # last target needs to be hit at last step

        distances = np.sum(min_distances, axis=0)  # shape (#samples, )
        distance = float(np.mean(distances))  # aggregate over all samples

        if self.target_circles is not None:
            all_distances = np.array([target_point.get_distances(positions_over_time[..., -1, :])
                                      for target_point in self.target_circles])
            all_distances = np.maximum(all_distances, 0)
            # shape (#targets, #samples, #steps)
            min_distances = np.min(all_distances, axis=-1)  # shape (#targets, #samples)
            min_distances[-1] = all_distances[-1, ..., -1]  # last target needs to be hit at last step

            distances = np.sum(min_distances, axis=0)  # shape (#samples, )
            boundary_distance = float(np.mean(distances))  # aggregate over all samples
            return distance, boundary_distance
        else:
            return distance, distance

    def update_state(self, action: np.array):
        self.update_active_targets()

    def get_observation(self, action):
        observation_type = self.observation_configuration.get("observation_type")
        if observation_type == "reward_like":
            # end-effector distance to each target,
            # as well as an aggregation of the velocity and acceleration of each joint
            # num_targets + 2 features

            if self.current_step > 0:
                squared_velocities = np.sum(action ** 2)
            else:
                squared_velocities = 0

            # acceleration
            if self.current_step > 1:
                squared_accelerations = np.sum((self.previous_action - action) ** 2)
            else:
                squared_accelerations = 0

            observations = np.array((*self.distance_to_targets,
                                     squared_velocities,
                                     squared_accelerations))
        elif observation_type == "linkwise":
            # same as reward_like, but for each intermediate joint and the end-effector
            # (num_targets + 2) * num_links features
            joint_positions = forward_kinematic(joint_angles=self.current_joint_angles,
                                                flatten_over_positions=False)[1:]
            target_positions = np.array([target.position for target in self.target_points])
            distance_to_targets = np.linalg.norm(joint_positions[:, None, ...] - target_positions[None, :, ...],
                                                 axis=-1)
            distance_to_targets = distance_to_targets.reshape(-1)  # flatten to 1d feature vector

            if self.current_step > 0:
                squared_velocities = action ** 2
            else:
                squared_velocities = np.zeros(action.shape)

            # acceleration
            if self.current_step > 1:
                squared_accelerations = (self.previous_action - action) ** 2
            else:
                squared_accelerations = np.zeros(action.shape)

            observations = np.concatenate((distance_to_targets,
                                           squared_velocities,
                                           squared_accelerations), axis=0)
        elif observation_type == "gym":
            # follows the openAI gym reacher observation space. Includes
            # (sin, cos) of each joint/angle
            # relative velocity of each joint/angle
            # (x,y) absolute position of each joint/angle
            # (x,y) position of each target
            # (x,y)-distance of end-effector to each target
            # total of 5*num_links + 4*num_targets features

            # sinoids
            joint_sines = np.sin(self.current_joint_angles)
            joint_cosines = np.cos(self.current_joint_angles)

            # velocity
            relative_velocities = action

            # absolute position
            joint_positions = forward_kinematic(joint_angles=self.current_joint_angles,
                                                flatten_over_positions=False)[1:]
            absolute_positions = joint_positions.reshape(-1)

            # target position
            target_positions = np.array([target.position for target in self.target_points])

            # end-effector distance
            end_effector_position = joint_positions[-1]
            target_dispositions = end_effector_position[None, ...] - target_positions
            target_dispositions = target_dispositions.reshape(-1)
            target_positions = target_positions.reshape(-1)

            observations = np.concatenate((joint_sines,
                                           joint_cosines,
                                           relative_velocities,
                                           absolute_positions,
                                           target_positions,
                                           target_dispositions),
                                          axis=0)
        else:
            raise ValueError(f"Unknown observation_type '{self.observation_configuration.get('observation_type')}'.")
        if self.observation_configuration.get("include_timestep"):
            # add normalized step (in [0,1]) as a feature
            observations = np.concatenate((observations, [self.current_step / self.total_steps]),
                                          axis=0)

        if self.observation_configuration.get("include_target_encoding"):
            observations = np.concatenate((observations, self.hit_targets[self.current_step]),
                                          axis=0)

        return observations

    def get_reward_dict(self) -> ScalarDict:
        """
        Compute the reward as an unnormalized log density (uld) 
        which the reward can be interpreted as in a variational inference setting
        Computes the reward for the point reacher at the end of its trajectory, or throws an error if called too early
        The reward consists of different Gaussian penalties, including distances to the target points, and 
        smoothness of the velocity and acceleration
        Args:
            
        Returns: A dictionary that contains the reward and its individual parts

        """
        assert self.is_done, "Can only call the reward function at the end of the trajectory/episode"
        if not self.evaluate_reward:
            return {k.TOTAL_REWARD: 0.0}

        angles_over_time = self.joint_angles
        positions_over_time = forward_kinematic(joint_angles=angles_over_time, flatten_over_positions=False)

        # target log density, i.e., "how well" the targets where hit
        target_log_density = self.get_target_penalty(positions_over_time=positions_over_time)

        # smoothness of the angle velocities over time
        if self.velocity_smoothness_gaussian is not None:
            angle_velocities = np.diff(angles_over_time, axis=-2)
            velocity_log_density = np.mean(self.velocity_smoothness_gaussian.logpdf(angle_velocities),
                                           axis=(-2, -1))
        else:
            velocity_log_density = 0

        # acceleration of the angles over time
        if self.acceleration_gaussian is not None:
            angle_acceleration = np.diff(np.diff(angles_over_time, axis=-2), axis=-2)
            acceleration_log_density = np.mean(self.acceleration_gaussian.logpdf(angle_acceleration),
                                               axis=(-2, -1))
        else:
            acceleration_log_density = 0

        total_reward = target_log_density + velocity_log_density + acceleration_log_density

        return_dict = {k.VELOCITY_LOG_DENSITY: velocity_log_density * self.reward_scaling_alpha,
                       k.ACCELERATION_LOG_DENSITY: acceleration_log_density * self.reward_scaling_alpha,
                       k.TARGET_LOG_DENSITY: target_log_density * self.reward_scaling_alpha,
                       k.TOTAL_REWARD: total_reward * self.reward_scaling_alpha}

        return return_dict

    def get_target_penalty(self, positions_over_time: np.ndarray) -> np.ndarray:
        """
        Calculates the penalty/reward for the minimum distance (and thus maximum log density) between end-effector
        position to targets at fixed time-steps.
        For N total targets, we want the reacher to reach target i after N/i of the trajectory. As such, we only
        need to compute the relevant end-effector positions for these steps.
        In the end, we sum over the penalties for each target to get a total penalty.
        Args:
            positions_over_time: The "input array" of positions over timesteps.
                Has shape (#samples, #steps, #angles+1, 2)

        Returns: An array of shape (#samples) where each entry corresponds to a sum of how well the end-effector
        reached the targets over the course of the trajectory
        """

        def first_positive(array: np.ndarray, axis: int, invalid_value: int = -1):
            """
            Find the first non-zero entry along axis, or return invalid_value otherwise
            Args:
                array:
                axis:
                invalid_value:

            Returns:

            """
            positive_mask = array > 0
            return np.where(positive_mask.any(axis=axis), positive_mask.argmax(axis=axis), invalid_value)

        end_effector_positions_over_time = positions_over_time[..., -1, :]
        target_lds = np.array([self.target_gaussian.logpdf(
            np.maximum(target_point.get_distances(end_effector_positions_over_time)[..., None], 0))
            for target_point in (self.target_circles if self.target_circles is not None else self.target_points)])
        target_lds[-1, :-1] = self.masked_log_density  # mask all but the last step for the last target
        # shape (#targets, #steps)

        # mask targets that have already been reached or can not yet be reached
        num_intermediate_targets = len(self.target_points) - 1
        for target_id in range(num_intermediate_targets):  # for each target
            satisfying_position_mask: np.ndarray = target_lds[target_id] > self.threshold_log_density
            num_steps = target_lds.shape[-1]

            first_satisfying_position = first_positive(array=satisfying_position_mask, axis=-1,
                                                       invalid_value=num_steps)
            last_satisfying_position = first_positive(array=np.diff(-1 * satisfying_position_mask, axis=-1),
                                                      axis=-1, invalid_value=num_steps)
            # first_satisfying_position gives the first step where the log density is larger than the threshold.
            # this corresponds to sufficiently close targets. The last satisfying position is the last consecutive
            # step that is still close enough to the target.

            if target_id < num_intermediate_targets:  # can access next target
                # mask values for the next targets until this one has been reached
                target_lds[(target_id + 1):, :first_satisfying_position] = self.masked_log_density
            # also mask values for the current target after "leaving" it for the first time
            target_lds[target_id, last_satisfying_position + 1:] = self.masked_log_density

            target_lds = np.max(target_lds, axis=-1)

        target_lds = np.sum(target_lds, axis=0)
        return target_lds

    @property
    def distance_to_targets(self):
        """
        Returns: The distance of the end-effector to the center of each target

        """
        end_effector_position = forward_kinematic(joint_angles=self.current_joint_angles,
                                                  flatten_over_positions=False)[-1]

        distance_to_targets = np.array([np.maximum(target_point.get_distances(end_effector_position), 0)
                                        for target_point in self.target_points])

        return distance_to_targets

    @property
    def is_done(self):
        return self.current_step == self.total_steps

    @property
    def current_joint_angles(self):
        if self.current_step == 0:
            return np.zeros(self.num_links)
        else:
            return self.joint_angles[self.current_step - 1]

    def render(self, mode="human", use_previous_values: bool = True):
        joint_angles = self.previous_joint_angles if use_previous_values else self.joint_angles
        active_targets = self.previous_active_targets if use_previous_values else self.hit_targets
        target_points = self.previous_targets if use_previous_values else self.target_points

        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        axes = plt.gca()
        axes.set_aspect(aspect="equal")
        axes.set_xlim([-self.num_links * 1.3, self.num_links * 1.3])
        axes.set_ylim([-self.num_links * 1.3, self.num_links * 1.3])
        plt.gca().add_patch(Rectangle(xy=(-0.1, -0.1), width=0.2, height=0.2, facecolor="grey", alpha=1, zorder=0))
        _ = [circle.plot(position=position, fill=False)
             for position, circle in enumerate(target_points)]

        joint_positions = forward_kinematic(joint_angles=joint_angles)
        all_colors = WheelColors(2 ** len(target_points))
        selected_colors = [all_colors(int(np.sum([target * (2 ** i)
                                                  for i, target in enumerate(current_active_target)])))
                           for current_active_target in active_targets[1:]]
        plot_time_series_trajectory(num_plotted_steps=self.total_steps,
                                    joint_positions=joint_positions,
                                    colors=selected_colors)

    def set_context_position(self, context_position: int):
        """

        Args:
            context_position: position for a context to choose. Useful when trying to do inference on a specific context


        Returns:

        """
        assert self.current_step == 0, f"May only set context id at start of trajectory. " \
                                       f"Tried to set at step '{self.current_step}'"
        self.set_target_points(context=self.contexts[context_position])
