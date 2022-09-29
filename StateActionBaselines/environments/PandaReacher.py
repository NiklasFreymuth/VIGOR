import gym
from baseline_util.Types import *
import baseline_util.Keys as k
from gym import spaces
from baseline_util.plot_trajectory import plot_target_distances
from environments.util.RobotModel import RobotModelFromPinochio
from functools import partial


def get_observation_dimension(observation_configuration_parameters: ConfigDict):
    observation_type = observation_configuration_parameters.get("observation_type")
    include_timestep = observation_configuration_parameters.get("include_timestep")
    include_target_encoding = observation_configuration_parameters.get("include_target_encoding")

    if observation_type == "geometric":
        observation_dimension = 4
    else:
        raise ValueError(f"Unknown observation_type '{observation_type}'")

    if include_timestep:
        observation_dimension += 1

    if include_target_encoding:
        observation_dimension += 2
    return observation_dimension


class PandaReacher(gym.Env):
    def __init__(self, environment_parameters: ConfigDict):
        """
        Initializes a pointreaching task, where a planar robot has to approximately reach a series of points from a
        constant starting position in a given order
        Args:
            environment_parameters: Dictionary of parameters. Contains
              "velocity_penalty": 1.0e-0,
              "acceleration_penalty": 1.0e-0,
              "radius": 0.05,
              "num_steps": 50,
              "n_dof": 6,
              "start_position": start_position,
              "observation_configuration":
                  {
                      "include_timestep": include_timestep,
                      "include_target_encoding": include_target_encoding
                  },
              "contexts": contexts
        """
        self.contexts = environment_parameters.get("contexts")

        self.dt = 1 / environment_parameters.get("num_steps")
        self.num_steps = environment_parameters.get("num_steps")
        self._velocity_penalty_lambda = environment_parameters.get("velocity_penalty")
        self._acceleration_penalty_lambda = environment_parameters.get("acceleration_penalty")
        self.start_position = environment_parameters.get("start_position")
        self.radius = environment_parameters.get("radius")

        # forward kinematics
        robot_model = RobotModelFromPinochio()
        self._robot_forward_kinematic = robot_model.get_forward_kinematic_position

        # action and observation spaces
        self.observation_configuration = environment_parameters["observation_configuration"]
        observation_dimension = get_observation_dimension(
            observation_configuration_parameters=self.observation_configuration)
        self.action_dimension = environment_parameters.get("n_dof")
        self.action_space = spaces.Box(low=-2 * np.pi, high=2 * np.pi,
                                       shape=(self.action_dimension,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(observation_dimension,), dtype=np.float32)

        # Initializing variables
        self.previous_action = np.zeros(self.action_dimension)
        self.joints_over_time = None  # keep history of positions for easy plotting and evaluating
        self.current_step = None
        self.goal_positions = None
        self.reached_targets = None
        self.hit_targets = None
        self.num_resets = 0

        self.previous_joints_over_time = None
        self.previous_goal_positions = None

        # whether to evaluate the reward function at the last step or not
        self.evaluate_reward = True

    def reset(self, context_id: Optional[int] = None, **kwargs) -> np.array:
        """
        Reset the environment by choosing a new context and resetting its step, state and collected actions/observations

        Returns:

        """
        if self.current_step == 0:
            # we test for step==0 to prevent sb3 from resetting our environment if we do not want it to
            return self.get_observation(action=np.zeros(self.action_dimension))

        # keep history from before last reset for rendering
        self.previous_joints_over_time = self.joints_over_time
        self.previous_goal_positions = self.goal_positions

        # get current context
        if context_id is not None:
            context = self.contexts[context_id]
        else:  # cycle through contexts
            context = self.contexts[self.num_resets % len(self.contexts)]

        # set this context
        self.goal_positions = context

        # reset state, step and saved actions
        self.joints_over_time = np.zeros(shape=(self.num_steps + 1, self.action_dimension))
        self.joints_over_time[0] = self.start_position
        self.current_step = 0
        self.num_resets = self.num_resets + 1

        self.reached_targets = np.zeros(len(self.goal_positions), dtype=bool)  # targets that have been reached
        self.hit_targets = np.zeros(shape=(self.num_steps + 1, len(self.goal_positions)))
        # targets that have been reached and then left again
        self.update_active_targets()

        self.previous_action = np.zeros(self.action_dimension)

        return self.get_observation(action=np.zeros(self.action_dimension))

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
        distance_mask = self.distance_to_targets <= self.radius 
        # potentially active targets

        # targets that have been reached at some point at or before the current timestep
        self.reached_targets = np.logical_or(distance_mask, self.reached_targets)
        last_reached_target = np.max(np.where(self.reached_targets), initial=-1)

        active_targets = np.zeros(2)
        if last_reached_target >= 0:
            active_targets[last_reached_target] = 1
        self.hit_targets[self.current_step] = active_targets

    def step(self, action: np.ndarray) -> Tuple[np.array, float, bool, dict]:
        """
        Takes a step in the PandaReacher environment.
        
        Args:
            action: Velocity to apply to the franka joints

        Returns: A tuple (observation, reward, done, additional_info), where
            * observation depends on the selected observation_type.
            * reward is 0 for all steps but the last, where it corresponds to a weighted sum of the log densities for
              target distance, smoothness and acceleration
            * done signals whether or not the environment is finished. It is 0 for all steps but the last.
            * additional_info is a dictionary containing optional information. It currently is empty except for the
              last step, where it contains
                * a decomposition of the reward,
                * the best distance to all targets

        """
        current_joints = self.current_joints_over_time
        self.current_step += 1
        next_joints = current_joints + action  # we do not add a dt here, as the action is learned by the agent
        self.joints_over_time[self.current_step] = next_joints

        self.update_active_targets()
        observation = self.get_observation(action=action)
        self.previous_action = action

        if self.is_done:
            reward_dict = self.get_reward_dict()
            reward = reward_dict.get(k.TOTAL_REWARD)
            additional_info = reward_dict
        else:
            reward = 0
            additional_info = {}
        return observation, reward, self.is_done, additional_info

    def get_observation(self, action: np.array):
        observation_type = self.observation_configuration.get("observation_type")
        if observation_type == "geometric":
            # velocity
            if self.current_step > 0:
                squared_velocities = np.mean(action ** 2)
            else:
                squared_velocities = 0

            # acceleration
            if self.current_step > 1:
                squared_accelerations = np.mean((self.previous_action - action) ** 2)
            else:
                squared_accelerations = 0
            observations = np.array((*self.distance_to_targets,
                                     squared_velocities,
                                     squared_accelerations))
        else:
            raise ValueError(f"Unknown observation_type '{self.observation_configuration.get('observation_type')}'.")

        if self.observation_configuration.get("include_timestep"):  # add normalized step (in [0,1]) as a feature
            observations = np.concatenate((observations, [self.current_step / self.num_steps]), axis=0)

        if self.observation_configuration.get("include_target_encoding"):
            observations = np.concatenate((observations, self.hit_targets[self.current_step]), axis=0)
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

        target_distance_penalty = self._get_target_distance_penalty(joints_over_time=self.joints_over_time[1:])

        acceleration_penalty, velocity_penalty = self._get_smoothness_penalties(joints_over_time=self.joints_over_time[1:])

        distance_to_target_centers, distance_to_target_boundaries = self._get_distances_to_targets(self.joints_over_time[1:])

        total_reward = -(target_distance_penalty + velocity_penalty + acceleration_penalty)
        return_dict = {k.TARGET_DISTANCE_PENALTY: target_distance_penalty,
                       k.VELOCITY_PENALTY: velocity_penalty,
                       k.ACCELERATION_PENALTY: acceleration_penalty,
                       k.TOTAL_REWARD: total_reward,
                       k.SUCCESS: float(distance_to_target_boundaries == 0),
                       k.DISTANCE_TO_CENTERS: distance_to_target_centers,
                       k.DISTANCE_TO_BOUNDARIES: distance_to_target_boundaries}
        return return_dict

    def _get_distances_to_targets(self, joints_over_time: np.array) -> Tuple[np.array, np.array]:
        positions_over_time = self.forward_kinematic(joints_over_time)

        distances_over_time = np.linalg.norm(positions_over_time[:, None, :] -
                                             self.goal_positions[None, :, :], axis=-1)
        # shape (#timesteps, #goal_positions)

        center_target_distance_penalty = self._get_minimum_target_distances(distances_over_time=distances_over_time)

        distances_to_boundary = np.maximum(0, distances_over_time - self.radius)
        boundary_target_distance_penalty = self._get_minimum_target_distances(distances_over_time=distances_to_boundary)

        return center_target_distance_penalty, boundary_target_distance_penalty

    def _get_minimum_target_distances(self, distances_over_time: np.array) -> np.array:
        best_goalwise_distances = np.empty(shape=(2,))
        best_goalwise_distances[0] = np.min(distances_over_time[:, 0])
        best_goalwise_distances[1] = distances_over_time[-1, 1]  # last step for second target
        # shape (#goal_positions)
        target_distance_penalty = np.sum(best_goalwise_distances, axis=-1)
        # shape (1, )
        return target_distance_penalty

    def _get_target_distance_penalty(self, joints_over_time):
        positions_over_time = self.forward_kinematic(joints_over_time)

        distances_over_time = np.linalg.norm(positions_over_time[:, None, :] -
                                             self.goal_positions[None, :, :], axis=-1)
        distances_over_time = np.maximum(0, distances_over_time - self.radius)
        # shape (#timesteps, #goal_positions)
        default_value = 1
        valid_distances = np.copy(distances_over_time)
        target_state_mask = np.empty(distances_over_time.shape)
        target_state_mask.fill(default_value)

        if np.any(distances_over_time[:, 0] == 0):  # first target is reached
            valid_distances[:-1, -1] = default_value  # only allow the last step for the last target
        else:
            valid_distances[:, -1] = default_value  # first target is never reached, so do not allow

        best_goalwise_distances = np.min(valid_distances, axis=-2)  # shape (#goal_positions)
        target_distance_penalty = np.sum(best_goalwise_distances, axis=-1)  # shape (1, )
        return target_distance_penalty

    def _get_smoothness_penalties(self, joints_over_time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes smoothness penalties for joint velocities and accelerations over time.
        While this is not explicitly mentioned in the tasks that we examine, adding such a smoothness term to the
        reward allows us to more easily spot anomalies in the behavior of the agents, such as too-fast movement or
        sudden jerks.
        Args:
            joints_over_time: Joint positions over time as an array of shape (..., #timesteps, #joints)

        Returns: A Tuple (acceleration_penalty, velocity_penalty), where each entry is of shape (..., ) and
            represents a penalty that is based on how fast the robot moves/accelerates over the course of the trajectory

        """
        if self._velocity_penalty_lambda > 0:
            # smoothness of the joint velocities over time
            joint_velocities = np.diff(joints_over_time, axis=-2)
            velocity_penalty = np.mean(self._velocity_penalty_lambda * joint_velocities ** 2)
        else:
            velocity_penalty = 0

        if self._acceleration_penalty_lambda > 0:
            # smoothness of joint acceleration over time
            joint_acceleration = np.diff(np.diff(joints_over_time, axis=-2), axis=-2)
            acceleration_penalty = np.mean(self._velocity_penalty_lambda * joint_acceleration ** 2)
        else:
            acceleration_penalty = 0
        return acceleration_penalty, velocity_penalty

    @property
    def distance_to_targets(self):
        """
        Returns: The distance of the end-effector to the center of each target

        """
        end_effector_position = self._robot_forward_kinematic(self.current_joints_over_time)
        # shape (3,)

        distance_to_targets = np.linalg.norm(end_effector_position - self.goal_positions, axis=-1)
        # shape (2,)

        return distance_to_targets

    def forward_kinematic(self, joints: np.array, **kwargs) -> np.array:
        """

        Args:
            joints: Batch of joint configurations of arbitrary shape (..., 7)

        Returns: Batch of cartesian coordinates corresponding to the joint configuraitos, shape (*joints.shape[:-1], 3)

        """
        cartesian_positions_over_time = np.apply_along_axis(partial(self._robot_forward_kinematic, **kwargs),
                                                            axis=-1, arr=joints)
        return cartesian_positions_over_time

    @property
    def is_done(self):
        return self.current_step == self.num_steps

    @property
    def current_joints_over_time(self):
        # we perform an index shift here, since the joints at step 0 are always at the starting position
        return self.joints_over_time[self.current_step]

    def render(self, mode="human", use_previous_values: bool = True):
        joints_over_time = self.previous_joints_over_time if use_previous_values else self.joints_over_time
        goal_positions = self.previous_goal_positions if use_previous_values else self.goal_positions

        positions_over_time = self.forward_kinematic(joints_over_time)
        # shape (#rollouts, #timesteps, 3)

        plot_target_distances(cartesian_positions=positions_over_time,
                              goal_positions=goal_positions, radius=self.radius)

        # plot_3d_trajectories(goal_positions=goal_positions, trajectory=positions_over_time,
        #                      radius=self.radius)

    def set_context_position(self, context_position: int):
        """

        Args:
            context_position: position for a context to choose. Useful when trying to do inference on a specific context


        Returns:

        """
        assert self.current_step == 0, f"May only set context id at start of trajectory. " \
                                       f"Tried to set at step '{self.current_step}'"
        self.goal_positions = self.contexts[context_position]
