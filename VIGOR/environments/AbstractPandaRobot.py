from abc import ABC

from environments.AbstractEnvironment import AbstractEnvironment
from util.Types import *
import matplotlib.pyplot as plt
import numpy as np
from algorithms.distributions.GMM import GMM
from mp_lib import det_promp
from recording.AdditionalPlot import AdditionalPlot

from util.robot_model.RobotModel import PandaRobotModelFromPinocchio
from util.ProMPUtil import mp_rollout
from functools import partial


def get_geometric_features(target_positions: np.array,
                           joints_over_time: np.array,
                           positions_over_time: np.array,
                           get_vector_norm: bool = True,
                           include_smoothness: bool = True) -> np.array:
    """

    Args:
        target_positions: Array of shape (#goals, {2,3}) of target balls in {2,3}d space
        joints_over_time: Batch of rollouts of shape (*rollout_shape, #timesteps, 7)
        positions_over_time: (vectorized/batched) positions over time of shape (*rollout_shape, #timesteps, (x,y[,z]))
        get_vector_norm: Whether to give an absolute distance (np.linalg.norm) [True] or (x,y[,z])-distances to each
            target position

    Returns: An array of geometric features over time. These features are calculated for each timestep and given as
        * the euclidean distance of the end-effector to each target iff get_vector_norm == "distances" OR
        * the relative (x,y[,z]) of the end-effector to each target iff get_vector_norm == "positions"
        * the mean squared joint velocity
        * the mean squared joint acceleration

    As an example, 2 targets and default configurations would result in features of shape (#rollouts, #timesteps, 4)

    """
    if get_vector_norm:
        distances_over_time = np.linalg.norm(positions_over_time[..., :, None, :] -
                                             target_positions[..., None, :, :], axis=-1)
    else:
        # give (x,y[,z]) coordinate
        distances_over_time = positions_over_time[..., :, None, :] - target_positions[..., None, :, :]
        distances_over_time = distances_over_time.reshape(*distances_over_time.shape[:-2], -1)

    if include_smoothness:
        squared_joint_velocities = np.mean(np.diff(joints_over_time, axis=-2) ** 2, axis=-1,
                                           keepdims=True)
        # shape: (#rollouts, #timesteps-1, 1)
        squared_joint_velocities = np.concatenate([np.zeros(shape=(*squared_joint_velocities.shape[:-2],
                                                                   1,
                                                                   squared_joint_velocities.shape[-1])),
                                                   squared_joint_velocities], axis=-2)
        # front zero-pad first step to get shape of (#rollouts, #timesteps, 1)

        squared_joint_accelerations = np.mean(np.diff(np.diff(joints_over_time, axis=-2), axis=-2) ** 2, axis=-1,
                                              keepdims=True)
        squared_joint_accelerations = np.concatenate([np.zeros(shape=(*squared_joint_accelerations.shape[:-2],
                                                                      2,
                                                                      squared_joint_accelerations.shape[-1])),
                                                      squared_joint_accelerations],
                                                     axis=-2)  # front zero-pad first 2 steps
        features = np.concatenate((distances_over_time,
                                   squared_joint_velocities,
                                   squared_joint_accelerations
                                   ),
                                  axis=-1)
        return features
    else:
        return distances_over_time


class AbstractPandaRobot(AbstractEnvironment, ABC):
    """
    Models an abstract environment with a Franka Emika Panda robot as the actor
    """

    def __init__(self, config: ConfigDict,
                 contexts: Dict[int, np.array],
                 parameter_dict: ConfigDict):
        """
        """
        super().__init__(config=config, contexts=contexts)
        self.feature_representation = config.get("modality")
        self.dt = 1 / parameter_dict.get("num_steps")
        self._velocity_penalty_lambda = parameter_dict.get("velocity_penalty")
        self._acceleration_penalty_lambda = parameter_dict.get("acceleration_penalty")

        if "start_position" in parameter_dict:
            self.start_position = parameter_dict.get("start_position")
        else:
            self.start_position = None

        self._input_space = parameter_dict.get("input_space")
        if self._input_space == "joint":
            robot_model = PandaRobotModelFromPinocchio()
            self._robot_forward_kinematic = robot_model.get_forward_kinematic_position
        elif self._input_space == "task":
            # if we already are in task space, we do not need to apply forward kinematics
            self._robot_forward_kinematic = lambda x, *args, **kwargs: x
        else:
            raise ValueError(f"Unknown input space '{self._input_space}")

        n_basis = parameter_dict.get("n_basis")
        n_dof = parameter_dict.get("n_dof")  # 7 degrees of freedom for the robot by default, but may use less
        self.num_parameters = n_basis * n_dof
        self.promp = det_promp.DeterministicProMP(n_basis=n_basis,
                                                  n_dof=n_dof,
                                                  width=0.005,
                                                  off=0.01,
                                                  zero_start="start_position" in parameter_dict,
                                                  zero_goal=False,
                                                  n_zero_bases=2)

    def reward(self, samples: np.array, context_id: int,
               return_as_dict: bool = False, **kwargs):
        """
        Wraps the uld() function by rolling out the proMP parameters to joints over time
        Args:
            samples: Array of shape (#samples, #promp_parameters)
            context_id:
            return_as_dict:
            **kwargs:

        Returns:

        """
        assert context_id in self.contexts.keys(), "Context id not in contexts"
        contexts = self.contexts[context_id]
        for shape in samples.shape[:-1]:
            contexts = np.repeat(contexts[None, :], shape, axis=0)
        joints_over_time = self.vectorized_mp_features(params=samples,
                                                       contexts=contexts,
                                                       feature_representation="joint")
        return self.uld(joints_over_time=joints_over_time, context_id=context_id,
                        return_as_dict=return_as_dict, **kwargs)

    def uld(self, joints_over_time: np.array, context_id: int,
            return_as_dict: bool = False, **kwargs) -> Union[ValueDict, np.array]:
        raise NotImplementedError("AbstractTeleoperationReacher does not implement 'uld()'")

    def _get_smoothness_penalties(self, joints_over_time: np.array) -> Tuple[np.array, np.array]:
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
            velocity_penalty = np.mean(self._velocity_penalty_lambda * joint_velocities ** 2, axis=(-2, -1))
        else:
            velocity_penalty = 0

        if self._acceleration_penalty_lambda > 0:
            # smoothness of joint acceleration over time
            joint_acceleration = np.diff(np.diff(joints_over_time, axis=-2), axis=-2)
            acceleration_penalty = np.mean(self._acceleration_penalty_lambda * joint_acceleration ** 2, axis=(-2, -1))
        else:
            acceleration_penalty = 0
        return acceleration_penalty, velocity_penalty

    def forward_kinematic(self, joints: np.array, **kwargs) -> np.array:
        """

        Args:
            joints: Batch of joint configurations of arbitrary shape (..., 7)

        Returns: Batch of cartesian coordinates corresponding to the joint configuraitos, shape (*joints.shape[:-1], 3)

        """
        cartesian_positions_over_time = np.apply_along_axis(partial(self._robot_forward_kinematic, **kwargs),
                                                            axis=-1, arr=joints)
        return cartesian_positions_over_time

    ###########################
    # movement primitive part #
    ###########################
    def vectorized_mp_features(self, params: np.array, contexts: np.array,
                               feature_representation: str,
                               dt: Optional[float] = None) -> np.array:
        """
        Computes the rollouts for the MP in a vectorized fashion and then parses them into features observed by the
        environment
        Args:
            params: Array of parameters for the MP. Has shape (num_samples, promp_dimension)
            feature_representation: The type of features to return.
                Defaults to self.feature_representation if not provided
            dt: Optional: Time resoltion to use. a dt of 0.01 leads to 100 timesteps

        Returns: An array of observed features for every set of ProMP parameters.
            The dimensionality/size of these observations depends on the environment

        """
        assert contexts.shape[:-1] == params.shape[:-1], f"Contexts and params must have the same shape, " \
                                                         f"given '{contexts.shape}' and '{params.shape}'"
        if params.ndim == 1:
            params = params[None, ...]

        joints_over_time = np.apply_along_axis(partial(self.mp_rollout, dt=dt), axis=-1, arr=params)

        if feature_representation == "joint":
            features = joints_over_time
        elif feature_representation == "cartesian":
            features = self.forward_kinematic(joints_over_time)
        elif feature_representation == "geometric":
            features = self.geometric_features(joints_over_time=joints_over_time, contexts=contexts)
        else:
            raise ValueError(f"Unknown feature representation '{feature_representation}'")
        return features

    def geometric_features(self, joints_over_time: np.array, contexts: np.array) -> np.array:
        raise NotImplementedError("AbstractTeleoperationReacher does not implement 'geometric_features'")

    def mp_rollout(self, params: np.array, dt: Optional[float] = None) -> np.array:
        """
        Wrapper for the (external) mp_rollout function that applies the current promp and time difference
        Args:
            params:
            dt: Optional parameter for the time difference between consecutive steps.
                Rollouts are normalized to length 1, so a dt of 0.02 would lead to 50 steps

        Returns:

        """
        if dt is None:
            dt = self.dt
        return mp_rollout(promp=self.promp, params=params, dt=dt, start_position=self.start_position)

    #########################
    # visualization utility #
    #########################

    def _set_plot_limits(self):
        axes = plt.gca()
        axes.set_xlim([0, int(1 / self.dt)])
        axes.set_ylim([0, 1])
        return axes

    ####################
    # Additional plots #
    ####################

    def get_additional_plots(self) -> List[AdditionalPlot]:
        """
        Collect all functions that should be used to draw additional plots. These functions take as argument the
        current policy and return the title of the plot that they draw

        Returns: A list of functions that can be called to draw plots. These functions take the current policy as
          argument, and draw in the current matplotlib figure. They return the title of the plot.

        """
        return super().get_additional_plots()

    ####################
    # metric recording #
    ####################

    def get_metrics(self, policy: GMM, context_id: int) -> ValueDict:
        return super().get_metrics(policy=policy, context_id=context_id)
