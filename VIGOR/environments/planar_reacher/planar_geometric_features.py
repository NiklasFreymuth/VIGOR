import numpy as np
from util.Types import *
from util.observation_util import append_contexts as append_contexts_function
from environments.planar_reacher.PlanarRobotForwardKinematic import forward_kinematic


def get_features_from_angles(angles_over_time: np.array, target_points: np.array,
                             feature_representation: str, append_contexts: bool = False):
    """
    We model the original task reward as faithfully as possible for timestep-wise demonstrations by using
        - the stepwise normalized distance of the EE (or all joints) to the target
        - for the simplified geometric: the average of squared angles.
        - for non-simplified geometric:
            - the average squared angle velocity (excluding the 0th angle). Zero-padded for the first step
            - the average squared angle acceleration (including the 0th angle). Zero-padded for the first 2 steps
    Args:
        angles_over_time: Has shape (..., #steps, #joints). E.g., (#contexts, #samples, #steps, #joints)
        target_points: The points to reach. Must be of shape (..., 2*num_targets).
        If the first dimensions are set, they are assumed to match #contexts and #samples

        feature_representation: How to represent the stepwise features/observations.

    Returns: A reward_like feature representation for the PlanarReacher. This representation is computed for
    every timestep and contains
    * a feature representing the mean velocity of the angles
    * a feature representing the mean acceleration of the angles
    * one distance feature for each target

    """
    if feature_representation in [None, "angles"]:
        if append_contexts:
            features = append_contexts_function(angles_over_time, target_points)
        else:
            features = angles_over_time
    elif feature_representation == "positions":
        positions_over_time = forward_kinematic(joint_angles=angles_over_time, flatten_over_positions=True)

        if append_contexts:
            features = append_contexts_function(positions_over_time, target_points)
        else:
            features = positions_over_time
    elif feature_representation == "plot":
        flattened_positions_over_time = forward_kinematic(joint_angles=angles_over_time, flatten_over_positions=False)
        if append_contexts:
            features = append_contexts_function(flattened_positions_over_time, target_points)
        else:
            features = flattened_positions_over_time
    elif feature_representation == "geometric":
        features = get_geometric_features(angles_over_time, target_points)
    elif feature_representation == "linkwise":
        features = get_linkwise_features(angles_over_time, target_points)
    else:
        raise ValueError(f"Unknown feature_representation: {feature_representation}")
    return features


def get_geometric_features(angles_over_time, target_points):
    # distances to target point segment
    positions_over_time = forward_kinematic(angles_over_time)  # shape (..., #steps, #angles+1, 2)
    end_effector_positions_over_time = positions_over_time[..., -1:, :]  # shape (..., #steps, 1, 2)
    target_points = target_points.reshape(*target_points.shape[:-1], -1, 2)  # extract individual target points
    target_points = target_points[..., None, :, :]  # broadcast over steps, i.e.,
    # shape (..., 1, #targets, target_dim)
    distances_to_target_points = np.linalg.norm(end_effector_positions_over_time - target_points,
                                                axis=-1)  # shape (..., #steps, #targets)
    angle_features = get_angle_features(angles_over_time=angles_over_time,
                                        aggregate=True)
    features = np.concatenate((distances_to_target_points,
                               *angle_features,
                               ),
                              axis=-1)
    return features


def get_linkwise_features(angles_over_time, target_points):
    angle_features = get_angle_features(angles_over_time=angles_over_time,
                                        aggregate=False)
    # distances to target point segment
    positions_over_time = forward_kinematic(angles_over_time)  # shape (..., #steps, #angles/links+1, 2)
    positions_over_time = positions_over_time[..., 1:, None, :]  # shape (..., #steps, #angles/links, 1, 2)
    target_points = target_points.reshape(*target_points.shape[:-1], -1, 2)  # extract individual target points
    target_points = target_points[..., None, None, :, :]  # broadcast over timesteps and links/joints, i.e.,
    # shape = (..., 1, 1, #targets, 2))
    distances_to_target_points = np.linalg.norm(positions_over_time - target_points,
                                                axis=-1)  # shape (..., #steps, #angles/links, #targets)
    distances_to_target_points = distances_to_target_points.reshape((*distances_to_target_points.shape[:-2], -1))
    features = np.concatenate((distances_to_target_points,
                               *angle_features
                               ),
                              axis=-1)
    return features


def get_angle_features(angles_over_time: np.ndarray,
                       aggregate: bool = True) -> List[np.ndarray]:
    """
    Computes the angle features for the geometric representation of the point_reacher. These are
    features depending on the angle velocities and acceleration
    Args:
        angles_over_time:
        aggregate: Whether to return the sum of angle features over all angles (default, True), or return each separately

    Returns:

    """
    # angle velocity smoothness
    angle_velocities = np.diff(angles_over_time, axis=-2)
    relevant_angle_velocities = angle_velocities
    squared_angle_velocities = relevant_angle_velocities ** 2
    squared_angle_velocities = np.concatenate([np.zeros(shape=(*squared_angle_velocities.shape[:-2],
                                                               1,
                                                               squared_angle_velocities.shape[-1])),
                                               squared_angle_velocities], axis=-2)  # front zero-pad first step

    # angle acceleration smoothness
    angle_accelerations = np.diff(angle_velocities, axis=-2)
    squared_angle_accelerations = angle_accelerations ** 2
    squared_angle_accelerations = np.concatenate([np.zeros(shape=(*squared_angle_accelerations.shape[:-2],
                                                                  2,
                                                                  squared_angle_accelerations.shape[-1])),
                                                  squared_angle_accelerations],
                                                 axis=-2)  # front zero-pad first 2 steps
    if aggregate:
        sum_of_squared_velocities = np.sum(squared_angle_velocities, axis=-1, keepdims=True)

        sum_of_squared_accelerations = np.sum(squared_angle_accelerations, axis=-1, keepdims=True)

        return [sum_of_squared_velocities, sum_of_squared_accelerations]
    else:
        return [squared_angle_velocities, squared_angle_accelerations]
