from typing import Union, List

import numpy as np


def forward_kinematic(joint_angles: Union[List[np.array], np.array], link_lengths=None,
                      flatten_over_positions: bool = False) -> Union[List[np.array], np.array]:
    """
    Calculates the forward kinematic of the robot by interpreting each input value as an angle
    Args:
        joint_angles: The angles of the joints. Can be of arbitrary shape, as long as the last dimension is over
        the relative angles of the robot. I.e., the shape is (..., angles)
        link_lengths: Length of each robot link. Default is 1.
        flatten_over_positions: Whether to return the forward kinematics as an array of shape (..., angles+1, 2) or
            as to concatenate the last to dimensions and thus return them via (..., 2*(angles+1))

    Returns: The positions as a list of arrays of the same shape as the joint_angles, with their x and y either being
    separate dimensions or flattened over

    """

    def _forward_kinematic(_joint_angles):
        if link_lengths is None:
            _link_lengths = np.ones(_joint_angles.shape[-1])
        else:
            _link_lengths = link_lengths

        angles = np.cumsum(_joint_angles, axis=-1)

        pos = np.zeros([*angles.shape[:-1], angles.shape[-1] + 1, 2])
        for i in range(angles.shape[-1]):
            pos[..., i + 1, 0] = pos[..., i, 0] + np.cos(angles[..., i]) * _link_lengths[i]
            pos[..., i + 1, 1] = pos[..., i, 1] + np.sin(angles[..., i]) * _link_lengths[i]
        if flatten_over_positions:
            pos = np.reshape(pos, (*pos.shape[:-2], -1))  # flattens over last two dimensions, i.e., x and y positions
        return pos

    if isinstance(joint_angles, list):
        # multiple rollouts of potentially different length
        return [_forward_kinematic(_joint_angles) for _joint_angles in joint_angles]
    elif isinstance(joint_angles, np.ndarray):
        # single rollout or vectorized rollout
        return _forward_kinematic(joint_angles)
    else:
        raise NotImplementedError("Unknown angle representation '{}'".format(type(joint_angles)))
