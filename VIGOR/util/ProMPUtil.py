import numpy as np
from mp_lib import det_promp
from typing import Optional, Union, List


def fit_promp(timestamps: np.array,
              trajectory: np.array,
              n_basis: int = 10,
              zero_start: bool = False,
              dt: Union[List[float], float] = 0.01,
              return_weights: bool = False,
              start_position: Optional[np.array] = None,
              n_zero_basis: int = 2):
    """
    Fit a promp on the given trajectory

    Args:
        timestamps:
        trajectory:
        n_basis:
        zero_start:
        dt:
        return_weights:
        start_position:
        n_zero_basis:

    Returns:

    """
    promp = det_promp.DeterministicProMP(n_basis=n_basis,
                                         n_dof=trajectory.shape[-1],
                                         width=0.005,
                                         off=0.01,
                                         zero_start=zero_start,
                                         zero_goal=False,
                                         n_zero_bases=n_zero_basis)

    promp.scale = np.max(timestamps)

    phi = promp._exponential_kernel(timestamps / promp.scale)[0]
    if zero_start:
        phi = phi[:, n_zero_basis:]  # remove the feature rows that would be zeroed out otherwise
        trajectory = trajectory - trajectory[0]  # move first step to 0
    weights = np.linalg.solve(np.dot(phi.T, phi) + 1.0e-6 * np.eye(phi.shape[1]), np.dot(phi.T, trajectory))

    if isinstance(dt, list):
        fit_rollout = [mp_rollout(promp=promp, params=weights, dt=dt_, start_position=start_position)
                       for dt_ in dt]
    else:
        fit_rollout = mp_rollout(promp=promp, params=weights, dt=dt, start_position=start_position)
    if return_weights:
        return weights, fit_rollout
    else:
        return fit_rollout


def mp_rollout(promp: det_promp.DeterministicProMP, params: np.array,
               dt: float = 0.01, start_position: Optional[np.array] = None) -> np.array:
    """
    Performs a single rollout using the DMP with the given parameters
    Args:
        params: Input parameters for the DMP

    Returns: A trajectory as angles over time

    """
    params = params.reshape(-1, promp.n_dof)
    promp.set_weights(1, params)
    _, trajectory, _, _ = promp.compute_trajectory(frequency=1 / dt, scale=1.)

    if start_position is not None:
        trajectory = trajectory + start_position

    return trajectory
