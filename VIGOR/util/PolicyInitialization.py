import numpy as np

from util.Normalization import normalize_samples
from algorithms.distributions.GMM import GMM
from util.Types import *


def initialize_gmm(num_components: int, dimension: int, covariance_scale: float = 1.0) -> GMM:
    """
    Initializes the GMM either with or without given samples by creating num_components Gaussians for it.
    Args:
        num_components: (Maximum) number of components for the GMM
        dimension: The dimension of the input data

    Returns: An initial GMM

    """

    weights = np.ones(shape=num_components) / num_components
    means = np.random.normal(loc=0.0, size=(num_components, dimension))
    covars = np.repeat(a=np.eye(dimension)[None, ...], repeats=num_components, axis=0) * covariance_scale
    return GMM(weights=weights, means=means, covars=covars)


def get_initial_gaussian_parameters(samples: np.array, rewards: np.array,
                                    stability_constant=1e-12) -> Tuple[np.array, np.array]:
    """
    Finds initial mean and covariance parameters for a set of samples that are weighted by some rewards
    Args:
        samples: The samples to consider for initialization
        rewards: Reward evaluations of these samples

    Returns: A tuple (new_mean, new_covariance)

    """
    best_sample = np.argmax(rewards)
    new_mean = samples[best_sample]
    distances_to_mean = np.linalg.norm(new_mean - samples, axis=1)
    rewards = normalize_samples(samples=rewards, value_range=[0, 1], return_dict=False)
    distances_to_mean = normalize_samples(samples=distances_to_mean, value_range=[0, 1], return_dict=False)
    weights = np.exp(rewards - distances_to_mean)
    weights += stability_constant  # add small constant
    weights = weights / np.sum(weights)  # normalize to 1
    new_covariance = np.cov(samples, rowvar=False, aweights=weights)
    new_covariance = new_covariance + np.eye(new_covariance.shape[0]) * stability_constant
    return new_covariance, new_mean
