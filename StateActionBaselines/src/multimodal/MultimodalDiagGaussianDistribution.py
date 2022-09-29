from typing import Tuple, Optional, Union

import torch
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, Independent, Normal

from stable_baselines3.common.distributions import Distribution, sum_independent_dims
from src.multimodal.MultimodalActionNet import MultiModalActionNet


class MultimodalDiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, num_components: int, action_dim: int, entropy_approximation_mode: str,
                 train_categorical_weights: bool):
        super(MultimodalDiagGaussianDistribution, self).__init__()
        self.num_components = num_components
        self.action_dim = action_dim
        self.entropy_approximation_mode = entropy_approximation_mode
        self.train_categorical_weights = train_categorical_weights

    def proba_distribution_net(self, num_components: int,
                               latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param num_components: Number of components of the mixture model
        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        action_net = MultiModalActionNet(num_components=num_components,
                                         latent_dimension=latent_dim,
                                         action_dimension=self.action_dim,
                                         train_categorical_weights=self.train_categorical_weights)
        log_std = nn.Parameter(torch.ones(num_components, self.action_dim) * log_std_init, requires_grad=True)
        return action_net, log_std

    def proba_distribution(self, weights: torch.Tensor, mean_actions: torch.Tensor,
                           log_std: torch.Tensor) -> "MultimodalDiagGaussianDistribution":
        """
        Create the distribution given its parameters (weights, mean, std)

        :param
        :param mean_actions:  Shape (#batch_size, #components, #action_dimension)?
        :param log_std:
        :return:
        """
        # mean_actions = torch.stack((mean_actions, mean_actions), dim=0)
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        mix = Categorical(weights, validate_args=False)
        comp = Independent(base_distribution=Normal(mean_actions, action_std),
                           reinterpreted_batch_ndims=1, validate_args=False)
        self.distribution = MixtureSameFamily(mix, comp, validate_args=False)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        if log_prob.dim() == 1:
            log_prob = log_prob[:, None]
        return sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        mixture_weights = self.distribution.mixture_distribution.probs
        if self.entropy_approximation_mode == "independent_gaussian":
            # assume that the Gaussians are far enough away to be pairwise independent
            independent_gaussian_entropy = self.distribution.component_distribution.entropy()
            weighted_gaussian_entropy = torch.einsum('ij, ij -> i', independent_gaussian_entropy, mixture_weights)

            mixture_entropy = self.distribution.mixture_distribution.entropy()
            approximate_entropy = weighted_gaussian_entropy + mixture_entropy
        elif self.entropy_approximation_mode == "samples":
            # a sampling based approach with self.distribution.rsample() does not work easily as rsample is not
            # implemented for MixtureSameFamily :(
            # Instead we need something similar to a gumbel softmax:
            #   https://discuss.pytorch.org/t/how-to-compute-gradient-of-sample-of-categorical-distribution/71970/2
            samples = self.distribution.component_distribution.rsample()
            evaluations = torch.stack([self.distribution.log_prob(samples[:, i, :])
                                       for i in range(samples.shape[1])], dim=-1)
            approximate_entropy = torch.einsum('ij, ij -> i', evaluations, mixture_weights)
        else:
            raise NotImplementedError(f"Entropy mode '{self.entropy_approximation_mode}' not provided")

        # previously: return sum_independent_dims(self.distribution.entropy())
        return approximate_entropy

    def sample(self) -> torch.Tensor:
        # rsample does not work for mixtures, so we work around it with this
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_params(self, weights: torch.Tensor, mean_actions: torch.Tensor,
                            log_std: torch.Tensor, deterministic: bool = False,
                            component_selection: Optional[Union[str, int]] = None) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(weights=weights, mean_actions=mean_actions, log_std=log_std)
        return self.get_actions(deterministic=deterministic, component_selection=component_selection)

    def get_actions(self, deterministic: bool = False,
                    component_selection: Optional[Union[str, int]] = None) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :param component_selection: How to select the component of the mixture to draw from. May be
            None for no component selection, i.e, random sampling over the full GMM
            "best" to always select the component with the highest weight, and then draw from that
            the id (starting from 0) of any component, to draw from this component.
        :return:
        """
        if component_selection is None:
            if deterministic:
                component_selection = "best"
            else:
                return self.distribution.sample()

        if isinstance(component_selection, str) and component_selection == "best":  # select best component
            indices = torch.argmax(self.distribution.mixture_distribution.probs, dim=-1)

        elif isinstance(component_selection, int):  # select component with given id
            indices = self.distribution.mixture_distribution.probs[:, component_selection].type(torch.LongTensor)
        else:
            raise ValueError(f"Unknown component selection strategy '{component_selection}'")
        # shape (batch_size)

        if deterministic:
            probs = self.distribution.component_distribution.mean
        else:
            probs = self.distribution.component_distribution.sample()
        # shape (batch_size, components, action_dimension)
        selected_samples = torch.stack([prob[index]
                                        for index, prob in zip(indices, probs)])

        return selected_samples

    def log_prob_from_params(self, weights: torch.Tensor, mean_actions: torch.Tensor,
                             log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param weights:
        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(weights, mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob
