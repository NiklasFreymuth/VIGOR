from algorithms.distributions.GMM import GMM
from algorithms.distributions.Gaussian import Gaussian
from algorithms.distributions.Categorical import Categorical
from algorithms.auxiliary.MoreGaussian import MoreGaussian
from algorithms.auxiliary.RepsCategorical import RepsCategorical
from algorithms.auxiliary.Regression import QuadFunc
from util.Functions import to_contiguous_float_array
from util.PolicyInitialization import get_initial_gaussian_parameters
from util.Normalization import log_normalize
from util.Types import *
import numpy as np
import copy

"""
Wrapper for learning/updating the GMM of EIM
"""

class GMMLearner:

    def __init__(self, eta_offset: float, policy_config: Dict[Key, Any], initial_model: GMM,
                 constrain_entropy: bool = False,
                 omega_offset: float = 0):
        """
        Initializes the GMM learner by specifying the dimensions of the problem, internal MORE parameters (eta, omega)
        and more
        Args:
            eta_offset: Offset to make the MORE equations hold with the given rewards/settings
            policy_config: A dict containing different information of how to construct and update the policy
            initial_model: Initial model to learn/train
            constrain_entropy: Whether to constrain the entropy of the new gaussian distributions or not
            Number of consecutive "bad" iterations allowed for a policy component before it gets deleted
        """
        self._dimension = initial_model.means.shape[-1]
        self._eta_offset = eta_offset
        self._omega_offset = omega_offset
        self._constrain_entropy = constrain_entropy

        self._maximum_time_to_live = policy_config.get("component_time_to_live")
        self._component_weight_threshold = policy_config.get("component_weight_threshold")
        self._num_maximum_components = policy_config.get("num_components")

        self._kl_bound = policy_config.get("kl_bound")
        self._samples_per_component = policy_config.get("samples_per_component")
        self._weight_update_type = policy_config.get("weight_update_type")

        self._initialize_model(model=initial_model)

    def _initialize_model(self, model: GMM):
        """
        Initializes a learner for every component of the model as well as the categorical distribution that gives
        the relationship between these components
        :param model: The GMM to initialize the learner for
        :return:
        """
        self._model = model

        # time to live:
        self._component_time_to_live = {}
        for component_id in self._model.component_ids:
            self._component_time_to_live[component_id] = self._maximum_time_to_live

        # learners
        self._component_learners = []
        for i in range(self._model.num_components):
            new_component_learner = self._get_new_component_learner()
            self._component_learners.append(new_component_learner)
        self._weight_learner = RepsCategorical(eta_offset=self._eta_offset, omega_offset=self._omega_offset,
                                               constrain_entropy=self._constrain_entropy)

    def _get_new_component_learner(self) -> MoreGaussian:
        new_component_learner = MoreGaussian(dim=self._dimension,
                                             eta_offset=self._eta_offset,
                                             omega_offset=self._omega_offset,
                                             constrain_entropy=self._constrain_entropy)
        return new_component_learner

    def update_model(self, reward_function: Callable[[np.array, GMM], np.array]):
        """
        Updates the model with the provided reward function. For this, the component weights and the components
        themselves are updated separately. We update the weights first since the components then depend on these weights
        Args:
            reward_function: The function to update based on

        Returns: An instance of this learner for compatibility with multiprocessing

        """
        self._update_weight(reward_function=reward_function)
        self._update_components(reward_function=reward_function)
        return self

    def _update_components(self, reward_function: Callable[[np.array, GMM], np.array]) -> None:
        """
        Updates the components of the model (GMM) given GMM components according to the MORE equations.
        This uses a local quadratic surrogate model that is built from the provided reward function.
        Args:
            reward_function: Function that evaluates samples of the GMM

        Returns:
        """
        samples = to_contiguous_float_array(self.model.sample_per_component(self._samples_per_component))

        rewards = to_contiguous_float_array(reward_function(samples, self.model))
        # calculate all rewards at once

        for i in range(self._model.num_components):
            component_rewards = rewards[i]
            if np.all(component_rewards == component_rewards[0]):
                # All inputs equal. Skipping update
                continue
            self._update_component(component=self._model.components[i],
                                   learner=self._component_learners[i],
                                   rewards=component_rewards,
                                   samples=samples[i])

    def _update_component(self, component: Gaussian, learner: MoreGaussian, rewards: np.array, samples: np.array):
        """
        Updates a singular Gaussian component using the MORE equations
        Args:
            component: The component to update
            learner: The class using the MORE equations to update this components
            rewards: Rewards to base the update on
            samples: Samples corresponding to these rewards

        Returns:

        """
        old_component = copy.deepcopy(component)

        quadratic_surrogate = QuadFunc(1e-12, normalize=True, unnormalize_output=False)

        quadratic_surrogate.fit(inputs=samples, outputs=rewards, weights=None, gaussian_mean=old_component.mean,
                                gaussian_chol_cov=old_component.chol_covar)
        # This is a numerical thing we did not use in the original paper: We do not undo the output normalization
        # of the regression, this will yield the same solution but the optimal lagrangian multipliers of the
        # MORE dual are scaled, so we also need to adapt the offset. This makes optimizing the dual much more
        # stable and indifferent to initialization
        old_eta_offset = learner.eta_offset
        old_omega_offset = learner.omega_offset
        learner.eta_offset = learner.eta_offset / quadratic_surrogate.o_std
        learner.omega_offset = learner.omega_offset / quadratic_surrogate.o_std
        new_mean, new_covar = learner.more_step(self._kl_bound, -1, component, quadratic_surrogate)
        if learner.success:
            component.update_parameters(new_mean, new_covar)
        learner.eta_offset = old_eta_offset
        learner.omega_offset = old_omega_offset

    def _update_weight(self, reward_function: Callable[[np.array, GMM], np.array]):
        """
        Updates the weights of the components of the marginal EIM distribution (the GMM).
        Samples from the GMM to compute the rewards and updates either in open or closed form.
        Only updates if the update is enabled and the model has at least 2 components.
        :param reward_function: Function to evaluate component samples by
        :return:
        """
        if self._weight_update_type in [False, None, "no_update", "none", "None"]:
            # no update needed, so we directly return
            return
        if self._model.num_components > 1:  # at least one component, so a weight update makes sense
            generated_samples = self.model.sample_per_component(self._samples_per_component//self.model.num_components)
            rewards = reward_function(generated_samples, self.model)
            rewards = np.mean(rewards, axis=1)
            rewards = np.squeeze(np.ascontiguousarray(np.stack(rewards, -1).astype(np.float64)))

            if self._weight_update_type in ["closed_form", "closed"]:
                self._update_weight_closed_form(rewards=rewards)
            elif self._weight_update_type in ["open", "open_form", "vanilla", "eim"]:
                self._update_weight_with_reps(rewards=rewards)
            else:
                raise NotImplementedError("Weight update type '{}' does not exist".format(self._weight_update_type))

    def _update_weight_with_reps(self, rewards):
        """
        Update weight according to MORE equations. This is effectively a reps problem
        :param rewards: component-wise rewards to update based on
        :return:
        """
        old_dist = Categorical(log_probabilities=self.model.log_weights)

        # -1 as entropy bound is a dummy as entropy is not constraint
        new_log_probabilities = self._weight_learner.reps_step(self._kl_bound, -1, old_dist, rewards)
        if self._weight_learner.success:
            self._model.weight_distribution.log_probabilities = new_log_probabilities

    def _update_weight_closed_form(self, rewards: np.array, separate_model: GMM = None):
        """
        Closed form weight update for the model to be trained. Essentially computes a entropy-regularized softmax weight
        update.
        Args:
            rewards: Component-wise rewards to update based on
            separate_model: (Optional) A separate model to update. If left unspecified, self._model will be updated
        Returns:

        """
        rewards -= np.mean(rewards)  # zero-mean the rewards
        old_weights = self.model.log_weights
        new_log_probabilities = (self._eta_offset * old_weights + rewards) / (self._eta_offset + self._omega_offset)
        new_log_probabilities = log_normalize(samples=new_log_probabilities)  # normalize
        if separate_model is None:
            self._model.weight_distribution.log_probabilities = new_log_probabilities
        else:
            separate_model.weight_distribution.log_probabilities = new_log_probabilities

    def add_component(self, reward_function: Callable, samples: np.array = None):
        """
        Heuristic for adding a new component to the model. A new component is added iff the current number of
        components is smaller than the maximum number allowed. For adding the new component, we either consider
        randomly drawn samples from this policy or the provided samples (if provided). The component is added to
        the most promising region as determined by the reward evaluations of these samples.
        Roughly speaking, the initial mean is the drawn sample with the highest reward.
        The initial covariance of the newly added components is determined using a boltzmann distribution over
        the distances of all drawn samples to the new component multiplied by their reward.
        As of now, a new component
        is added whenever this function is called as long as current_components < num_components.
        For this, we draw sample_per_component samples for every component and add the new component at the
        position of maximum reward.
        Args:
            reward_function: A reward function that takes a 1d array of samples
            and evaluates each of them. A higher evaluation is better.
            Will be used to determine the most promising region to add the component at
            samples: (Optional) array of samples to consider for the new component mean. The sample with the best
            reward evaluation will become the new mean. Similarly, the covariance will depend on how the other samples
            relate to this best sample and how their rewards look like

        Returns: The id of the newly added component if one is added, None otherwise

        """
        if self.model.num_components < self._num_maximum_components:  # can add new component
            if samples is None:
                samples = self.model.sample_per_component(num_samples_per_component=self._samples_per_component)
                samples = np.concatenate(samples, axis=0)
            rewards = reward_function(samples, policy=self.model)

            new_covariance, new_mean = get_initial_gaussian_parameters(samples=samples, rewards=rewards)
            new_component_id = self._add_new_component(initial_mean=new_mean, initial_covar=new_covariance)
            return new_component_id
        else:
            return None

    def _add_new_component(self, initial_mean: np.array, initial_covar: np.array) -> int:
        """
        Adds a component specified by the given weight, mean and covariance. The weight is absolute, and the other
        component will adapt their weight based on this
        :param initial_mean:
        :param initial_covar:
        :return: The id of the created component
        """
        initial_weight = self._component_weight_threshold
        component_id = self._model.add_component(initial_weight=initial_weight,
                                                 initial_mean=initial_mean,
                                                 initial_covar=initial_covar)
        if component_id is not None:  # new component was successfully added
            new_component_learner = self._get_new_component_learner()
            self._component_learners.append(new_component_learner)
            self._component_time_to_live[component_id] = self._maximum_time_to_live
        return component_id

    def delete_components(self):
        """
        Applies one iteration of the component deletion heuristic. This reduces the time_to_live of every component
        of this GMM whose weight is below a given threshold by 1. If the ttl for a component reaches a value <0, this
        component is deleted.
        Returns:

        """
        weights = self.model.weights
        deletable_component_positions = [i for i, component_weight in enumerate(weights)
                                         if component_weight < self._component_weight_threshold]
        for position in reversed(range(len(weights))):  # iterate over all positions backwards due to shifting indices
            component_id = self._model.get_component_id(position)
            if position in deletable_component_positions:  # component weight below threshold
                self._component_time_to_live[component_id] -= 1
                if self._component_time_to_live[component_id] < 0:
                    self.delete_component(position=position)
            else:  # reset time to live
                self._component_time_to_live[component_id] = self._maximum_time_to_live

    def delete_component(self, position: int):
        """
        Deletes the component at the specified position of the model/gmm. The other components "fill up" the weight
        that becomes missing due to this deletion
        Args:
            position: Position of the component to be deleted

        Returns:

        """
        component_id = self._model.get_component_id(position)
        self._model.remove_component(position=position)
        # remove entries from dicts
        del self._component_learners[position]
        del self._component_time_to_live[component_id]

    @property
    def model(self) -> GMM:
        return self._model
