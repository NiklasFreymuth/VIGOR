import numpy as np
from util.Types import *
from util.Defaults import TRAIN
from util.Normalization import normalize_samples
from util.Types import *
import numpy as np
from algorithms.VIGOR import VIGOR
from algorithms.distributions.GMM import GMM
from algorithms.AbstractReward import AbstractReward
from util.Defaults import TRAIN, TEST, DREX_TRAIN
from util.Functions import get_from_nested_dict
from functools import partial
from algorithms.distributions.GMM import GMM
from algorithms.AbstractReward import AbstractReward

"""
DRex - https://dsbrown1331.github.io/CoRL2019-DREX/
"""


class DRex(VIGOR):

    def __init__(self,
                 config: dict,
                 expert_observations: np.ndarray,
                 train_contexts: np.ndarray,
                 test_contexts: np.ndarray,
                 policy_dimension: int,
                 action_to_observation_function: MappingFunction,
                 train_policies: List[GMM]):
        if len(test_contexts) > 0:  # has test contexts
            assert train_contexts.shape[-1] == test_contexts.shape[-1], \
                "Train and test contexts must match in context dim, got {} and {}".format(train_contexts.shape,
                                                                                          test_contexts.shape)

        super().__init__(config=config, expert_observations=expert_observations,
                         train_contexts=train_contexts,
                         validation_contexts=None,
                         test_contexts=test_contexts,
                         policy_dimension=policy_dimension,
                         action_to_observation_function=action_to_observation_function)
        self._train_policies = train_policies
        self._reward_config = config.get("vigor").get("reward")
        self._regression_network_history = None
        self._reward_alpha = get_from_nested_dict(config, list_of_keys=["vigor", "reward_alpha"])

        self._drex_reward = None
        self._reward_parameters = {}
        self._reward = None

        self.has_stepwise_reward = False

    def initialize_training(self):
        """
        For TestOnlyVIGOR, we do not need to initialize train policies or a DRE. Instead, we take the provided values and
        create what we need directly. For simplicity reasons, we also directly do the regression.
        Returns:

        """
        self._initialize_policies()
        self._drex_reward = DRexReward(input_shape=self.expert_observations.shape[2:],
                                       network_config=self.config.get("network"),
                                       network_type=self._get_additional_dre_information().get("network_type"),
                                       train_contexts=self._dre_contexts,
                                       reward_config=self._reward_config,
                                       action_to_observation_function=self._action_to_observation_function,
                                       policies=self.policies.get(TRAIN))

        self._regression_network_history = self._drex_reward.regress_reward(current_policies=self.policies.get(TRAIN))
        reward_pieces = self._drex_reward.get_reward()
        self._reward_parameters = reward_pieces.get("parameters")
        self._reward = reward_pieces.get("callable")

    @property
    def policies(self) -> Dict[str, List[GMM]]:
        return {TRAIN: self._train_policies,
                TEST: [policy_learner.model for policy_learner in self.policy_learners.get(TEST)],
                DREX_TRAIN: [policy_learner.model for policy_learner in self.policy_learners.get(DREX_TRAIN)]
                }

    @property
    def reward_parameters(self):
        return self._reward_parameters

    def get_policy_update_reward(self, samples: np.array, context: np.array, policy: GMM,
                                 policy_mode: Optional[str] = None) -> np.array:
        """
        The update reward for TestOnlyVIGOR is only ever used for test contexts.
        Args:
            samples: A 2-dimensional array of policy samples
            context: A 1-dimensional array of contexts
            policy: The policy that was used for the samples
            policy_mode: the kind of policy used. Either "train", "validation" or "test"

        Returns: The reward evaluation to update this policy

        """
        contexts = np.repeat(context[None, :], repeats=len(samples), axis=0)
        reward_part = self._reward_alpha * self.recovered_reward_function(samples, contexts)
        policy_part = policy.log_density(samples=samples)
        reward = reward_part - policy_part
        return reward

    @property
    def recovered_reward_function(self):
        return self._reward

    @property
    def policy_modes(self):
        return [TEST, DREX_TRAIN]

    @property
    def dre_networks(self):
        raise ValueError("No Density Ratio Estimator for DRex")

    def train_iteration(self, iteration: int):
        """
        Here, every iteration effectively boils down to the EIM updates on multiple contexts at once.
        Args:
            iteration:

        Returns:

        """
        if iteration > 0:
            self._regression_network_history = None  # only need to record history once

        for mode in self.policy_modes:
            if self._reward_config.get(f"{mode}_policies_heuristics"):
                self._apply_policies_heuristics(policy_mode=mode,
                                                policy_iteration=iteration)
            for context, policy_learner in zip(self.contexts.get(mode), self.policy_learners.get(mode)):
                policy_learner.update_model(reward_function=partial(self._get_reward_per_component,
                                                                    context=context,
                                                                    policy_mode=mode))


class DRexReward(AbstractReward):
    def __init__(self,
                 input_shape: Union[tuple, int],
                 train_contexts: np.ndarray,
                 reward_config: Dict[Key, Any],
                 network_config: Dict[Key, Any],
                 network_type: str,
                 policies: List[GMM],
                 action_to_observation_function: MappingFunction = None):
        super(AbstractReward, self).__init__(input_shape=input_shape,
                                             dre_contexts=train_contexts,
                                             reward_config=reward_config,
                                             network_config=network_config,
                                             network_type=network_type,
                                             action_to_observation_function=action_to_observation_function
                                             )
        self._add_sampling_policies(new_policies=policies)
        drex_config = reward_config.get("drex")
        self._maximum_variance_multiplier: float = drex_config.get("maximum_variance_multiplier")
        self._num_variance_levels: int = drex_config.get("num_variance_levels")
        self._num_steps = input_shape[0]

    def _get_data(self, current_policies: List[GMM]) -> List[Dict[Key, np.array]]:
        """
        Internal method to compute and provide the data to fit the regression network with. The data is sampled from
        an internal list of sampling policies, with can be updated with new policies via the add_sampling_policies
        method.
        Assumes that the policies are provided in the same order as the original contexts
        Args:
            current_policies: The policies to evaluate for the targets. These are the "freshest" policies of the
            underlying EIM optimization. Note that these may differ from the sampling policies that this class maintains

        Returns: A dictionary of numpy arrays {samples, targets}

        """
        assert self._maximum_variance_multiplier >= 1, "May not decrease variance. Got '{}'".format(
            self._maximum_variance_multiplier)
        noise_multipliers = np.linspace(start=1, stop=self._maximum_variance_multiplier, num=self._num_variance_levels)
        samples_per_noise_level = int(self._num_regression_samples / len(noise_multipliers))
        num_samples_per_policy = int(
            self._num_regression_samples / (len(noise_multipliers) * self.num_dre_contexts))

        targets = []
        observations = []

        for noise_multiplier in noise_multipliers:
            # create samples of varying variance, starting with the "original" sample variance.

            current_samples = self._sample(num_samples_per_policy=num_samples_per_policy,
                                           current_policies=current_policies,
                                           sample_variance_multiplier=noise_multiplier * self._sample_variance_multiplier)

            # contextualize and flatten samples
            current_contexts = np.repeat(self._dre_contexts[:, None, :], repeats=num_samples_per_policy, axis=1)

            # get observations from current input parameterization
            current_observations = self._action_to_observation_function(current_samples, current_contexts)
            current_observations = current_observations.reshape(-1, *current_observations.shape[-2:])
            observations.append(current_observations)
            targets.append(np.full(shape=current_observations.shape[0], fill_value=1 / noise_multiplier))
            # target normalization does not really make sense for DRex

        # target shape: (#variance_levels, #contexts*samples_per_context)
        # observation shape: (#variance_levels, #contexts*samples_per_context, *observation_shape)
        observations = np.concatenate(observations, axis=0)
        targets = np.concatenate(targets, axis=0)
        # target shape: (#variance_levels*contexts*samples_per_context)
        # observation shape: (#variance_levels*contexts*samples_per_context, *observation_shape)

        if self._sample_normalization == "normal":
            observations, self._input_normalization_parameters = normalize_samples(samples=observations,
                                                                                   normalization_type="normal",
                                                                                   return_dict=True)
        data_list = self._allocate_samples(observations, targets)  # allocate to networks
        return data_list

    def _sample(self, num_samples_per_policy: int, current_policies: List[GMM],
                sample_variance_multiplier: float) -> np.array:
        """
        Samples num_samples_per_policies samples from each of the policies in self._sampling_policies.
        If self._sample_variance_multiplier >1, then the policies covariance will be multiplied by said multiplier
        for the sampling, effective providing a more diverse set of samples for each policy
        Args:
            num_samples_per_policy: Number of samples to draw for each policy
            current_policies: The current list of policies. If self._fresh_policy_sample_rate > 0, then some of the
            samples will be taken from these policies
            sample_variance_multiplier: Multiplier for the sample variance.

        Returns:

        """
        assert self._fresh_policy_sample_rate == 1, \
            "Can not draw old samples with DRex, " \
            "but got fresh sample rate <1 of {}".format(self._fresh_policy_sample_rate)
        num_fresh_samples_per_policy = int(num_samples_per_policy * self._fresh_policy_sample_rate)
        assert current_policies is not None, "Must provide current policies to sample fresh samples"
        current_sampling_policies = [GMM(weights=current_gmm.weights,
                                         means=current_gmm.means,
                                         covars=sample_variance_multiplier * current_gmm.covars)
                                     for current_gmm in current_policies]
        samples = np.array([current_policy.sample(num_fresh_samples_per_policy)
                            for current_policy in current_sampling_policies])
        return samples
