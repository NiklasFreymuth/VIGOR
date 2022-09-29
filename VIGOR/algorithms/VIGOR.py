import copy
import torch
from torch import nn
import numpy as np

from joblib import Parallel, delayed  # delayed prevents the method from executing before being parallelized
import matplotlib.pyplot as plt

from util.Types import *
import util.Defaults as d
from util.Defaults import TRAIN, TEST, VALIDATION, DREX_TRAIN
import util.PolicyInitialization as policy_initialization
from util.colors.SmartColors import SmartColors
from util.VisualizationUtil import plot_component_weights
from util.Functions import get_from_nested_dict, save_concatenate, sigmoid
from util.pytorch.UtilityFunctions import detach

from recording.AdditionalPlot import AdditionalPlot

from algorithms.dre.VIGOR_DRE import VIGOR_DRE

from algorithms.VigorUtil import get_dre_aggregation
from algorithms.auxiliary.GMM_Learner import GMMLearner
from algorithms.distributions.GMM import GMM

"""
Main Code for VIGOR
"""


class VIGOR:

    def __init__(self, config: dict,
                 expert_observations: np.ndarray,
                 train_contexts: np.ndarray,
                 validation_contexts: np.ndarray,
                 test_contexts: np.ndarray,
                 policy_dimension: int,
                 action_to_observation_function: callable):
        assert expert_observations.shape[0] == train_contexts.shape[0], \
            "Expert observations must be arranged by contexts, got {} and {}".format(expert_observations.shape,
                                                                                     train_contexts.shape)

        self.config = config
        self._policy_dimension = policy_dimension

        self.expert_observations = expert_observations
        self._action_to_observation_function = action_to_observation_function

        # initializing DRE values
        self._num_dres = self.config.get("network").get("num_dres")
        dre_aggregation_str = self.config.get("network").get("dre_aggregation")

        self._dre_aggregation = get_dre_aggregation(num_dres=self._num_dres, dre_aggregation_str=dre_aggregation_str)

        self._dres = None
        self._dre_histories = None


        if validation_contexts is not None and len(validation_contexts) > 0:
            self._dre_contexts = np.concatenate((train_contexts, validation_contexts), axis=0)
        else:
            self._dre_contexts = train_contexts

        self.contexts = {TRAIN: train_contexts,
                         VALIDATION: validation_contexts if validation_contexts is not None else [],
                         TEST: test_contexts if test_contexts is not None else [],
                         DREX_TRAIN: train_contexts if train_contexts is not None else []}

        self.policy_learners = {}

        self._has_validation_contexts = len(validation_contexts) > 0 if validation_contexts is not None else False

        self.n_update_jobs = get_from_nested_dict(config,
                                                  list_of_keys=["vigor", "n_update_jobs"],
                                                  default_return=1)

        ### recording utility ###
        architecture = get_from_nested_dict(config,
                                            list_of_keys=["network", "time_series", "architecture"],
                                            default_return=False)
        is_full_variant = get_from_nested_dict(config,
                                               list_of_keys=["network", "time_series",
                                                             "hollistic_mlp", "variant"],
                                               default_return=False) == "full"
        self.has_stepwise_reward = not (architecture and is_full_variant)

        ########################
        ### additional plots ###
        ########################
        self._stepwise_dre_evaluations = None
        self._policy_weight_colors = None
        self._component_weights = None
        self._component_weight_iteration = None

        ####################
        # metric recording #
        ####################
        self._num_evaluation_samples = config.get("meta_task").get("num_evaluation_samples")
        self.policy_weight_update_type = get_from_nested_dict(self.config, ["policy", "weight_update_type"])


    @property
    def policies(self) -> Dict[str, List[GMM]]:
        return {TRAIN: [policy_learner.model for policy_learner in self.policy_learners.get(TRAIN)],
                VALIDATION: [policy_learner.model for policy_learner in self.policy_learners.get(VALIDATION)]}

    @property
    def policy_modes(self):
        return [TRAIN, VALIDATION]

    @property
    def dre_networks(self) -> List[nn.Module]:
        return [dre.network for dre in self._dres]

    def initialize_training(self):
        self._initialize_policies()
        self._initialize_dres()

    def _initialize_policies(self):
        """
        Initializes a GMM policy and its learner for every train and every test context
        Returns:

        """
        policy_config = self.config.get("policy")
        num_components = policy_config.get("num_components")

        self._policy_component_addition_frequency = policy_config.get("component_addition_frequency")
        self._policy_component_weight_threshold = policy_config.get("component_weight_threshold")
        assertion_string = "Can not have a weight threshold of {} for {} components".format(
            self._policy_component_weight_threshold, num_components)  # make sure that deletions make sense
        assert self._policy_component_weight_threshold * num_components < 1, assertion_string

        for mode in self.policy_modes:
            self.policy_learners[mode] = []  # set key for mode
            for _ in self.contexts.get(mode):  # create a policy for every context
                current_gmm = policy_initialization.initialize_gmm(
                    num_components=num_components,
                    dimension=self._policy_dimension,
                    covariance_scale=1)
                self.policy_learners.get(mode).append(self._get_gmm_learner(gmm=current_gmm))

    def _get_gmm_learner(self, gmm: GMM) -> GMMLearner:
        eta_offset = 1.0
        omega_offset = 0.0
        policy_learner = GMMLearner(eta_offset=eta_offset,
                                    omega_offset=omega_offset,
                                    policy_config=self.config.get("policy"),
                                    initial_model=copy.deepcopy(gmm))
        return policy_learner

    def _initialize_dres(self):
        """
        Wrapper for initializing the correct dre and setting up variables for it
        """

        expert_observations = np.concatenate(self.expert_observations, axis=0)  # flatten over contexts

        # shuffle the demonstrations from different contexts into each other for fairness in DRE training
        copied_expert_observations = copy.deepcopy(expert_observations)
        np.random.shuffle(copied_expert_observations)

        dre_policies = self.policies.get(TRAIN) + self.policies.get(VALIDATION)

        self._dres = [VIGOR_DRE(config=self.config,
                               expert_observations=copied_expert_observations,
                               eim_contexts=self._dre_contexts,
                               additional_information=self._get_additional_dre_information(),
                               eim_policies=dre_policies,
                               action_to_observation_function=self._action_to_observation_function)
                      for _ in range(self._num_dres)]

    def _get_additional_dre_information(self):
        return {"network_type": "sequence"}

    def train_iteration(self, iteration: int):
        """
        Train (and record) the EIM algorithm for one iteration.
            - Trains the density ratio estimator
            - Updates component weights
            - Updates components (means and covariances)
        :param iteration: The current iteration. Needed for recording purposes
        :return: Nothing, as the updates are performed on the class variables.
        """
        self._dre_histories = [dre.train() for dre in self._dres]

        # update validation policies
        if self._has_validation_contexts:
            self._apply_policy_update_step(step=iteration, policy_mode=d.VALIDATION, apply_heuristics=True)

        # update train policies
        self._apply_policy_update_step(step=iteration, policy_mode=d.TRAIN, apply_heuristics=True)

    def _apply_policy_update_step(self, step: int, policy_mode: str, apply_heuristics: bool = True):
        if apply_heuristics:
            self._apply_policies_heuristics(policy_mode=policy_mode,
                                            policy_iteration=step)

        # parallel updates!
        self.policy_learners[policy_mode] = Parallel(self.n_update_jobs)(delayed(policy_learner.update_model)
                                                                         (partial(self._get_reward_per_component,
                                                                                  context=context,
                                                                                  policy_mode=policy_mode))
                                                                         for context, policy_learner in
                                                                         zip(self.contexts.get(policy_mode),
                                                                             self.policy_learners.get(policy_mode)
                                                                             )
                                                                         )

    def get_policy_update_reward(self, samples: np.array, context: np.array, policy: GMM,
                                 policy_mode: Optional[str] = None) -> np.array:
        """
        Args:
            samples: A 2-dimensional array of policy samples
            context: A 1-dimensional array of contexts
            policy: The policy that was used for the samples
            policy_mode: the kind of policy used. Either d.TRAIN, d.VALIDATION or d.TEST

        Returns: The reward evaluation to update this policy

        """
        contexts = np.repeat(context[None, :], samples.shape[0], axis=0)
        samples = self._action_to_observation_function(samples, contexts=contexts)
        [dre_network.eval() for dre_network in self.dre_networks]
        samples = torch.tensor(samples.astype(np.float32))
        reward = self._dre_aggregation([detach(dre_network(samples).squeeze(axis=1))
                                        for dre_network in self.dre_networks])
        return reward

    def _apply_policies_heuristics(self, policy_mode: str, policy_iteration: int):
        """
        For the pseudo-contextual setting, the policy heuristic are applied to every policy independently. This function
        thus takes a list of policy learners and calls the addition and deletion procedures for each one
        Args:
            policy_mode: Either TRAIN or TEST. Determines which policies to update
            policy_iteration: The iteration of the policies. This may be different from the outer iteration of the
            algorithm, because we can update a policy multiple times between DRE updates

        Returns:

        """
        policy_learners = self.policy_learners.get(policy_mode)
        contexts = self.policy_learners.get(policy_mode)

        # delete components
        if self._policy_component_weight_threshold > 0:
            [policy_learner.delete_components() for policy_learner in policy_learners]

        # add components
        do_addition = self._policy_component_addition_frequency > 0 and policy_iteration > 0 \
                      and policy_iteration % self._policy_component_addition_frequency == 0
        if do_addition:
            for position, (policy_learner, context) in enumerate(zip(policy_learners, contexts)):
                reward_function = lambda x: self.get_policy_update_reward(x,
                                                                          context=context,
                                                                          policy=policy_learner.model,
                                                                          policy_mode=policy_mode)
                # add w/o expert demonstrations
                policy_learner.add_component(reward_function=reward_function, samples=None)

    def _get_reward_per_component(self, samples_per_component: Union[np.array, List[np.array]], policy: GMM,
                                  **kwargs) -> np.array:
        """
        Utility function for evaluating the reward for every component by simply calling the individual reward
        function for every component
        Overwrites the vanilla_EIM. Uses self._dre_reward as an approximated target_uld
        and respects the log probabilities of the model
        :param samples_per_component: A 3d array [#components, #samples, #policy_dimensions] of samples
        :return: The component-wise list of rewards
        """
        rewards = np.array([self.get_policy_update_reward(x, policy=policy,
                                                          **kwargs) for x in samples_per_component])
        return rewards

    #############################
    # additional plot functions #
    #############################
    def get_additional_plots(self) -> List[AdditionalPlot]:
        """
        Collect all functions that should be used to draw additional plots. These functions take as argument the
        current policy and return the title of the plot that they draw

        Returns: A list of functions that can be called to draw plots. These functions take the current policy as
          argument, and draw in the current matplotlib figure.

        """
        plots = [
            AdditionalPlot(function=self.component_weights,
                           is_policy_based=True,
                           uses_iteration_wise_figures=False,
                           is_expensive=False,
                           uses_context=True),
            AdditionalPlot(function=self.feature_histogram,
                           is_policy_based=True,
                           uses_iteration_wise_figures=True,
                           is_expensive=False,
                           uses_context=True)
        ]
        if self._dres is not None:
            plots.append(AdditionalPlot(function=self.stepwise_dre_evaluations,
                                        is_policy_based=True,
                                        uses_iteration_wise_figures=False,
                                        is_expensive=False,
                                        uses_context=True)
                         )
        if self.has_stepwise_reward:
            plots.append(
                AdditionalPlot(function=self.stepwise_recovered_rewards,
                               is_policy_based=True,
                               uses_iteration_wise_figures=False,
                               is_expensive=False,
                               uses_context=True)
            )
        return plots

    def feature_histogram(self, policy: GMM, context: np.array):
        samples = policy.sample(self._num_evaluation_samples)
        contexts = np.repeat(context[None, :], repeats=self._num_evaluation_samples, axis=0)
        observations = self._action_to_observation_function(samples, contexts=contexts).astype(np.float32)
        flat_observations = observations.reshape((-1, observations.shape[-1]))
        plt.ylabel("Frequency")
        for feature_position in range(flat_observations.shape[-1]):
            label = str(feature_position)
            plt.hist(flat_observations[..., feature_position], bins=20, label=label,
                     histtype="step")
        plt.legend(loc="upper right")

    def stepwise_dre_evaluations(self, policy: GMM, context: np.array) -> np.array:
        samples = policy.sample(num_samples=self._num_evaluation_samples)
        contexts = np.repeat(context[None, :], repeats=self._num_evaluation_samples, axis=0)
        observations = self._action_to_observation_function(samples, contexts=contexts).astype(np.float32)
        observations = torch.tensor(observations.astype(np.float32))
        [dre_network.eval() for dre_network in self.dre_networks]
        stepwise_dre_evaluations = self._dre_aggregation(
            [np.squeeze(detach(dre_network(observations, as_dict=True)[d.STEPWISE_LOGITS]))
             for dre_network in self.dre_networks])
        stepwise_reward_means = np.mean(stepwise_dre_evaluations, keepdims=True, axis=0)
        stepwise_reward_means = 2 * sigmoid(0.5 * stepwise_reward_means) - 1  # low-temperature tanh

        reward_image = self._get_dre_evaluation_reward_image(stepwise_reward_means, context=context)

        plt.ylabel("Timestep in trajectory")
        plt.xlabel("EIM Iteration")
        heatmap = plt.imshow(reward_image, cmap=plt.get_cmap("jet"), aspect="auto", vmin=-1, vmax=1)
        colorbar = plt.colorbar(heatmap)
        colorbar.set_label(r"$0.5\cdot tanh(\mathbb{E}_{x\in\pi}\lbrack\phi(x)\rbrack)$")

    def stepwise_recovered_rewards(self, policy: GMM, context: np.array, **reward_kwargs):
        # currently not supported for DREX
        samples = policy.sample(num_samples=self._num_evaluation_samples)
        contexts = np.repeat(context[None, :], repeats=self._num_evaluation_samples, axis=0)
        observations = self._action_to_observation_function(samples, contexts=contexts, **reward_kwargs)
        observations = torch.tensor(observations.astype(np.float32))

        [dre_network.eval() for dre_network in self.dre_networks]
        stepwise_dre_evaluations = self._dre_aggregation(
            [np.squeeze(detach(dre_network(observations, as_dict=True)[d.STEPWISE_LOGITS]))
             for dre_network in self.dre_networks])

        stepwise_reward_means = np.mean(stepwise_dre_evaluations, keepdims=True, axis=0)
        stepwise_reward_means = 2 * sigmoid(0.5 * stepwise_reward_means) - 1  # low-temperature tanh

        reward_image = self._get_dre_evaluation_reward_image(stepwise_reward_means, context=context)

        plt.ylabel("Timestep in trajectory")
        plt.xlabel("EIM Iteration")
        heatmap = plt.imshow(reward_image, cmap=plt.get_cmap("jet"), aspect="auto", vmin=-1, vmax=1)
        colorbar = plt.colorbar(heatmap)
        colorbar.set_label(r"$0.5\cdot tanh(\mathbb{E}_{x\in\pi}\lbrack\phi(x)\rbrack)$")

    def _get_dre_evaluation_reward_image(self, stepwise_reward_means: np.array, context: np.array) -> np.array:
        context = tuple(context.flatten())  # as tuple to allow this to be a dictionary key
        if self._stepwise_dre_evaluations is None:
            self._stepwise_dre_evaluations = {}  # initialize
        self._stepwise_dre_evaluations[context] = save_concatenate(self._stepwise_dre_evaluations.get(context),
                                                                   stepwise_reward_means)
        reward_image = self._stepwise_dre_evaluations.get(context).T
        return reward_image

    def component_weights(self, policy, **reward_kwargs):
        assert "context" in reward_kwargs, "Need to provide context for policy component weights in VIGOR"
        context = tuple(reward_kwargs.get("context").flatten())  # as tuple to allow this to be a dictionary key

        if self._component_weight_iteration is None:
            self._component_weight_iteration = {}
        if context not in self._component_weight_iteration:
            self._component_weight_iteration[context] = 0

        if self._policy_weight_colors is None:
            self._policy_weight_colors = {}
        if context not in self._policy_weight_colors:
            self._policy_weight_colors[context] = SmartColors()
        self._policy_weight_colors[context].assign_colors(ids=policy.component_ids)

        if self._component_weights is None:
            self._component_weights = {}
        if context not in self._component_weights:
            self._component_weights[context] = None
        self._component_weights[context] = plot_component_weights(policy=policy,
                                                                  component_weights=self._component_weights[context],
                                                                  policy_colors=self._policy_weight_colors[context],
                                                                  iteration=self._component_weight_iteration[context],
                                                                  plot_title=False)

        self._component_weight_iteration[context] = self._component_weight_iteration[context] + 1

    ####################
    # metric recording #
    ####################

    def get_policy_metrics(self, policy: GMM, context: np.array) -> ValueDict:
        metrics = {"weighted_component_entropy": self.weighted_component_entropy(policy=policy),
                   "expected_mixture_entropy": self.expected_mixture_entropy(policy=policy),
                   "total_drawn_samples": self.total_policy_samples(policy=policy),
                   "weighted_recovered_algorithm_reward": self.policy_algorithm_reward(policy=policy,
                                                                                       weighted=True,
                                                                                       context=context),
                   }

        if self.policy_weight_update_type not in [False, None, "no_update", "none", "None"]:
            metrics["equiweighted_expected_mixture_entropy"] = self.expected_mixture_entropy(
                policy=policy.get_equiweighted_gmm())
            metrics["equiweighted_recovered_algorithm_reward"] = self.policy_algorithm_reward(policy=policy,
                                                                                              weighted=False,
                                                                                              context=context)
        if get_from_nested_dict(self.config, ["policy", "num_components"], raise_error=True) > 1:
            metrics["normalized_policy_entropy"] = self.normalized_component_entropy(policy=policy)

            added_components, removed_components = self.policy_changed_components(policy=policy)
            metrics["num_added_policy_components"] = len(added_components)
            metrics["num_removed_policy_components"] = len(removed_components)

        if self._dres is not None:
            metrics["density_ratio_estimator_evaluation"] = self.density_ratio_estimator_policy_evaluation(
                policy=policy,
                context=context)
            metrics["density_ratio_estimator_confidence"] = self.density_ratio_estimator_confidence(policy=policy,
                                                                                                    context=context)

        return metrics

    def policy_changed_components(self, policy: GMM) -> Tuple[List[int], List[int]]:
        ids = policy.component_ids
        current_ids = set(ids)
        removed_components = sorted(list(policy.previous_component_ids - current_ids))
        added_components = sorted(list(current_ids - policy.previous_component_ids))
        policy.previous_component_ids = current_ids
        return added_components, removed_components

    def weighted_component_entropy(self, policy: GMM) -> np.array:
        entropies = np.array([c.entropy for c in policy.components])

        weights = policy.weight_distribution.probabilities
        weighted_component_entropy = np.sum(entropies * weights)
        return weighted_component_entropy

    def policy_algorithm_reward(self, policy: GMM, context: np.array, weighted: bool) -> np.array:
        if not weighted:
            policy = policy.get_equiweighted_gmm()

        rewards = self.get_policy_update_reward(samples=policy.sample(self._num_evaluation_samples),
                                                policy=policy,
                                                context=context)
        return np.mean(rewards)

    def total_policy_samples(self, policy: GMM) -> np.array:
        return policy.total_drawn_samples

    def normalized_component_entropy(self, policy: GMM) -> np.array:
        return policy.component_entropy / np.log(policy.num_components)

    def expected_mixture_entropy(self, policy: GMM, **reward_kwargs):
        log_densities = np.empty(policy.num_components)
        for position, component in enumerate(policy.components):
            action_samples = component.sample(self._num_evaluation_samples)
            component_log_densities = policy.log_density(action_samples)
            log_densities[position] = np.mean(component_log_densities)
        entropy = -np.sum(np.exp(policy.log_weights) * log_densities)
        return entropy

    def get_dre_metrics(self) -> ValueDict:
        metrics = {
            "trained_epochs": np.mean([len(dre_history.get(d.TOTAL_LOSS)) for dre_history in self._dre_histories]),
            "loss": np.mean([dre_history.get(d.TOTAL_LOSS)[-1] for dre_history in self._dre_histories]),
            "accuracy": np.mean([dre_history.get(d.ACCURACY)[-1] for dre_history in self._dre_histories]),
            "excitement": self.density_ratio_estimator_excitement(),
            "expert_confidence": self.density_ratio_estimator_confidence(policy="expert", context=None),
        }
        return metrics

    def density_ratio_estimator_policy_evaluation(self, policy: GMM, context: np.array) -> np.array:
        [dre_network.eval() for dre_network in self.dre_networks]
        policy_samples = policy.sample(num_samples=self._num_evaluation_samples)

        contexts = np.repeat(context[None, :], repeats=self._num_evaluation_samples, axis=0)
        policy_samples = torch.Tensor(self._action_to_observation_function(policy_samples, contexts=contexts))
        policy_evaluation = self._dre_aggregation([detach(dre_network(policy_samples))
                                                   for dre_network in self.dre_networks])
        return np.mean(policy_evaluation)

    def density_ratio_estimator_confidence(self, policy: Union[str, GMM], context: Optional[np.array]):
        if isinstance(policy, str) and policy == "expert":
            samples = self._dres[0].expert_observations
        elif isinstance(policy, GMM):
            assert context is not None
            samples = policy.sample(num_samples=self._num_evaluation_samples)
            contexts = np.repeat(context[None, :], repeats=self._num_evaluation_samples, axis=0)
            samples = self._action_to_observation_function(samples, contexts=contexts)
        else:
            raise ValueError(f"Unknown value for 'policy': {policy}")
        samples = torch.Tensor(samples)
        evaluation = self._dre_aggregation([detach(dre_network(samples)) for dre_network in self.dre_networks])
        confidence = np.mean(sigmoid(evaluation))
        return confidence

    def density_ratio_estimator_excitement(self) -> float:
        """
        Calculates the excitement of the Density Ratio Estimator, which is defined as the Root Mean Square of its
          outputs/density ratio estimates for the expert. The higher this value is, the more "excited" the estimator
          is to see an average expert demonstration. A high value thus corresponds to a confident DRE
        Returns:

        """
        observations = torch.Tensor(self._dres[0].expert_observations)
        evaluations = self._dre_aggregation([detach(dre_network(observations)) for dre_network in self.dre_networks])
        root_mean_squared_evaluations = float(np.sqrt(np.mean(evaluations ** 2)))
        return root_mean_squared_evaluations
