"""
Abstract class for task data. Ensures that all Data classes provide an init() and a function
to estimate the unnormalized log density.
"""
import abc
from util.colors.SmartColors import SmartColors
import numpy as np
from util.Types import *
from algorithms.distributions.GMM import GMM
from recording.AdditionalPlot import AdditionalPlot
import matplotlib.pyplot as plt
from util.Functions import get_from_nested_dict


class AbstractEnvironment(abc.ABC):

    def __init__(self, *, config: ConfigDict, contexts: Dict[int, np.array]):
        self.contexts = contexts  # mapping from context_id to numpy array
        self._num_evaluation_samples = get_from_nested_dict(dictionary=config,
                                                            list_of_keys=["meta_task", "num_evaluation_samples"],
                                                            raise_error=True)
        self._colors = SmartColors()

        # recording/additional plots
        self._reward_decompositions = {}

    def reward(self, samples, context_id: int, return_as_dict: bool = False, **kwargs) -> np.array:
        """
        Wrapper for self.uld() that transforms the samples into an appropriate input representation
        Args:
            samples: Samples to evaluate
            return_as_dict: Whether to return the evaluation of the reward as a dictionary of its individual components
              (True) or not (default, False)

        Returns:

        """
        raise NotImplementedError

    def uld(self, samples, context_id: int, return_as_dict: bool = False, **kwargs) -> Union[ValueDict, np.array]:
        """
        unnormalized log density of the distribution that describes the task
        Args:
            samples: Samples to evaluate
            return_as_dict: Whether to return the evaluation of the reward as a dictionary of its individual components
              (True) or not (default, False)

        Returns:

        """
        raise NotImplementedError

    def vectorized_mp_features(self, params: np.array, contexts: np.array,
                               feature_representation: str) -> np.array:
        raise NotImplementedError


    #########################
    # visualization utility #
    #########################

    @abc.abstractmethod
    def plot(self, policy: GMM, context_id: int, **kwargs):
        raise NotImplementedError("Can not plot from AbstractTask")

    @abc.abstractmethod
    def plot_samples(self, policy_samples: np.array, context_id: int, **kwargs):
        raise NotImplementedError("Can not plot policy samples from AbstractTask")

    def reward_decomposition(self, policy: GMM, context_id: int):
        policy_samples = policy.sample(num_samples=self._num_evaluation_samples)
        decomposed_reward = self.reward(policy_samples, context_id=context_id, return_as_dict=True)
        if context_id not in self._reward_decompositions:
            self._reward_decompositions[context_id] = {"means": {}, "mins": {}, "maxs": {}}
        reward_decompositions = self._reward_decompositions[context_id]

        reward_means = {k: np.mean(v) for k, v in decomposed_reward.items()}
        reward_mins = {k: np.min(v) for k, v in decomposed_reward.items()}
        reward_maxs = {k: np.max(v) for k, v in decomposed_reward.items()}

        for position, reward_part_name in enumerate(decomposed_reward.keys()):
            if reward_part_name not in reward_decompositions["means"].keys():  # initilization
                reward_decompositions["means"][reward_part_name] = []
            if reward_part_name not in reward_decompositions["mins"].keys():
                reward_decompositions["mins"][reward_part_name] = []
            if reward_part_name not in reward_decompositions["maxs"].keys():
                reward_decompositions["maxs"][reward_part_name] = []

            reward_decompositions["means"][reward_part_name].append(reward_means[reward_part_name])  # update
            reward_decompositions["mins"][reward_part_name].append(reward_mins[reward_part_name])
            reward_decompositions["maxs"][reward_part_name].append(reward_maxs[reward_part_name])

            means = np.array(reward_decompositions["means"][reward_part_name])
            mins = np.array(reward_decompositions["mins"][reward_part_name])
            maxs = np.array(reward_decompositions["maxs"][reward_part_name])
            plt.plot(range(len(means)), means, label=reward_part_name.title().replace("_", " "),
                     color=self._colors(position))
            plt.plot(range(len(mins)), mins, "--", color=self._colors(position))
            plt.plot(range(len(maxs)), maxs, "--", color=self._colors(position))
        plt.legend(loc="upper left")
        plt.yscale("symlog")
        plt.xlabel("Iteration")
        plt.ylabel("Log Density")

    def get_additional_plots(self) -> List[AdditionalPlot]:
        """
        Collect all functions that should be used to draw additional plots. These functions take as argument the
        current policy and return the title of the plot that they draw

        Returns: A list of functions that can be called to draw plots. These functions take the current policy as
          argument, and draw in the current matplotlib figure. They return the title of the plot.

        """
        plots = [AdditionalPlot(function=self.reward_decomposition,
                                is_policy_based=True,
                                uses_context_id=True,
                                uses_iteration_wise_figures=False,
                                is_expensive=False),
                 ]
        return plots

    ####################
    # metric recording #
    ####################

    def negative_policy_elbo(self, policy: GMM, context_id: int, **reward_kwargs) -> np.array:
        """
        Computes the negative evidence lower bound of the policy to the reward function.
        This can be seen as an unnormalized numerical KL divergence,
        Args:
            policy:
            context_id: The context id of the policy
            **reward_kwargs: Additional parameters to pass to the reward function

        Returns:

        """
        model_samples = policy.sample(self._num_evaluation_samples)
        evaluated_policy_log_density = policy.log_density(model_samples)
        evaluated_reward_log_density = self.reward(samples=model_samples,
                                                   context_id=context_id, **reward_kwargs)
        negative_elbo = np.mean(evaluated_policy_log_density - evaluated_reward_log_density)
        return negative_elbo

    def get_reward_dict(self, policy: GMM, context_id: int, **kwargs) -> ValueDict:
        component_reward_dicts = {}
        for component in policy.components:
            samples = component.sample(self._num_evaluation_samples)
            reward_dict = self.reward(samples, context_id=context_id, return_as_dict=True, **kwargs)
            # Dictionary with 1 value per metric

            for (key, value) in reward_dict.items():
                metric_list = component_reward_dicts.setdefault(key, [])  # key might exist already
                metric_list.append(np.mean(value))
            # Dictionary with #components entries per metric

        component_reward_dicts = {key: sorted(value) for key, value in component_reward_dicts.items()}
        component_reward_dicts = {f"c{idx}_{key}": value
                                  for key, metric in component_reward_dicts.items()
                                  for idx, value in enumerate(metric)}

        mean_reward_dict = self.reward(policy.means, context_id=context_id, return_as_dict=True, **kwargs)
        mean_reward_dict = {"mean_" + key: np.dot(value, policy.weights) for key, value in mean_reward_dict.items()}

        samples_reward_dict = self.reward(policy.sample(self._num_evaluation_samples), context_id=context_id,
                                          return_as_dict=True, **kwargs)
        samples_reward_dict = {"samples_" + key: np.mean(value) for key, value in samples_reward_dict.items()}
        return {**component_reward_dicts, **mean_reward_dict, **samples_reward_dict}

    def get_metrics(self, policy: GMM, context_id: int) -> ValueDict:
        """
        Compute all environment-specific metrics for the given policy
        Args:
            policy: A gaussian mixture model policy parameterizing the actions of this environment
            context_id: The context id of the current policy

        Returns: A dictionary {metric_name: salar_evaluation} for a number of relevant metrics

        """
        return self.get_reward_dict(policy=policy, context_id=context_id)
