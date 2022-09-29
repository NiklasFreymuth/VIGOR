import copy
import torch
import numpy as np
from algorithms.distributions.GMM import GMM
from util.Types import *
import util.Defaults as d
from util.Functions import joint_shuffle, merge_dictionaries
from util.observation_util import append_contexts
from util.pytorch.UtilityFunctions import detach
from util.pytorch.NetworkBuilder import NetworkBuilder
from util.pytorch.architectures.Network import Network
from util.pytorch.training.NetworkTrainer import NetworkTrainer
from util.pytorch.training.TrainingUtil import dataloader_from_dictionary_data
from util.Normalization import normalize_samples, normalize_from_dict
from util.save.SaveDefaults import NORMALIZATION_PARAMETER_FILE

def reward_function(policy_samples: np.array, contexts: np.array,
                    action_to_observation_function: Optional[MappingFunction],
                    network_list: List[Network],
                    return_dict: bool = False,
                    network_aggregation_method: str = "min",
                    normalization_parameters: Optional[dict] = None) -> np.array:
    """
    Wrapper for applying the action_to_observation_function to any policy samples before calling the regressed
    network. This produces f_regressed(f_obs(x))
    Args:
        policy_samples: Samples to be evaluated.
        contexts: Contexts for these samples
        action_to_observation_function: The mapping function between policy space and the input space of the network
        network_list: A list of networks to evaluate by. Each network is a regression over one or multiple EIM policies
        and their last DRE. Will evaluate all networks and choose the minimum of them for the reward. This is similar
        to the double Q networks in e.g., Soft Actor Critic
        return_dict: Whether to return additional information for the networks in the form of a dictionary
        normalization_parameters: An optional dict to normalize features with before evaluating them. Will be applied
        after the action_to_observation_function if it is provided.

    Returns: The evaluations of the network of the (possibly mapped) input samples as a numpy array

    """
    observations = action_to_observation_function(policy_samples, contexts)

    if normalization_parameters is not None:
        observations = normalize_from_dict(samples=observations, normalization_parameters=normalization_parameters)

    observations = torch.tensor(observations.astype(np.float32))
    [network.eval() for network in network_list]  # set all networks to evaluation mode

    if return_dict:
        evaluation_dicts = [network(observations, as_dict=True) for network in network_list]
        predictions = np.array([np.atleast_1d(np.squeeze(detach(network_dict.get(d.PREDICTIONS))))
                                for network_dict in evaluation_dicts])

        if network_aggregation_method == "min":
            aggregation_positions = np.argmin(predictions, axis=0)
        elif network_aggregation_method == "mean":
            aggregation_positions = None  # just do regular mean later on
        elif network_aggregation_method == "median":
            aggregation_positions = np.argsort(predictions, axis=0)[len(predictions) // 2]  # "argmedian"
        elif network_aggregation_method == "max":
            aggregation_positions = np.argmax(predictions, axis=0)
        else:
            raise ValueError("Unknown aggregation method '{}'".format(network_aggregation_method))

        filtered_evaluation_dict = {}
        for key in evaluation_dicts[0].keys():
            entries_by_key = np.array([np.atleast_1d(np.squeeze(detach(network_dict.get(key))))
                                       for network_dict in evaluation_dicts])

            if network_aggregation_method == "mean":
                entries_by_key = np.mean(entries_by_key, axis=0)
            else:
                entries_by_key = entries_by_key[aggregation_positions, np.arange(entries_by_key.shape[1])]
            filtered_evaluation_dict[key] = entries_by_key
        return filtered_evaluation_dict
    else:
        # take the minimum of all evaluations. The rest is just wrapping to make sure we return proper numpy arrays
        unaggregated_evaluations = [np.atleast_1d(np.squeeze(detach(network(observations))))
                                    for network in network_list]
        if network_aggregation_method == "min":
            evaluations = np.min(unaggregated_evaluations, axis=0)
        elif network_aggregation_method == "mean":
            evaluations = np.mean(unaggregated_evaluations, axis=0)
        elif network_aggregation_method == "median":
            evaluations = np.median(unaggregated_evaluations, axis=0)
        elif network_aggregation_method == "max":
            evaluations = np.max(unaggregated_evaluations, axis=0)
        else:
            raise ValueError("Unknown aggregation method '{}'".format(network_aggregation_method))
    return evaluations


class AbstractReward:
    def __init__(self,
                 input_shape: Union[tuple, int],
                 dre_contexts: np.ndarray,
                 reward_config: Dict[Key, Any],
                 network_config: Dict[Key, Any],
                 network_type: str,
                 action_to_observation_function: MappingFunction = None,
                 ):
        self._num_regression_samples = reward_config.get("num_regression_samples")
        self._action_to_observation_function = action_to_observation_function
        self._dre_contexts = dre_contexts

        self._network_aggregation_method = reward_config.get("network_aggregation_method")

        self._num_regression_networks = np.maximum(reward_config.get("num_regression_networks"), 1)  # at least 1 network
        # overwrite default config w/ regression config
        regression_network_config = merge_dictionaries(network_config,
                                                       reward_config.get("regression_network"))
        self._initialize_regression_networks(input_shape=input_shape,
                                             regression_network_config=regression_network_config,
                                             network_type=network_type,
                                             regression_loss_function=reward_config.get("regression_loss_function"))

        self._sample_variance_multiplier = reward_config.get("sample_variance_multiplier")
        assert self._sample_variance_multiplier > 0, "must have a positive multiplier for the sample variance"

        self._sampling_policy_mixture_rate = reward_config.get("sampling_policy_mixture_rate")
        assert self._sampling_policy_mixture_rate == "equal" or 0 <= self._sampling_policy_mixture_rate <= 1, \
            "Need sampling_policy_mixture_rate==equal or in [0,1], got '{}'".format(self._sampling_policy_mixture_rate)
        self._fresh_policy_sample_rate = reward_config.get("fresh_policy_sample_rate")
        assert 0 <= self._fresh_policy_sample_rate <= 1, "Need fresh_policy_sample_rate in [0,1], got '{}'".format(
            self._fresh_policy_sample_rate)

        self._target_normalization = reward_config.get("target_normalization")
        self._sample_normalization = reward_config.get("sample_normalization")
        self._input_normalization_parameters = None

        self._sampling_policies = None
        self._num_added_policy_lists = 0

    def _initialize_regression_networks(self, input_shape: Union[tuple, int],
                                        regression_network_config: Dict[Key, Any],
                                        network_type: str,
                                        regression_loss_function: str):
        """
        Initializes the regression network(s) as well as relevant parameters/configuration details for it
        Args:
            input_shape: The shape of the inputs for the network
            regression_network_config: Full config defining the regression network
            network_type: The type of the network. Something like "LSTM" or "Feedforward"
            regression_loss_function: Name of the loss function to use. Can be "mse" for a regular regression, or
            "drex" for a comparison-based loss

        Returns: Nothing, but initializes some values

        """
        # initialize some training parameters
        self._batch_size = regression_network_config.get("batch_size")
        self._validation_split = regression_network_config.get("validation_split")

        self._trainers = []
        for _ in range(self._num_regression_networks):
            network_builder = NetworkBuilder(input_shape=input_shape,
                                             network_config=regression_network_config,
                                             network_type=network_type,
                                             loss=regression_loss_function)
            network, primary_loss_function = network_builder()
            self._trainers.append(NetworkTrainer(network=network, network_config=regression_network_config,
                                                 primary_loss_function=primary_loss_function))

    def regress_reward(self, current_policies: List[GMM]) -> List[Dict[Key, Any]]:
        """
        Use the current phi and the provided policies to create a NN that regresses from (observed) samples of the
        policy to its log density (+ potentially the evaluation of the current phi, which is the fitted DRE).
        This effectively moves the evaluation of the policy (its log density) from policy space to feature space
        Args:
            current_policies: The training policies for the iteration at which to regress. The "freshest" policies

        Returns: A dictionary containing the training history of the regression network

        """
        list_of_histories = []
        list_of_data = self._get_data(current_policies=current_policies)
        for position, (data, trainer) in enumerate(zip(list_of_data, self._trainers)):
            train_data_loader, validation_data_loader = dataloader_from_dictionary_data(data=data,
                                                                                        batch_size=self._batch_size,
                                                                                        validation_split=self._validation_split)
            current_history = trainer.fit(train_data_loader=train_data_loader,
                                          validation_data_loader=validation_data_loader)
            list_of_histories.append(current_history)
        return list_of_histories

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
        num_samples_per_policy = int(self._num_regression_samples / self.num_dre_contexts)
        samples = self._sample(num_samples_per_policy=num_samples_per_policy, current_policies=current_policies,
                               sample_variance_multiplier=self._sample_variance_multiplier)

        # contextualize and flatten samples and get observations
        contextualized_samples = append_contexts(samples=samples, contexts=self._dre_contexts, flatten=False)
        observations = np.array([self._action_to_observation_function(policy_wise_contextualized_samples)
                                 for policy_wise_contextualized_samples in contextualized_samples])

        targets = self._get_targets(current_policies=current_policies, observations=observations, samples=samples)
        flattened_targets = np.concatenate(targets, axis=0)
        flattened_observations = np.concatenate(observations, axis=0)

        flattened_targets = self._normalize_targets(flattened_targets)
        if self._sample_normalization == "normal":
            flattened_observations, self._input_normalization_parameters = normalize_samples(
                samples=flattened_observations,
                normalization_type="normal",
                return_dict=True)
        data_list = self._allocate_samples(flattened_observations, flattened_targets)  # allocate to networks
        return data_list

    def _get_targets(self, current_policies: List[GMM], observations: np.ndarray, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method needs to be implemented by the subclass")

    def _normalize_targets(self, targets: np.ndarray) -> np.ndarray:
        if self._target_normalization in ["zm", "mean", "zero_mean"]:
            targets = targets - np.mean(targets)
        elif self._target_normalization == "normal":
            targets = normalize_samples(samples=targets, normalization_type="normal", return_dict=False)
        return targets

    def _allocate_samples(self, observations: np.ndarray, targets: np.ndarray) -> List[Dict[Key, np.ndarray]]:
        """
        Allocates the given samples (observations, targets) to the networks depending on the
        self._sample_to_network_allocation parameter
        Args:
            observations: The data/observations for the regression
            targets: The values to regress to

        Returns: A tuple (observation_list, target_list), where each entry is a list of length num_regression_networks
          containing the observations/targets for each network in each entry

        """
        assert len(observations) == len(targets), "Must get equal amount of data and labels"
        num_networks = self.num_regression_networks
        observation_target_tuples = [joint_shuffle(observations, targets) for _ in range(num_networks)]
        observation_list, target_list = zip(*observation_target_tuples)

        data_list = []
        for current_observations, current_targets in zip(observation_list, target_list):
            # shuffle each set of observations for fairness in case there are multiple policies/contexts used
            current_observations, current_targets = joint_shuffle(current_observations, current_targets)
            data_list.append({d.SAMPLES: current_observations, d.TARGETS: current_targets})
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
            sample_variance_multiplier: Multiplier for the sample variance of the "fresh" samples

        Returns:

        """
        assert self._sampling_policies is not None, "Need to provide sampling policies first"
        num_fresh_samples_per_policy = int(num_samples_per_policy * self._fresh_policy_sample_rate)
        if num_fresh_samples_per_policy > 0:  # also use fresh samples, i.e., sample from the current policy
            assert current_policies is not None, "Must provide current policies to sample fresh samples"
            if sample_variance_multiplier == 1:
                current_sampling_policies = copy.deepcopy(current_policies)
            else:
                current_sampling_policies = [GMM(weights=current_gmm.weights,
                                                 means=current_gmm.means,
                                                 covars=sample_variance_multiplier * current_gmm.covars)
                                             for current_gmm in current_policies]
            fresh_samples = np.array([current_policy.sample(num_fresh_samples_per_policy)
                                      for current_policy in current_sampling_policies])
            num_old_samples_per_policy = num_samples_per_policy - num_fresh_samples_per_policy
            if num_old_samples_per_policy > 0:  # for the old policies, the variance multiplier is already accounted for
                old_samples = np.array([sampling_policy.sample(num_old_samples_per_policy)
                                        for sampling_policy in self._sampling_policies])
                samples = np.concatenate((fresh_samples, old_samples), axis=1)
                # shuffle each context to mix the fresh samples into the old ones
                [np.random.shuffle(policy_wise_samples)
                 for policy_wise_samples in samples]
            else:  # do not draw old samples
                samples = fresh_samples
        else:  # only use old samples
            samples = np.array([sampling_policy.sample(num_samples_per_policy)
                                for sampling_policy in self._sampling_policies])
        return samples

    def _add_sampling_policies(self, new_policies: List[GMM]):
        """
        The VIGOR regression keeps an internal list of policies to sample data from. Each policy of this list corresponds
         to a single context and is a mixture over all the sampling policies previously added via this method.
         The weightings of this mixture determine how much often older policies are being sampled from, and can be
         set via the sampling_policy_mixture_rate parameter.
        Assumes that the policies are provided in the same order as the original contexts
        Args:
            new_policies: A list of current train policies

        Returns:

        """
        self._num_added_policy_lists += 1
        if not self._sample_variance_multiplier == 1:
            new_sampling_policies = [GMM(weights=current_gmm.weights,
                                         means=current_gmm.means,
                                         covars=self._sample_variance_multiplier * current_gmm.covars)
                                     for current_gmm in new_policies]
        else:
            new_sampling_policies = new_policies
        if self._sampling_policies is None:  # no sampling policies provided yet
            self._sampling_policies = copy.deepcopy(new_sampling_policies)
        else:  # merge with existing policies
            merging_weight = self._sampling_policy_mixture_rate
            if merging_weight == 1:
                self._sampling_policies = copy.deepcopy(new_sampling_policies)  # simply overwrite
            elif merging_weight == 0:
                pass  # do not merge for weight==0
            else:
                if merging_weight == "equal":
                    merging_weight = 1 / self._num_added_policy_lists
                for sampling_policy, new_sampling_policy in zip(self._sampling_policies, new_sampling_policies):
                    sampling_policy: GMM
                    sampling_policy.merge(copy.deepcopy(new_sampling_policy), new_weight=merging_weight)

    def get_reward(self) -> Dict[Key, Any]:
        """
        The reward for vigor is the minimum evaluation of all the regression networks.

        Returns: a dict consisting of
            callable: The reward as a callable function of the policy samples
            parameters: A list of all the networks

        """
        return {"callable": partial(reward_function, network_list=self.regression_networks,
                                    action_to_observation_function=self._action_to_observation_function,
                                    network_aggregation_method=self._network_aggregation_method,
                                    normalization_parameters=self._input_normalization_parameters),
                "parameters": {"networks": self.regression_networks,
                               NORMALIZATION_PARAMETER_FILE: self._input_normalization_parameters}
                }

    @property
    def num_dre_contexts(self) -> int:
        return len(self._dre_contexts)

    @property
    def num_regression_networks(self) -> int:
        return self._num_regression_networks

    @property
    def regression_networks(self) -> List[Network]:
        regression_networks: List[Network] = [trainer.network for trainer in self._trainers]
        return regression_networks
