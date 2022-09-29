import numpy as np
from util.Types import *
from algorithms.dre.VIGORTrainer import VIGORTrainer
from util.pytorch.architectures.Network import Network
from algorithms.distributions.GMM import GMM
from util.Functions import get_from_nested_dict


class VIGOR_DRE:
    def __init__(self, config: Dict[str, Any],
                 expert_observations: np.array,
                 eim_contexts: np.array,
                 additional_information: Dict[str, Any],
                 eim_policies: List[GMM],
                 action_to_observation_function: callable,
                 ):
        self._eim_contexts = eim_contexts
        self._eim_policies = eim_policies
        self._action_to_observation_function = action_to_observation_function
        self.expert_observations = expert_observations.astype(np.float32)

        # numbers of provided expert demonstrations and policy samples to draw for the DRE training
        self._num_expert_samples = len(expert_observations)
        self.uniform_policy_dre_samples = get_from_nested_dict(config, ["network", "uniform_policy_dre_samples"],
                                                               raise_error=True)

        self.learner_sampling_ratio = 1  # same amount of learner and expert samples

        self._trainer = VIGORTrainer(config=config,
                                     network_type=additional_information.get("network_type"),
                                     expert_observations=self.expert_observations)

    def train(self):
        """
        Train the density ratio estimator by drawing data, giving this data to the Trainer()-class and then training
        it.
        May be overwritten by other approaches.
        Returns: The training history of the network

        """
        self._set_training_observations()
        dre_history = self._trainer.prepare_data_and_fit()
        return dre_history

    def _set_training_observations(self):
        """
        Overwrites the set_training_samples of EIM because the action_to_observation_function now also takes the
        contexts into account. If no action_to_observation_function is provided, we concatenate the data with the
        contexts it is sampled from
        """
        all_actions = []
        num_samples_per_context = self.learner_sampling_ratio * self._num_expert_samples / len(self._eim_contexts)
        num_samples_per_context = int(np.ceil(num_samples_per_context))
        # distribute samples evenly over contexts

        for context, sampling_policy in zip(self._eim_contexts,
                                            self._eim_policies):  # draw samples for each context
            if self.uniform_policy_dre_samples:
                sampling_policy = sampling_policy.get_equiweighted_gmm()
            learner_actions = sampling_policy.sample(num_samples=num_samples_per_context)
            # learner_actions = append_contexts(samples=learner_actions, contexts=context)
            all_actions.append(learner_actions)
        all_actions = np.array(all_actions)
        contexts = np.repeat(self._eim_contexts[:, None, :], repeats=num_samples_per_context, axis=1)
        all_observations = self._action_to_observation_function(params=all_actions, contexts=contexts)
        all_observations = np.concatenate(all_observations, axis=0)
        self._trainer.learner_observations = all_observations

    @property
    def network(self) -> Network:
        """
        Simple wrapper to refer to the network trained by self._trainer
        Returns: The network trained by the NetworkTrainer

        """
        return self._trainer.network
