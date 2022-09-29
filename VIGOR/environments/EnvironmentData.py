import os
import numpy as np

from environments.EnvironmentDataUtil import get_data_dict
from util.Types import *
from util.Defaults import ENVIRONMENTS_FOLDER
from util.Functions import get_from_nested_dict

from algorithms.distributions.GMM import GMM
"""
Wrapper class containing all important data for the task, including expert demonstrations etc.
"""


class EnvironmentData:
    """
    Version of EnvironmentData compatible with the pseudo-contextual setting
    """
    def __init__(self, config: dict):
        self._config = config
        self._task_config = config.get("task")
        self._task_name = self._task_config.get("task")
        self._algorithm_name = config.get("algorithm")
        self._data_dict = None
        self._target_function = None
        self._environment_specifics = self._get_environment_specifics()

    def _get_environment_specifics(self):
        from environments.get_environment_specifics import get_environment_specifics
        environment_specifics = get_environment_specifics(task_name=self._task_name,
                                                          config=self._config,
                                                          data_dict=self.data_dict)
        return environment_specifics


    def _get_data_dict(self) -> Dict[Key, Any]:
        return get_data_dict(config=self._config)

    @property
    def policy_dimension(self):
        assert "policy_dimension" in self._environment_specifics, "Need to specify policy dimension in task_specfics"
        return self._environment_specifics.get("policy_dimension")

    @property
    def environment_specifics(self) -> Dict[Key, Any]:
        return self._environment_specifics

    @property
    def data_dict(self):
        """
        Loads the demonstrations and possibly observations and normalization parameters of the given task.
        The values are pre-loaded once and then stored in an internal variable
        Returns: A dicionary containing training data, possibly observations and normalization parameters

        """
        if self._data_dict is None:  # cache this, i.e., only create it once
            self._data_dict = self._get_data_dict()
        return self._data_dict

    @property
    def algorithm_name(self) -> str:
        return self._algorithm_name

    @property
    def environment_reward_function(self):
        return self._environment_specifics.get("reward")

    @property
    def expert_observations(self):
        return self.data_dict.get("train_observations")

    @property
    def config(self):
        return self._config


    @property
    def train_contexts(self) -> np.ndarray:
        return self.data_dict.get("train_contexts")

    @property
    def validation_contexts(self) -> np.ndarray:
        return self.data_dict.get("validation_contexts")

    @property
    def test_contexts(self) -> np.ndarray:
        return self.data_dict.get("test_contexts")

    @property
    def drex_train_contexts(self) -> np.ndarray:
        return self.data_dict.get("drex_train_contexts")

    @property
    def ground_truth_uld(self) -> Callable:
        """
        Since the ground truth uld depends on the context for the pseudo-contextual setting, we implement it
        via a look-up table over target functions
        Returns: A callable function that takes samples and a context and outputs the uld of the samples for the
        specified context

        """

        def _uld(x: np.array, context: Union[np.array, int], **kwargs) -> np.array:
            if isinstance(context, int):
                context_id = context
            else:
                context_id = self._environment_specifics.get("reverse_context_ids").get(tuple(context.flatten()))
            return partial(self._environment_specifics.get("reward"), samples=x, context_id=context_id, **kwargs)
        return _uld

    @property
    def action_to_observation_function(self) -> Callable:
        """
        Takes a policy sample and turns it into an observation (e.g. an image).
        For pseudo-contextual data, we also need to add a context into the function since the observation may depend
         on it
        For time series data, the action_to_observation_function is a little more complicated. It is always provided
        (since we must map from the policy to the time series somehow), and basically performs a rollout over time
         of the parameters given some controller, which are then further transformed depending on the context

        """
        assert "action_to_observation_function" in self._environment_specifics, \
            "Need action_to_observation_function for pseudo-contextual setting"
        return self._environment_specifics.get("action_to_observation_function")

    def get_expert_policies(self) -> List[GMM]:
        environment_config = self.config.get("task")
        train_context_ids = self.data_dict.get("train_context_ids")
        base_path = os.path.join(ENVIRONMENTS_FOLDER,
                                 environment_config.get("task"),
                                 environment_config.get("data_source"))

        use_em_gmms = self.config.get("algorithm") == "DRex" \
                      and get_from_nested_dict(self.config, list_of_keys=["vigor", "reward", "drex", "use_em_gmm"],
                                               default_return=False)
        if use_em_gmms:
            drex_config = get_from_nested_dict(self.config,
                                               list_of_keys=["vigor", "drex"],
                                               raise_error=True)
            em_gmm_config = drex_config.get("em_gmm")

            if "fit_name" in em_gmm_config and em_gmm_config.get(
                    "fit_name") is not None:  # fit name is given explicitly
                fit_name = em_gmm_config.get("fit_name")
            else:
                assert "num_source_samples" in em_gmm_config, "Need to provide number of source samples if providing" \
                                                              "number of components"
                assert "num_fit_components" in em_gmm_config, "Must provide num_fit_components in drex config"
                num_fit_components = em_gmm_config.get("num_fit_components")
                num_source_samples = em_gmm_config.get("num_source_samples")
                fit_name = f"promp_{num_fit_components}_{num_source_samples}"

            policy_save_path = os.path.join(base_path, "EM", str(fit_name))
            policy_file_name = "gmms.pkl"
        else:
            policy_save_path = base_path
            policy_file_name = "gmms.pkl"

        if os.path.isfile(os.path.join(policy_save_path, policy_file_name)):  # newer version
            import pickle
            with open(os.path.join(policy_save_path, policy_file_name), "rb") as gmm_file:
                policies = pickle.load(gmm_file)
        elif os.path.isfile(os.path.join(policy_save_path, "weights.npy")):
            # policy is saved as individual GMM weights, means and covariances
            weights = np.load(os.path.join(policy_save_path, "weights.npy"))
            means = np.load(os.path.join(policy_save_path, "means.npy"))
            covars = np.load(os.path.join(policy_save_path, "covars.npy"))
            # covariances are saved as a sparse array of shape (#contexts, #components, #dimension).
            # we create a diagonal covariance per component,
            # i.e., parse it to shape (#contexts, #components, #dimension, #dimension)
            covars = np.array([[np.diag(covar) for covar in inner_covars]
                               for inner_covars in covars])
            policies = [GMM(weights=current_weights, means=current_means, covars=current_covars) for
                        current_weights, current_means, current_covars in zip(weights, means, covars)]

        else:
            raise FileNotFoundError("File '{}' for ground truth policies not found".format(policy_save_path))

        # select used train contexts
        policies = [policies[idx] for idx in train_context_ids]

        return policies
