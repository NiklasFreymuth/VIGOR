import os

import torch
import numpy as np

from algorithms.distributions.GMM import GMM
from util.Types import *
from algorithms.distributions.AbstractGMM import AbstractGMM
from util.pytorch.architectures.Network import Network
import util.save.SaveDefaults as defaults
from util.save.SaveDefaults import format_iteration


def create_path(path: str):
    if not os.path.exists(path):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)


def _save_dict(path: str, file_name: str, dict_to_save: dict, overwrite: bool = False):
    """
    Saves the given dict
    Args:
        path: Full/Absolute path to save to
        file_name: Name of the file to save the dictionary to
        dict_to_save: The dictionary to save
        overwrite: Whether to overwrite an existing file

    Returns:

    """
    if not file_name.endswith(".npz"):
        file_name += ".npz"
    create_path(path)
    file_to_save = os.path.join(path, file_name)
    if overwrite or not os.path.isfile(file_to_save):
        np.savez_compressed(file_to_save, **dict_to_save)


def _save_gmm_raw(path, gmm: AbstractGMM, file_name):
    log_weights = gmm.log_weights
    means = np.stack([c.mean for c in gmm.components], axis=0)
    covariances = np.stack([c.covar for c in gmm.components], axis=0)
    model_dict = {"log_weights": log_weights, "means": means, "covars": covariances}
    _save_dict(path=path, file_name=file_name, dict_to_save=model_dict)


class SaveUtility:
    def __init__(self, save_directory: str):
        self._save_directory = os.path.join(save_directory, defaults.SAVE_DIRECTORY)
        self._has_saved_before = False

    def save_contexts(self, data_dict: ConfigDict):
        context_ids = {k: v
                       for k, v in data_dict.items()
                       if "context_id" in k
                       and v is not None and (not (isinstance(v, list)) or len(v) == 0)}
        _save_dict(path=self._save_directory,
                   file_name="context_ids",
                   dict_to_save=context_ids)

    def save(self, iteration: int,
             policy: AbstractGMM = None,
             train_policies: Optional[Dict[int, AbstractGMM]] = None,
             validation_policies: Optional[Dict[int, AbstractGMM]] = None,
             test_policies: Optional[Dict[int, AbstractGMM]] = None,
             drex_train_policies: Optional[Dict[int, AbstractGMM]] = None,
             discriminators: Optional[List[Network]] = None,
             reward_parameters: dict = None):
        """
        Saves the given model and reward parameters in a directory path/name.
        Can be either given a single policy or a bunch of train/test policies
        :param iteration: the current iteration of the algorithm. Will be used to name the saved files
        :param policy: The current policy for non-contextual tasks
        :param train_policies: A list of training policies used in the pseudo-contextual setting
        :param validation_policies: A list of validation policies used in the pseudo-contextual setting
        :param test_policies: A list of test policies trained alongside the training
            policies in the pseudo-contextual setting
        :param discriminators: The ensemble of discriminator models used for EIM.
        :param reward_parameters: Other reward parameters used
        :return:
        """
        file_name = format_iteration(iteration)

        if policy is not None:  # only one policy
            self._save_eim_policy(file_name=file_name, policy=policy)
        if train_policies is not None:  # pseudo_contextual setting
            for position, _policy in train_policies.items():
                self._save_eim_policy(policy=_policy, file_name=file_name + "_train{:03d}".format(position))
        if validation_policies is not None:  # pseudo_contextual setting
            for position, _policy in validation_policies.items():
                self._save_eim_policy(policy=_policy, file_name=file_name + "_validation{:03d}".format(position))
        if test_policies is not None:
            for position, _policy in test_policies.items():
                self._save_eim_policy(policy=_policy, file_name=file_name + "_test{:03d}".format(position))
        if drex_train_policies is not None:
            for position, _policy in drex_train_policies.items():
                self._save_eim_policy(policy=_policy, file_name=file_name + "_drex_train{:03d}".format(position))

        if reward_parameters is not None:
            self._save_reward_parameters(iteration=iteration, reward_parameters=reward_parameters)
        if discriminators is not None:
            [self._save_network(network=discriminator,
                                file_name=defaults.EIM_DISCRIMINATOR + format_iteration(iteration) + f"_{position}")
             for position, discriminator in enumerate(discriminators)]
        self._has_saved_before = True

    def _save_reward_parameters(self, iteration: int, reward_parameters):
        """
        Saves the reward model, i.e. a pytorch model for the generative reward or
        a dict of pytorch models + some potential prior
        :param reward_parameters: Model(s) and parameters that provide the reward
        :param iteration: Iteration this is called at
        :return:
        """
        if isinstance(reward_parameters, dict):
            if "phis" in reward_parameters.keys():  # cumulative reward
                self._save_cumulative_reward_parameters(iteration=iteration, reward_parameters=reward_parameters)
            elif "networks" in reward_parameters.keys():  # regressive reward
                for position, regression_network in enumerate(reward_parameters.get("networks")):
                    file_name = defaults.REGRESSIVE_REWARD + "{}_".format(position) + format_iteration(iteration)
                    self._save_network(network=regression_network,
                                       file_name=file_name, config_prefix="regression_")
                if not self._has_saved_before:
                    if defaults.NORMALIZATION_PARAMETER_FILE in reward_parameters.keys() and \
                            reward_parameters.get(defaults.NORMALIZATION_PARAMETER_FILE) is not None:
                        reward_folder = os.path.join(self._save_directory, defaults.REWARD_FOLDER)
                        create_path(reward_folder)
                        if not self._has_saved_before:
                            _save_dict(path=reward_folder,
                                       file_name="regression_" + defaults.NORMALIZATION_PARAMETER_FILE,
                                       dict_to_save=reward_parameters.get(defaults.NORMALIZATION_PARAMETER_FILE))
        else:
            raise NotImplementedError(
                "Reward parameters {} of instance {} can not be saved".format(reward_parameters,
                                                                              type(reward_parameters)))

    def _save_cumulative_reward_parameters(self, iteration: int, reward_parameters: dict):
        """
        Saves the cumulative reward model, which is represented as a dict {q0, [phi_0, ..., phi_n]}.
        For simplicity, only the latest phi will be saved. This assumes that this method is called after every iteration
        :param reward_parameters: a dict {q0, [phi_0, ..., phi_n]}
        :param iteration: iteration to save to
        :return:
        """
        # only need to save last phi
        last_phi = reward_parameters.get("phis")[-1]
        self._save_network(network=last_phi, file_name=defaults.PHI_NAME + format_iteration(iteration))

        if not self._has_saved_before:
            # prior only needs to be saved once
            if "prior" in reward_parameters:
                prior = reward_parameters.get("prior")
                _save_gmm_raw(path=os.path.join(self._save_directory, defaults.REWARD_FOLDER), gmm=prior,
                              file_name=defaults.REWARD_PRIOR_NAME)

    def _save_eim_policy(self, policy, file_name: str):
        """
        Wrapper for saving the EIM policy.
        The policy will be a GMM for the "normal" case and might be different for e.g. the conditional case
        :param policy: The policy to save
        :param file_name: Name of the file to save to
        :return: Nothing, but the policy will be saved accordingly
        """
        if isinstance(policy, AbstractGMM):
            self._save_gmm(folder=defaults.POLICY_FOLDER, policy=policy, file_name=file_name)
        else:
            raise NotImplementedError("Saving not implemented for class {}".format(policy.__class__))

    def _save_gmm(self, folder: str, policy: AbstractGMM, file_name: str):
        """
        Saves the current AbstractGMM model's weights, means and covariances
        :param policy: The GMM. Its parameters are the weights, means and covariances of the gaussian components
        :param file_name: Name of the file to save to
        :return:
        """
        if not file_name.startswith(defaults.POLICY_NAME):
            file_name = defaults.POLICY_NAME + file_name
        if not file_name.endswith(".npz"):
            file_name += ".npz"

        path = os.path.join(self._save_directory, folder)
        _save_gmm_raw(path=path, gmm=policy, file_name=file_name)

    def _save_network(self, network: Network, file_name: str, config_prefix: str = ""):
        """
        Saves the given network's state_dict at the specified location. Note that this dict can only be loaded into an
        already instantiated model, which requires the model config/hyperparameters to be saved as well.
        This is only saved if save_config=True
        Args:
            network: The pytorch network to save
            file_name: Name of the file to save to
            config_prefix: A potential prefix for the network config
        Returns:

        """
        reward_folder = os.path.join(self._save_directory, defaults.REWARD_FOLDER)
        create_path(reward_folder)
        if not self._has_saved_before:
            _save_dict(path=reward_folder,
                       file_name=config_prefix + defaults.NETWORK_KWARGS_FILE, dict_to_save=network.kwargs)
        if not file_name.endswith(".pt"):
            file_name = file_name + ".pt"
        save_path = os.path.join(reward_folder, file_name)
        torch.save(network.state_dict(), save_path)


def save_diagonal_gmms(gmms: List[GMM], save_path):
    """
    Save a list of GMMs with diagonal covariance matrix by saving their weights, means and covariances separately as
    numpy arrays.
    Will save
    *  a list of weights of shape (#gmms, #components)
    *  a list of means of shape (#gmms, #components, #dimension)
    *  a list of diagonal covariances of shape (#gmms, #components, #dimension)
    Args:
        gmms: List of Gaussian Mixture Model instances
        save_path: Path to save to

    Returns:

    """
    print(f"Saving GMMs to path {save_path}")
    create_path(save_path)
    np.save(os.path.join(save_path, "weights"), arr=np.array([gmm.weights for gmm in gmms]))
    np.save(os.path.join(save_path, "means"), arr=np.array([gmm.means for gmm in gmms]))
    np.save(os.path.join(save_path, "covars"), arr=np.array([[np.diag(covar) for covar in gmm.covars] for gmm in gmms]))