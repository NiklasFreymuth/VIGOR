import os
from typing import Dict, Any, Union, Optional

import numpy as np

from util.Defaults import ENVIRONMENTS_FOLDER
from util.Functions import joint_shuffle
from util.Types import ConfigDict


def get_samples(data_path: str, shuffle: bool = True) -> np.array:
    """
    load samples from the given npy file. Wrapper for np.load.
    Assumes that the samples lie in "tasks"/task/filename, e.g.
    "tasks/planar_robot/data/point_reaching/2.npy"
    :param data_path: path to the specified data
    :param shuffle: Whether to shuffle the data or not

    :return: The samples as a numpy array
    """
    if not data_path.endswith((".npy", ".npz")):
        if os.path.isfile(data_path + ".npy"):
            data_path += ".npy"
        elif os.path.isfile(data_path + ".npz"):
            data_path += ".npz"
        else:
            raise FileNotFoundError("data file at path '{}' does not exist".format(data_path))
    samples = np.load(data_path)

    if not type(samples) == np.ndarray:  # "decompress" .npz file
        samples = samples["arr_0"]

    if samples.ndim == 1:  # fit format for 1d data
        samples = samples.reshape(-1, 1)
    if shuffle:
        np.random.shuffle(samples)
    return samples


def get_data_dict(config: ConfigDict) -> Dict[str, Any]:
    """
    Retrieves a data dictionary for pseudo-contextual environments. pseudo-contextual tasks always use an observation of the
    data, meaning that the observation_config may not be None
    Args:
        config: Dictionary. Contains
            task_config:{
                data_source:
                task:
            }
            meta_task_config:
            modality:

    Returns:

    """
    task_config = config.get("task")

    task = task_config.get("task")
    data_source = task_config.get("data_source")

    modality = config.get("modality")

    meta_task_config = config.get("meta_task")

    sample_size: Union[int, float] = meta_task_config.get("sample_size")
    shuffle_demonstrations: Union[bool, str] = meta_task_config.get("shuffle_demonstrations")
    num_train_contexts: int = meta_task_config.get("num_train_contexts")
    num_validation_contexts: int = meta_task_config.get("num_validation_contexts")
    num_test_contexts: int = meta_task_config.get("num_test_contexts")
    return_expert_promp_weights: bool = meta_task_config.get("return_expert_promp_weights", False)

    base_path = os.path.join(ENVIRONMENTS_FOLDER, task, data_source)

    assert modality is not None, "Need observation_config for pseudo-contextual setting, got None"
    contexts = get_samples(data_path=os.path.join(base_path, "contexts.npy"), shuffle=False)
    observations = get_samples(data_path=os.path.join(base_path, modality), shuffle=False)
    if return_expert_promp_weights:
        expert_promp_weights = get_samples(data_path=os.path.join(base_path, "promp_weights"), shuffle=False)
    else:
        expert_promp_weights = None

    assert num_train_contexts <= len(observations), f"Can not use more train contexts than available. " \
                                                    f"Requested '{num_train_contexts}' of '{len(observations)}"

    if task == "planar_reacher":
        # constraint number of observations by task to make comparison "fairer" between different numbers of contexts
        observations = observations[:12]

    if sample_size <= 1:  # sample size as a percentage
        sample_size = int(sample_size * len(observations[1]))

    if isinstance(shuffle_demonstrations, str) and shuffle_demonstrations == "stratified":
        # use stratified sampling for box pusher to make sure all big context modes are represented
        assert task == "box_pusher", "Can only do stratified context shuffling for box pusher task"
        # shuffle within each context (in-place)
        if return_expert_promp_weights:
            for position, (observation, promp_weight) in enumerate(zip(observations, expert_promp_weights)):
                observation, promp_weight = joint_shuffle(observation, promp_weight)
                observations[position] = observation
                expert_promp_weights[position] = promp_weight
        else:
            [np.random.shuffle(observation) for observation in observations]

        divisor = num_train_contexts // 4
        rest = num_train_contexts % 4

        train_context_ids = []
        for position, conditions in enumerate((np.logical_and(contexts[:, 2] > 0, contexts[:, 1] > 0),
                                               np.logical_and(contexts[:, 2] > 0, contexts[:, 1] < 0),
                                               np.logical_and(contexts[:, 2] < 0, contexts[:, 1] > 0),
                                               np.logical_and(contexts[:, 2] < 0, contexts[:, 1] < 0))):
            all_ids, = np.where(conditions)
            num_choices = divisor
            if rest > position:
                num_choices = num_choices + 1
            chosen_ids = np.random.choice(all_ids, num_choices, replace=False)
            train_context_ids.extend(chosen_ids)
        train_context_ids = np.array(train_context_ids)

    elif shuffle_demonstrations:
        # shuffle within each context (in-place)
        if return_expert_promp_weights:
            for position, (observation, promp_weight) in enumerate(zip(observations, expert_promp_weights)):
                observation, promp_weight = joint_shuffle(observation, promp_weight)
                observations[position] = observation
                expert_promp_weights[position] = promp_weight
        else:
            [np.random.shuffle(observation) for observation in observations]
        # choose num_train_context out of len(observations) contexts
        train_context_ids = np.random.choice(len(observations), num_train_contexts, replace=False)
    else:
        train_context_ids = np.arange(len(contexts))[:num_train_contexts]
        # note that the contexts and observations are pre-shuffled in a joint fashion, also making this unbiased
        # still, not shuffling allows the methods to implicitly overfit on some contexts

    train_contexts = contexts[train_context_ids]  # choose contexts
    train_contexts = train_contexts.reshape(train_contexts.shape[0], -1)  # flatten context
    train_observations = observations[train_context_ids, :sample_size]
    if return_expert_promp_weights:
        expert_promp_weights = expert_promp_weights[train_context_ids, :sample_size]

    validation_context_ids: Optional[list] = meta_task_config.get("validation_context_ids", None)
    if num_validation_contexts >= 1:
        if shuffle_demonstrations:
            remaining_context_ids = list(set(np.arange(len(contexts))) - set(train_context_ids))
            validation_context_ids = np.random.choice(remaining_context_ids, num_validation_contexts, replace=False)
        elif validation_context_ids is not None:  # specific validation context ids provided
            assert isinstance(validation_context_ids, list), \
                "Must provide a list of validation context ids, got '{}' instead".format(validation_context_ids)
            assert len(validation_context_ids) == num_validation_contexts, \
                "Length of validation context ids must match number of context. " \
                "Given {}, expected {}".format(validation_context_ids, num_validation_contexts)
        else:  # "default" case. Pick the contexts after the train contexts
            validation_context_ids = np.arange(len(contexts))[
                                     num_train_contexts:num_train_contexts + num_validation_contexts]

        validation_contexts = contexts[validation_context_ids]
        validation_contexts = validation_contexts.reshape(len(validation_contexts), -1)
    else:
        validation_contexts = []

    test_context_ids: Optional[list] = meta_task_config.get("test_context_ids", None)
    if num_test_contexts >= 1:  # split off given number of contexts
        if shuffle_demonstrations:
            remaining_context_ids = set(np.arange(len(contexts))) - set(train_context_ids)
            if validation_context_ids is not None:
                remaining_context_ids = remaining_context_ids - set(validation_context_ids)
            remaining_context_ids = list(remaining_context_ids)
            test_context_ids = np.random.choice(remaining_context_ids, num_test_contexts, replace=False)
        elif test_context_ids is not None:  # specific test context ids provided
            assert isinstance(test_context_ids, list), \
                "Must provide a list of test context ids, got '{}' instead".format(test_context_ids)
            assert len(test_context_ids) == num_test_contexts, \
                "Length of test context ids must match number of context. " \
                "Given {}, expected {}".format(test_context_ids, num_test_contexts)
        else:  # "default" case. Pick the last contexts
            test_context_ids = np.arange(len(contexts))[-num_test_contexts:]

        test_contexts = contexts[test_context_ids]
        test_contexts = test_contexts.reshape(len(test_contexts), -1)
    else:
        test_contexts = []

    data_dict = {"train_observations": train_observations,

                 "train_contexts": train_contexts,
                 "train_context_ids": train_context_ids,

                 "validation_contexts": validation_contexts,
                 "validation_context_ids": validation_context_ids,

                 "test_contexts": test_contexts,
                 "test_context_ids": test_context_ids,

                 "drex_train_contexts": train_contexts,
                 "drex_train_context_ids": train_context_ids}
    if return_expert_promp_weights:
        data_dict["expert_promp_weights"] = expert_promp_weights
    return data_dict