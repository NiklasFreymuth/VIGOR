import numpy as np
from util.Types import *
from environments.panda_reacher.PandaReacher import PandaReacher
import os
from util.Defaults import ENVIRONMENTS_FOLDER
from functools import partial
from util.Functions import save_concatenate


def get_panda_reacher_specifics(config: ConfigDict, data_dict: ValueDict) -> Dict[Key, Any]:
    """
    Gathers necessary information for the non-contextual teleoperation reacher
    Args:

    Returns:

    """
    modality = config.get("modality")
    task_config = config.get("task")
    data_source = task_config.get("data_source")
    task = task_config.get("task")
    base_path = os.path.join(ENVIRONMENTS_FOLDER, task,  data_source)
    contexts = np.load(os.path.join(base_path, "contexts.npy"))

    all_context_ids = save_concatenate(data_dict.get("train_context_ids"),
                                       data_dict.get("validation_context_ids"),
                                       data_dict.get("test_context_ids"))

    context_dict = {context_id: contexts[context_id].flatten() for context_id in all_context_ids}

    parameter_dict = {"velocity_penalty": 1.0e-0,
                      "acceleration_penalty": 1.0e-0,
                      "radius": 0.05,
                      "n_basis": 8,
                      "num_steps": 50,
                      "input_space": "joint",
                      "start_position": np.load(os.path.join(base_path, "start_position.npy")),
                      "n_dof": 6,
                      }

    reacher = PandaReacher(config=config,
                           contexts=context_dict,
                           parameter_dict=parameter_dict)

    reverse_context_ids = {tuple(value.flatten()): key for key, value in context_dict.items()}
    action_to_observation_function = partial(reacher.vectorized_mp_features,
                                             feature_representation=modality)
    environment_specifics = {"action_to_observation_function": action_to_observation_function,
                             "reward": reacher.reward,
                             "policy_dimension": reacher.num_parameters,
                             "environment": reacher,
                             "context_ids": context_dict,
                             "reverse_context_ids": reverse_context_ids,
                             }
    return environment_specifics
