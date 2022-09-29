import numpy as np
import os
from util.Types import *
from environments.planar_reacher.PlanarReacher import PlanarReacher
from util.Functions import save_concatenate

from util.Defaults import ENVIRONMENTS_FOLDER


def get_planar_reacher_specifics(config: ConfigDict, data_dict: ConfigDict) -> Dict[Key, Any]:
    """
    Gathers information for the planar reacher task
    Args:

    Returns:

    """
    task_config = config.get("task")
    task = task_config.get("task")
    data_source = task_config.get("data_source")
    modality = config.get("modality")
    data_folder = os.path.join(ENVIRONMENTS_FOLDER, task)

    contexts = np.load(os.path.join(data_folder, data_source, "contexts.npy"))

    parameter_dict = {
        "num_targets": 2,
        "all_target_points": contexts,
        "acceleration_penalty_lambda": 1,
        "velocity_penalty_lambda": 1,
        "radius": 0.5,
        "num_links": 5,
        "total_steps": 30
    }

    all_context_ids = save_concatenate(data_dict.get("train_context_ids"),
                                       data_dict.get("validation_context_ids"),
                                       data_dict.get("test_context_ids"))

    context_dict = {context_id: contexts[context_id] for context_id in all_context_ids}

    multi_point_reacher = PlanarReacher(config=config,
                                        contexts=context_dict,
                                        parameter_dict=parameter_dict,
                                        )

    reward = multi_point_reacher.reward
    action_to_observation_function = partial(multi_point_reacher.vectorized_mp_features,
                                             feature_representation=modality,
                                             append_contexts=True)

    reverse_context_ids = {tuple(value): key for key, value in context_dict.items()}
    environment_specifics = {"action_to_observation_function": action_to_observation_function,
                             "reward": reward,
                             "num_links": parameter_dict.get("num_links"),
                             "policy_dimension": multi_point_reacher.num_parameters,
                             "radius": parameter_dict.get("radius"),
                             "environment": multi_point_reacher,
                             "context_ids": context_dict,
                             "reverse_context_ids": reverse_context_ids,
                             }
    return environment_specifics


def get_multi_point_reacher_parameters(task_description_dict: Dict[str, Any], data_source) -> Dict[str, Any]:
    data_folder = task_description_dict.get("data_folder")

    contexts = np.load(os.path.join(data_folder, data_source, "contexts.npy"))

    zero_start = True
    weight_scale_multiplier = 2
    radius = 0.5

    return {
        "num_targets": 2,
        "all_target_points": contexts,
        "target_distance_threshold": max(1, radius),
        "weight_scale_multiplier": weight_scale_multiplier,
        "radius": radius,
        "zero_start": zero_start
    }
