from environments.EnvironmentData import EnvironmentData
from algorithms.VIGOR import VIGOR
from util.Types import *


def get_algorithm(config: Dict[Key, Any], environment_data: EnvironmentData) -> VIGOR:
    algorithm_name = environment_data.algorithm_name.lower()
    if algorithm_name == "vigor":
        from algorithms.VIGOR import VIGOR
        algorithm = VIGOR(config=config,
                          expert_observations=environment_data.expert_observations,
                          train_contexts=environment_data.train_contexts,
                          validation_contexts=environment_data.validation_contexts,
                          test_contexts=environment_data.test_contexts,
                          policy_dimension=environment_data.policy_dimension,
                          action_to_observation_function=environment_data.action_to_observation_function)
    elif algorithm_name == "drex":
        from algorithms.DRex import DRex
        train_policies = environment_data.get_expert_policies()
        algorithm = DRex(config=config, expert_observations=environment_data.expert_observations,
                         train_contexts=environment_data.train_contexts,
                         test_contexts=environment_data.test_contexts,
                         policy_dimension=environment_data.policy_dimension,
                         action_to_observation_function=environment_data.action_to_observation_function,
                         train_policies=train_policies
                         )
    else:
        raise NotImplementedError("Unknown algorithm '{}'".format(environment_data.algorithm_name))
    return algorithm
