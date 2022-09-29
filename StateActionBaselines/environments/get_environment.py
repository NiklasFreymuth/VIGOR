from baseline_util.Types import *
from baseline_util.Keys import PANDA_REACHER, PLANAR_REACHER, BOX_PUSHER, ONLINE_BOX_PUSHER
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import copy
import os


def get_environment(config: ConfigDict, context_ids: np.array) -> VecFrameStack:
    """
    Build an environment as specified by the provided config.
    Args:
        config: Contains the name of the task/environment, as well as information about the kind of observation and
            which features to use
        context_ids: Context ids to test on

    Returns: The specified environment, wrapped in a DummyVecEnv and with FrameStacks taken from Stable Baselines

    """
    task: str = config.get("task")
    if task == ONLINE_BOX_PUSHER:
        contexts = np.load(f"data/{BOX_PUSHER}/contexts.npy")[context_ids]
    else:
        contexts = np.load(f"data/{task}/contexts.npy")[context_ids]
    include_timestep: bool = config.get("include_timestep")
    include_target_encoding: bool = config.get("include_target_encoding")

    if task == PLANAR_REACHER:
        observation_type: str = config.get("observation_type")
        environment_parameters = {"num_links": 5,
                                  "radius": 0.5,
                                  "total_steps": 30,
                                  "reward_scaling_alpha": 1e-1,
                                  "velocity_std": 3e-2,
                                  "acceleration_std": 1e-3,
                                  "target_std": 1e-4,
                                  "observation_configuration":
                                      {
                                          "observation_type": observation_type,
                                          "include_timestep": include_timestep,
                                          "include_target_encoding": include_target_encoding
                                      },
                                  "contexts": contexts
                                  }
        from environments.PlanarReacher import PlanarReacher
        environment = PlanarReacher(environment_parameters=environment_parameters)
    elif task == PANDA_REACHER:
        start_position = np.load(os.path.join("data", "panda_reacher", "start_position.npy"))
        environment_parameters = {"velocity_penalty": 1.0e-0,
                                  "acceleration_penalty": 1.0e-0,
                                  "radius": 0.05,
                                  "num_steps": 50,
                                  "n_dof": 6,
                                  "start_position": start_position,
                                  "observation_configuration":
                                      {
                                          "observation_type": "geometric",
                                          "include_timestep": include_timestep,
                                          "include_target_encoding": include_target_encoding
                                      },
                                  "contexts": contexts
                                  }

        from environments.PandaReacher import PandaReacher
        environment = PandaReacher(environment_parameters=environment_parameters)
    else:
        raise ValueError(f"Unknown task '{task}'")

    wrapped_environment = DummyVecEnv([lambda: copy.deepcopy(environment)])
    num_framestacks = config["num_framestacks"]
    wrapped_environment = VecFrameStack(venv=wrapped_environment, n_stack=num_framestacks)
    return wrapped_environment
