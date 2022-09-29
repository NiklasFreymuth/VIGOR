import numpy as np
from gym import spaces
from imitation.algorithms import bc
from imitation.data.types import Transitions

from util.Types import ConfigDict


def train_algorithm(demonstrations: Transitions, wandb_run, config: ConfigDict):
    """
    Trains an imitation learning algorithm on the provided expert demonstrations using the given environment.
    The reward of the environment is never used, i.e., it only provides observations for the actions of the learner.
    Args:
        demonstrations: The expert demonstrations to learn from. Are provided as a set of obs-act-obs-done tuples
        wandb_run: An instance of wandb.init() that is used to track the progress of the run.
        config: Config containing hyperparameters for the algorithm.

    Returns: A policy (stable_baselines3.common.policies.ActorCriticPolicy()) that is trained to imitate the expert
      actions

    """
    from imitation.util import logger
    import stable_baselines3.common.logger as sb_logger
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy
    verbose = config.get("verbose", False)

    _logger = sb_logger.Logger(folder=f"recordings/{wandb_run.id}",
                               output_formats=[logger.WandbOutputFormat()])
    if verbose:
        algorithm_logger = logger.HierarchicalLogger(default_logger=_logger)
    else:
        algorithm_logger = logger.HierarchicalLogger(default_logger=_logger, format_strs=())

    network_architecture = [config.get("neurons_per_layer")] * config.get("num_layers")

    observation_space = spaces.Box(low=-2 * np.pi, high=2 * np.pi,
                                   shape=(demonstrations.obs.shape[-1],), dtype=np.float32)
    action_space = spaces.Box(low=-np.inf, high=np.inf,
                              shape=(demonstrations.acts.shape[-1],), dtype=np.float32)
    if config.get("algorithm") == "bc":
        bc_policy = ActorCriticPolicy(observation_space=observation_space,
                                      action_space=action_space,
                                      lr_schedule=lambda _: torch.finfo(torch.float32).max,
                                      net_arch=network_architecture)
    elif config.get("algorithm") == "mbc":
        from trajectory_bc_baselines.multimodal.MultimodalActorCriticPolicy import MultimodalActorCriticPolicy
        bc_policy = MultimodalActorCriticPolicy(observation_space=observation_space,
                                                action_space=action_space,
                                                lr_schedule=lambda _: torch.finfo(torch.float32).max,
                                                net_arch=network_architecture,
                                                num_components=config.get("num_components"),
                                                entropy_approximation_mode=config.get(
                                                    "entropy_approximation_mode"),
                                                train_categorical_weights=config.get(
                                                    "train_categorical_weights"))
    else:
        raise ValueError(f"Unknown algorithm '{config.get('algorithm')}'")

    if verbose:
        print(f"Length of train demonstrations: {len(demonstrations)}")
        print(f"Shape of actions: {demonstrations.acts.shape}")
        print(f"Shape of observations: {demonstrations.obs.shape}")
        print(f"Shape of next observations: {demonstrations.next_obs.shape}")
        print(f"Shape of dones: {demonstrations.dones.shape}")
        print(f"batch_size: {config.get('batch_size')}")
        print(f"ent_weight: {config.get('ent_weight')}")
        print(f"l2_weight: {config.get('l2_weight')}")
        print(f"learning rate: {config.get('learning_rate')}")
        print(f"observation space: {observation_space}")
        print(f"action space: {action_space}")
        print(f"logger: {algorithm_logger}")
    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=demonstrations,
        custom_logger=algorithm_logger,
        policy=bc_policy,
        batch_size=config.get("batch_size"),
        ent_weight=config.get("ent_weight"),
        l2_weight=config.get("l2_weight"),
        optimizer_kwargs={"lr": config.get("learning_rate")}
    )

    bc_trainer.train(progress_bar=verbose,
                     n_epochs=config.get("n_epochs"))
    # number of passes through the expert data
    policy = bc_trainer.policy
    #  bc_trainer.policy is the trained policy. Can be evaluated on different environments
    return policy