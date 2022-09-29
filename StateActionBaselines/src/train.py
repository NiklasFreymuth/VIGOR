import matplotlib

matplotlib.use('agg')

import stable_baselines3 as sb3
from imitation.algorithms import bc
from imitation.algorithms.adversarial import gail
from imitation.data import rollout
from imitation.data.types import Transitions
from stable_baselines3.common.vec_env import VecEnv
from baseline_util.Keys import PLANAR_REACHER, PANDA_REACHER, BOX_PUSHER, ONLINE_BOX_PUSHER
from baseline_util.Types import *
from src.collect_demonstrations import get_expert_trajectories


def get_demonstrations(config) -> Tuple[Transitions, np.array, np.array]:
    """
    Get expert demonstrations in a format that is readable for the imitation library
    Args:
        config: Config for this run, including information on which expert data to use

    Returns: A batch of obs-act-obs-done transitions

    """

    train_context_ids, test_context_ids = get_contexts(config)
    trajectories = get_expert_trajectories(config=config, context_ids=train_context_ids)

    # Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
    # This is a more general dataclass containing unordered
    # (observation, actions, next_observation) transitions.
    transitions = rollout.flatten_trajectories(trajectories)

    return transitions, train_context_ids, test_context_ids


def get_contexts(config) -> Tuple[np.array, np.array]:
    num_train_contexts: int = config.get("num_train_contexts")
    num_test_contexts: int = config.get("num_test_contexts", 6)

    task = config.get("task")
    if task == PLANAR_REACHER:
        num_total_train_contexts = 12  # only pick first 12 train contexts for planar reacher

    elif task in [PANDA_REACHER, BOX_PUSHER]:
        num_total_train_contexts = len(np.load(f"data/{task}/joint.npz")["arr_0"])
    elif task == ONLINE_BOX_PUSHER:
        num_total_train_contexts = len(np.load(f"data/{BOX_PUSHER}/joint.npz")["arr_0"])
    else:
        raise ValueError(f"Unknown task '{task}'")

    if task in [BOX_PUSHER, ONLINE_BOX_PUSHER]:

        contexts = np.load(f"data/{BOX_PUSHER}/contexts.npy")
        # stratified sampling for box pusher by default
        divisor = num_train_contexts // 4
        rest = num_train_contexts % 4

        train_context_ids = []
        for position, conditions in enumerate((np.logical_and(contexts[:, 2] > 0, contexts[:, 1] > 0),
                                               np.logical_and(contexts[:, 2] > 0, contexts[:, 1] < 0),
                                               np.logical_and(contexts[:, 2] < 0, contexts[:, 1] > 0),
                                               np.logical_and(contexts[:, 2] < 0, contexts[:, 1] < 0))):
            all_satisfying_ids, = np.where(conditions)
            num_choices = divisor
            if rest > position:
                num_choices = num_choices + 1
            chosen_ids = np.random.choice(all_satisfying_ids, num_choices, replace=False)
            train_context_ids.extend(chosen_ids)
        train_context_ids = np.array(train_context_ids)
        np.random.shuffle(train_context_ids)

    else:
        # always choose from available train contexts
        train_context_ids = np.random.choice(num_total_train_contexts, num_train_contexts, replace=False)

    if task == ONLINE_BOX_PUSHER:
        num_total_contexts = len(np.load(f"data/{BOX_PUSHER}/contexts.npy"))
    else:
        num_total_contexts = len(np.load(f"data/{task}/contexts.npy"))

    remaining_context_ids = list(set(np.arange(num_total_contexts)) - set(train_context_ids))
    test_context_ids = np.random.choice(remaining_context_ids, num_test_contexts, replace=False)

    return train_context_ids, test_context_ids


def train_algorithm(environment: VecEnv, demonstrations: Transitions, wandb_run, algorithm: str,
                    algorithm_config: ConfigDict, verbose: bool = False):
    """
    Trains an imitation learning algorithm on the provided expert demonstrations using the given environment.
    The reward of the environment is never used, i.e., it only provides observations for the actions of the learner.
    Args:
        environment: The gym environment to train on.
        demonstrations: The expert demonstrations to learn from. Are provided as a set of obs-act-obs-done tuples
        wandb_run: An instance of wandb.init() that is used to track the progress of the run.
        algorithm: Which algorithm to use. Supports
            "bc" for vanilla deep behavioral cloning
            "gail" for generative adversarial imitation learning
            "airl" for adversarial inverse reinforcement learning
        algorithm_config: Config containing hyperparameters for the algorithm.
          The hyperparameters differ for each algorithm.
        verbose: Whether to print training progress or not. Defaults to False

    Returns: A policy (stable_baselines3.common.policies.ActorCriticPolicy()) that is trained to imitate the expert
      actions

    """
    from imitation.util import logger
    import stable_baselines3.common.logger as sb_logger
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy

    _logger = sb_logger.Logger(folder=f"recordings/{wandb_run.id}",
                               output_formats=[logger.WandbOutputFormat()])
    if verbose:
        algorithm_logger = logger.HierarchicalLogger(default_logger=_logger)
    else:
        algorithm_logger = logger.HierarchicalLogger(default_logger=_logger, format_strs=())

    network_architecture = [algorithm_config.get("neurons_per_layer")] * algorithm_config.get("num_layers")
    if algorithm in ["bc", "mbc"]:
        # Train BC on expert data.
        # BC also accepts as `demonstrations` any PyTorch-style DataLoader that iterates over
        # dictionaries containing observations and actions.
        # taken from the imitation package except for the architecture

        if algorithm == "bc":
            bc_policy = ActorCriticPolicy(observation_space=environment.observation_space,
                                          action_space=environment.action_space,
                                          lr_schedule=lambda _: torch.finfo(torch.float32).max,
                                          net_arch=network_architecture)
        elif algorithm == "mbc":
            from src.multimodal.MultimodalActorCriticPolicy import MultimodalActorCriticPolicy
            bc_policy = MultimodalActorCriticPolicy(observation_space=environment.observation_space,
                                                    action_space=environment.action_space,
                                                    lr_schedule=lambda _: torch.finfo(torch.float32).max,
                                                    net_arch=network_architecture,
                                                    num_components=algorithm_config.get("num_components"),
                                                    entropy_approximation_mode=algorithm_config.get(
                                                        "entropy_approximation_mode"),
                                                    train_categorical_weights=algorithm_config.get(
                                                        "train_categorical_weights"))
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'")

        if verbose:
            print(f"Length of train demonstrations: {len(demonstrations)}")
            print(f"Shape of actions: {demonstrations.acts.shape}")
            print(f"Shape of observations: {demonstrations.obs.shape}")
            print(f"Shape of next observations: {demonstrations.next_obs.shape}")
            print(f"Shape of dones: {demonstrations.dones.shape}")
            print(f"batch_size: {algorithm_config.get('batch_size')}")
            print(f"ent_weight: {algorithm_config.get('ent_weight')}")
            print(f"l2_weight: {algorithm_config.get('l2_weight')}")
            print(f"learning rate: {algorithm_config.get('learning_rate')}")
            print(f"observation space: {environment.observation_space}")
            print(f"action space: {environment.action_space}")
            print(f"logger: {algorithm_logger}")
        bc_trainer = bc.BC(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            demonstrations=demonstrations,
            custom_logger=algorithm_logger,
            policy=bc_policy,
            batch_size=algorithm_config.get("batch_size"),
            ent_weight=algorithm_config.get("ent_weight"),
            l2_weight=algorithm_config.get("l2_weight"),
            optimizer_kwargs={"lr": algorithm_config.get("learning_rate")}
        )

        bc_trainer.train(progress_bar=verbose, n_epochs=algorithm_config.get("n_epochs"))
        # number of passes through the expert data
        policy = bc_trainer.policy
        #  bc_trainer.policy is the trained policy. Can be evaluated on different environments

    elif algorithm == "gail":
        # Train GAIL on expert data.
        # GAIL, and AIRL also accept as `demonstrations` any Pytorch-style DataLoader that
        # iterates over dictionaries containing observations, actions, and next_observations.
        from imitation.rewards.reward_nets import BasicRewardNet
        reward_net = BasicRewardNet(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            hid_sizes=network_architecture
        )

        learner_config = algorithm_config.get("learner")
        learner_architecture = [learner_config.get("neurons_per_layer")] * learner_config.get("num_layers")
        if learner_config.get("share_network", True):
            policy_kwargs = dict(net_arch=learner_architecture)
        else:
            policy_kwargs = dict(net_arch=[dict(pi=learner_architecture, vf=learner_architecture)])

        if algorithm_config.get("learning_algorithm").lower() == "ppo":
            learner = sb3.PPO(policy="MlpPolicy",
                              env=environment,
                              verbose=verbose,
                              n_steps=learner_config.get("n_steps"),
                              batch_size=learner_config.get("batch_size"),
                              policy_kwargs=policy_kwargs)
        elif algorithm_config.get("learning_algorithm").lower() == "sac":
            learner = sb3.SAC(policy="MlpPolicy",
                              env=environment,
                              buffer_size=learner_config.get("buffer_size"),
                              batch_size=learner_config.get("batch_size"),
                              # policy_kwargs={**policy_kwargs, "log_std_init": -2},
                              use_sde=learner_config.get("use_sde"),
                              tau=learner_config.get("tau"),
                              learning_rate=learner_config.get("learning_rate"),
                              learning_starts=10000,
                              verbose=verbose,
                              )
        else:
            raise ValueError(f"Unknown algorithm_config '{algorithm_config}'")
        gail_trainer = gail.GAIL(
            venv=environment,  # gives the environment, but overwrites the reward part of the step() to a learned reward
            demonstrations=demonstrations,  # expert demonstrations
            demo_batch_size=algorithm_config.get("demo_batch_size"),  # number of expert samples per batch
            gen_algo=learner,  # algorithm that is used to generate the learner data
            custom_logger=algorithm_logger,
            reward_net=reward_net,  # can take a pytorch module of observations and actions that outputs a scalar
        )
        gail_trainer.train(total_timesteps=algorithm_config.get("total_timesteps"))
        policy = gail_trainer.policy
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'")
    return policy
