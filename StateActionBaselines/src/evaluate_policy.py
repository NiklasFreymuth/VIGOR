from typing import Union, Tuple, Optional, List, Dict

import numpy as np
import wandb
from matplotlib import pyplot as plt
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecFrameStack

from environments.get_environment import get_environment


def evaluate_policy(config, environment, policy, test_context_ids, train_context_ids, wandb_run):
    train_evaluations = evaluate_policy_on_environment(policy=policy, environment=environment,
                                                       context_ids=train_context_ids,
                                                       deterministic=True, render=True, plot_title="Training Contexts")
    log_evaluations(wandb_run=wandb_run, evaluations=train_evaluations, prefix="train")
    test_environment = get_environment(config=config, context_ids=test_context_ids)
    test_evaluations = evaluate_policy_on_environment(policy=policy, environment=test_environment,
                                                      context_ids=test_context_ids, deterministic=True,
                                                      render=True, plot_title="Test Contexts")
    log_evaluations(wandb_run=wandb_run, evaluations=test_evaluations, prefix="test")

    # evaluate policy samples
    if config.get("task") in ["box_pusher", "online_box_pusher"]:
        num_samples = 5
    else:
        num_samples = 100
    num_visualized_samples = 3

    all_train_evaluations = []
    all_test_evaluations = []
    for idx in range(num_samples):
        # hacky way to also evaluate 100 samples of each policy, including plots for the first 3
        current_train_evaluations = evaluate_policy_on_environment(policy=policy, environment=environment,
                                                                   context_ids=train_context_ids,
                                                                   deterministic=False,
                                                                   render=idx < num_visualized_samples,
                                                                   plot_title=f"Training Contexts Sample#{idx}")
        if idx < num_visualized_samples:
            wandb_run.log({f"train_sample{idx}_visualization": plt})

        current_test_evaluations = evaluate_policy_on_environment(policy=policy, environment=test_environment,
                                                                  context_ids=test_context_ids,
                                                                  deterministic=False,
                                                                  render=idx < num_visualized_samples,
                                                                  plot_title=f"Test Contexts Sample#{idx}")
        if idx < num_visualized_samples:
            wandb_run.log({f"test_sample{idx}_visualization": plt})

        all_train_evaluations.extend(current_train_evaluations)
        all_test_evaluations.extend(current_test_evaluations)

    log_evaluations(wandb_run=wandb_run, evaluations=all_train_evaluations,
                    prefix=f"train_samples", log_visualization=False)
    log_evaluations(wandb_run=wandb_run, evaluations=all_test_evaluations,
                    prefix=f"test_samples", log_visualization=False)

    # evaluate multimodal policy
    if config.get("algorithm") == "mbc":
        evaluate_multimodal_policy(config=config, train_environment=environment,
                                   num_samples=num_samples, policy=policy, test_context_ids=test_context_ids,
                                   test_environment=test_environment, train_context_ids=train_context_ids,
                                   wandb_run=wandb_run)


def evaluate_multimodal_policy(config, train_environment, num_samples, policy, test_context_ids, test_environment,
                               train_context_ids, wandb_run):
    num_components = config.get("mbc").get("num_components")

    for mode, environment, context_ids in zip(["train", "test"],
                                              [train_environment, test_environment],
                                              [train_context_ids, test_context_ids]):
        all_evaluations = {}
        num_contexts = len(context_ids)

        for component_id in range(num_components):
            for sample_id in range(num_samples):
                current_evaluations = evaluate_policy_on_environment(policy=policy,
                                                                     environment=environment,
                                                                     context_ids=context_ids,
                                                                     deterministic=(False, component_id),
                                                                     render=False)
                # list of dictionaries over each context for single-sample evaluation

                for context_position, value_dictionary in enumerate(current_evaluations):
                    for key, value in value_dictionary.items():
                        if key not in all_evaluations.keys():
                            all_evaluations[key] = np.empty(shape=(num_contexts, num_components, num_samples))
                        all_evaluations[key][context_position, component_id, sample_id] = value

        # all_evaluations is a dictionary over metrics, where each entry has shape (#contexts, #components, #samples)

        logging_dict = {}
        for key, value in all_evaluations.items():
            # aggregate over samples
            value = np.mean(value, axis=-1)
            # shape: (#contexts, #components)

            # sort by component performance
            value = np.array([np.sort(component_array) for component_array in value])
            # shape: (#contexts, #components), but now sorted

            value = np.mean(value, axis=0)
            # shape: (#components)

            for idx, component_value in enumerate(value):
                logging_dict[f"{mode}/c{idx}_{key}"] = component_value
        wandb_run.log(logging_dict)



def evaluate_policy_on_environment(policy: BasePolicy, environment: VecFrameStack, context_ids: np.array,
                                   deterministic: Union[Tuple[bool, Union[str, int]], bool] = False,
                                   render: bool = False, save: bool = False,
                                   plot_title: Optional[str] = None) -> List[Dict]:
    policy.eval()
    if render:
        fig = plt.figure(figsize=(32, 18))
        if plot_title is None:
            plot_title = "Baseline evaluation"
        fig.suptitle(plot_title)

    last_infos = []
    for context_position, context_id in enumerate(context_ids):
        observations = environment.reset()
        environment.envs[0].set_context_position(context_position=context_position)  # go over all contexts
        environment.envs[0].evaluate_reward = True
        done = False
        while not done:
            actions = policy.predict(observation=observations, deterministic=deterministic)[0]
            observations, rewards, done, info = environment.step(actions=actions)

        info_dict = info[0]
        info_dict.pop("terminal_observation", None)
        last_infos.append(info_dict)
        if render:
            fig.add_subplot(int((len(context_ids) + 2) // 3), 3, context_position + 1)

            plt.title(f"Context ID: {context_id}")
            environment.render()
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("")
            plt.ylabel("")
    if render:
        plt.tight_layout()
        if save:
            plt.savefig(f"{plot_title.replace(' ', '_').lower()}.pdf", format="pdf",
                        dpi=200, transparent=False, bbox_inches='tight', pad_inches=0)
    return last_infos


def log_evaluations(wandb_run, evaluations: List[Dict], prefix: str = "", log_visualization: bool = True):
    """
    Log the evaluations of the current run in the wandb dashboard.
    Args:
        wandb_run:
        evaluations:
        prefix:

    Returns:

    """
    if log_visualization:
        wandb_run.log({f"{prefix}_visualization": plt})
    train_table = wandb.Table(columns=["context_id"] + list(evaluations[0].keys()),
                              data=[[position] + list(context_evaluation.values())
                                    for position, context_evaluation in enumerate(evaluations)])
    wandb_run.log({f"{prefix}_table": train_table})
    mean_evaluations = {f"{prefix}/mean_{k}": np.mean([evaluation[k]
                                                       for evaluation in evaluations])
                        for k in evaluations[0].keys()}
    wandb_run.log(mean_evaluations)
