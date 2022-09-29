from util.Types import *

import numpy as np
import wandb
from matplotlib import pyplot as plt
from stable_baselines3.common.policies import BasePolicy

from environments.AbstractEnvironment import AbstractEnvironment
from environments.planar_reacher.PlanarReacher import PlanarReacher
from environments.panda_reacher.PandaReacher import PandaReacher
from environments.box_pusher.TeleoperationBoxPusher import TeleoperationBoxPusher


def evaluate_policy(config: ConfigDict, environment: AbstractEnvironment, policy: BasePolicy,
                    train_context_ids: np.array, train_contexts: np.array,
                    test_context_ids: np.array, test_contexts: np.array, wandb_run):
    # evaluate policy means
    train_evaluations = evaluate_policy_on_environments(policy=policy, environment=environment,
                                                        num_samples_per_context=1, num_visualized_samples=1,
                                                        context_ids=train_context_ids,
                                                        contexts=train_contexts,
                                                        deterministic=True, render=True,
                                                        plot_title="Train Means")
    log_evaluations(wandb_run=wandb_run, evaluations=train_evaluations, prefix="train")

    test_evaluations = evaluate_policy_on_environments(policy=policy, environment=environment,
                                                       num_samples_per_context=1, num_visualized_samples=1,
                                                       context_ids=test_context_ids,
                                                       contexts=test_contexts,
                                                       deterministic=True, render=True,
                                                       plot_title="Test Means")
    log_evaluations(wandb_run=wandb_run, evaluations=test_evaluations, prefix="test")

    # evaluate policy samples
    if config.get("task").get("task") == "box_pusher":
        num_samples_per_context = 5
        num_visualized_samples = 1
    else:
        num_samples_per_context = 100
        num_visualized_samples = 5

    sample_train_evaluations = evaluate_policy_on_environments(policy=policy, environment=environment,
                                                               num_samples_per_context=num_samples_per_context,
                                                               num_visualized_samples=num_visualized_samples,
                                                               context_ids=train_context_ids,
                                                               contexts=train_contexts,
                                                               deterministic=False, render=True,
                                                               plot_title="Train Samples")
    log_evaluations(wandb_run=wandb_run, evaluations=sample_train_evaluations,
                    prefix=f"train_samples")

    sample_test_evaluations = evaluate_policy_on_environments(policy=policy, environment=environment,
                                                              num_samples_per_context=num_samples_per_context,
                                                              num_visualized_samples=num_visualized_samples,
                                                              context_ids=test_context_ids,
                                                              contexts=test_contexts,
                                                              deterministic=False, render=True,
                                                              plot_title="Test Samples")
    log_evaluations(wandb_run=wandb_run, evaluations=sample_test_evaluations,
                    prefix=f"test_samples")

    # evaluate individual components for multimodal policy
    if config.get("algorithm") == "mbc":
        evaluate_multimodal_policy(config=config,
                                   environment=environment,
                                   num_samples_per_context=num_samples_per_context,
                                   policy=policy,
                                   train_context_ids=train_context_ids,
                                   train_contexts=train_contexts,
                                   test_context_ids=test_context_ids,
                                   test_contexts=test_contexts,
                                   wandb_run=wandb_run)


def evaluate_multimodal_policy(config, environment: AbstractEnvironment,
                               train_context_ids: np.array,
                               train_contexts: np.array,
                               test_context_ids: np.array,
                               test_contexts: np.array,
                               num_samples_per_context: np.array,
                               policy,
                               wandb_run):
    num_components = config.get("num_components")

    for mode, contexts, context_ids in zip(["train", "test"],
                                           [train_contexts, test_contexts],
                                           [train_context_ids, test_context_ids]):
        all_evaluations = {}
        num_contexts = len(context_ids)
        for component_id in range(num_components):
            current_evaluations = evaluate_policy_on_environments(policy=policy,
                                                                  environment=environment,
                                                                  num_samples_per_context=num_samples_per_context,
                                                                  context_ids=context_ids,
                                                                  contexts=contexts,
                                                                  deterministic=(False, component_id),
                                                                  render=False)
            # list over contexts

            for context_position, value_dictionary in enumerate(current_evaluations):
                for key, value in value_dictionary.items():
                    if key not in all_evaluations.keys():
                        all_evaluations[key] = np.empty(shape=(num_contexts, num_components))
                    all_evaluations[key][context_position, component_id] = value

        # all_evaluations is a dictionary over metrics, where each entry has shape (#contexts, #components)

        logging_dict = {}
        for key, value in all_evaluations.items():

            # sort by component performance
            value = np.array([np.sort(component_array) for component_array in value])
            # shape: (#contexts, #components), but now sorted

            value = np.mean(value, axis=0)
            # shape: (#components)

            for idx, component_value in enumerate(value):
                logging_dict[f"{mode}/c{idx}_{key}"] = component_value


        logging_dict = {k.replace("_", " ").title(): v for k, v in logging_dict.items()}
        wandb_run.log(logging_dict)


def evaluate_policy_on_environments(policy: BasePolicy, environment, context_ids: np.array,
                                    contexts: np.array,
                                    num_samples_per_context: int,
                                    deterministic: Union[Tuple[bool, Union[str, int]], bool] = False,
                                    render: bool = False, save: bool = False, num_visualized_samples: int = 1,
                                    plot_title: Optional[str] = None, ) -> List[Dict]:
    """

    Args:
        policy:
        environment:
        context_ids:
        contexts:
        num_samples_per_context:
        deterministic:
        render:
        save:
        num_visualized_samples:
        plot_title:

    Returns: A list of dictionaries of evaluations per context. I.e., a list over contexts, where each entry is a
        dictionary with metrics for this context. The metrics are aggregated over num_samples_per_context samples

    """
    policy.eval()
    if render:
        fig = plt.figure(figsize=(32, 18))
        if plot_title is None:
            plot_title = "Baseline evaluation"
        fig.suptitle(plot_title)

    reward_dicts = []
    
    for context_position, (context_id, context) in enumerate(zip(context_ids, contexts)):
        observation = np.array([context] * num_samples_per_context)
        promp_parameters = policy.predict(observation=observation, deterministic=deterministic)[0]

        reward_dict = environment.reward(samples=promp_parameters, context_id=context_id, return_as_dict=True)
        reward_dict = {key: np.mean(value) for key, value in reward_dict.items()}
        reward_dicts.append(reward_dict)

        if render:
            fig.add_subplot(int((len(context_ids) + 2) // 3), 3, context_position + 1)

            plt.title(f"Context ID: {context_id}")

            if isinstance(environment, PlanarReacher):
                environment.plot_samples(policy_samples=promp_parameters[:num_visualized_samples],
                                         context_id=context_id)
            elif isinstance(environment, PandaReacher):
                environment._plot_sample_target_projection(policy_samples=promp_parameters[:num_visualized_samples],
                                                           context_id=context_id,
                                                           draw_dashes=False)
            elif isinstance(environment, TeleoperationBoxPusher):
                environment._plot_mujoco_image(policy_sample=promp_parameters[0], context_id=context_id)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("")
            plt.ylabel("")
    if render:
        plt.tight_layout()
        if save:
            plt.savefig(f"{plot_title.replace(' ', '_').lower()}.pdf", format="pdf",
                        dpi=200, transparent=False, bbox_inches='tight', pad_inches=0)
    return reward_dicts


def log_evaluations(wandb_run, evaluations: List[Dict], prefix: str = ""):
    """
    Log the evaluations of the current run in the wandb dashboard.
    Args:
        wandb_run:
        evaluations:
        prefix:

    Returns:

    """
    wandb_run.log({f"{prefix}_visualization": plt})
    train_table = wandb.Table(columns=["context_id"] + list(evaluations[0].keys()),
                              data=[[position] + list(context_evaluation.values())
                                    for position, context_evaluation in enumerate(evaluations)])
    wandb_run.log({f"{prefix}_table": train_table})
    mean_evaluations = {f"{prefix}/mean_{k}": np.mean([evaluation[k]
                                                       for evaluation in evaluations])
                        for k in evaluations[0].keys()}
    mean_evaluations = {k.replace("_", " ").title(): v for k, v in mean_evaluations.items()}
    wandb_run.log(mean_evaluations)
