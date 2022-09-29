from typing import Optional, Dict

import numpy as np
from matplotlib import pyplot as plt

from algorithms.distributions.GMM import GMM
from util.colors.SmartColors import SmartColors


def plot_component_weights(policy: GMM, component_weights: Optional[Dict[int, np.array]],
                           policy_colors: SmartColors, iteration: int, plot_title: bool = True) -> Dict[int, np.array]:
    """
    Takes a dictionary of previous policy weights and the current policy. Extends the dictionary to the current
    iteration and plots the policy weights as a stackplot.
    Args:
        policy:
        component_weights:
        policy_colors:
        iteration:

    Returns: The component_weights dictionary but with new entries depending on the last iteration and  whether new
    components were added or old ones deleted

    """
    num_positions = iteration + 1
    # find out old and new weights
    current_component_weights = policy.components_as_dict
    if component_weights is None:  # initialize
        component_weights = {}
    for component_id in component_weights.keys():
        if component_id in current_component_weights:
            component_weights[component_id].append(current_component_weights[component_id])
        else:
            component_weights[component_id].append(0)
    for component_id in current_component_weights.keys():
        if component_id not in component_weights:
            # new weight (and thus new component)
            component_weights[component_id] = [0] * num_positions
            component_weights[component_id][-1] = current_component_weights[component_id]

    y = component_weights.values()
    colors = ["black"] * len(component_weights.keys())
    current_color = 0
    handles = []
    labels = []
    for component_position, component_id in enumerate(component_weights.keys()):
        if component_weights[component_id][-1] > 0:  # active component
            colors[component_position] = policy_colors(color_id=component_id)
            labels.append(component_id)
            handles.append(colors[component_position])
            current_color += 1
        else:
            labels.append('_nolegend_')
    plt.stackplot(range(num_positions), y, colors=colors, labels=labels)
    if plot_title:
        plt.title("Component Weights")
    plt.legend(loc='upper left')
    return component_weights


def create_feature_histogram(modality_samples: np.array, contexts: np.array, modality_name: str,
                             num_visualized_contexts: int, num_plotted_rows: int = 5,
                             num_plotted_columns: int = 5) -> None:
    fig = plt.figure(figsize=(num_plotted_rows * 16 * 0.8, num_plotted_rows * 9 * 0.8))
    fig.suptitle("Feature histogram for modality '{}'".format(modality_name))
    for i, (context, modality) in enumerate(zip(contexts[:num_visualized_contexts],
                                                modality_samples[:num_visualized_contexts])):
        flattened_modality = modality.reshape((-1, modality.shape[-1]))
        print("Plotting feature histogram {} for modality '{}'".format(i, modality_name))
        fig.add_subplot(num_plotted_rows, num_plotted_columns, i + 1)
        plt.title("Context: {}".format(np.round(context, 2)))
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
        for feature_position in range(flattened_modality.shape[-1]):
            plt.hist(flattened_modality[..., feature_position], bins=20, label=str(feature_position),
                     histtype="step")
        plt.legend(loc="upper right")
    plt.tight_layout()