import matplotlib

matplotlib.use('Agg')

from recording.get_recorder import get_recorder
from environments.EnvironmentData import EnvironmentData
from util.ProcessConfig import process_config

import numpy as np
import torch
from algorithms.get_algorithm import get_algorithm


def get_default_config() -> dict:
    return {"algorithm": "VIGOR",
            "iterations": 301,
            "steps_per_iteration": 1,
            "recording": {
                "make_videos": False,
                "draw_expensive_plots": True,
                "wandb_logging": "default",
                "checkpoint_save_frequency": 1,
                "pseudo_contextual": {
                    "record_train_policies": True,
                    "record_validation_policies": True,
                    "record_test_policies": False,
                    "plotted_train_contexts": 6,
                    "plotted_validation_contexts": 6,
                    "plotted_test_contexts": 0,
                },
            },
            "policy": {
                "component_time_to_live": 5000,  # set this very high because we do not add/delete components here
                "component_addition_frequency": 5,
                "component_weight_threshold": 0.01,
                "num_components": 5,
                "weight_update_type": None,
                "kl_bound": 0.2,
                "samples_per_component": 512,
            },
            "network": {
                "num_dres": 5,
                "dre_aggregation": "mean",
                "uniform_policy_dre_samples": True,
                "batch_size": 64,
                "validation_split": 0.1,
                "verbose": 0,
                "early_stopping": {
                    "patience": 10,
                    "restore_best": True,
                    "warmup": 10,
                },
                "epochs": 50,
                "feedforward": {
                    "max_neurons_per_layer": "tied",
                },
                "learning_rate": 3.0e-4,
                "regularization": {
                    "batch_norm": False,
                    "dropout": 0.2,
                    "l2_norm": 0,
                    "spectral_norm": False,
                    "activation_function": "leakyrelu",
                },
                "time_series": {
                    "architecture": "1d_cnn",
                    "1d_cnn": {
                        "kernel_size": 5,
                        "num_layers": 2,
                        "num_channels": 32,
                        "padding": "zero",
                        "stepwise_aggregation_method": "sum",
                    },
                    "stepwise_loss": True,

                },
            },
            "meta_task": {
                "sample_size": 5,
                "num_test_contexts": 0,
                "shuffle_demonstrations": True,
                "num_evaluation_samples": 100,
                "num_train_contexts": 6,
                "num_validation_contexts": 6,
            },
            "task": {
                "data_source": "promp_fits",
                "task": "planar_reacher",
            },
            "modality": "geometric",
            }


def main():
    # get config and random seed
    experiment_name = "quickstart"

    config = get_default_config()
    config = process_config(current_config=config)
    np.random.seed(seed=0)
    torch.manual_seed(seed=0)

    # get algorithm and recorder
    environment_data = EnvironmentData(config=config)
    algorithm = get_algorithm(config=config, environment_data=environment_data)
    recorder = get_recorder(algorithm=algorithm, config=config, runname=experiment_name,
                            directory_name=experiment_name, environment_data=environment_data)

    algorithm.initialize_training()
    recorder.initialize_recording()
    for iteration in range(config.get("iterations")):
        for step in range(config.get("steps_per_iteration")):
            algorithm.train_iteration(iteration=iteration * config.get("steps_per_iteration") + step)
        recorder()


if __name__ == "__main__":
    main()
