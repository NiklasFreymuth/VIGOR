import matplotlib
from trajectory_bc_baselines.train_algorithm import train_algorithm
import os
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from trajectory_bc_baselines.evaluate_policy import evaluate_policy

matplotlib.use('Agg')
from cw2.cw_data import cw_logging
from cw2 import cluster_work, experiment

import gc
from environments.EnvironmentData import EnvironmentData
from recording.LoggingUtil import log_heading, get_logger, remove_handlers

import numpy as np
import torch
from recording.WAndBWrapper import WAndBWrapper
from util import Defaults
if os.name == "posix":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # run on CPU
else:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class BaselineExperiment(experiment.AbstractExperiment):

    def initialize(self, config: dict, rep: int, logger: cw_logging.AbstractLogger) -> None:
        self._logger = get_logger("Clusterwork")
        self._logger.info("Ready to start repetition {}.".format(rep))
        runname = config["_experiment_name"] + "_" + str(rep)
        run_notification = "Current Run: {}".format(runname).upper()
        log_heading(logger=self._logger, string_to_frame=run_notification)

    def run(self, config: dict, rep: int, logger) -> None:
        experiment_name = config.get("_experiment_name")
        self._directory_name = config.get("_nested_dir") if config.get("_nested_dir") else config.get(
            "_experiment_name")

        config = config.get("params")  # move into custom config

        np.random.seed(seed=rep)
        torch.manual_seed(seed=rep)

        config["seed"] = rep

        task_name = config.get("task").get("task")
        runname = experiment_name + "_" + str(rep)
        if self._directory_name is not None:
            recording_dir = os.path.join(Defaults.RECORDING_DIR, task_name, self._directory_name, runname)
        else:
            recording_dir = os.path.join(Defaults.RECORDING_DIR, task_name, runname)

        # gather data
        environment_data = EnvironmentData(config=config)
        data_dict = environment_data.data_dict
        expert_promp_weights = environment_data.data_dict.get("expert_promp_weights")
        train_contexts = data_dict.get("train_contexts")
        train_context_ids = data_dict.get("train_context_ids")
        test_contexts = data_dict.get("test_contexts")
        test_context_ids = data_dict.get("test_context_ids")

        environment = environment_data.environment_specifics.get("environment")

        # collect demonstrations
        trajectories = []
        for promp_weights, train_context_id, train_context in zip(expert_promp_weights,
                                                                  train_context_ids,
                                                                  train_contexts):
            for sample in promp_weights:
                # need to append next_obs for compatibility, but they are never used for bc
                trajectories.append(TrajectoryWithRew(obs=np.array([train_context, train_context]),
                                                      acts=np.array([sample]),
                                                      infos=None,
                                                      terminal=True,
                                                      rews=np.zeros(1)))
        transitions = rollout.flatten_trajectories(trajectories)

        wandb_run = WAndBWrapper(config=config, recording_dir=recording_dir, runname=runname).logger
        # train bc agent
        policy = train_algorithm(demonstrations=transitions,
                                 config=config,
                                 wandb_run=wandb_run)

        # evaluate
        evaluate_policy(config=config, environment=environment, policy=policy,
                        train_context_ids=train_context_ids, train_contexts=train_contexts,
                        test_context_ids=test_context_ids, test_contexts=test_contexts,
                        wandb_run=wandb_run)

        wandb_run.finish()

    def finalize(self, surrender=None, crash=False):
        if crash:
            self._logger.error("Error with run. Finalizing!")
        try:
            gc.collect()
            remove_handlers(logger=self._logger)
        except Exception as e:
            try:
                self._logger.error("Error finalizing: {}".format(e))
            except Exception as e:
                print(f"Error finalizing: {e}")


if __name__ == "__main__":
    # from util.PrintTrace import add_print_trace
    # add_print_trace()
    # vscode sanity check
    pass
    cw = cluster_work.ClusterWork(BaselineExperiment)
    cw.run()
