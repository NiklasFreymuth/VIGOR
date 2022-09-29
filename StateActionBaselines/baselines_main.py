from src.evaluate_policy import evaluate_policy
from src.train import get_demonstrations, train_algorithm
from environments.get_environment import get_environment
import wandb
import os

if os.name == "posix":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # run on CPU

import matplotlib
import os

matplotlib.use('Agg')
from cw2.cw_data import cw_logging
from cw2 import cluster_work, experiment

import gc
from baseline_util.cw2_logging_util import log_heading, get_logger, remove_handlers

import numpy as np
import torch

if os.name == "posix":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # run on CPU
else:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_wandb_run(config: dict, runname: str):
    project_name = f"VIGOR_{config.get('task')}"
    if "__" in runname:
        runname = runname.split("__", 1)[1]
    runname = runname.replace("True", "1")
    runname = runname.replace("False", "0")
    groupname = runname.rsplit("_", 1)[0]
    if len(groupname) > 127:
        groupname = groupname[-127:]

    # reset env
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_START_METHOD",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

    start_method = "thread"

    for i in range(10):
        try:
            entity = config.get("recording").get("wandb_entity", None)
            logger = wandb.init(entity=entity,
                                project=project_name,
                                tags=[config.get("algorithm")],
                                config=config,  # file config
                                group=groupname,  # group repetitions of same exp
                                name=runname,
                                job_type=config.get("identifier", "box_pusher_state_action"),
                                reinit=False,  # do not allow multiple init calls for the same process
                                settings=wandb.Settings(start_method=start_method))  # _disable_stats=True,
            return logger  # if starting the run is successful, exit the loop (and in this case the function)
        except Exception as e:
            # implement a simple randomized exponential backoff if starting a run fails
            from time import sleep
            from random import random
            import warnings
            waiting_time = ((random() / 50) + 0.01) * (2 ** i)
            # wait between 0.01 and 10.24 seconds depending on the random seed and the iteration of the exponent

            warnings.warn("Problem with starting wandb: {}. Trying again in {} seconds".format(e, waiting_time))
            sleep(waiting_time)


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

        runname = experiment_name + "_" + str(rep)

        wandb_run = get_wandb_run(config=config, runname=runname)

        # train bc agent
        train_demonstrations, train_context_ids, test_context_ids = get_demonstrations(config=config)
        environment = get_environment(config=config, context_ids=train_context_ids)
        policy = train_algorithm(environment=environment, demonstrations=train_demonstrations,
                                 algorithm=config.get("algorithm"),
                                 algorithm_config=config.get(config.get("algorithm")),
                                 wandb_run=wandb_run,
                                 verbose=config.get("verbose", False))
        evaluate_policy(config, environment, policy, test_context_ids, train_context_ids, wandb_run)

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
    cw = cluster_work.ClusterWork(BaselineExperiment)
    cw.run()
