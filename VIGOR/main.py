import matplotlib

matplotlib.use('Agg')
from cw2.cw_data import cw_logging
from cw2 import cluster_work, experiment

import gc
from recording.get_recorder import get_recorder
from environments.EnvironmentData import EnvironmentData
from recording.LoggingUtil import log_heading, get_logger, remove_handlers
from recording import Recorder
from algorithms.VIGOR import VIGOR
from util.ProcessConfig import process_config

import numpy as np
import torch
from util.Types import *


def run_algorithm_iteration(num_iterations: int, algorithm: VIGOR,
                            recorder: Recorder, steps_per_iteration: int = 1) -> Dict[Key, Any]:
    """
    Runs the given algorithm and yields iteration-wise results in the form of a dict of metrics.
    Also records a lot of details about the algorithm using the provided recorder
    Args:
        num_iterations: The number of iterations to run the algorithm for
        algorithm: The algorithm to run
        recorder: Records most of what happens with the algorithm
        steps_per_iteration: Number of algorithm steps to perform in each iteration. The recorder is only applied
        after all steps have been performed.

    Returns: Yields metrics after every iteration

    """
    algorithm.initialize_training()
    recorder.initialize_recording()
    for iteration in range(num_iterations):
        for step in range(steps_per_iteration):
            algorithm.train_iteration(iteration=iteration * steps_per_iteration + step)
        results = recorder()
        yield results


class IRLExperiment(experiment.AbstractExperiment):

    def initialize(self, config: dict, rep: int, logger: cw_logging.AbstractLogger) -> None:
        self._logger = get_logger("Clusterwork")
        self._logger.info("Ready to start repetition {}.".format(rep))
        runname = config["_experiment_name"] + "_" + str(rep)
        run_notification = "Current Run: {}".format(runname).upper()
        log_heading(logger=self._logger, string_to_frame=run_notification)

    def run(self, config: dict, rep: int, logger) -> None:
        experiment_name = config.get("_experiment_name")
        self._directory_name = config.get("_nested_dir") if config.get("_nested_dir") else experiment_name

        config = config.get("params")  # move into custom config
        from util.Functions import get_from_nested_dict
        seed = get_from_nested_dict(dictionary=config, list_of_keys=["meta_task", "seed"], raise_error=True)
        if seed == "default":
            seed = rep
            config["meta_task"]["seed"] = rep
        pytorch_seed = get_from_nested_dict(dictionary=config, list_of_keys=["meta_task", "pytorch_seed"],
                                            raise_error=True)
        if pytorch_seed == "default":
            pytorch_seed = rep
            config["meta_task"]["pytorch_seed"] = rep
        elif pytorch_seed == "paired":
            pytorch_seed = seed
            config["meta_task"]["pytorch_seed"] = seed

        np.random.seed(seed=seed)
        torch.manual_seed(seed=pytorch_seed)

        config = process_config(current_config=config)
        self._runname = experiment_name + "_" + str(rep)
        self._algorithm, self._recorder = self._prepare_algorithm_and_recording(config=config)
        results = [x for x in
                   run_algorithm_iteration(num_iterations=config.get("iterations"), algorithm=self._algorithm,
                                           recorder=self._recorder,
                                           steps_per_iteration=config.get("steps_per_iteration"))]

    def _prepare_algorithm_and_recording(self, config: dict) -> Tuple[VIGOR, Recorder.Recorder]:
        from algorithms.get_algorithm import get_algorithm
        environment_data = EnvironmentData(config=config)
        algorithm = get_algorithm(config=config, environment_data=environment_data)
        recorder = get_recorder(algorithm=algorithm, config=config, runname=self._runname,
                                directory_name=self._directory_name, environment_data=environment_data)
        return algorithm, recorder

    def finalize(self, surrender=None, crash=False):
        if crash:
            self._logger.error("Error with run. Finalizing!")
        try:
            self._recorder.finalize()
        except Exception as e:
            self._logger.error("Error finalizing: {}".format(e))
        finally:
            del self._recorder
            del self._algorithm
            gc.collect()
            remove_handlers(logger=self._logger)


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(IRLExperiment)
    cw.run()
