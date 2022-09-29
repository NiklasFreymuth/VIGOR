import os

from util import Defaults
from util.Types import *

from algorithms.VIGOR import VIGOR
from recording.Recorder import Recorder
from environments.EnvironmentData import EnvironmentData


def get_recorder(algorithm: VIGOR, runname: str, directory_name: str, config: Dict[str, Any],
                 environment_data: EnvironmentData) -> Recorder:
    """
    Provides a recorder object given specification about the running algorithm and the task
    Args:
        algorithm: Instance of the current algorithm. Always based on VIGOR
        runname: Name of the current run
        directory_name: Name of the recording directory. Everything recorded in this run is saved there
        config: Dict containing the configuration of the run
        environment_data: Class that holds most of the necessary information for the run, such as the relevant data,
        normalization parameters, and eventually things such as the target function or mapping functions
        debug: Whether to add a "debug" directory or not

    Returns:

    """

    task_name = config.get("task").get("task")
    if directory_name is not None:
        recording_dir = os.path.join(Defaults.RECORDING_DIR, task_name, directory_name, runname)
    else:
        recording_dir = os.path.join(Defaults.RECORDING_DIR, task_name, runname)

    recorder = Recorder(algorithm=algorithm, config=config,
                        recording_dir=recording_dir,
                        runname=runname, environment_data=environment_data)

    return recorder
