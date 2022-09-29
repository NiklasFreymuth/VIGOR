import wandb
import os


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_START_METHOD",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


class WAndBWrapper:
    def __init__(self, config: dict, recording_dir: str, runname: str):
        project_name = "VIGOR_" + config.get("task").get("task")
        if "__" in runname:
            runname = runname.split("__", 1)[1]
        runname = runname.replace("True", "1")
        runname = runname.replace("False", "0")
        groupname = runname.rsplit("_", 1)[0]
        if len(groupname) > 127:
            groupname = groupname[-127:]
        reset_wandb_env()
        start_method = "thread"

        for i in range(10):
            try:
                entity = config.get("recording").get("wandb_entity", None)
                self.logger = wandb.init(entity=entity,
                                         project=project_name,
                                         tags=[config.get("task").get("task"), config.get("algorithm")],
                                         config=config,  # file config
                                         group=groupname,  # group repetitions of same exp
                                         name=runname,
                                         job_type=recording_dir.split(os.path.sep)[-2],
                                         dir=recording_dir,
                                         reinit=False,  # do not allow multiple init calls for the same process
                                         settings=wandb.Settings(start_method=start_method))  # _disable_stats=True,
                return  # if starting the run is successful, exit the loop (and in this case the function)
            except Exception as e:
                # implement a simple randomized exponential backoff if starting a run fails
                from time import sleep
                from random import random
                import warnings
                waiting_time = ((random() / 50) + 0.01) * (2 ** i)
                # wait between 0.01 and 10.24 seconds depending on the random seed and the iteration of the exponent

                warnings.warn("Problem with starting wandb: {}. Trying again in {} seconds".format(e, waiting_time))
                sleep(waiting_time)

    def log(self, metrics: dict, iteration: int = None):
        """
        Parses and logs the given dict of recorder metrics to wandb.
        Args:
            metrics: A dictionary of metrics to log. Logs all scalar values automatically,
            and can parse some pairs/lists with appropriate keys
            iteration: (Optional) Algorithm Iteration to log at

        Returns:

        """
        metrics = {k.replace("_", " ").title(): v for k, v in metrics.items()}
        self.logger.log(data=metrics, step=iteration)

    def finalize(self):
        self.logger.finish()
