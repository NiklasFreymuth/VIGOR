import os
import yaml
import csv
from pprint import pformat
from timeit import default_timer as timer
from recording.LoggingUtil import get_logger, log_heading
from util.Functions import flatten_dict


class WrapRecording:
    """
        Module for handling the framework of the current recording.
         Will print and save the initial config as well as the final results
    """

    def __init__(self, runname, recording_dir):
        self._logger = get_logger(name="Wrapper", path=recording_dir)
        self._start_time = None
        self._end_time = None
        self._config = None
        self._runname = runname
        self._recording_dir = recording_dir

    def initialize(self, config):
        self._config = config
        self._start_time = timer()
        log_heading(logger=self._logger, string_to_frame="Starting experiment")
        self._print_dict(config)
        self._save_to_yaml(config, "config")

    def finalize(self, results: dict, final_iteration: int = 0):
        self._end_time = timer()
        log_heading(logger=self._logger, string_to_frame="results")
        self._logger.info("Ending experiment with timestamp: {}".format(self._end_time))
        results["Elapsed_time"] = self._end_time - self._start_time
        self._save_to_csv(results=results, iterations=final_iteration+1)  # since we start counting at 0
        self._print_dict(results)
        self._save_to_yaml(results, "results")
        self._logger.handlers = []

    def _save_to_csv(self, results, iterations: int = 1):
        """
        Saves the config and the results of the current run to an overall .csv file containing all runs of
        the current experiment.
        :param results: the results of the run
        :return:
        """
        full_data = {**{"Runname": self._runname, "Iterations": iterations, "Results": results}, **self._config}
        full_data = flatten_dict(full_data)
        csv_dir = os.path.dirname(self._recording_dir)

        import pathlib
        csv_name = pathlib.PurePath(self._recording_dir).parent.name
        csv_path = os.path.join(csv_dir, csv_name + ".csv")
        try:
            if not os.path.isfile(csv_path):
                #  initialize new csv file
                with open(csv_path, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=full_data)
                    writer.writeheader()
                    writer.writerow(full_data)
            else:
                with open(csv_path, 'a', newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=full_data)
                    writer.writerow(full_data)
        except IOError as e:
            print("I/O error: {}".format(e))

    def _save_to_yaml(self, input_dict, input_type):
        """
        SaveUtility the current dictionary to a input_type.yaml
        :param input_dict: The dictionary given to the module
        :param input_type: Type of the input. Currently either "config" or "results"
        :return:
        """
        filename = os.path.join(self._recording_dir, input_type + ".yaml")
        with open(filename, "w") as file:
            if isinstance(input_dict, dict):
                yaml.dump(input_dict, file, sort_keys=True, indent=4)
            else:
                yaml.dump(input_dict.__dict__, file, sort_keys=True, indent=4)

    def _print_dict(self, input_dict):
        """
        Print the input dictionary given to this module
        :param input_dict: The dictionary given to the module
        :return:
        """
        self._logger.info("\n" + pformat(input_dict, indent=2))
