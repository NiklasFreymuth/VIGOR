import matplotlib
from recording.RecordingUtil import merge_folder_to_file, parse_last_results

matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path

import util.Defaults as d
import recording.LoggingUtil as LoggingUtil

from util.Types import *
from util.Functions import get_from_nested_dict
from util.save.SaveUtility import SaveUtility

from recording.WAndBWrapper import WAndBWrapper
from recording.MetricRecording import MetricRecording
from recording.WrapRecording import WrapRecording
from recording.AdditionalPlotWrapper import AdditionalPlotWrapper
from recording.RecordingUtil import is_vigor
from recording.environment_visualization.EnvironmentVisualization import EnvironmentVisualization
from recording.AdditionalPlot import AdditionalPlot

from algorithms.VIGOR import VIGOR
from algorithms.DRex import DRex

from environments.AbstractEnvironment import AbstractEnvironment
from environments.EnvironmentData import EnvironmentData


class Recorder:
    """
    Custom Logger class responsible for logging all results of the training
    """

    def __init__(self, algorithm: VIGOR, config: dict,
                 recording_dir, runname, environment_data: EnvironmentData, save_format="pdf", img_dpi=200):
        self._algorithm = algorithm
        self._iteration = -1
        self._data_dict = environment_data.data_dict
        self._checkpoint_save_frequency = get_from_nested_dict(config,
                                                               list_of_keys=["recording", "checkpoint_save_frequency"],
                                                               default_return=1)
        self._recording_dir = recording_dir

        if not os.path.exists(recording_dir):
            print('Directory ' + recording_dir + ' not found. Creating directory')
            os.makedirs(recording_dir)
        self.logger = LoggingUtil.get_logger(name="Recorder", path=recording_dir)

        # set up the different things to actually record
        self._wrap_recording = WrapRecording(runname=runname, recording_dir=recording_dir)
        self._make_videos = get_from_nested_dict(dictionary=config,
                                                 list_of_keys=["recording", "make_videos"],
                                                 default_return=True)
        self._wandb_logger = self._get_wandb_logger(config, recording_dir, runname)

        self._environment: AbstractEnvironment = environment_data.environment_specifics.get("environment")

        self._initialize_task_visualization(algorithm=algorithm,
                                            environment=self._environment,
                                            environment_data=environment_data, config=config)

        self._metric_recording = MetricRecording(algorithm=algorithm,
                                                 environment_data=environment_data,
                                                 recording_dir=recording_dir)

        self._additional_plot_wrapper = AdditionalPlotWrapper(algorithm=algorithm,
                                                              environment_data=environment_data)
        self._draw_expensive_additional_plots = get_from_nested_dict(dictionary=config,
                                                                     list_of_keys=["recording",
                                                                                   "draw_expensive_plots"],
                                                                     raise_error=True)

        # give visualization of initial setup and save config of task
        self._wrap_recording.initialize(config=config)

        # save utility
        self._save_utility = SaveUtility(save_directory=self._recording_dir)
        self._save_utility.save_contexts(data_dict=environment_data.data_dict)

        assert save_format in ["png", "jpg", "svg", "pdf"], "unknown save_format {}".format(save_format)
        self._save_format = save_format
        self._img_dpi = img_dpi

        # context ids etc.
        self._context_ids = environment_data.environment_specifics.get("context_ids")
        self._reverse_context_ids = environment_data.environment_specifics.get("reverse_context_ids")

        pseudo_contextual_recording_dict = get_from_nested_dict(environment_data.config,
                                                                list_of_keys=["recording", "pseudo_contextual"],
                                                                raise_error=True)

        self._visualization_modes = []
        record_train_policies = pseudo_contextual_recording_dict.get("record_train_policies", True)
        if record_train_policies:
            self._visualization_modes.append(d.TRAIN)

        record_validation_policies = pseudo_contextual_recording_dict.get("record_validation_policies", False)
        if record_validation_policies and len(environment_data.validation_contexts) > 0:
            self._visualization_modes.append(d.VALIDATION)

        record_test_policies = pseudo_contextual_recording_dict.get("record_test_policies", False)
        if record_test_policies and len(environment_data.test_contexts) > 0:
            self._visualization_modes.append(d.TEST)

        record_drex_train_policies = pseudo_contextual_recording_dict.get("record_drex_train_policies", False)
        if record_drex_train_policies and len(environment_data.drex_train_contexts) > 0 \
                and isinstance(algorithm, DRex):
            self._visualization_modes.append(d.DREX_TRAIN)

    def _get_wandb_logger(self, config, recording_dir, runname) -> Optional[WAndBWrapper]:
        wandb_logger = None
        wandb_logging = get_from_nested_dict(dictionary=config, list_of_keys=["recording", "wandb_logging"],
                                             default_return=True)
        if isinstance(wandb_logging, str) and wandb_logging in ["default", "auto"]:
            wandb_logging = os.name == "posix"  # do wandb_logging on linux, but not on windows per default
        if wandb_logging:
            try:
                wandb_logger = WAndBWrapper(config=config, recording_dir=recording_dir, runname=runname)
            except Exception as e:
                self.logger.warning("Error with wandb logging: {}".format(e))
        return wandb_logger

    def _initialize_task_visualization(self, **kwargs):
        algorithm = kwargs.get("algorithm")
        environment = kwargs.get("environment")
        environment_data = kwargs.get("environment_data")
        config = kwargs.get("config")
        context_recording_dict = config.get("recording").get("pseudo_contextual")
        self._task_visualization = EnvironmentVisualization(algorithm=algorithm,
                                                            plot_environment=environment,
                                                            environment_specifics=environment_data.environment_specifics,
                                                            context_recording_dict=context_recording_dict,
                                                            data_dict=environment_data.data_dict)

    def initialize_recording(self):
        self._initialize_visualization()
        self._save_model_checkpoints()

    def _initialize_visualization(self):
        for mode in self._visualization_modes:  # any combination of [d.TRAIN, "evaluation", d.TEST]
            try:
                self._plot_wrapper(name="task_vis",
                                   plot_function=partial(self._task_visualization.__call__, mode=mode),
                                   iteration=self._iteration,
                                   subdir=os.path.join(d.TASK_VISUALIZATION_DIRECTORY, mode),
                                   save_name="init", set_figure_in_function=True)
            except Exception as e:
                self.logger.warning("Error with plotting {} policies: {}".format(mode, e))

    def __call__(self):
        """
        Logs the current training iteration based on the state of self._algorithm
        Returns:

        """
        self._iteration += 1  # starts at -1 so the first iteration is 0
        self._record_timestamp()

        if self._iteration == 0 and isinstance(self._algorithm, VIGOR):
            self.logger.info("Recording Network Schematic")
            self._record_network_architecture()

        if self._iteration == 0 or self._iteration % self._checkpoint_save_frequency == 0:
            # we only save the checkpoints every self._checkpoint_save_frequency iterations to save on some compute
            # and disc storage
            self.logger.info("Saving Models")
            self._save_model_checkpoints()

        current_metrics = self._record_iteration()

        if self._wandb_logger is not None:
            try:
                self.logger.info("Logging metrics to WAndB")
                self._wandb_logger.log(metrics=current_metrics, iteration=self._iteration)
            except Exception as e:
                self.logger.warning("Error with wandb logging: {}".format(e))

        self.logger.info("--- Finished Recording Iteration {:5d} ---".format(self._iteration))
        return current_metrics

    def _record_iteration(self) -> np.array:
        self.logger.info("Calculating Metrics")

        current_metrics = self._record_metrics()

        self._visualize_task()
        self._visualize_additional_plots()
        return current_metrics

    def _record_metrics(self) -> ValueDict:
        recorded_metrics = self._metric_recording.__call__()
        np.savez_compressed(os.path.join(self._recording_dir, d.METRIC_FILE_NAME + ".npz"), **recorded_metrics)
        self._plot_wrapper(name="metrics", plot_function=self._metric_recording.plot, set_figure_in_function=True)
        current_metrics = self._metric_recording.get_current_metrics()

        self._plot_wrapper(name="multimetrics",
                           plot_function=self._metric_recording.multi_plot,
                           set_figure_in_function=True)
        multi_metrics = self._metric_recording.get_current_multi_metrics()
        return {**current_metrics, **multi_metrics}

    def _visualize_task(self):
        self.logger.info("Plotting Task Visualization")
        for mode in self._visualization_modes:  # any combination of [d.TRAIN, "evaluation", d.TEST]
            try:
                self._plot_wrapper(name="task_vis",
                                   plot_function=lambda x: self._task_visualization.__call__(x, mode=mode),
                                   iteration=self._iteration,
                                   subdir=os.path.join(d.TASK_VISUALIZATION_DIRECTORY, mode),
                                   save_name="{:04d}".format(self._iteration), set_figure_in_function=True)
            except Exception as e:
                self.logger.warning("Error with plotting {} policies: {}".format(mode, e))

    def _visualize_additional_plots(self):
        additional_plot_list = self._get_additional_plot_list()
        for additional_plot in additional_plot_list:
            try:
                if not additional_plot.is_expensive or self._draw_expensive_additional_plots:
                    if additional_plot.uses_iteration_wise_figures:
                        save_name = "{:04d}".format(self._iteration)
                        subdir = os.path.join(d.ADDITIONAL_PLOT_DIR, additional_plot.name.title())
                        add_title = True
                    else:
                        save_name = additional_plot.name
                        subdir = d.ADDITIONAL_PLOT_DIR
                        add_title = False

                    if additional_plot.is_policy_based:
                        plot_function = partial(self._additional_plot_wrapper.policy_based_plot_wrapper,
                                                additional_plot=additional_plot)
                        add_title = False
                    else:
                        plot_function = additional_plot.function

                    self.logger.info(f"Plotting {additional_plot.name.title()}")
                    self._plot_wrapper(name=additional_plot.name,
                                       plot_function=plot_function,
                                       iteration=self._iteration,
                                       subdir=subdir,
                                       save_name=save_name,
                                       set_figure_in_function=True,
                                       add_title=add_title)
                    # todo this currently raises MatPlotLib warnings, presumably because the figures are not set correctly
                    #   the code itself works fine, however.

            except Exception as e:
                self.logger.warning("Error with additional plot '{}': '{}'".format(additional_plot.name, e))

    def _get_additional_plot_list(self) -> List[AdditionalPlot]:
        """
        Collect plots from policy and environment
        Returns:

        """
        environment_additional_plots = self._environment.get_additional_plots()
        algorithm_additional_plots = self._algorithm.get_additional_plots()
        return environment_additional_plots + algorithm_additional_plots

    def _record_network_architecture(self):
        if is_vigor(self._algorithm):
            self._algorithm: VIGOR
            self.logger.info("Network architecture: {}".format(self._algorithm.dre_networks[0]))
            try:
                total_network_parameters = sum(p.numel()
                                               for p in self._algorithm.dre_networks[0].parameters())
                trainable_parameters = sum(p.numel()
                                           for p in self._algorithm.dre_networks[0].parameters() if p.requires_grad)
                self.logger.info(f"Total parameters: {total_network_parameters}")
                self.logger.info(f"Trainable parameters: {trainable_parameters}")
            except Exception as e:
                print(f"Error counting network parameters: {e}")

            if isinstance(self._algorithm, DRex):
                reward_parameters = self._algorithm.reward_parameters
                self.logger.info("Number of regression networks: {}".format(len(reward_parameters.get("networks"))))
                self.logger.info("Regression Network Architecture: {}".format(reward_parameters.get("networks")[0]))

    def _record_timestamp(self):
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%m.%d %H:%M")
        self.logger.info("--- Iteration {:5d}: {} ---".format(self._iteration, timestamp))

    def _save_model_checkpoints(self):
        """
        Saves the EIM policy and possibly a reward model in a designated directory of the current run.
        :return:
        """
        """
        Saves the EIM policy and possibly a reward model in a designated directory of the current run.
        :return:
        """
        assert isinstance(self._algorithm, VIGOR)

        all_policies = {}
        for mode in [d.TRAIN, d.VALIDATION, d.TEST, d.DREX_TRAIN]:
            policies = self._algorithm.policies.get(mode)
            if policies is not None and len(policies) > 0:
                policies = {idx: policy
                            for idx, policy in zip(self._data_dict.get(f"{mode}_context_ids"),
                                                   self._algorithm.policies.get(mode))}
            else:
                policies = {}
            all_policies[mode] = policies

        from recording.RecordingUtil import is_vigor
        if is_vigor(self._algorithm):
            discriminators = self._algorithm.dre_networks
        else:
            discriminators = None

        if isinstance(self._algorithm, DRex):
            reward_parameters = self._algorithm.reward_parameters
        else:
            reward_parameters = None

        self._save_utility.save(iteration=self._iteration,
                                train_policies=all_policies.get(d.TRAIN),
                                validation_policies=all_policies.get(d.VALIDATION),
                                test_policies=all_policies.get(d.TEST),
                                drex_train_policies=all_policies.get(d.DREX_TRAIN),
                                discriminators=discriminators,
                                reward_parameters=reward_parameters)

    def _plot_wrapper(self, name: str, plot_function: Callable,
                      iteration: Optional[int] = None,
                      figsize: Optional[Tuple[int, int]] = None,
                      subdir: Optional[str] = None,
                      save_name: Optional[str] = None,
                      set_figure_in_function: bool = False,
                      add_title: bool = False):
        """
        Wraps around the functions used for the recording to determine how to plot and where to save the resulting plots
        to.
        Args:
            name: Name of the plot
            plot_function: Function that returns a plot
            iteration: Current algorithm iteration

        Returns:
        """
        if set_figure_in_function:
            if iteration is None:
                fig = plot_function()
            else:
                fig = plot_function(iteration)
        else:
            fig = plt.figure(name, figsize=figsize)
            if iteration is not None:
                plot_function(iteration)
            else:
                plot_function()

        if add_title:
            plt.title(f"{name.title().replace('_', ' ')} {iteration:04d}")

        if isinstance(fig, dict):  # train, test, validation test figures
            for mode, figure in fig.items():
                self._save_figure(fig=figure, name=name, save_name=save_name,
                                  subdir=mode + "_" + subdir if subdir is not None else None)
        else:
            self._save_figure(fig=fig, name=name, save_name=save_name, subdir=subdir)

    def _save_figure(self, fig: plt.Figure, name: str, save_name: str, subdir: str):
        try:
            save_dir = self._recording_dir
            if subdir is not None:
                save_dir = os.path.join(save_dir, subdir)
                if not os.path.exists(save_dir):
                    path = Path(save_dir)
                    path.mkdir(parents=True)
            if save_name is None:
                save_name = name

            if fig is not None:
                fig.savefig(os.path.join(save_dir, save_name + ".{}".format(self._save_format)),
                            format=self._save_format, dpi=self._img_dpi,
                            transparent=True, bbox_inches='tight', pad_inches=0.0)
        except Exception as e:
            self.logger.warning("Error plotting: {}".format(e))
        finally:
            try:
                plt.close(fig)
            except Exception as e:
                self.logger.warning("Error closing figure: {}".format(e))

    def finalize(self):
        recorded_metrics = self._finalize_metrics()
        np.savez_compressed(os.path.join(self._recording_dir, d.FINAL_METRIC_NAME + ".npz"), **recorded_metrics)

        last_results = parse_last_results(recorded_metrics)
        self._wrap_recording.finalize(results=last_results, final_iteration=self._iteration)

        if self._wandb_logger is not None:
            try:
                self._wandb_logger.finalize()
            except Exception as e:
                print("Error with finalizing wandb logging: {}".format(e))

        self._merge_pdfs()

        self.logger.handlers = []
        return last_results

    def _finalize_metrics(self):
        return self._metric_recording.finalize()

    def _merge_pdfs(self):
        try:
            self.logger.info("Merging PDF files")
            try:
                self.logger.info("Merging visualizations")
                merge_folder_to_file(target_path=os.path.join(self._recording_dir, d.TASK_VISUALIZATION_DIRECTORY),
                                     delete_folder=True, recursion_depth=1, make_video=self._make_videos)
            except Exception as e:
                self.logger.warning("Error with merging visualizations: {}".format(e))

            try:
                for potential_prefix in ["", "train_", "test_", "validation_"]:
                    if os.path.isdir(os.path.join(self._recording_dir, potential_prefix + d.ADDITIONAL_PLOT_DIR)):
                        self.logger.info("Merging {}additional_plots".format(potential_prefix))
                        additional_plot_dir = os.path.join(self._recording_dir,
                                                           potential_prefix + d.ADDITIONAL_PLOT_DIR)
                        merge_folder_to_file(target_path=additional_plot_dir, delete_folder=True,
                                             recursion_depth=1, make_video=self._make_videos)
            except Exception as e:
                self.logger.warning("Error with merging additional plots: {}".format(e))
        except Exception as e:
            self.logger.warning("Error with merging PDF files: {}".format(e))
