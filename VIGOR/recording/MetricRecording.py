from timeit import default_timer as timer
from algorithms.DRex import DRex
from util.Plot import draw_subplot
import os
import numpy as np
import matplotlib.pyplot as plt
from recording.LoggingUtil import get_logger
import util.Defaults as d
from util.Types import *
from util.Functions import symlog_spacing, format_title, get_from_nested_dict, prefix_keys
from algorithms.VIGOR import VIGOR
from environments.EnvironmentData import EnvironmentData
from util.colors.WheelColors import WheelColors
from matplotlib import ticker as ticker


class MetricRecording:
    """
        Basic wrapper for the recording of different metrics.
        Is initialized using a dictionary of all metrics that are to be recorded and will
        record all these metrics whenever called
    """

    def __init__(self, algorithm: VIGOR,
                 environment_data: EnvironmentData,
                 recording_dir: str):
        self._environment_specifics = environment_data.environment_specifics
        self._environment = self._environment_specifics.get("environment")
        self._algorithm = algorithm
        self._logger = get_logger(name="Metrics", path=recording_dir)
        self._recorded_metrics = {}
        self._previous_time = timer()

        self._recorded_multi_metrics = {}
        self._recorded_multi_means = {}  # scalar version of the metrics above
        assert isinstance(environment_data, EnvironmentData)
        self._contexts = {d.TRAIN: environment_data.train_contexts,
                          d.VALIDATION: environment_data.validation_contexts,
                          d.TEST: environment_data.test_contexts}

        self._eval_modes = []  # which modes (train, validation, test) to evaluate
        pseudo_contextual_recording_dict = get_from_nested_dict(environment_data.config,
                                                                list_of_keys=["recording", "pseudo_contextual"],
                                                                raise_error=True)
        record_train_policies = pseudo_contextual_recording_dict.get("record_train_policies", True)
        if record_train_policies:  # must always have more than 1 train context
            self._eval_modes.append(d.TRAIN)

        record_validation_policies = pseudo_contextual_recording_dict.get("record_validation_policies", False)
        if record_validation_policies and len(self._contexts.get(d.VALIDATION)) > 0:
            self._eval_modes.append(d.VALIDATION)

        record_test_policies = pseudo_contextual_recording_dict.get("record_test_policies", False)
        if record_test_policies and len(self._contexts.get(d.TEST)) > 0:
            self._eval_modes.append(d.TEST)

        if isinstance(self._algorithm, DRex):
            self._contexts[d.DREX_TRAIN] = environment_data.drex_train_contexts
            record_drex_train_policies = pseudo_contextual_recording_dict.get("record_drex_train_policies", False)
            if record_drex_train_policies and len(self._contexts.get(d.DREX_TRAIN)) > 0:
                self._eval_modes.append(d.DREX_TRAIN)

    def __call__(self):
        """
        Logs the current training iteration by requesting metrics from the algorithm and the environment.
        These metrics can be defined in a function 'get_metrics()', that takes no arguments for the algorithm, and
        the current algorithm policy for the environment. They must be returned as part of a dictionary
        {metric_name: scalar_evaluation}
        """
        default_metrics = prefix_keys(self._get_default_metrics(), prefix="default")

        if isinstance(self._algorithm, DRex):
            metrics = default_metrics
        elif isinstance(self._algorithm, VIGOR):
            # regular VIGOR metrics do not include any metrics that depend on the policy or environment
            dre_metrics = prefix_keys(self._algorithm.get_dre_metrics(), ["algorithm", "density_ratio_estimator"])
            metrics = {**default_metrics, **dre_metrics}
        else:
            algorithm_metrics = prefix_keys(self._algorithm.get_metrics(), prefix="algorithm")
            environment_metrics = prefix_keys(self._environment.get_metrics(policy=self._algorithm.policy),
                                              prefix="environment")
            metrics = {**default_metrics, **algorithm_metrics, **environment_metrics}

        for metric_name, evaluation in metrics.items():
            self._record_evaluation(metric_name=metric_name, current_evaluation=evaluation)

        # multimetrics
        for eval_mode in self._eval_modes:  # for train/validation/test data
            policies = self._algorithm.policies.get(eval_mode)
            contexts = self._contexts.get(eval_mode)

            metrics = {}
            for context, policy in zip(contexts, policies):
                context_id = self._get_context_id(context=context)
                current_environment_metrics = prefix_keys(dictionary=self._environment.get_metrics(policy=policy,
                                                                                                   context_id=context_id),
                                                          prefix="environment")
                current_algorithm_metrics = prefix_keys(dictionary=self._algorithm.get_policy_metrics(policy=policy,
                                                                                                      context=context),
                                                        prefix="algorithm")
                current_metrics = {**current_algorithm_metrics, **current_environment_metrics}

                for key, value in current_metrics.items():
                    if key not in metrics.keys():
                        metrics[key] = []
                    metrics[key].append(value)

            for key, value in metrics.items():
                self._record_evaluation(metric_name=key, current_evaluation=value, eval_mode=eval_mode)

        return {**self._recorded_multi_metrics, **self._recorded_metrics}

    def get_current_metrics(self) -> Dict[Key, Any]:
        current_metrics = {k: v[-1] for k, v in self._recorded_metrics.items()}
        return current_metrics

    def get_current_multi_metrics(self) -> dict:
        current_multi_metrics = {key: value[-1] for key, value in self._recorded_multi_means.items()}
        return current_multi_metrics

    def plot(self) -> plt.Figure:
        axis = None

        fig = plt.figure("metrics", figsize=(12, len(self._recorded_metrics)))
        for position, (metric_name, metric_evaluations) in enumerate(sorted(self._recorded_metrics.items())):
            title = format_title(title=metric_name)
            data = np.array(metric_evaluations)
            symlog = "elbo" in metric_name
            axis, _ = draw_subplot(num_subplots=len(self._recorded_metrics),
                                   position=position,
                                   title=title,
                                   data=data,
                                   symlog=symlog,
                                   shared_axis=axis,
                                   num_y_ticks=d.Y_AXIS_TICKS,
                                   xlabel="Iterations")
        plt.tight_layout()  # make sure that the subplots fit into the figure
        return fig

    def multi_plot(self):
        fig = plt.figure("metrics", figsize=(12, len(self._recorded_multi_metrics)))
        old_axis = None
        num_contexts = len(self._contexts.get(d.TRAIN)) + \
                       len(self._contexts.get(d.VALIDATION)) + \
                       len(self._contexts.get(d.TEST))
        colors = WheelColors(num_colors=num_contexts)
        for position, (metric_name, metric_evaluation) in enumerate(sorted(self._recorded_multi_metrics.items())):
            title = format_title(title=metric_name)
            ax: plt.Axes = plt.subplot(len(self._recorded_multi_metrics) + 2, 1, position + 1, sharex=old_axis)
            ax.grid(b=True, which="major", color="lightgray", linestyle="-")

            #  shared x ticks
            if old_axis is None:  # the first plot
                ax.set_xlabel("Iteration")
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()
                ax.tick_params(axis="x", which="major", bottom=False, top=True)
            elif position + 1 < len(
                    self._recorded_multi_metrics) + 1:  # in-between plots have no labels but ticks pointing inwards
                transparent_color = (1, 1, 1, 0)
                ax.tick_params(axis="x", direction="in", which="major", bottom=True, top=True, pad=-15,
                               labelcolor=transparent_color, labelsize=0, zorder=100)
            else:  # last plot
                ax.set_xlabel("Iteration")

            # y axis spacing
            ax.tick_params(labelright=True)
            use_symlog = "elbo" in metric_name
            if use_symlog:  # rescale appropriate metrics for nicer vis
                plt.yscale("symlog", linthresh=1e-4)
                data = np.concatenate([metric_evaluation[eval_mode]
                                       for eval_mode in self._eval_modes],
                                      axis=1)
                spacing = symlog_spacing(data=data, num_positions=5, epsilon=0.1)
                plt.yticks(spacing)
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))

            # setting label
            h = plt.ylabel(title)
            h.set_rotation(0)
            ax.yaxis.set_label_coords(-0.2, 0.3)

            # actually plot
            x_positions = np.arange(len(metric_evaluation[self._eval_modes[0]]))
            # essentially gets the number of iterations

            for eval_mode in self._eval_modes:
                data = np.array(metric_evaluation[eval_mode]).T
                if eval_mode == d.TRAIN:
                    linestyle = "-"
                elif eval_mode == d.VALIDATION:
                    linestyle = "-."
                elif eval_mode == d.TEST:
                    linestyle = "--"
                elif eval_mode == d.DREX_TRAIN:
                    linestyle = ":"
                else:
                    raise ValueError("Unknown eval_mode {}".format(eval_mode))
                for policy_position, policy_data in enumerate(data):
                    if eval_mode == d.VALIDATION:
                        policy_color_id = policy_position + len(self._contexts.get(d.TRAIN))
                    elif eval_mode == d.TEST:
                        policy_color_id = policy_position + \
                                          len(self._contexts.get(d.TRAIN)) + \
                                          len(self._contexts.get(d.VALIDATION))
                    else:
                        policy_color_id = policy_position
                    if len(x_positions) == len(policy_data):
                        plt.plot(x_positions, policy_data, color=colors(policy_color_id), alpha=0.4,
                                 linestyle=linestyle)
                if eval_mode + "/" + metric_name in self._recorded_multi_metrics:
                    plt.plot(x_positions, self._recorded_multi_means[eval_mode + "/" + metric_name], color="black",
                             linestyle=linestyle)
            old_axis = ax

        plt.Axes = plt.subplot(len(self._recorded_multi_metrics) + 2, 1, len(self._recorded_multi_metrics) + 2)
        plt.axis("off")
        plt.legend([plt.Line2D([0], [0], color="black", linewidth=4, marker="o", linestyle=""),
                    plt.Line2D([0], [0], color="black", linewidth=4),
                    plt.Line2D([0], [0], color="black", linewidth=4, linestyle="-."),
                    plt.Line2D([0], [0], color="black", linewidth=4, linestyle="--")],
                   ["Means", "Train Context", "Validation Context", "Test Context"],
                   ncol=4, fontsize="large", loc="center")
        colors.draw_colorbar(label="Policy Position", horizontal=True)
        plt.tight_layout()  # makes the subplots fit into the figure
        return fig

    def _record_evaluation(self, metric_name: str, current_evaluation: List[np.ndarray],
                           eval_mode: Optional[str] = None):
        mean_evaluation = np.mean(current_evaluation)
        if eval_mode is None:
            self._logger.info(f"{format_title(metric_name, linebreaks=False):<80} {mean_evaluation:.4f}")
        else:
            self._logger.info(
                f"{format_title(eval_mode + '_' + metric_name, linebreaks=False):<80} {mean_evaluation:.4f}")

        if eval_mode is None:
            if metric_name not in self._recorded_metrics:
                self._recorded_metrics[metric_name] = []
            self._recorded_metrics[metric_name].append(current_evaluation)
        else:  # multimetric setting
            if metric_name not in self._recorded_multi_metrics.keys():
                self._recorded_multi_metrics[metric_name] = {eval_mode_: [] for eval_mode_ in self._eval_modes}
            self._recorded_multi_metrics[metric_name][eval_mode].append(current_evaluation)

            if eval_mode + "/" + metric_name not in self._recorded_multi_means.keys():
                self._recorded_multi_means[eval_mode + "/" + metric_name] = []
            self._recorded_multi_means[eval_mode + "/" + metric_name].append(mean_evaluation)

    def _get_default_metrics(self) -> ValueDict:
        default_metrics = {"wallclock_duration": self._get_wallclock_duration()}
        if os.name == "posix":  # record memory usage per iteration. Only works on linux
            import resource
            default_metrics["memory_usage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        return default_metrics

    def _get_wallclock_duration(self):
        current_time = timer()
        duration = current_time - self._previous_time
        self._previous_time = current_time
        return duration

    def _get_context_id(self, context: np.array) -> int:
        assert "reverse_context_ids" in self._environment_specifics, \
            "Need to provide inverse id mapping in task specifics"
        return self._environment_specifics.get("reverse_context_ids")[tuple(context.flatten())]

    def finalize(self) -> Dict[Key, Any]:
        self._logger.handlers = []
        return {**self._recorded_metrics, **self._recorded_multi_means}
