import matplotlib.pyplot as plt

import numpy as np
from algorithms.VIGOR import VIGOR
from environments.EnvironmentData import EnvironmentData
from algorithms.DRex import DRex
from util.Types import *
from util.Functions import get_from_nested_dict
from util.Defaults import TRAIN, VALIDATION, TEST, DREX_TRAIN
from recording.AdditionalPlot import AdditionalPlot


class AdditionalPlotWrapper:

    def __init__(self, algorithm: VIGOR, environment_data: EnvironmentData, ):
        self._algorithm = algorithm
        self._config = environment_data.config

        self._reverse_context_ids = environment_data.environment_specifics.get("reverse_context_ids")

        if isinstance(self._algorithm, VIGOR):
            pseudo_contextual_recording_dict = get_from_nested_dict(environment_data.config,
                                                                    list_of_keys=["recording", "pseudo_contextual"],
                                                                    raise_error=True)
            self._record_policies = {mode: pseudo_contextual_recording_dict.get(f"record_{mode}_policies")
                                     for mode in [TRAIN, VALIDATION, TEST]}

            if isinstance(self._algorithm, DRex):
                self._record_policies[DREX_TRAIN] = pseudo_contextual_recording_dict.get(f"record_{DREX_TRAIN}_policies")

            self._num_plotted_contexts = {}
            for mode, record_mode in self._record_policies.items():
                if record_mode:
                    num_plotted_contexts = pseudo_contextual_recording_dict.get("plotted_{}_contexts".format(mode), 3)
                    num_contexts = len(self._algorithm.contexts.get(mode))
                    self._num_plotted_contexts[mode] = np.minimum(num_plotted_contexts, num_contexts)

        else:
            self._record_policies = {TRAIN: True}
            self._num_plotted_contexts = {TRAIN: 1}

    def policy_based_plot_wrapper(self, iteration: int,
                                  additional_plot: AdditionalPlot) -> Union[plt.Figure, Dict[Key, plt.Figure]]:
        """
        Wraps all policy_based plots in a figure. If the task is pseudo-contextual, then num_plotted_train_contexts
        plots will be plotted in a grid. Otherwise, the (non-contextual) policy of the algorithm will be used for
        a sole plot
        Args:
            iteration: The iteration of the algorithm
            additional_plot: The additional plot function to wrap the figure around.
            Its function will be called once for non-contextual and num_plotted_train_contexts time for
            pseudo-contextual settings

        Returns: A plt.fig() instance containing whatever plot_function_to_wrap plotted in it

        """
        if isinstance(self._algorithm, VIGOR):  # multiple contexts!
            figure_dict = {}
            for mode in self._num_plotted_contexts.keys():
                figure_dict[mode] = self._policy_based_multi_plot_wrapper(iteration=iteration,
                                                                          additional_plot=additional_plot,
                                                                          mode=mode)
            return figure_dict
        else:
            fig = plt.figure()
            if additional_plot.projection is not None:
                fig.add_subplot(111, projection=additional_plot.projection)
            plot_function_to_wrap = additional_plot.function
            title = plot_function_to_wrap(policy=self._algorithm.policy)
            plt.title(title)
            return fig

    def _policy_based_multi_plot_wrapper(self, iteration: int, additional_plot: AdditionalPlot,
                                         mode: str = TRAIN) -> plt.Figure:
        """
        Wraps all policy_based plots in a figure. If the task is pseudo-contextual, then num_plotted_train_contexts
        plots will be plotted in a grid. Otherwise, the (non-contextual) policy of the algorithm will be used for
        a sole plot
        Args:
            iteration: The iteration of the algorithm
            additional_plot: The additional_plot to wrap the figure around.
             Its function will be called once for non-contextual and
             num_plotted_MODE_contexts time for pseudo-contextual settings

        Returns: A plt.fig() instance containing whatever plot_function_to_wrap plotted in it

        """
        assert isinstance(self._algorithm, VIGOR)
        plot_function_to_wrap = additional_plot.function

        num_plotted_contexts = self._num_plotted_contexts.get(mode)

        policies = self._algorithm.policies.get(mode)[:num_plotted_contexts]
        contexts = self._algorithm.contexts.get(mode)[:num_plotted_contexts]
        context_ids = [self._reverse_context_ids.get(tuple(context.flatten())) for context in contexts]

        num_horizontal_plots = int(np.min((3, num_plotted_contexts)))
        num_vertical_plots = int(np.ceil(num_plotted_contexts / 3))
        fig = plt.figure(figsize=(num_horizontal_plots * 8, np.min((40, num_vertical_plots * 6))))

        for position, (policy, context, context_id) in enumerate(zip(policies, contexts, context_ids)):
            fig.add_subplot(num_vertical_plots, num_horizontal_plots, position + 1,
                            projection=additional_plot.projection)
            if additional_plot.uses_context:
                plot_function_to_wrap(policy=policy, context=context)
            elif additional_plot.uses_context_id:
                plot_function_to_wrap(policy=policy, context_id=context_id)
            else:
                plot_function_to_wrap(policy=policy)
            plt.title("Context {}: {}".format(context_id, np.round(context, decimals=2)))
        fig.suptitle(f"{additional_plot.name.title()} at iteration {iteration}")
        return fig
