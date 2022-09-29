import numpy as np
from util.Types import *
import util.Defaults as d
import matplotlib.pyplot as plt
from algorithms.VIGOR import VIGOR
from recording.environment_visualization.EnvironmentVisualizationUtil import get_plot_title
from environments.AbstractEnvironment import AbstractEnvironment
from environments.panda_reacher.PandaReacher import PandaReacher


class EnvironmentVisualization:
    """
        Wrapper for visualizing multiple different contexts for the same overall kind of task.
        For example, one may have different targets for a point reaching robot or different hole positions for the
        hole reacher.
    """

    def __init__(self, algorithm: VIGOR, plot_environment: AbstractEnvironment,
                 data_dict: ConfigDict, environment_specifics: Dict[Key, Any],
                 context_recording_dict: Dict[str, Any]):
        """

        Args:
            algorithm: In instance of VIGOR or its derivates EIM and VIRL
            environment_specifics: A dictionary over (potentially contextual) information of the task
        """
        self._algorithm = algorithm
        self._environment_specifics = environment_specifics

        self._plot_environment = plot_environment

        self._contexts = {}
        self._context_ids = {}

        for mode in [d.TRAIN, d.VALIDATION, d.TEST, d.DREX_TRAIN]:
            num_plotted_contexts = context_recording_dict.get(f"plotted_{mode}_contexts")
            self._contexts[mode] = data_dict.get(f"{mode}_contexts")[:num_plotted_contexts]

            context_ids = data_dict.get(f"{mode}_context_ids")
            if context_ids is None:
                context_ids = []
            self._context_ids[mode] = context_ids[:num_plotted_contexts]

        self.projection_mode = "3d" if isinstance(plot_environment, PandaReacher) else None

    def __call__(self, iteration: int, mode: str = d.TRAIN) -> plt.figure:
        """
        Logs the current training iteration. For this, all necessary components of the recorded algorithm must
        be provided.
        Args:
            iteration: The outer iteration of the recorded algorithm
            mode: Whether this call refers to plotting the training data "train", the "validation" or the "test" data.

        Returns: A figure separated in a number of subplots where each subplot corresponds to one context
        """
        contexts = self._contexts.get(mode)
        context_ids = self._context_ids.get(mode)
        title_prefix = mode.title()

        num_contexts = len(contexts)
        assert num_contexts >= 1, "need at least 1 context for plotting"
        num_horizontal_plots = int(np.min((3, num_contexts)))
        num_vertical_plots = int(np.ceil(num_contexts / 3))

        fig = plt.figure(figsize=(num_horizontal_plots * 8, np.min((40, num_vertical_plots * 6))))
        fig.suptitle(get_plot_title(iteration=iteration, prefix=title_prefix))

        for position, (context, context_id) in enumerate(zip(contexts, context_ids)):
            fig.add_subplot(num_vertical_plots, num_horizontal_plots, position + 1, projection=self.projection_mode)

            policy = self._algorithm.policies.get(mode)[position]
            self._plot_environment.plot(policy=policy, context_id=context_id)
            plt.title("Position {}: {}".format(position, np.round(context, 2)))
        return fig

    def _get_context_id(self, context: np.array) -> int:
        assert "reverse_context_ids" in self._environment_specifics, "Need to provide inverse id mapping in task specifics"
        return self._environment_specifics.get("reverse_context_ids")[tuple(context.flatten())]
