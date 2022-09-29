from dataclasses import dataclass
from util.Types import *
from algorithms.distributions.GMM import GMM


@dataclass
class AdditionalPlot:
    """
    Class for maintaining an additional plot function and flags/utility to go with it
    """
    function: Callable[[GMM], None]
    is_policy_based: bool
    uses_iteration_wise_figures: bool
    is_expensive: bool
    uses_context: bool = False
    uses_context_id: bool = False
    custom_name: Optional[str] = None
    projection: Optional[str] = None

    @property
    def name(self):
        return self.custom_name if self.custom_name is not None else self.function.__name__
