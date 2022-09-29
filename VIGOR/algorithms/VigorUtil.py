from functools import partial
import numpy as np

def get_dre_aggregation(num_dres: int, dre_aggregation_str: str) -> callable:
    if num_dres == 1 or dre_aggregation_str == "mean":
        # if we only have 1 dre, all aggregation methods are the same
        dre_aggregation = partial(np.mean, axis=0)
    elif dre_aggregation_str == "min":
        dre_aggregation = partial(np.min, axis=0)
    elif dre_aggregation_str == "median":
        dre_aggregation = partial(np.median, axis=0)
    elif dre_aggregation_str == "max":
        dre_aggregation = partial(np.max, axis=0)
    else:
        raise ValueError(f"Unknown dre aggregation method '{dre_aggregation_str}'")
    return dre_aggregation
