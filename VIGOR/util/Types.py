from typing import Union, Any, Dict, Tuple, List, Callable, Type, Optional, Iterable
from numpy import array as np_array
from torch import Tensor
from functools import partial

MappingFunction = Union[
    Callable[[np_array], Union[List[np_array], np_array]],
    Callable[[np_array, np_array], Union[List[np_array], np_array]]
]
LossFunction = Union[
    Callable[[Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor]
]
Key = Union[str, int, Tuple]
ConfigDict = Dict[Key, Any]
ValueDict = Dict[Key, Any]
