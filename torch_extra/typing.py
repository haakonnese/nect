import builtins
from typing import List, Tuple, Union

import torch

_int = builtins.int
_size = Union[torch.Size, List[_int], Tuple[_int, ...]]
