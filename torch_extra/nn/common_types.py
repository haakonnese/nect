from typing import Tuple, TypeVar, Union

T = TypeVar("T")

_scalar_or_tuple_4_t = Union[T, Tuple[T, T, T, T]]
_size_4_t = _scalar_or_tuple_4_t[int]
