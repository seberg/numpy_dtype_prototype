import numpy as np

from .dtypes import (
        DType, BasicNumericDType, UnitDType, dtype_is_finalized,
        discover_dtype, promote_types, get_cast_func, dtype as asdtype)


__all__ = ["array", "discover_arr_dtype"]


class array:
    def __init__(self, array, dtype=None):
        if isinstance(dtype, DType):
            if not dtype_is_finalized(dtype):
                dtype = discover_arr_dtype(array, dtype=dtype)

        elif dtype is None:
            dtype = discover_arr_dtype(array, dtype=dtype)

        assert isinstance(dtype, DType)

        # Hardcode these for now for prototyping:
        if isinstance(dtype, BasicNumericDType):
            np_dtype = dtype._np_dtype
        elif isinstance(dtype, UnitDType):
            np_dtype = dtype._base._np_dtype
        else:
            raise NotImplementedError("did not implement dtype fuzzing!")

        self.arr = np.asarray(array, np_dtype)
        self.dtype = dtype

    def __repr__(self):
        return f"{repr(self.arr)}\n    -- {self.dtype}"

    def astype(self, dtype, casting="unsafe", copy=True):
        dtype = asdtype(dtype)
        new_dtype, func = get_cast_func(self.dtype, dtype, casting)

        new_ndarray = func(self.arr)
        print(new_dtype, new_ndarray)
        return array(new_ndarray, dtype=new_dtype)


def discover_arr_dtype(obj, default_dtype=True, scalar_only=True, dtype=None):
    obj_arr = np.asarray(obj, dtype=object)

    minimal = True if not default_dtype else False  # This seems off for scalar logic...

    if obj_arr.size != 0:
        if dtype is None:
            dtype = discover_dtype(obj_arr.item(0), minimal=minimal)
        for val in obj_arr.flat[0 if dtype is None else 1:]:
            next_dtype = discover_dtype(val, minimal=minimal)
            dtype = promote_types(dtype, next_dtype)
    elif dtype is None:
         raise TypeError("No dtype given, should guess float64...")

    if default_dtype:
        dtype = dtype.casting__default_type()
    elif scalar_only and obj_arr.ndim != 0:
        dtype = dtype.casting__default_type()

    return dtype
