import numpy as np
from .ndarray import array, discover_arr_dtype
from . import dtypes as _dtypes


class UFuncImpl:
    def __init__(self, name, signature):
        self.name = name
        self._signature = signature
        self._ufunc = getattr(np.core.umath, name)

    def __repr__(self):
        return f"<UFuncImpl MockUp for {self.name} {self.signature}>"

    def __call__(self, dtypes, in_arrs, out_arrs):
        # Would be overloaded, or a default implementation based on an inner
        # loop. (I.e. a subclass of UFuncImpl which uses the current machinery
        # of inner loop definitions plus setup, etc.)
        # Not sure if the dtypes would be passed in here explicitly.
        np_in_arrs = [arr.arr for arr in in_arrs]
        np_out_arrs = [None if arr is None else arr.arr for arr in out_arrs]

        np_out_arrs = self._ufunc(*np_in_arrs, out=np_out_arrs[0],
                                  signature=self._signature)

        if isinstance(np_out_arrs, tuple):
            raise NotImplementedError("multiple outputs not implemented")

        out_dtype = dtypes[-1]

        return array(np_out_arrs, out_dtype)


class UFunc:
    def __init__(self, name):
        # More stuff (what we have now)
        self.name = name
        self._loops = {}

    def __repr__(self):
        return f"umath.{self.name}"

    def _register_loop(self, types, resolver):
        if types in self._loops:
            raise RuntimeError("multipel identical resolvers not allowed.")
        self._loops[types] = resolver

    def resolve_loop(self, dtypes, specified_dtypes=None):
        """This function should be able to cache, at least often!"""
        if specified_dtypes is not None:
            raise NotImplementedError("need to implement specific dtypes!")

        print(dtypes)
        categories = tuple(None if dt is None else dt._dispatch_category for dt in dtypes)
        print(categories)

        for cats, resolver in self._loops.items():
            # just find first match (for now)
            for c1, c2 in zip(categories, cats):
                if c1 == c2 or c1 is None or c2 is None:
                   continue
                break
            else:
                return resolver(dtypes)

        raise TypeError(f"no matching loop found for {tuple(dtypes)}.")

    def __call__(self, arr1, arr2, out=None):
        arrs = [arr1, arr2, out]

        dtypes = [None, None, None]
        discovered = [False, False, False]

        for i, arr in enumerate(arrs):
            if arr is None:
                continue

            if isinstance(arr, array):
                dtypes[i] = arr.dtype
            else:
                dtypes[i] = discover_arr_dtype(arr, default_dtype=False)
                discovered[i] = True

        # May be cacheable:
        dtypes, ufunc_impl = self.resolve_loop(dtypes)

        for i, disc in enumerate(discovered):
            if not disc:
                continue
            if arrs[i] == None:
                continue
            arrs[i] = array(arrs[i], dtypes[i])

        for dt in dtypes:
            if not _dtypes.dtype_is_finalized(dt):
                raise RuntimeError("dtypes must all be finalized!")

        return ufunc_impl(dtypes, arrs[:2], tuple(arrs[2:]))


def register_loop(ufunc, dtypes):
    def reg(func):
        ufunc._register_loop(dtypes, func)
    return reg


add = UFunc("add")
multiply = UFunc("multiply")


def _simple_uniform_resolver(dtypes):
    if dtypes[-1] is not None:
        common_type = dtypes[-1]
        for dtype in dtypes[:-1]:
            if not dtypes.can_cast(dtypes, common_type, casting="safe"):
                raise TypeError("no loop!")  # should not be possible
    else:
        common_type = _dtypes.promote_types(dtypes[0], dtypes[1])
        common_type.casting__default_type()  # in case we got scalar stuff.

    c = common_type._np_dtype.char
    signature = f"{c}{c}->{c}"
    ufunc_impl = UFuncImpl("add", signature)
    return (common_type,) * 3, ufunc_impl


register_loop(add, (_dtypes.BasicNumericDType,) * 3)(_simple_uniform_resolver)
register_loop(multiply, (_dtypes.BasicNumericDType,) * 3)(_simple_uniform_resolver)


# Unit loops:

@register_loop(add, (_dtypes.UnitDType,) * 3)
def add_unit_resolver(dtypes):
    if dtypes[0]._unit != dtypes[1]._unit:
        raise ValueError("units do not match.")

    unit = dtypes[0]._unit
    if dtypes[-1] is not None and dtypes[-1]._unit != unit:
        raise ValueError("output unit does not match.")

    base_dts = [None if dt is None else dt._base for dt in dtypes]

    dtypes, ufunc_impl = add.resolve_loop(base_dts)
    dtypes = [_dtypes.UnitDType(dt, unit) for dt in dtypes]
    return dtypes, ufunc_impl
