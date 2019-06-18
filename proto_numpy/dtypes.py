"""
Some general notes/TODOs:
-------------------------

1. Making these normal methods, means we need to instanciate classes, etc.
   before using them, even if they are not true finalized dtypes. That
   seems rather OK, e.g. dtype will simply try to convert a class into
   a "non-finalized" instance (this means classes need to support this,
   although they could do so through `__new__` plausibly).
   TODO: At this time at least, I have not create such naive ones for
         normally non-naive classes!
2. The instance checks in the code should probably be exact type checks!
3. Should we even allow multiple levels of inheritance? Can we forbid it?
4. Most of the slots need to be frozen at instanciation time (or even at type
   creation time)!
"""

import numpy as np
import numbers
import warnings


def _assert_dtypes(*dtypes):
    for dtype in dtypes:
        if not isinstance(dtype, DType):
            raise ValueError(f"{dtype} is not a valid dtype instance.")


def dtype(dtype):
    if isinstance(dtype, DType):
        return dtype

    if isinstance(dtype, str):
        # Fall back to numpy logic here...
        return BasicNumericDType(dtype)

    if isinstance(dtype, type):
        if issubclass(dtype, DType):
            return dtype()  # Try to create a naive version of it.

        if dtype in DTypeMeta._dtype_discovery:
            # a scalar class is associated correctly:
            dtype_category = DTypeMeta._dtype_discovery[dtype]
            # create a naive dtype and then ask for the corresponding default,
            # a bit convoluted, but does that really matter?
            return dtype_category.casting__discover_type_from_class(dtype)


def can_cast(from_dtype, to_dtype, casting="safe"):
    _assert_dtypes(from_dtype, to_dtype)

    first_try = to_dtype.casting__can_cast_to(from_dtype, casting)
    if first_try is not NotImplemented:
        return first_try

    second_try = from_dtype.casting__can_cast_from(to_dtype, casting)
    if second_try is not NotImplemented:
        return second_try

    warnings.warn("no casting is defined for the dtypes "
                  f"{from_dtype} and {to_dtype}.", RuntimeWarning)
    return False


def _raise_no_cast_possible(from_dtype, to_dtype, casting="safe"):
    raise TypeError(f"no casting is defined for {from_dtype} being cast "
                    f"to {to_dtype} under the rule '{casting}'")


def get_cast_func(from_dtype, to_dtype, casting="safe"):
    """
    Currently, returns a function working on numpy arrays, in reality,
    it should be much like a UFuncImpl and should expose at least a strided
    implementation, if not a few other fast implementations.
    """
    # on the C-level, we would typically know the safe casting stuff already
    _assert_dtypes(from_dtype, to_dtype)

    first_try = to_dtype.casting__get_cast_func_to(from_dtype, casting)
    if first_try:
        return first_try
    elif first_try is not NotImplemented:
        _raise_no_cast_possible(from_dtype, to_dtype, casting)

    second_try = from_dtype.casting__get_cast_func_from(to_dtype, casting)
    if second_try:
        return second_try

    _raise_no_cast_possible(from_dtype, to_dtype, casting)


def promote_types(dtype1, dtype2):
    # common type may be a better name, but this is the current one.
    _assert_dtypes(dtype1, dtype2)

    first_try = dtype1.casting__common_type(dtype2)
    if first_try is not NotImplemented:
        return first_try

    second_try = dtype2.casting__common_type(dtype1)
    if second_try is not NotImplemented:
        return second_try

    # TODO: Fall back to self cast logic, but should that atually happen?
    # We need to try/except this,
    if can_cast(dtype2, dtype1, "safe"):
        return dtype1

    if can_cast(dtype1, dtype2, "safe"):
        return dtype2

    raise TypeError(f"{dtype1} and {dtype2} could not be promoted to a "
                      "common dtype.")


class DTypeMeta(type):
    # global registration:
    _dtype_discovery = {}

    def __init__(cls, *args, **kwargs):
        if len(cls.__mro__[1:]) == 1:
            # only object is a base, the base dtype is not a category,
            # nor does it have direct instances...
            pass
        elif cls._dispatch_category is None:
            prev = cls
            for next in cls.__mro__[1:]:
                # I suppose this may break for strange type hierarchy
                # we may want to forbid multiple inheritence if possible.
                if next is DType:
                    cls._dispatch_category = prev
                prev = next

        for pytype in cls._discover_pytypes:
            if pytype in DTypeMeta._dtype_discovery:
                raise TypeError("cannot register two dtypes for one "
                                 "python type.")
            DTypeMeta._dtype_discovery[pytype] = cls

        super().__init__(*args, **kwargs)


def discover_dtype(pyscalar, minimal=False):
    pytype = type(pyscalar)
    if pytype in DTypeMeta._dtype_discovery:
        dtype_category = DTypeMeta._dtype_discovery[pytype]
        dtype = dtype_category.casting__discover_type(pyscalar)
    elif isinstance(pyscalar, numbers.Integral):
        # This is actually somewhat broken, what would we do with ABCs?
        # We may have to loop all options and then add the result to the
        # cache?
        dtype_category = DTypeMeta._dtype_discovery[numbers.Integral]
        dtype = dtype_category.casting__discover_type(pyscalar)
    else:
        raise TypeError(f"cannot automatically coerce {pyscalar} to dtype.")

    if minimal:
        return dtype

    return dtype.casting__default_type()


def dtype_is_finalized(dtype):
    return dtype._finalized


class DType(metaclass=DTypeMeta):
    # Class level slots:
    _dispatch_category = None
    # on the C-side would be registration, maybe Python as well:
    _discover_pytypes = []

    # instance level slots (although they could be finalized/inherited
    # from the class level).
    _finalized = False
    _itemsize = -1
    _byteorder = "="  # TODO: Add this?
    _aligned = True  # TODO: Add this?
    _type = None  # We must have a type association (TODO: Add this)
    # ... many other slots probably

    # TODO: Solve this differently/move to metaclass...
    def finalize_descriptor(self):
        # freeze flexible descriptor slots, these are independend from
        # casting information which are frozen during/at type creation?
        assert self._itemsize >= 0
        self._finalized = True

    @property
    def itemsize(self):
        return self._itemsize

    # Casting information, these would be a single, extensible
    # slot/struct on the C-side, similar to the `tp_number` slot, etc.
    def casting__can_cast_to(self, other, casting="safe"):
        # TODO: Naming is can cast to self from other, which reads ok here
        #       but stranger in the actual code calling this :/
        res = self.casting__get_cast_func_to(other, casting)
        if type(res) is tuple:
            return True  # it returned a new dtype and casting function.
        return res

    def casting__can_cast_from(self, other, casting="safe"):
        res = self.casting__get_cast_func_from(other, casting)
        if type(res) is tuple:
            return True  # it returned a new dtype and casting function.
        return res

    def casting__get_cast_func_to(self, other, casting="safe"):
        """Maybe these should be consolidated into the cast slots themselves.
        The only reason for not doing it is to speed up the type resolution
        step itself, since it does not require casting. Unfortunately, it
        seems like we may sometimes have to figure ou the casting logic twice.
        """
        return NotImplemented

    def casting__get_cast_func_from(self, other, casting="safe"):
        return NotImplemented

    def casting__common_type(self, other):
        """Operator used for promotion/common type operation.
        Making this an operator and not registration means that outside
        packages cannot override it though.
        As a (too simple) example, it is not possible that an outside
        package would define `int16, uint16 -> int24` which is safe to
        do in principle.
        """
        return NotImplemented

    def casting__default_type(self):
        if self._finalized:
            # Finalized will always just be self:
            return self
        raise TypeError(f"flexible/category dtype {self} has no default dtype.")

    @classmethod
    def casting__discover_type(cls, pyscalar):
        # Fall back to non-value based discovery by default:
        return cls.casting__discover_type_from_class(type(pyscalar))

    @classmethod
    def casting__discover_type_from_class(cls, pytype):
        raise TypeError(f"dtype {cls} discovery must be implemented!")

    # We should/could have another private struct on the C-side?
    # although if we use functions to fill up the above, we just could hide
    # them away completely.
    # It seems we need to put them in a struct though, since we cannot grow
    # the dtypes size, if we want to allow C-side subclassing without
    # recompilation


class BasicNumericDType(DType):
    def __init__(self, base=None):
        if base is None:
            # A naive basic dtype, pretty useless.
            self._np_dtype._base = None
            # TODO: It is an interesting issue if/what to do here...
            #       A naive basic type makes sense (not so much here, but
            #       generally, the question of casting from a float64 to
            #       a generic Unit is reasonable. In that case, need to return
            #       a new dtype though!)
            raise NotImplementedError(
                    "Have to think about this, it is interesting...")
            return

        base = np.dtype(base)
        if not issubclass(base.type, np.number):
            raise TypeError("Expect a numpy numerical dtype.")
        self._np_dtype = base
        self._itemsize = base.itemsize
        super().finalize_descriptor()

    def __repr__(self):
        return f"BasicNumericDType({self._np_dtype})"

    def casting__get_cast_func_to(self, other, casting="safe"):
        # Would be easy to provide a fast slot for this one.
        if not isinstance(other, BasicNumericDType):
            return NotImplemented

        if np.can_cast(other._np_dtype, self._np_dtype, casting=casting):
            return self, lambda arr: arr.astype(self._np_dtype)
        else:
            return False

    def casting__common_type(self, other):
        if not isinstance(other, BasicNumericDType):
            return NotImplemented

        return BasicNumericDType(np.promote_types(self._np_dtype, other._np_dtype))


class PyFloatDescriptor(DType):
    # UFuncs dispatch this the same as a BasicNumericDtype...
    # is this too ugly?! Does some subclassing structure make sense?!
    _dispatch_category = BasicNumericDType
    _discover_pytypes = {float}

    def __init__(self, value):
        self._abs_value = abs(float(value))
        self._default = BasicNumericDType(np.float64)

    def __repr__(self):
        return f"PyFloatDescriptor({self._abs_value})"

    def casting_can_cast_from(self, other, casting="safe"):
        # This is oversimplified and needs value based logic like ints!
        # We do not provide any cast functions for this dtype, since it
        # can never be attached to an array (never finalized).
        if can_cast(BasicNumericDType(np.float128), casting=casting):
            return True
        return NotImplemented

    def casting__default_type(self):
        return self._default

    def casting__common_type(self, other):
        if isinstance(other, PyFloatDescriptor):
            abs_value = max(self._abs_value, other._abs_value)
            new = PyFloatDescriptor(abs_value)
            return new
        elif isinstance(other, PyIntDescriptor):
            # ignore value of it for now :).
            return self

        return NotImplemented

    @classmethod
    def casting__discover_type(cls, pyvalue):
        return cls(pyvalue)


class PyIntDescriptor(DType):
    # UFuncs dispatch this the same as a BasicNumericDtype...
    # is this too ugly?! Does some subclassing structure make sense?!
    _dispatch_category = BasicNumericDType
    _discover_pytypes = {int, numbers.Integral}  # May hardcode PyInt...

    def __init__(self, value, max_value=None):
        # Should never be instanciated by a user directly.
        # TODO: need to hide it away?
        if not isinstance(value, int):
            raise ValueError("Can only define this for python integers!")
        self._min_value = value
        if max_value is None:
            self._max_value = value
        elif not isinstance(max_value, int):
            raise ValueError("Can only define this for python integers!")
        else:
            self._max_value = max_value

        if not (np.can_cast(value, "int64") or np.can_cast(value, "uint64")):
            raise TypeError(
                    "value cannot be represented by PyIntDescriptor; "
                    "explicitly use object dtype or somesuch thing.")

        # Likely should actually try intp first, to simplify logic for
        # 32 bit platforms (different discussion), has to be default integer
        # for now, so "long"...
        if np.can_cast(self._max_value, "int64"):
            self._default = BasicNumericDType(np.int64)
        else:
            self._default = BasicNumericDType(np.uint64)

    def __repr__(self):
        return f"PyIntDescriptor({self._min_value}, {self._max_value})"

    def __iter_types(self):
        # Iterate through all plausible types as a fallback,
        # we should only need to do this for the smallest int and uint.
        for width in ["8", "16", "32", "64"]:
            for type_ in ["uint", "int"]:
                np_dtype = np.dtype(type_ + width)
                dtype = BasicNumericDType(np_dtype)
                # Check our own type first:
                if not np.can_cast(self._min_value, np_dtype, casting="safe"):
                    continue
                if not np.can_cast(self._max_value, np_dtype, casting="safe"):
                    continue

                yield np_dtype, dtype

    def casting__can_cast_from(self, other, casting="safe"):
        # Would require faster paths for basic dtypes of course,
        # but generally we ask this second, so if a usertype makes a faster
        # decision, we pick that up.
        # We do not provide any cast functions for this dtype, since it
        # can never be attached to an array (never finalized).
        for np_dtype, dtype in self.__iter_types():
            # Check other dtype:
            if not can_cast(dtype, other, casting=casting):
                continue

            return True

        return NotImplemented

    def casting__default_type(self):
        return self._default

    def casting__common_type(self, other):
        if isinstance(other, PyIntDescriptor):
            min_value = min(self._min_value, other._min_value)
            max_value = max(self._max_value, other._max_value)
            new = PyIntDescriptor(min_value, max_value)
            # TODO: Generally, this will not be blazing fast, to optimize it
            #       a bit, we can allow a non-finalized dtype to be mutable
            #       e.g. a non-finalized string could get a length which
            #       adapts (instead of returning a new string every time).
            return new

        # TODO: User types will have to define this, or can we provide a
        #       reasonable fallback iteration approach?
        #       This would also work if we define `can_cast_to` but that
        #       makes casting slow, since user types should specialize it
        #       normally?
        elif isinstance(other, BasicNumericDType):
            for np_dtype, dtype in self.__iter_types():
                # Check other dtype:
                if not can_cast(other, dtype, casting="safe"):
                    continue

                return dtype

        return NotImplemented

    @classmethod
    def casting__discover_type(cls, pyvalue):
        return cls(pyvalue)


class UnitDType(DType):
    _base = None
    _unit = ""

    def __init__(self, base, unit=None):
        # TODO: Unit is maybe missing the "no-base" flexible dtype
        #       which could even have a unit as to allow:
        #       `float32_arr.astype(UnitDtype(unit="m"))`
        self._base = dtype(base)
        self._itemsize = self._base._itemsize

        self._unit = unit
        if self._unit is not None:
            # If unit is None, this is a flexible dtype!
            super().finalize_descriptor()

    def __repr__(self):
        return f"UnitDType({self._base._np_dtype}, '{self._unit}')"

    def casting__get_cast_func_to(self, other, casting="safe"):
        if isinstance(other, UnitDType):
            if other._unit != self._unit:
                return False
            else:
                res = self._base.casting__get_cast_func_to(other._base, casting)

        elif isinstance(other, BasicNumericDType):
            if casting != "unsafe":
                return False
            res = self._base.casting__get_cast_func_to(other, casting)
        else:
            return NotImplemented

        if not res:
            return res

        new_dtype = UnitDType(res[0], self._unit)
        return new_dtype, res[1]

    def casting__get_cast_func_from(self, other, casting="safe"):
        if not isinstance(other, BasicNumericDType):
            return NotImplemented  # UnitDtype is caught in can_cast_to.

        if casting != "unsafe":
            return False  # Could be True if Unit is empty ""?

        if can_cast(self._base, other, casting):
            return get_cast_func(self._base, other, casting)

    def casting__common_type(self, other):
        if isinstance(other, BasicNumericDType):
            pass
        elif isinstance(other, UnitDType):
            if other._unit != self._unit:
                raise TypeError("can only find common type for identical units.")

            other = other._base
        else:
            return NotImplemented

        new_dtype = promote_types(self._base, other)
        new_dtype = UnitDType(new_dtype)
        return new_dtype(self._unit)  # could return self if already good

