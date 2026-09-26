"""Value <-> bucket-index mappings, mirroring `ddsketch.mapping`.

Upstream computes one key per Python call; here the same expressions run over
whole arrays inside Mojo (`dds_keys_*` / `dds_values_*`). The scalar `key` and
`value` methods are kept for API parity and go through the same kernels on
length-1 arrays, so there is exactly one implementation of each formula.
"""

import math
import sys

import numpy as np

from . import _lib

_FLOAT_INFO_MIN = sys.float_info.min
_FLOAT_INFO_MAX = sys.float_info.max


class KeyMapping:
    """Base class: the arithmetic shared by every mapping.

    Attributes match upstream: `gamma`, `min_possible`, `max_possible`.
    """

    #: selects the flavour in the scalar kernel; 0 log, 1 linear, 2 cubic
    _mode = 0

    #: name of the Mojo symbol pair this mapping dispatches to
    _pair = ""

    def __init__(self, relative_accuracy, offset=0.0):
        if relative_accuracy <= 0 or relative_accuracy >= 1:
            raise ValueError(
                "Relative accuracy must be between 0 and 1, got %r"
                % relative_accuracy
            )
        self.relative_accuracy = relative_accuracy
        self._offset = float(offset)

        gamma_mantissa = 2 * relative_accuracy / (1 - relative_accuracy)
        self.gamma = 1 + gamma_mantissa
        self._multiplier = 1 / math.log1p(gamma_mantissa)
        self.min_possible = _FLOAT_INFO_MIN * self.gamma
        self.max_possible = _FLOAT_INFO_MAX / self.gamma

    @classmethod
    def from_gamma_offset(cls, gamma, offset=0.0):
        return cls((gamma - 1.0) / (gamma + 1.0), offset=offset)

    @property
    def scale(self) -> float:
        """Upstream's `2 / (1 + gamma)` factor applied to every bucket value."""
        return 2.0 / (1 + self.gamma)

    def key(self, value) -> int:
        return _lib.key_scalar(value, self._mode, self._multiplier,
                               self._offset)

    def value(self, key) -> float:
        return _lib.value_scalar(key, self._mode, self._multiplier,
                                 self._offset, self.scale)

    def keys(self, values) -> np.ndarray:
        raise NotImplementedError

    def values(self, keys) -> np.ndarray:
        raise NotImplementedError


class LogarithmicMapping(KeyMapping):
    """Memory-optimal mapping: log2 of the value times the multiplier."""

    _pair = "log"

    def __init__(self, relative_accuracy, offset=0.0):
        super().__init__(relative_accuracy, offset=offset)
        self._multiplier *= math.log(2)

    def keys(self, values) -> np.ndarray:
        return _lib.keys_log(values, self._multiplier, self._offset)

    def values(self, keys) -> np.ndarray:
        return _lib.values_log(keys, self._multiplier, self._offset, self.scale)


class LinearlyInterpolatedMapping(KeyMapping):
    """log2 approximated by linear interpolation of the significand."""

    _mode = 1
    _pair = "linear"

    def keys(self, values) -> np.ndarray:
        return _lib.keys_linear(values, self._multiplier, self._offset)

    def values(self, keys) -> np.ndarray:
        return _lib.values_linear(keys, self._multiplier, self._offset,
                                  self.scale)


class CubicallyInterpolatedMapping(KeyMapping):
    """log2 approximated by a cubic polynomial of the significand."""

    _mode = 2
    _pair = "cubic"

    _A = 6.0 / 35.0
    _B = -3.0 / 5.0
    _C = 10.0 / 7.0

    def __init__(self, relative_accuracy, offset=0.0):
        super().__init__(relative_accuracy, offset=offset)
        self._multiplier /= self._C

    def keys(self, values) -> np.ndarray:
        return _lib.keys_cubic(values, self._multiplier, self._offset)

    def values(self, keys) -> np.ndarray:
        return _lib.values_cubic(keys, self._multiplier, self._offset,
                                 self.scale)
