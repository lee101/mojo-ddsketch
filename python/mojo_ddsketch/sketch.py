"""Quantile sketches with relative-error guarantees, mirroring `ddsketch`.

`BaseDDSketch` keeps the upstream structure: a mapping, a positive-value store,
a negative-value store keyed by absolute value, and a zero counter. The
per-value loops are replaced by `add_many` and `get_quantiles`, which run the
mapping and the store arithmetic over whole arrays in Mojo.
"""

import numpy as np

from .mapping import LogarithmicMapping
from .store import (
    CollapsingHighestDenseStore,
    CollapsingLowestDenseStore,
    DenseStore,
)

DEFAULT_REL_ACC = 0.01
DEFAULT_BIN_LIMIT = 2048


class BaseDDSketch:
    """Base implementation: mapping and storage supplied by the subclass."""

    name = "DDSketch"

    def __init__(self, mapping, store, negative_store, zero_count=0.0):
        self._mapping = mapping
        self._store = store
        self._negative_store = negative_store
        self._zero_count = float(zero_count)

        self._relative_accuracy = mapping.relative_accuracy
        self._count = (
            self._negative_store.count + self._zero_count + self._store.count
        )
        self._min = float("inf")
        self._max = float("-inf")
        self._sum = 0.0

    def __repr__(self) -> str:
        return (
            "store: %s, negative_store: %s, zero_count: %s, count: %s, "
            "sum: %s, min: %s, max: %s"
            % (self._store, self._negative_store, self._zero_count, self._count,
               self._sum, self._min, self._max)
        )

    @property
    def count(self) -> float:
        return self._count

    @property
    def num_values(self) -> float:
        return self._count

    @property
    def avg(self) -> float:
        return self._sum / self._count

    @property
    def sum(self) -> float:  # noqa: A003
        return self._sum

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    @property
    def relative_accuracy(self) -> float:
        return self._relative_accuracy

    # -- ingestion ----------------------------------------------------------

    def add(self, val, weight=1.0) -> None:
        """Add one value. See `add_many` for the batched path."""
        if weight <= 0.0:
            raise ValueError("weight must be a positive float, got %r" % weight)
        val = float(val)
        min_possible = self._mapping.min_possible
        if val > min_possible:
            self._store.add(self._mapping.key(val), weight)
        elif val < -min_possible:
            self._negative_store.add(self._mapping.key(-val), weight)
        else:
            self._zero_count += weight

        self._count += weight
        self._sum += val * weight
        if val < self._min:
            self._min = val
        if val > self._max:
            self._max = val

    def add_many(self, values, weights=None) -> None:
        """Add a whole array of values, mapping and accumulating in Mojo.

        This is the ported path: the upstream package has no bulk API and must
        be called once per value.
        """
        values = np.ascontiguousarray(values, dtype=np.float64).ravel()
        if weights is None:
            weights = np.ones(values.size, dtype=np.float64)
        else:
            weights = np.ascontiguousarray(weights, dtype=np.float64).ravel()
            if weights.size != values.size:
                raise ValueError(
                    "weights and values must have the same length, got %d and %d"
                    % (weights.size, values.size)
                )
        if values.size == 0:
            return
        if not np.all(weights > 0.0):
            raise ValueError("weight must be a positive float")

        min_possible = self._mapping.min_possible
        pos = values > min_possible
        neg = values < -min_possible
        zero = ~(pos | neg)

        if pos.any():
            self._store.add_many(self._mapping.keys(values[pos]), weights[pos])
        if neg.any():
            self._negative_store.add_many(
                self._mapping.keys(-values[neg]), weights[neg]
            )
        if zero.any():
            self._zero_count += float(weights[zero].sum())

        self._count += float(weights.sum())
        self._sum += float(np.dot(values, weights))
        self._min = min(self._min, float(values.min()))
        self._max = max(self._max, float(values.max()))

    # -- quantiles ----------------------------------------------------------

    def get_quantile_value(self, quantile):
        """Value at `quantile`, or None for an empty sketch or a rank outside
        [0, 1] -- upstream's contract."""
        if quantile < 0 or quantile > 1 or self._count == 0:
            return None
        return float(self.get_quantiles([quantile])[0])

    def get_quantiles(self, quantiles):
        """Vectorised `get_quantile_value`.

        Returns None for an empty sketch. Ranks outside [0, 1] come back as NaN
        rather than None, because an array has no way to say "no value".
        """
        quantiles = np.ascontiguousarray(quantiles, dtype=np.float64).ravel()
        out = np.full(quantiles.size, np.nan, dtype=np.float64)
        if self._count == 0 or quantiles.size == 0:
            return None if self._count == 0 else out

        in_range = (quantiles >= 0.0) & (quantiles <= 1.0)
        ranks = quantiles * (self._count - 1)
        neg_count = self._negative_store.count
        neg = in_range & (ranks < neg_count)
        zero = in_range & ~neg & (ranks < self._zero_count + neg_count)
        pos = in_range & ~neg & ~zero

        if zero.any():
            out[zero] = 0.0
        if neg.any():
            reversed_rank = neg_count - ranks[neg] - 1.0
            keys = self._negative_store.keys_at_rank(reversed_rank, lower=False)
            out[neg] = -self._mapping.values(keys)
        if pos.any():
            keys = self._store.keys_at_rank(
                ranks[pos] - self._zero_count - neg_count
            )
            out[pos] = self._mapping.values(keys)
        return out

    # -- merge --------------------------------------------------------------

    def _mergeable(self, other) -> bool:
        return self._mapping.gamma == other._mapping.gamma

    def _copy(self, sketch) -> None:
        self._store.copy(sketch._store)
        self._negative_store.copy(sketch._negative_store)
        self._zero_count = sketch._zero_count
        self._min = sketch._min
        self._max = sketch._max
        self._count = sketch._count
        self._sum = sketch._sum

    def merge(self, sketch) -> None:
        if not self._mergeable(sketch):
            raise ValueError(
                "Cannot merge two DDSketches with different parameters, got %r "
                "and %r" % (self._mapping.gamma, sketch._mapping.gamma)
            )
        if sketch.count == 0:
            return
        if self._count == 0:
            self._copy(sketch)
            return

        self._store.merge(sketch._store)
        self._negative_store.merge(sketch._negative_store)
        self._zero_count += sketch._zero_count

        self._count += sketch._count
        self._sum += sketch._sum
        if sketch._min < self._min:
            self._min = sketch._min
        if sketch._max > self._max:
            self._max = sketch._max


class DDSketch(BaseDDSketch):
    """The default implementation: unlimited bins, logarithmic mapping."""

    def __init__(self, relative_accuracy=None):
        if relative_accuracy is None:
            relative_accuracy = DEFAULT_REL_ACC
        super().__init__(
            mapping=LogarithmicMapping(relative_accuracy),
            store=DenseStore(),
            negative_store=DenseStore(),
            zero_count=0.0,
        )


class LogCollapsingLowestDenseDDSketch(BaseDDSketch):
    """Bounded memory: the lowest bins collapse once `bin_limit` is reached."""

    def __init__(self, relative_accuracy=None, bin_limit=None):
        if relative_accuracy is None:
            relative_accuracy = DEFAULT_REL_ACC
        if bin_limit is None or bin_limit < 0:
            bin_limit = DEFAULT_BIN_LIMIT
        super().__init__(
            mapping=LogarithmicMapping(relative_accuracy),
            store=CollapsingLowestDenseStore(bin_limit),
            negative_store=CollapsingLowestDenseStore(bin_limit),
            zero_count=0.0,
        )


class LogCollapsingHighestDenseDDSketch(BaseDDSketch):
    """Bounded memory: the highest bins collapse once `bin_limit` is reached."""

    def __init__(self, relative_accuracy=None, bin_limit=None):
        if relative_accuracy is None:
            relative_accuracy = DEFAULT_REL_ACC
        if bin_limit is None or bin_limit < 0:
            bin_limit = DEFAULT_BIN_LIMIT
        super().__init__(
            mapping=LogarithmicMapping(relative_accuracy),
            store=CollapsingHighestDenseStore(bin_limit),
            negative_store=CollapsingHighestDenseStore(bin_limit),
            zero_count=0.0,
        )
