"""Dense bin stores, mirroring `ddsketch.store`.

The bin block is a contiguous `float64` array. Every O(nbins) pass over it --
re-centring, accumulating a batch of keys, the running-count scan, the rank
search, and the collapse fold -- runs in Mojo; the range bookkeeping (which
`offset` to use, when to grow) stays here because it is pure control flow.
"""

import math

import numpy as np

from . import _lib

CHUNK_SIZE = 128


class DenseStore:
    """Keeps every bin between `min_key` and `max_key`, growing in chunks."""

    def __init__(self, chunk_size=CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.offset = 0
        self.bins = np.zeros(0, dtype=np.float64)
        self.count = 0.0
        self.min_key = None
        self.max_key = None
        self._cum = None

    def __len__(self) -> int:
        return int(self.bins.size)

    def length(self) -> int:
        return int(self.bins.size)

    def __repr__(self) -> str:
        return (
            "{%s}, min_key:%s, max_key:%s, offset:%s"
            % (
                ", ".join(
                    "%s: %s" % (k, v) for k, v in sorted(self.bin_dict().items())
                ),
                self.min_key,
                self.max_key,
                self.offset,
            )
        )

    def bin_dict(self) -> dict:
        """{key: count} for the non-empty bins, for inspection and tests."""
        if self.max_key is None:
            return {}
        keys = np.arange(self.min_key, self.max_key + 1, dtype=np.int64)
        counts = self.bins[keys - self.offset]
        return {
            int(k): float(c)
            for k, c in zip(keys, counts)
            if c != 0.0
        }

    # -- range management ---------------------------------------------------

    def _get_new_length(self, new_min_key, new_max_key) -> int:
        desired = new_max_key - new_min_key + 1
        return self.chunk_size * int(math.ceil(desired / self.chunk_size))

    def _extend_range(self, key, second_key=None) -> None:
        if second_key is None:
            second_key = key
        if self.min_key is None:
            new_min_key = min(key, second_key)
            new_max_key = max(key, second_key)
        else:
            new_min_key = min(key, second_key, self.min_key)
            new_max_key = max(key, second_key, self.max_key)

        if self.length() == 0:
            self.bins = np.zeros(self._get_new_length(new_min_key, new_max_key),
                                 dtype=np.float64)
            self.offset = new_min_key
            self._adjust(new_min_key, new_max_key)
        elif (new_min_key >= self.min_key
                and new_max_key < self.offset + self.length()):
            self.min_key = new_min_key
            self.max_key = new_max_key
        else:
            new_length = self._get_new_length(new_min_key, new_max_key)
            if new_length > self.length():
                grown = np.zeros(new_length, dtype=np.float64)
                grown[:self.length()] = self.bins
                self.bins = grown
            self._adjust(new_min_key, new_max_key)

    def _adjust(self, new_min_key, new_max_key) -> None:
        self._center_bins(new_min_key, new_max_key)
        self.min_key = new_min_key
        self.max_key = new_max_key

    def _center_bins(self, new_min_key, new_max_key) -> None:
        middle_key = new_min_key + (new_max_key - new_min_key + 1) // 2
        self._shift_bins(self.offset + self.length() // 2 - middle_key)

    def _shift_bins(self, shift: int) -> None:
        if shift and self.length():
            _lib.shift_bins(self.bins, shift)
            self.offset -= shift
            self._cum = None

    # -- ingestion ----------------------------------------------------------

    def _get_index(self, key: int) -> int:
        if self.min_key is None or key < self.min_key or key > self.max_key:
            self._extend_range(key)
        return key - self.offset

    def add(self, key: int, weight: float = 1.0) -> None:
        idx = self._get_index(int(key))
        self.bins[idx] += weight
        self.count += weight
        self._cum = None

    def add_many(self, keys: np.ndarray, weights: np.ndarray) -> None:
        """Accumulate a whole batch of keys in one kernel call."""
        keys = np.ascontiguousarray(keys, dtype=np.int64)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if keys.size != weights.size:
            raise ValueError(
                "keys and weights must have the same length, got %d and %d"
                % (keys.size, weights.size)
            )
        if keys.size == 0:
            return
        self._extend_range(int(keys.min()), int(keys.max()))
        # Out-of-range keys only exist once the store has collapsed, and there
        # the clipped index is the collapsed bin, matching the store's _get_index.
        if getattr(self, "is_collapsed", False):
            keys = np.clip(keys, self.min_key, self.max_key)
        self._cum = None
        _lib.add_bulk(self.bins, keys, weights, self.offset)
        self.count += float(weights.sum())

    # -- rank selection -----------------------------------------------------

    def _running(self) -> np.ndarray:
        if self._cum is None:
            self._cum = _lib.rank_scan(self.bins)
        return self._cum

    def key_at_rank(self, rank: float, lower: bool = True) -> int:
        return int(self.keys_at_rank([rank], lower=lower)[0])

    def keys_at_rank(self, ranks, lower: bool = True) -> np.ndarray:
        ranks = np.ascontiguousarray(ranks, dtype=np.float64)
        if ranks.size == 0:
            return np.zeros(0, dtype=np.int64)
        fallback = 0 if self.max_key is None else int(self.max_key)
        return _lib.keys_at_rank(self._running(), ranks, self.offset, fallback,
                                 lower)

    # -- merge --------------------------------------------------------------

    def copy(self, store) -> None:
        self.bins = np.array(store.bins, dtype=np.float64, copy=True)
        self.count = store.count
        self.min_key = store.min_key
        self.max_key = store.max_key
        self.offset = store.offset
        self._cum = None

    def merge(self, store) -> None:
        if store.count == 0:
            return
        if self.count == 0:
            self.copy(store)
            return
        if store.min_key < self.min_key or store.max_key > self.max_key:
            self._extend_range(store.min_key, store.max_key)
        self._cum = None
        n = store.max_key - store.min_key + 1
        _lib.merge_add(self.bins, store.bins, n,
                       store.min_key - self.offset, store.min_key - store.offset)
        self.count += store.count


class CollapsingLowestDenseStore(DenseStore):
    """Drops the lowest bins into the first one once `bin_limit` is reached."""

    def __init__(self, bin_limit, chunk_size=CHUNK_SIZE):
        if bin_limit < 1:
            # Upstream indexes an empty bin list here and dies with IndexError.
            raise ValueError("bin_limit must be >= 1, got %r" % bin_limit)
        super().__init__(chunk_size=chunk_size)
        self.bin_limit = bin_limit
        self.is_collapsed = False

    def copy(self, store) -> None:
        self.bin_limit = store.bin_limit
        self.is_collapsed = store.is_collapsed
        super().copy(store)

    def _get_new_length(self, new_min_key, new_max_key) -> int:
        desired = new_max_key - new_min_key + 1
        return min(
            self.chunk_size * int(math.ceil(desired / self.chunk_size)),
            self.bin_limit,
        )

    def _get_index(self, key: int) -> int:
        if self.min_key is not None and key < self.min_key:
            if self.is_collapsed:
                return 0
            self._extend_range(key)
            if self.is_collapsed:
                return 0
        elif self.min_key is None or key > self.max_key:
            # Only keys below min_key collapse into the first bin; a key above
            # max_key grows the range and lands in the last bin.
            self._extend_range(key)
        return key - self.offset

    def _adjust(self, new_min_key, new_max_key) -> None:
        if new_max_key - new_min_key + 1 > self.length():
            new_min_key = new_max_key - self.length() + 1
            if self.min_key is None or new_min_key >= self.max_key:
                self.offset = new_min_key
                self.min_key = new_min_key
                self.bins[:] = 0.0
                self.bins[0] = self.count
            else:
                shift = self.offset - new_min_key
                if shift < 0:
                    collapse_start = self.min_key - self.offset
                    collapse_end = new_min_key - self.offset
                    _lib.fold_range(self.bins, collapse_start, collapse_end,
                                    collapse_end)
                    self.min_key = new_min_key
                    self._shift_bins(shift)
                else:
                    self.min_key = new_min_key
                    self._shift_bins(shift)
            self.max_key = new_max_key
            self.is_collapsed = True
        else:
            self._center_bins(new_min_key, new_max_key)
            self.min_key = new_min_key
            self.max_key = new_max_key
        self._cum = None

    def merge(self, store) -> None:
        if store.count == 0:
            return
        if self.count == 0:
            self.copy(store)
            return
        if store.min_key < self.min_key or store.max_key > self.max_key:
            self._extend_range(store.min_key, store.max_key)
        self._cum = None

        collapse_start = store.min_key - store.offset
        collapse_end = min(self.min_key, store.max_key + 1) - store.offset
        if collapse_end > collapse_start:
            _lib.fold_src(self.bins, store.bins, collapse_start, collapse_end, 0)
        else:
            collapse_end = collapse_start

        start = collapse_end + store.offset
        n = store.max_key - start + 1
        _lib.merge_add(self.bins, store.bins, n, start - self.offset,
                       collapse_end)
        self.count += store.count


class CollapsingHighestDenseStore(DenseStore):
    """Drops the highest bins into the last one once `bin_limit` is reached."""

    def __init__(self, bin_limit, chunk_size=CHUNK_SIZE):
        if bin_limit < 1:
            # Upstream indexes an empty bin list here and dies with IndexError.
            raise ValueError("bin_limit must be >= 1, got %r" % bin_limit)
        super().__init__(chunk_size=chunk_size)
        self.bin_limit = bin_limit
        self.is_collapsed = False

    def copy(self, store) -> None:
        self.bin_limit = store.bin_limit
        self.is_collapsed = store.is_collapsed
        super().copy(store)

    def _get_new_length(self, new_min_key, new_max_key) -> int:
        desired = new_max_key - new_min_key + 1
        return min(
            self.chunk_size * int(math.ceil(desired / self.chunk_size)),
            self.bin_limit,
        )

    def _get_index(self, key: int) -> int:
        if self.min_key is not None and key > self.max_key:
            if self.is_collapsed:
                return self.length() - 1
            self._extend_range(key)
            if self.is_collapsed:
                return self.length() - 1
        elif self.min_key is None or key < self.min_key:
            self._extend_range(key)
        return key - self.offset

    def _adjust(self, new_min_key, new_max_key) -> None:
        if new_max_key - new_min_key + 1 > self.length():
            new_max_key = new_min_key + self.length() - 1
            if self.max_key is None or new_max_key <= self.min_key:
                self.offset = new_min_key
                self.max_key = new_max_key
                self.bins[:] = 0.0
                self.bins[-1] = self.count
            else:
                shift = self.offset - new_min_key
                if shift > 0:
                    collapse_start = new_max_key - self.offset + 1
                    collapse_end = self.max_key - self.offset + 1
                    _lib.fold_range(self.bins, collapse_start, collapse_end,
                                    collapse_start - 1)
                    self.max_key = new_max_key
                    self._shift_bins(shift)
                else:
                    self.max_key = new_max_key
                    self._shift_bins(shift)
            self.min_key = new_min_key
            self.is_collapsed = True
        else:
            self._center_bins(new_min_key, new_max_key)
            self.min_key = new_min_key
            self.max_key = new_max_key
        self._cum = None

    def merge(self, store) -> None:
        if store.count == 0:
            return
        if self.count == 0:
            self.copy(store)
            return
        if store.min_key < self.min_key or store.max_key > self.max_key:
            self._extend_range(store.min_key, store.max_key)
        self._cum = None

        collapse_end = store.max_key - store.offset + 1
        collapse_start = max(self.max_key + 1, store.min_key) - store.offset
        if collapse_end > collapse_start:
            _lib.fold_src(self.bins, store.bins, collapse_start, collapse_end,
                          self.length() - 1)
        else:
            collapse_start = collapse_end

        n = collapse_start + store.offset - store.min_key
        _lib.merge_add(self.bins, store.bins, n, store.min_key - self.offset,
                       store.min_key - store.offset)
        self.count += store.count
