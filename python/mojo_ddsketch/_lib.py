"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-ddsketch.so"

KEY_NONE = np.int64(np.iinfo(np.int64).min)
"""Written by the key kernels for inputs that are not indexable. Matches the
Mojo side's `MIN_KEY`."""


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    keys_argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
        ctypes.c_double, ctypes.c_int64,
    ]
    for name in ("dds_keys_log", "dds_keys_linear", "dds_keys_cubic"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = keys_argtypes

    values_argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_int64,
    ]
    for name in ("dds_values_log", "dds_values_linear", "dds_values_cubic"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = values_argtypes

    fn = lib.dds_key_scalar
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_double, ctypes.c_int64, ctypes.c_double,
                   ctypes.c_double]

    fn = lib.dds_value_scalar
    fn.restype = ctypes.c_double
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
                   ctypes.c_double, ctypes.c_double]

    fn = lib.dds_add_bulk
    fn.restype = None
    fn.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]

    fn = lib.dds_shift_bins
    fn.restype = None
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    fn = lib.dds_rank_scan
    fn.restype = None
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    fn = lib.dds_keys_at_rank
    fn.restype = None
    fn.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]

    fn = lib.dds_fold_range
    fn.restype = None
    fn.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
    ]

    fn = lib.dds_fold_src
    fn.restype = None
    fn.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]

    fn = lib.dds_merge_add
    fn.restype = None
    fn.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
    ]
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _keys(symbol, values, multiplier, offset) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float64)
    n = values.size
    out = np.full(n, KEY_NONE, dtype=np.int64)
    if n:
        symbol(_addr(values), n, ctypes.c_double(multiplier),
               ctypes.c_double(offset), _addr(out))
    return out


def _values(symbol, keys, multiplier, offset, scale) -> np.ndarray:
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    n = keys.size
    out = np.empty(n, dtype=np.float64)
    if n:
        symbol(_addr(keys), n, ctypes.c_double(multiplier),
               ctypes.c_double(offset), ctypes.c_double(scale), _addr(out))
    return out


def key_scalar(value, mode, multiplier, offset=0.0) -> int:
    """KeyMapping.key for one value: the same kernel, no buffer."""
    return int(lib.dds_key_scalar(float(value), ctypes.c_int64(mode),
                                  ctypes.c_double(multiplier),
                                  ctypes.c_double(offset)))


def value_scalar(key, mode, multiplier, offset, scale) -> float:
    """KeyMapping.value for one key: the same kernel, no buffer."""
    return float(lib.dds_value_scalar(int(key), ctypes.c_int64(mode),
                                      ctypes.c_double(multiplier),
                                      ctypes.c_double(offset),
                                      ctypes.c_double(scale)))


def keys_log(values, multiplier, offset=0.0):
    return _keys(lib.dds_keys_log, values, multiplier, offset)


def keys_linear(values, multiplier, offset=0.0):
    return _keys(lib.dds_keys_linear, values, multiplier, offset)


def keys_cubic(values, multiplier, offset=0.0):
    return _keys(lib.dds_keys_cubic, values, multiplier, offset)


def values_log(keys, multiplier, offset, scale):
    return _values(lib.dds_values_log, keys, multiplier, offset, scale)


def values_linear(keys, multiplier, offset, scale):
    return _values(lib.dds_values_linear, keys, multiplier, offset, scale)


def values_cubic(keys, multiplier, offset, scale):
    return _values(lib.dds_values_cubic, keys, multiplier, offset, scale)


def add_bulk(bins: np.ndarray, keys: np.ndarray, weights: np.ndarray,
             offset: int) -> None:
    """In-place `bins[key - offset] += weight` over the whole batch."""
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    n = keys.size
    if n and bins.size:
        lib.dds_add_bulk(_addr(bins), bins.size, _addr(keys), _addr(weights),
                         n, ctypes.c_int64(offset))


def shift_bins(bins: np.ndarray, shift: int) -> None:
    """In-place re-centring move; see `dds_shift_bins`."""
    if bins.size:
        lib.dds_shift_bins(_addr(bins), bins.size, ctypes.c_int64(shift))


def rank_scan(bins: np.ndarray) -> np.ndarray:
    """Inclusive prefix sum of the bin counters."""
    bins = np.ascontiguousarray(bins, dtype=np.float64)
    out = np.empty(bins.size, dtype=np.float64)
    if bins.size:
        lib.dds_rank_scan(_addr(bins), bins.size, _addr(out))
    return out


def keys_at_rank(cum: np.ndarray, ranks: np.ndarray, offset: int, max_key: int,
                 lower: bool) -> np.ndarray:
    """Batch `key_at_rank` over pre-computed running counts."""
    cum = np.ascontiguousarray(cum, dtype=np.float64)
    ranks = np.ascontiguousarray(ranks, dtype=np.float64)
    out = np.empty(ranks.size, dtype=np.int64)
    if ranks.size:
        lib.dds_keys_at_rank(
            _addr(cum), cum.size, _addr(ranks), ranks.size,
            ctypes.c_int64(offset), ctypes.c_int64(max_key),
            ctypes.c_int64(1 if lower else 0), _addr(out),
        )
    return out


def fold_range(bins: np.ndarray, start: int, end: int, target: int) -> None:
    """Zero `bins[start:end]` and add its total into `bins[target]`."""
    if bins.size:
        lib.dds_fold_range(_addr(bins), bins.size, ctypes.c_int64(start),
                           ctypes.c_int64(end), ctypes.c_int64(target))


def fold_src(dst: np.ndarray, src: np.ndarray, start: int, end: int,
             target: int) -> None:
    """Add `src[start:end]`'s total into `dst[target]`."""
    src = np.ascontiguousarray(src, dtype=np.float64)
    if dst.size and src.size:
        lib.dds_fold_src(_addr(dst), dst.size, _addr(src), src.size,
                         ctypes.c_int64(start), ctypes.c_int64(end),
                         ctypes.c_int64(target))


def merge_add(dst: np.ndarray, src: np.ndarray, n: int, dst_off: int,
              src_off: int) -> None:
    """`dst[dst_off + i] += src[src_off + i]` for i in range(n)."""
    src = np.ascontiguousarray(src, dtype=np.float64)
    if n and dst.size and src.size:
        lib.dds_merge_add(_addr(dst), dst.size, _addr(src), n,
                          ctypes.c_int64(dst_off), ctypes.c_int64(src_off))
