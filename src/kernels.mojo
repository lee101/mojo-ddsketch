"""DDSketch kernels: logarithmic bucket mapping and dense-store bin arithmetic.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

The semantics mirror `ddsketch.mapping.KeyMapping` and `ddsketch.store.DenseStore`
from the upstream package, restricted to whole-array ("bulk") form. The per-value
Python bookkeeping -- growing the store, choosing `offset`, collapsing bins --
stays in the Python shim, which owns every array.
"""

from std.math import abs, ceil, exp2, floor, frexp, ldexp, log2, pow, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime MIN_KEY = -9223372036854775807 - 1
"""Sentinel written by the key kernels for inputs that are not indexable
(positive and finite). `Int64.MIN` can never be a bucket key, so the shim can
route on it without a second output buffer."""

comptime CUBIC_A = 6.0 / 35.0
comptime CUBIC_B = -3.0 / 5.0
comptime CUBIC_C = 10.0 / 7.0
comptime FLOAT64_MAX = 1.7976931348623157e308
"""Largest finite double; `v < FLOAT64_MAX` rejects +inf, and `v > 0.0` already
rejects NaN, so the two comparisons are the indexability test."""


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def _cbrt(x: Float64) -> Float64:
    """ddsketch.mapping._cbrt: signed cube root via pow, not the correctly
    rounded cbrt, so the cubic mapping tracks the upstream series."""
    var y = pow(abs(x), 1.0 / 3.0)
    if x < 0.0:
        return -y
    return y


# ---------------------------------------------------------------------------
# value <-> key, the three upstream KeyMapping flavours
# ---------------------------------------------------------------------------
#
# All three compute key = int(ceil(_log_gamma(value)) + offset), with
# `multiplier` carrying mapping._multiplier and `offset` mapping._offset, and
# value = _pow_gamma(key - offset) * scale with `scale` holding upstream's
# 2 / (1 + gamma) factor. `mode` selects the flavour: 0 logarithmic,
# 1 linearly interpolated, 2 cubically interpolated.
#
# The scalar and array entry points below share one implementation, so the
# per-value and whole-array paths cannot drift apart.


def _key_of(v: Float64, mode: Int, multiplier: Float64,
            offset: Float64) -> Int64:
    if v <= 0.0 or v >= FLOAT64_MAX:
        return MIN_KEY
    if mode == 0:
        return Int64(ceil(log2(v) * multiplier) + offset)
    var r = frexp(v)
    var s = 2.0 * Float64(r[0]) - 1.0
    var lg = s
    if mode == 2:
        lg = ((CUBIC_A * s + CUBIC_B) * s + CUBIC_C) * s
    lg = lg + (Float64(r[1]) - 1.0)
    return Int64(ceil(lg * multiplier) + offset)


def _value_of(k: Int64, mode: Int, multiplier: Float64, offset: Float64,
              scale: Float64) -> Float64:
    var x = (Float64(k) - offset) / multiplier
    if mode == 0:
        return scale * exp2(x)
    if mode == 1:
        var e = Int32(floor(x) + 1.0)
        return scale * ldexp((x - Float64(e) + 2.0) / 2.0, e)
    var exponent = Int32(floor(x))
    var delta_0 = CUBIC_B * CUBIC_B - 3.0 * CUBIC_A * CUBIC_C
    var delta_1 = 2.0 * CUBIC_B * CUBIC_B * CUBIC_B
    delta_1 -= 9.0 * CUBIC_A * CUBIC_B * CUBIC_C
    delta_1 -= 27.0 * CUBIC_A * CUBIC_A * (x - Float64(exponent))
    var cardano = _cbrt(
        (delta_1 - sqrt(delta_1 * delta_1 - 4.0 * delta_0 * delta_0 * delta_0))
        / 2.0
    )
    var significand_plus_one = 1.0 - (
        CUBIC_B + cardano + delta_0 / cardano
    ) / (3.0 * CUBIC_A)
    return scale * ldexp(significand_plus_one / 2.0, exponent + 1)


@export("dds_key_scalar")
def dds_key_scalar(v: Float64, mode: Int, multiplier: Float64,
                   offset: Float64) abi("C") -> Int64:
    """KeyMapping.key for a single value, without going through a buffer."""
    return _key_of(v, mode, multiplier, offset)


@export("dds_value_scalar")
def dds_value_scalar(k: Int, mode: Int, multiplier: Float64, offset: Float64,
                     scale: Float64) abi("C") -> Float64:
    """KeyMapping.value for a single key, without going through a buffer."""
    return _value_of(Int64(k), mode, multiplier, offset, scale)


@export("dds_keys_log")
def dds_keys_log(vals_addr: Int, n: Int, multiplier: Float64, offset: Float64,
                 dst_addr: Int) abi("C"):
    """LogarithmicMapping: _log_gamma(v) = log2(v) * multiplier, for a whole
    array of values."""
    var vals = fp(vals_addr)
    var dst = ip(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = _key_of(vals[unsafe_offset=i], 0, multiplier,
                                       offset)


@export("dds_keys_linear")
def dds_keys_linear(vals_addr: Int, n: Int, multiplier: Float64,
                    offset: Float64, dst_addr: Int) abi("C"):
    """LinearlyInterpolatedMapping: log2 approximated by extracting the floor
    of log2 from the binary representation and interpolating linearly."""
    var vals = fp(vals_addr)
    var dst = ip(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = _key_of(vals[unsafe_offset=i], 1, multiplier,
                                       offset)


@export("dds_keys_cubic")
def dds_keys_cubic(vals_addr: Int, n: Int, multiplier: Float64,
                   offset: Float64, dst_addr: Int) abi("C"):
    """CubicallyInterpolatedMapping: log2 approximated by a cubic polynomial in
    the significand (sketches-java)."""
    var vals = fp(vals_addr)
    var dst = ip(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = _key_of(vals[unsafe_offset=i], 2, multiplier,
                                       offset)


@export("dds_values_log")
def dds_values_log(keys_addr: Int, n: Int, multiplier: Float64, offset: Float64,
                   scale: Float64, dst_addr: Int) abi("C"):
    """LogarithmicMapping: _pow_gamma(x) = 2 ** (x / multiplier)."""
    var keys = ip(keys_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = _value_of(keys[unsafe_offset=i], 0, multiplier,
                                         offset, scale)


@export("dds_values_linear")
def dds_values_linear(keys_addr: Int, n: Int, multiplier: Float64,
                      offset: Float64, scale: Float64, dst_addr: Int) abi("C"):
    """Inverse of dds_keys_linear's log2 approximation."""
    var keys = ip(keys_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = _value_of(keys[unsafe_offset=i], 1, multiplier,
                                         offset, scale)


@export("dds_values_cubic")
def dds_values_cubic(keys_addr: Int, n: Int, multiplier: Float64, offset: Float64,
                     scale: Float64, dst_addr: Int) abi("C"):
    """Inverse of dds_keys_cubic, from Cardano's formula as upstream."""
    var keys = ip(keys_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = _value_of(keys[unsafe_offset=i], 2, multiplier,
                                         offset, scale)



# ---------------------------------------------------------------------------
# dense store arithmetic
# ---------------------------------------------------------------------------


@export("dds_add_bulk")
def dds_add_bulk(bins_addr: Int, nbins: Int, keys_addr: Int, weights_addr: Int,
                 n: Int, offset: Int) abi("C"):
    """Accumulate: bins[key - offset] += weight for every (key, weight) pair."""
    var bins = fp(bins_addr)
    var keys = ip(keys_addr)
    var w = fp(weights_addr)
    var off = Int64(offset)
    for i in range(n):
        var idx = keys[unsafe_offset=i] - off
        if idx >= 0 and idx < Int64(nbins):
            bins[unsafe_offset=idx] += w[unsafe_offset=i]


@export("dds_shift_bins")
def dds_shift_bins(bins_addr: Int, nbins: Int, shift: Int) abi("C"):
    """DenseStore._shift_bins: move the bin block in place, zero filling the
    vacated end. A positive shift drops the highest bins, a negative one the
    lowest, which is what re-centring the store needs."""
    var bins = fp(bins_addr)
    if nbins <= 0:
        return
    if shift > 0:
        var s = min(shift, nbins)
        for i in range(nbins - 1, s - 1, -1):
            bins[unsafe_offset=i] = bins[unsafe_offset=i - s]
        for i in range(0, s):
            bins[unsafe_offset=i] = 0.0
    elif shift < 0:
        var s = min(-shift, nbins)
        for i in range(0, nbins - s):
            bins[unsafe_offset=i] = bins[unsafe_offset=i + s]
        for i in range(nbins - s, nbins):
            bins[unsafe_offset=i] = 0.0


@export("dds_rank_scan")
def dds_rank_scan(bins_addr: Int, nbins: Int, dst_addr: Int) abi("C"):
    """Inclusive prefix sum of the bins: the running count that
    DenseStore.key_at_rank walks with."""
    var bins = fp(bins_addr)
    var dst = fp(dst_addr)
    var acc = Float64(0.0)
    for i in range(0, nbins):
        acc += bins[unsafe_offset=i]
        dst[unsafe_offset=i] = acc


@export("dds_keys_at_rank")
def dds_keys_at_rank(cum_addr: Int, nbins: Int, ranks_addr: Int, n: Int,
                     offset: Int, max_key: Int, lower: Int,
                     dst_addr: Int) abi("C"):
    """Batch DenseStore.key_at_rank by binary search over the running counts.

    `lower != 0` selects `running_ct > rank`, `lower == 0` selects
    `running_ct >= rank + 1`; the predicates are the upstream ones, evaluated
    on the same float threshold. Returns `max_key` when no bin qualifies,
    exactly as the upstream linear scan does.
    """
    var cum = fp(cum_addr)
    var ranks = fp(ranks_addr)
    var keys = ip(dst_addr)
    var off = Int64(offset)
    var fallback = Int64(max_key)
    for t in range(0, n):
        var rank = ranks[unsafe_offset=t]
        var thr = rank + 1.0
        var found = -1
        var lo = 0
        var hi = nbins - 1
        while lo <= hi:
            var mid = (lo + hi) // 2
            var c = cum[unsafe_offset=mid]
            var ok: Bool
            if lower != 0:
                ok = c > rank
            else:
                ok = c >= thr
            if ok:
                found = mid
                hi = mid - 1
            else:
                lo = mid + 1
        if found >= 0:
            keys[unsafe_offset=t] = Int64(found) + off
        else:
            keys[unsafe_offset=t] = fallback


@export("dds_fold_range")
def dds_fold_range(bins_addr: Int, nbins: Int, start: Int, end: Int,
                   target: Int) abi("C"):
    """Zero bins[start:end] and add their total into bins[target]: the
    collapsing stores' `_adjust` fold, in one pass."""
    var bins = fp(bins_addr)
    var acc = Float64(0.0)
    var s = max(start, 0)
    var e = min(end, nbins)
    for i in range(s, e):
        acc += bins[unsafe_offset=i]
        bins[unsafe_offset=i] = 0.0
    if target >= 0 and target < nbins:
        bins[unsafe_offset=target] += acc


@export("dds_fold_src")
def dds_fold_src(dst_addr: Int, dst_nbins: Int, src_addr: Int, src_nbins: Int,
                 src_start: Int, src_end: Int, dst_target: Int) abi("C"):
    """Add the total of src[src_start:src_end] into dst[dst_target]: the
    collapsing stores' `merge`, which folds the incoming keys that fall past
    the surviving range into the first or last bin."""
    var dst = fp(dst_addr)
    var src = fp(src_addr)
    var acc = Float64(0.0)
    for i in range(max(src_start, 0), min(src_end, src_nbins)):
        acc += src[unsafe_offset=i]
    if dst_target >= 0 and dst_target < dst_nbins:
        dst[unsafe_offset=dst_target] += acc


@export("dds_merge_add")
def dds_merge_add(dst_addr: Int, dst_nbins: Int, src_addr: Int, n: Int,
                  dst_off: Int, src_off: Int) abi("C"):
    """Elementwise merge: dst[dst_off + i] += src[src_off + i] for i in range(n),
    skipping indices outside either block, as DenseStore.merge does."""
    var dst = fp(dst_addr)
    var src = fp(src_addr)
    for i in range(0, n):
        var d = dst_off + i
        var s = src_off + i
        if d >= 0 and d < dst_nbins and s >= 0:
            dst[unsafe_offset=d] += src[unsafe_offset=s]
