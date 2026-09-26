"""Correctness-gated benchmark for mojo-ddsketch.

Every case checks numerical agreement with its reference before timing, so a
regression in the Mojo kernels shows up as a correctness failure rather than a
suspiciously good number. The reference is the fastest reasonable alternative:
vectorised NumPy for the array kernels, and the real `ddsketch` API for the
sketch-level cases, which has no bulk form of its own.

The two sides of a case are timed alternately, best of `ROUNDS`, because this
box is shared: measuring one side and then the other lets another process's
load land on whichever ran second.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import ddsketch  # noqa: E402

import mojo_ddsketch as m  # noqa: E402

ALPHA = 0.01
ROUNDS = 5


def _time_pair(reference, mine, rounds=ROUNDS):
    """Alternate the two candidates and keep each one's best wall clock."""
    ref_best = float("inf")
    mine_best = float("inf")
    for _ in range(rounds):
        for fn, is_ref in ((reference, True), (mine, False)):
            t0 = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - t0
            if is_ref:
                ref_best = min(ref_best, elapsed)
            else:
                mine_best = min(mine_best, elapsed)
    return ref_best, mine_best


def bench_keys(n: int = 1 << 20):
    """Value -> bucket index for a whole array. The reference is the same
    formula in vectorised NumPy, which is what anyone would write in Python."""
    rng = np.random.default_rng(0)
    values = np.exp(rng.uniform(-30.0, 30.0, n))
    mapping = m.LogarithmicMapping(ALPHA)
    mult, off = mapping._multiplier, mapping._offset

    got = mapping.keys(values)
    expect = np.ceil(np.log2(values) * mult + off).astype(np.int64)
    mismatches = int(np.count_nonzero(got != expect))
    # A value sitting exactly on a bucket boundary can round either way; the
    # contract is that the two agree on the bucket, not on the boundary case.
    assert mismatches <= n * 1e-6, (mismatches, n)
    np.testing.assert_array_equal(mapping.values(got), mapping.values(expect))

    return ("keys n=%d" % n,
            lambda: np.ceil(np.log2(values) * mult + off).astype(np.int64),
            lambda: mapping.keys(values))


def bench_keys_linear(n: int = 1 << 20):
    """The same job with the interpolated mapping, whose log2 is arithmetic
    rather than a libm call. Included because it isolates what the
    logarithmic mapping is actually paying for."""
    rng = np.random.default_rng(6)
    values = np.exp(rng.uniform(-30.0, 30.0, n))
    mapping = m.LinearlyInterpolatedMapping(ALPHA)
    mult, off = mapping._multiplier, mapping._offset

    def numpy_keys():
        mantissa, exponent = np.frexp(values)
        significand = 2.0 * mantissa - 1.0
        return np.ceil((significand + (exponent - 1.0)) * mult + off).astype(
            np.int64
        )

    np.testing.assert_array_equal(mapping.keys(values), numpy_keys())
    return ("keys (linear map) n=%d" % n, numpy_keys,
            lambda: mapping.keys(values))


def bench_add_many(n: int = 1 << 20):
    """Ingesting a whole array of values. Upstream can only be called once per
    value, so this measures the API difference honestly rather than pretending
    a like-for-like bulk loop exists."""
    rng = np.random.default_rng(1)
    values = np.exp(rng.uniform(-5.0, 20.0, n))
    qs = np.array([0.5, 0.95, 0.99])

    mine = m.DDSketch(ALPHA)
    mine.add_many(values)
    theirs = ddsketch.DDSketch(ALPHA)
    for value in values:
        theirs.add(float(value))
    np.testing.assert_allclose(
        mine.get_quantiles(qs),
        [theirs.get_quantile_value(float(q)) for q in qs],
        rtol=1e-9,
    )

    return ("add_many n=%d" % n,
            lambda: [theirs.add(float(v)) for v in values],
            lambda: m.DDSketch(ALPHA).add_many(values))


def bench_store_add(n: int = 1 << 20):
    """The store accumulate on its own, against np.add.at, the scatter-add
    NumPy offers."""
    rng = np.random.default_rng(2)
    keys = rng.integers(-500, 500, n).astype(np.int64)
    weights = np.ones(n, dtype=np.float64)
    reference = np.bincount(keys + 512, minlength=1024)

    store = m.DenseStore()
    store.add_many(keys, weights)
    np.testing.assert_array_equal(store.bins, reference)

    scatter = np.zeros(1024, dtype=np.float64)
    return ("store add n=%d" % n,
            lambda: np.add.at(scatter, keys + 512, weights),
            lambda: m.DenseStore().add_many(keys, weights))


def bench_rank_select(nbins: int = 4096, nq: int = 20000):
    """Running-count scan plus a rank select for many quantiles at once,
    against searchsorted over a cumulative sum."""
    rng = np.random.default_rng(3)
    bins = rng.random(nbins) * 100.0
    ranks = np.sort(rng.random(nq) * bins.sum())
    store = m.DenseStore()
    store.bins = bins.copy()
    store.count = float(bins.sum())
    store.min_key = 0
    store.max_key = nbins - 1
    store.offset = 0
    store._cum = None

    np.testing.assert_array_equal(
        store.keys_at_rank(ranks),
        np.array([store.key_at_rank(float(r)) for r in ranks]),
    )

    return ("rank select nbins=%d nq=%d" % (nbins, nq),
            lambda: np.searchsorted(np.cumsum(bins), ranks, side="right"),
            lambda: store.keys_at_rank(ranks))


def bench_quantiles(n: int = 1 << 20, nq: int = 1000):
    """A thousand quantiles out of a sketch holding a million values, against
    the upstream one-call-at-a-time API."""
    rng = np.random.default_rng(5)
    values = np.exp(rng.uniform(-5.0, 20.0, n))
    qs = np.linspace(0.0, 1.0, nq)
    mine, theirs = m.DDSketch(ALPHA), ddsketch.DDSketch(ALPHA)
    mine.add_many(values)
    for value in values:
        theirs.add(float(value))
    np.testing.assert_allclose(
        mine.get_quantiles(qs),
        [theirs.get_quantile_value(float(q)) for q in qs],
        rtol=1e-9,
    )
    return ("get_quantiles n=%d x%d" % (n, nq),
            lambda: [theirs.get_quantile_value(float(q)) for q in qs],
            lambda: mine.get_quantiles(qs))


def bench_scalar_add(n: int = 200000):
    """One value at a time: the API-compatible path, against upstream's."""
    rng = np.random.default_rng(4)
    values = np.exp(rng.uniform(-5.0, 20.0, n))
    q = 0.99

    def build_mine():
        sketch = m.DDSketch(ALPHA)
        for value in values:
            sketch.add(float(value))
        return sketch

    def build_theirs():
        sketch = ddsketch.DDSketch(ALPHA)
        for value in values:
            sketch.add(float(value))
        return sketch

    a, b = build_mine(), build_theirs()
    np.testing.assert_allclose(a.get_quantile_value(q),
                               b.get_quantile_value(q), rtol=1e-9)

    return "scalar add n=%d" % n, build_theirs, build_mine


CASES = (bench_keys, bench_keys_linear, bench_add_many, bench_store_add,
         bench_rank_select, bench_quantiles, bench_scalar_add)


def main():
    print(f"{'case':<34}{'reference':>12}{'mojo-ddsketch':>20}{'ratio':>10}")
    print("-" * 76)
    for build in CASES:
        label, reference, mine = build()
        ref, got = _time_pair(reference, mine)
        ratio = ref / got if got else float("nan")
        print(f"{label:<34}{ref * 1e3:>10.2f}ms{got * 1e3:>18.2f}ms"
              f"{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
