"""Parity tests for the quantile sketches.

The sketches are compared against the real `ddsketch` classes: same input
values, same quantiles. They are also checked against the contract the package
advertises -- a relative error no worse than `relative_accuracy` on every
quantile -- which is what a wrong rank offset or a mis-routed negative value
would break.
"""

import ddsketch
import numpy as np
import pytest

import mojo_ddsketch as m

ALPHA = 0.02
ALPHAS = [0.5, 0.1, 0.02, 0.01, 0.001]
QUANTILES = np.concatenate([
    np.linspace(0.0, 1.0, 101),
    np.array([0.001, 0.01, 0.5, 0.99, 0.999]),
])

SKETCHES = [
    (m.DDSketch, ddsketch.DDSketch),
    (m.LogCollapsingLowestDenseDDSketch, ddsketch.LogCollapsingLowestDenseDDSketch),
    (m.LogCollapsingHighestDenseDDSketch, ddsketch.LogCollapsingHighestDenseDDSketch),
]


def ref_quantiles(theirs, qs):
    return np.array([theirs.get_quantile_value(float(q)) for q in qs])


def assert_close(got, expect, alpha, label=""):
    """Quantile estimates agree to well inside the advertised accuracy, and
    never to more than the two libm's differ."""
    assert np.all(np.isfinite(got)), label
    np.testing.assert_allclose(got, expect, rtol=1e-9, atol=0.0,
                               err_msg=label)


@pytest.mark.parametrize("mine_cls,ref_cls", SKETCHES)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_quantile_parity_positive(mine_cls, ref_cls, alpha):
    rng = np.random.default_rng(1234)
    values = np.exp(rng.normal(0.0, 4.0, 20000))
    mine, theirs = mine_cls(alpha), ref_cls(alpha)
    mine.add_many(values)
    for value in values:
        theirs.add(float(value))
    assert_close(mine.get_quantiles(QUANTILES),
                 ref_quantiles(theirs, QUANTILES), alpha)


@pytest.mark.parametrize("mine_cls,ref_cls", SKETCHES)
def test_scalar_and_batch_paths_agree(mine_cls, ref_cls):
    """`add` and `add_many` are the same sketch written two ways."""
    rng = np.random.default_rng(77)
    values = np.exp(rng.normal(0.0, 2.0, 5000))
    batched, scalar = mine_cls(ALPHA), mine_cls(ALPHA)
    batched.add_many(values)
    for value in values:
        scalar.add(float(value))
    np.testing.assert_allclose(batched.get_quantiles(QUANTILES),
                               scalar.get_quantiles(QUANTILES), rtol=0.0,
                               atol=0.0)
    assert batched.count == scalar.count
    assert batched.min == scalar.min
    assert batched.max == scalar.max
    # add_many sums with a NumPy reduction and add with a sequential +=, so
    # the totals agree to rounding rather than bit for bit.
    assert batched.sum == pytest.approx(scalar.sum, rel=1e-14)


def test_summary_statistics_match_upstream():
    rng = np.random.default_rng(9)
    values = np.exp(rng.normal(0.0, 3.0, 5000))
    mine, theirs = m.DDSketch(ALPHA), ddsketch.DDSketch(ALPHA)
    mine.add_many(values)
    for value in values:
        theirs.add(float(value))
    assert mine.count == theirs.count
    assert mine.num_values == theirs.num_values
    assert mine.min == theirs._min
    assert mine.max == theirs._max
    assert mine.sum == pytest.approx(theirs.sum, rel=1e-14)
    assert mine.avg == pytest.approx(theirs.avg, rel=1e-14)
    assert mine.name == theirs.name
    assert mine.relative_accuracy == ALPHA


def test_mixed_signs_and_zeros_match_upstream():
    """Negative values go to a mirrored store and zeros to a counter; a wrong
    rank offset between the three shows up immediately here."""
    rng = np.random.default_rng(31)
    positive = np.exp(rng.normal(0.0, 3.0, 4000))
    values = np.concatenate([
        positive,
        -np.exp(rng.normal(0.0, 3.0, 3000)),
        np.zeros(25),
        np.array([0.0, -0.0, 1e-320, -1e-320, 1e300, -1e300]),
    ])
    mine, theirs = m.DDSketch(ALPHA), ddsketch.DDSketch(ALPHA)
    mine.add_many(values)
    for value in values:
        theirs.add(float(value))
    assert_close(mine.get_quantiles(QUANTILES),
                 ref_quantiles(theirs, QUANTILES), ALPHA)
    assert mine._zero_count == theirs._zero_count
    assert mine.count == theirs.count


def test_weighted_parity():
    rng = np.random.default_rng(55)
    values = np.exp(rng.normal(0.0, 2.0, 4000))
    weights = rng.uniform(0.25, 4.0, values.size)
    mine, theirs = m.DDSketch(ALPHA), ddsketch.DDSketch(ALPHA)
    mine.add_many(values, weights)
    for value, weight in zip(values, weights):
        theirs.add(float(value), float(weight))
    assert_close(mine.get_quantiles(QUANTILES),
                 ref_quantiles(theirs, QUANTILES), ALPHA)
    assert mine.count == pytest.approx(theirs.count, rel=1e-13)


def test_negative_weights_rejected():
    sketch = m.DDSketch(ALPHA)
    with pytest.raises(ValueError):
        sketch.add(1.0, 0.0)
    with pytest.raises(ValueError):
        sketch.add_many(np.array([1.0, 2.0]), np.array([1.0, -1.0]))


def test_empty_and_out_of_range():
    empty = m.DDSketch(ALPHA)
    assert empty.get_quantile_value(0.5) is None
    assert empty.get_quantiles([0.0, 1.0]) is None
    assert empty.add_many(np.zeros(0)) is None

    sketch = m.DDSketch(ALPHA)
    sketch.add_many(np.array([1.0, 2.0, 3.0]))
    assert sketch.get_quantile_value(-0.1) is None
    assert sketch.get_quantile_value(1.1) is None
    out = sketch.get_quantiles([-0.5, 0.5, 2.0])
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert np.isfinite(out[1])


def test_relative_accuracy_contract():
    """The accuracy contract, checked two ways.

    A value landing in bucket k is reported as (1 - alpha) * gamma**k, so every
    reported value is within `alpha` of some sample that really is in the
    sketch. That is the guarantee the package makes, and it holds for this port
    and for the upstream package alike.

    The second check compares against np.quantile over the middle of the
    distribution, where adjacent order statistics are close. The allowance is
    3 * alpha rather than alpha because the sketch's rank is q * (n - 1) with a
    lower tie-break, so it may report the bucket of the neighbouring order
    statistic, and in the far tail those are far apart.
    """
    for alpha in ALPHAS:
        rng = np.random.default_rng(int(alpha * 1e6) + 1)
        values = np.exp(rng.normal(0.0, 5.0, 50000))
        sketch, theirs = m.DDSketch(alpha), ddsketch.DDSketch(alpha)
        sketch.add_many(values)
        for value in values:
            theirs.add(float(value))
        exact = np.quantile(values, QUANTILES)
        mid = (QUANTILES >= 0.05) & (QUANTILES <= 0.95)
        for got in (sketch.get_quantiles(QUANTILES),
                    ref_quantiles(theirs, QUANTILES)):
            nearest = np.array(
                [np.min(np.abs(g - values) / values) for g in got]
            )
            assert np.all(nearest <= alpha), (alpha, nearest.max())
            assert np.all(np.abs(got - exact)[mid] / exact[mid] <= 3 * alpha)


def test_extreme_quantiles_land_on_the_extremes():
    """q=0 and q=1 must report the observed minimum and maximum within the
    accuracy, which is where a wrong rank offset is most visible."""
    rng = np.random.default_rng(12)
    values = np.exp(rng.normal(0.0, 6.0, 20000))
    sketch = m.DDSketch(ALPHA)
    sketch.add_many(values)
    got = sketch.get_quantiles([0.0, 1.0])
    assert got[0] == pytest.approx(sketch.min, rel=ALPHA)
    assert got[1] == pytest.approx(sketch.max, rel=ALPHA)


@pytest.mark.parametrize("mine_cls,ref_cls", SKETCHES)
def test_merge_parity(mine_cls, ref_cls):
    rng = np.random.default_rng(2024)
    values = np.exp(rng.normal(0.0, 4.0, 8000))
    left, right = values[:4000], values[4000:]
    mine_a, mine_b = mine_cls(ALPHA), mine_cls(ALPHA)
    mine_a.add_many(left)
    mine_b.add_many(right)
    theirs_a, theirs_b = ref_cls(ALPHA), ref_cls(ALPHA)
    for value in left:
        theirs_a.add(float(value))
    for value in right:
        theirs_b.add(float(value))
    mine_a.merge(mine_b)
    theirs_a.merge(theirs_b)
    assert_close(mine_a.get_quantiles(QUANTILES),
                 ref_quantiles(theirs_a, QUANTILES), ALPHA)
    assert mine_a.count == theirs_a.count
    assert mine_a.min == theirs_a._min
    assert mine_a.max == theirs_a._max


def test_merge_matches_equivalent_single_sketch():
    """Merging two halves must be indistinguishable from adding all the values
    at once, up to the floating point of the totals."""
    rng = np.random.default_rng(88)
    values = np.exp(rng.normal(0.0, 3.0, 6000))
    whole = m.DDSketch(ALPHA)
    whole.add_many(values)
    a, b = m.DDSketch(ALPHA), m.DDSketch(ALPHA)
    a.add_many(values[:2500])
    b.add_many(values[2500:])
    a.merge(b)
    np.testing.assert_allclose(a.get_quantiles(QUANTILES),
                               whole.get_quantiles(QUANTILES), rtol=0.0,
                               atol=0.0)
    assert a.count == whole.count


def test_merge_rejects_incompatible_gamma():
    a = m.DDSketch(0.01)
    b = m.DDSketch(0.02)
    b.add(1.0)
    with pytest.raises(ValueError):
        a.merge(b)


def test_merge_into_empty_adopts_the_source():
    target = m.DDSketch(ALPHA)
    source = m.DDSketch(ALPHA)
    values = np.exp(np.linspace(-5.0, 5.0, 500))
    source.add_many(values)
    target.merge(source)
    fresh = m.DDSketch(ALPHA)
    fresh.add_many(values)
    np.testing.assert_array_equal(target.get_quantiles(QUANTILES),
                                  fresh.get_quantiles(QUANTILES))


@pytest.mark.parametrize("mine_cls,ref_cls", SKETCHES[1:])
def test_collapsing_sketches_with_tight_bin_limit(mine_cls, ref_cls):
    """With a bin limit far below the key span both implementations collapse,
    and they must collapse identically -- including the accuracy they lose."""
    rng = np.random.default_rng(303)
    values = np.exp(rng.normal(0.0, 8.0, 20000))
    mine, theirs = mine_cls(ALPHA, bin_limit=64), ref_cls(ALPHA, bin_limit=64)
    mine.add_many(values)
    for value in values:
        theirs.add(float(value))
    assert mine._store.is_collapsed is True
    assert_close(mine.get_quantiles(QUANTILES),
                 ref_quantiles(theirs, QUANTILES), ALPHA)
    # Documented cost of collapsing: the lowest quantiles are no longer within
    # the accuracy guarantee, which is exactly why the parity check above has
    # to compare against the reference rather than against np.quantile.
    exact = np.quantile(values, QUANTILES)
    got = mine.get_quantiles(QUANTILES)
    assert np.max(np.abs(got - exact) / exact) > ALPHA


def test_zero_and_subnormal_routing():
    """Values below mapping.min_possible are zeros, not buckets; both sides
    must agree on the exact cut."""
    values = np.array([0.0, 5e-324, -5e-324, 1e-310, 1.0, -1.0])
    mine, theirs = m.DDSketch(ALPHA), ddsketch.DDSketch(ALPHA)
    mine.add_many(values)
    for value in values:
        theirs.add(float(value))
    assert mine._zero_count == theirs._zero_count
    assert mine.count == theirs.count
