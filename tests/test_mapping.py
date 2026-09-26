"""Parity tests for the value <-> bucket mapping kernels.

The mapping is the numeric heart of a DDSketch, so these compare every formula
against the real `ddsketch.mapping` on a fixed, seeded sample. Keys are exact
integers: the Mojo expressions are the upstream expressions, evaluated in the
same order, so a mismatch means a real bug (wrong multiplier, dropped element,
transposed index) rather than rounding.
"""

import numpy as np
import pytest
from ddsketch.mapping import CubicallyInterpolatedMapping as RefCubic
from ddsketch.mapping import LinearlyInterpolatedMapping as RefLinear
from ddsketch.mapping import LogarithmicMapping as RefLog

import mojo_ddsketch as m

ALPHA = 0.01
N = 20000


@pytest.fixture(scope="module")
def sample():
    """Log-uniform positive sample: spans 60 binades, so a wrong multiplier or
    a missing exponent term shows up immediately."""
    rng = np.random.default_rng(20240917)
    return np.exp(rng.uniform(-30.0, 30.0, N))


def _ref_keys(ref, values):
    return np.array([ref.key(float(v)) for v in values], dtype=np.int64)


def _ref_values(ref, keys):
    return np.array([ref.value(int(k)) for k in keys], dtype=np.float64)


PAIRS = [
    (m.LogarithmicMapping, RefLog, "log", 1e-12),
    (m.LinearlyInterpolatedMapping, RefLinear, "linear", 0.0),
    (m.CubicallyInterpolatedMapping, RefCubic, "cubic", 1e-8),
]


@pytest.mark.parametrize("mine_cls,ref_cls,tag,vtol", PAIRS)
def test_keys_exact(sample, mine_cls, ref_cls, tag, vtol):
    mine = mine_cls(ALPHA)
    ref = ref_cls(ALPHA)
    assert mine._multiplier == ref._multiplier
    assert mine.gamma == ref.gamma
    np.testing.assert_array_equal(mine.keys(sample), _ref_keys(ref, sample))


@pytest.mark.parametrize("mine_cls,ref_cls,tag,vtol", PAIRS)
def test_values_close(sample, mine_cls, ref_cls, tag, vtol):
    """`vtol` is the spread Mojo's libm has against CPython's for each
    formula: exp2 vs 2**x, an exact ldexp, and a pow-based cube root."""
    mine = mine_cls(ALPHA)
    ref = ref_cls(ALPHA)
    keys = np.arange(-60, 61, dtype=np.int64)
    np.testing.assert_allclose(mine.values(keys), _ref_values(ref, keys),
                               rtol=vtol, atol=0.0)


@pytest.mark.parametrize("mine_cls,ref_cls,tag,vtol", PAIRS)
def test_bucket_contains_value(sample, mine_cls, ref_cls, tag, vtol):
    """The defining property: key = ceil(log_gamma(v)) puts v in
    (gamma**(k-1), gamma**k], and value() returns the bucket scaled by
    1 - alpha. A wrong scale factor breaks this even when the keys are right."""
    mine = mine_cls(ALPHA)
    keys = mine.keys(sample)
    lower = mine.values(keys - 1) / (1.0 - ALPHA)
    upper = mine.values(keys) / (1.0 - ALPHA)
    assert np.all(sample > lower)
    assert np.all(sample <= upper)


@pytest.mark.parametrize("mine_cls,ref_cls,tag,vtol", PAIRS)
def test_batch_matches_scalar(sample, mine_cls, ref_cls, tag, vtol):
    """A dropped SIMD tail or a wrong stride shows up here and nowhere else."""
    mine = mine_cls(ALPHA)
    np.testing.assert_array_equal(
        mine.keys(sample), np.array([mine.key(v) for v in sample], dtype=np.int64)
    )
    keys = np.arange(-30, 31, dtype=np.int64)
    np.testing.assert_array_equal(
        mine.values(keys), np.array([mine.value(k) for k in keys])
    )


def test_non_indexable_inputs_are_sentinel():
    """Zero, negatives, +-inf and NaN have no bucket; the kernels must say so
    rather than returning garbage from log2(0)."""
    mapping = m.LogarithmicMapping(ALPHA)
    probe = np.array([0.0, -0.0, -1.0, -1e300, np.inf, -np.inf, np.nan])
    assert np.all(mapping.keys(probe) == m._lib.KEY_NONE)
    rng = np.random.default_rng(11)
    good = np.exp(rng.uniform(-30.0, 30.0, 1000))
    assert not np.any(mapping.keys(good) == m._lib.KEY_NONE)


def test_offset_shifts_keys_and_cancels_in_value():
    """A non-zero offset shifts keys up but must leave the value of a shifted
    key unchanged, since value() subtracts the offset again."""
    plain = m.LogarithmicMapping(ALPHA)
    shifted = m.LogarithmicMapping(ALPHA, offset=3.0)
    rng = np.random.default_rng(5)
    v = np.exp(rng.uniform(-10.0, 10.0, 5000))
    np.testing.assert_array_equal(shifted.keys(v), plain.keys(v) + 3)
    np.testing.assert_allclose(shifted.values(plain.keys(v) + 3),
                               plain.values(plain.keys(v)), rtol=1e-15)


def test_from_gamma_offset_roundtrip():
    ref = RefLog(0.02)
    mine = m.LogarithmicMapping.from_gamma_offset(ref.gamma, offset=0.0)
    assert mine.relative_accuracy == pytest.approx(0.02)
    assert mine.gamma == ref.gamma
    # The round trip through (gamma - 1) / (gamma + 1) moves the multiplier by
    # an ulp, which is far too little to move a key.
    assert mine._multiplier == pytest.approx(ref._multiplier, rel=1e-15)
    rng = np.random.default_rng(6)
    v = np.exp(rng.uniform(-10.0, 10.0, 2000))
    np.testing.assert_array_equal(mine.keys(v), _ref_keys(ref, v))


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.0, 2.0])
def test_invalid_relative_accuracy(bad):
    for cls in (m.LogarithmicMapping, m.LinearlyInterpolatedMapping,
                m.CubicallyInterpolatedMapping):
        with pytest.raises(ValueError):
            cls(bad)


def test_indexable_bounds_match_upstream():
    mine = m.LogarithmicMapping(ALPHA)
    ref = RefLog(ALPHA)
    assert mine.min_possible == ref.min_possible
    assert mine.max_possible == ref.max_possible
    assert mine.scale == 2.0 / (1.0 + mine.gamma)
