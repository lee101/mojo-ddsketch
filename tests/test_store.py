"""Parity tests for the dense bin stores.

Every store class is driven through the same scripted sequence of keys and
compared against `ddsketch.store`, bin for bin: same counts at the same keys,
same `offset`, same length, same collapse flag. Unit weights make the counts
exact integers, so a difference here is a real indexing bug -- an off-by-one in
the fold, a shift in the wrong direction, a lost bin.
"""

import random

import ddsketch.store as ref_store
import numpy as np
import pytest

import mojo_ddsketch as m


def ref_bins(store):
    """{key: count} using upstream's own key convention, key = offset + i."""
    return {i + store.offset: c for i, c in enumerate(store.bins) if c}


CLASSES = [
    (m.DenseStore, ref_store.DenseStore, ()),
    (m.CollapsingLowestDenseStore, ref_store.CollapsingLowestDenseStore, (64,)),
    (m.CollapsingHighestDenseStore, ref_store.CollapsingHighestDenseStore, (64,)),
    (m.CollapsingLowestDenseStore, ref_store.CollapsingLowestDenseStore, (2048,)),
    (m.CollapsingHighestDenseStore, ref_store.CollapsingHighestDenseStore, (2048,)),
]


def _script(seed, lo, hi, count):
    rng = random.Random(seed)
    return [(rng.randint(lo, hi), 1.0) for _ in range(count)]


def _assert_same(mine, theirs, label=""):
    assert mine.bin_dict() == ref_bins(theirs), label
    assert mine.offset == theirs.offset, label
    assert mine.length() == len(theirs.bins), label
    assert mine.count == pytest.approx(theirs.count, rel=1e-15), label
    assert mine.min_key == theirs.min_key, label
    assert mine.max_key == theirs.max_key, label
    if hasattr(theirs, "is_collapsed"):
        assert mine.is_collapsed == theirs.is_collapsed, label


@pytest.mark.parametrize("mine_cls,ref_cls,args", CLASSES)
def test_add_scalar_parity(mine_cls, ref_cls, args):
    """A narrow key range so the chunk growth and re-centring shifts run many
    times, in both directions."""
    ops = _script(4242, -500, 500, 600)
    mine, theirs = mine_cls(*args), ref_cls(*args)
    for key, weight in ops:
        mine.add(key, weight)
        theirs.add(key, weight)
    _assert_same(mine, theirs)


@pytest.mark.parametrize("mine_cls,ref_cls,args", CLASSES)
def test_add_wide_range_parity(mine_cls, ref_cls, args):
    """A wide range with a small bin_limit forces the collapse path."""
    ops = _script(99, -5000, 5000, 400)
    mine, theirs = mine_cls(*args), ref_cls(*args)
    for key, weight in ops:
        mine.add(key, weight)
        theirs.add(key, weight)
    _assert_same(mine, theirs)
    if hasattr(theirs, "is_collapsed"):
        assert mine.is_collapsed is True


@pytest.mark.parametrize("mine_cls,ref_cls,args", CLASSES)
def test_add_many_parity(mine_cls, ref_cls, args):
    """The batched kernel must land every weight in the same bin the scalar
    path would: catches a bad clip, a bad offset, or an unclipped collapsed key.
    """
    rng = random.Random(7)
    mine, theirs = mine_cls(*args), ref_cls(*args)
    for _ in range(12):
        keys = np.array([rng.randint(-4000, 4000) for _ in range(200)],
                        dtype=np.int64)
        weights = np.ones(keys.size, dtype=np.float64)
        mine.add_many(keys, weights)
        for key in keys:
            theirs.add(int(key), 1.0)
    _assert_same(mine, theirs)
    assert mine.count == 2400.0


@pytest.mark.parametrize("mine_cls,ref_cls,args", CLASSES)
def test_key_at_rank_parity(mine_cls, ref_cls, args):
    ops = _script(31337, -900, 900, 500)
    mine, theirs = mine_cls(*args), ref_cls(*args)
    for key, weight in ops:
        mine.add(key, weight)
        theirs.add(key, weight)
    for rank in np.linspace(0.0, mine.count, 37):
        assert mine.key_at_rank(float(rank)) == theirs.key_at_rank(float(rank))
        assert (mine.key_at_rank(float(rank), lower=False)
                == theirs.key_at_rank(float(rank), lower=False))
    # Ranks outside the population fall back to max_key, as upstream does.
    for rank in (-1.0, mine.count + 10.0):
        assert (mine.key_at_rank(rank)
                == theirs.key_at_rank(rank))


@pytest.mark.parametrize("mine_cls,ref_cls,args", CLASSES)
def test_keys_at_rank_matches_scalar(mine_cls, ref_cls, args):
    """The binary search over the running counts must return exactly what the
    upstream linear scan returns, for both tie-breaking rules."""
    ops = _script(2024, -700, 700, 400)
    mine, theirs = mine_cls(*args), ref_cls(*args)
    for key, weight in ops:
        mine.add(key, weight)
        theirs.add(key, weight)
    ranks = np.concatenate([
        np.linspace(0.0, mine.count, 51),
        np.array([0.0, 0.5, 1.0, 2.0, 1e-9, mine.count - 1.0]),
    ])
    for lower in (True, False):
        batch = mine.keys_at_rank(ranks, lower=lower)
        expect = np.array(
            [theirs.key_at_rank(float(r), lower=lower) for r in ranks],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(batch, expect)


@pytest.mark.parametrize("mine_cls,ref_cls,args", CLASSES)
def test_merge_parity(mine_cls, ref_cls, args):
    left = _script(1, -2000, 2000, 200)
    right = _script(2, -2000, 2000, 200)
    mine, theirs = mine_cls(*args), ref_cls(*args)
    for key, weight in left:
        mine.add(key, weight)
        theirs.add(key, weight)
    donor_mine, donor_theirs = mine_cls(*args), ref_cls(*args)
    for key, weight in right:
        donor_mine.add(key, weight)
        donor_theirs.add(key, weight)
    mine.merge(donor_mine)
    theirs.merge(donor_theirs)
    _assert_same(mine, theirs)


@pytest.mark.parametrize("mine_cls,ref_cls,args", CLASSES)
def test_merge_weighted_parity(mine_cls, ref_cls, args):
    """Random weights: the totals are summed in a different order than the
    upstream sequential `+=`, so this is a tolerance check, not an exact one."""
    rng = random.Random(8)
    mine, theirs = mine_cls(*args), ref_cls(*args)
    donor_mine, donor_theirs = mine_cls(*args), ref_cls(*args)
    for target in (mine, theirs, donor_mine, donor_theirs):
        pass
    for _ in range(200):
        key, weight = rng.randint(-2000, 2000), rng.random()
        mine.add(key, weight)
        theirs.add(key, weight)
    for _ in range(200):
        key, weight = rng.randint(-2000, 2000), rng.random()
        donor_mine.add(key, weight)
        donor_theirs.add(key, weight)
    mine.merge(donor_mine)
    theirs.merge(donor_theirs)
    assert mine.offset == theirs.offset
    assert mine.length() == len(theirs.bins)
    got, expect = mine.bin_dict(), ref_bins(theirs)
    assert set(got) == set(expect)
    for key, value in got.items():
        assert value == pytest.approx(expect[key], rel=1e-12)


@pytest.mark.parametrize("mine_cls,ref_cls,args", CLASSES)
def test_merge_empty_and_into_empty(mine_cls, ref_cls, args):
    mine, theirs = mine_cls(*args), ref_cls(*args)
    empty_mine, empty_theirs = mine_cls(*args), ref_cls(*args)
    mine.merge(empty_mine)
    theirs.merge(empty_theirs)
    assert mine.count == 0.0

    full_mine, full_theirs = mine_cls(*args), ref_cls(*args)
    for key in (-3, 5, 9, 9, -40):
        full_mine.add(key, 1.0)
        full_theirs.add(key, 1.0)
    mine.merge(full_mine)
    theirs.merge(full_theirs)
    _assert_same(mine, theirs)


@pytest.mark.parametrize("mine_cls,ref_cls,args", CLASSES)
def test_copy_parity(mine_cls, ref_cls, args):
    src_mine, src_theirs = mine_cls(*args), ref_cls(*args)
    for key in (-11, 0, 4, 77):
        src_mine.add(key, 1.0)
        src_theirs.add(key, 1.0)
    mine, theirs = mine_cls(*args), ref_cls(*args)
    mine.copy(src_mine)
    theirs.copy(src_theirs)
    _assert_same(mine, theirs)
    # The copy must be independent of its source.
    mine.add(1000, 1.0)
    assert src_mine.count == 4.0


def test_weighted_merge_into_collapsed_store_keeps_total():
    """Folding must conserve mass: the total across all bins equals count."""
    for cls in (m.CollapsingLowestDenseStore, m.CollapsingHighestDenseStore):
        store = cls(32)
        rng = random.Random(3)
        for _ in range(500):
            store.add(rng.randint(-1000, 1000), rng.random())
        assert sum(store.bin_dict().values()) == pytest.approx(store.count,
                                                                rel=1e-12)


def test_bin_limit_must_be_positive():
    for cls in (m.CollapsingLowestDenseStore, m.CollapsingHighestDenseStore):
        with pytest.raises(ValueError):
            cls(0)


def test_add_many_rejects_mismatched_weights():
    store = m.DenseStore()
    with pytest.raises(ValueError):
        store.add_many(np.array([1, 2, 3], dtype=np.int64),
                       np.ones(2, dtype=np.float64))
