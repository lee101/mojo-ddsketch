# mojo-ddsketch

`mojo-ddsketch` is the compute-oriented subset of
[ddsketch](https://github.com/DataDog/sketches-python) with the bucket mapping
and the dense-store bin arithmetic implemented in Mojo. It keeps the upstream
class and method shapes for everything it covers, and adds the two whole-array
entry points the upstream package has no equivalent for: `add_many` and
`get_quantiles`.

The Python package is named `mojo_ddsketch`, so it installs alongside the real
`ddsketch` and the tests compare the two directly.

```python
import numpy as np
import mojo_ddsketch as mds

sketch = mds.DDSketch(relative_accuracy=0.01)
sketch.add_many(np.random.lognormal(0.0, 1.0, 1_000_000))
sketch.get_quantiles([0.5, 0.95, 0.99])
sketch.get_quantile_value(0.99)
```

## Why this package

A DDSketch is two numeric kernels wrapped in a small amount of bookkeeping.
The mapping turns a `float64` into a bucket index with `ceil(log_gamma(v))` and
back again with `gamma**k`; the store is a dense array of float counters that
gets scattered into, scanned, and rank-searched. Upstream implements all of it
one value at a time in Python, and has no bulk API at all, so ingesting a
million observations means a million Python method calls. That is the surface
this port moves into Mojo.

The bookkeeping around the kernels -- deciding when to grow the store, which
`offset` re-centres the bins on, when to collapse -- is control flow, not
compute, and stays in Python where it is readable.

## Covered subset

| area | implemented API | kernel |
| --- | --- | --- |
| Bucket mapping | `LogarithmicMapping`, `LinearlyInterpolatedMapping`, `CubicallyInterpolatedMapping`; `.key`/`.value` scalars and `.keys`/`.values` arrays | `dds_keys_log`, `dds_keys_linear`, `dds_keys_cubic`, `dds_values_log`, `dds_values_linear`, `dds_values_cubic`, `dds_key_scalar`, `dds_value_scalar` |
| Dense store | `DenseStore` add / add_many / key_at_rank / keys_at_rank / merge / copy | `dds_add_bulk`, `dds_rank_scan`, `dds_keys_at_rank`, `dds_merge_add` |
| Store growth | chunked growth and re-centring on `offset` | `dds_shift_bins` |
| Collapsing stores | `CollapsingLowestDenseStore`, `CollapsingHighestDenseStore` | `dds_fold_range`, `dds_fold_src` |
| Sketches | `DDSketch`, `LogCollapsingLowestDenseDDSketch`, `LogCollapsingHighestDenseDDSketch`: add, add_many, merge, get_quantile_value, get_quantiles, count / sum / avg / min / max | the above, plus `dds_add_bulk` for the value routing |

All three sketch classes, all three store classes and all three mappings from
`ddsketch.mapping` are covered, at every relative accuracy, with unit and
fractional weights, over positive, negative, zero, subnormal and
non-indexable inputs.

### Not implemented

- `ddsketch.pb` -- the protobuf serialisation format (`proto.proto_ddsketch`,
  `serialize`, `deserialize`). That is encoding and IO, not compute; use the
  real `ddsketch.pb` for it.
- `ddsketch.ddsketch.BaseDDSketch.__repr__` formatting details, and the
  `name`/`num_values` property spellings beyond what is listed above.
- Any GPU path. The kernels are memory- and libm-bound, and a device transfer
  per batch would dominate.

### Deliberate divergences

- `add_many` sums the weights with a NumPy reduction and the values with
  `np.dot`, while `add` accumulates sequentially, so the running `sum` differs
  between the two by float rounding (measured: 1e-16 relative). Upstream has
  no batch path to disagree with.
- The collapsing stores reject `bin_limit < 1` with a `ValueError`. Upstream
  reaches an empty bin list there and dies with an `IndexError`.
- `get_quantiles` returns `None` for an empty sketch, like
  `get_quantile_value`, and `NaN` for individual ranks outside `[0, 1]`,
  because an array has no way to say "no value".
- The store keeps its running-count scan cached until the next mutation, so
  repeated quantile queries over an unchanged sketch scan once.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-ddsketch.so`. Set `PYTHONPATH=python`
when using the package outside a Pixi task. Against the shared toolchain:

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

## Tests

102 tests, all comparing against the real `ddsketch` package:

| file | what it pins down |
| --- | --- |
| `tests/test_mapping.py` | keys exact against `ddsketch.mapping` for all three mappings on 20 000 log-uniform values; values within the libm spread of each formula; the bucket-contains-value invariant; batch equals scalar; sentinel for non-indexable input; offset handling; `from_gamma_offset`; argument validation |
| `tests/test_store.py` | bin-for-bin parity with `ddsketch.store` for all three store classes over scalar adds, batch adds, wide ranges that force a collapse, merges (exact with unit weights, `rtol=1e-12` with random weights), copies, and rank selection in both tie-breaking modes |
| `tests/test_sketch.py` | quantile parity for all three sketch classes at five relative accuracies, mixed signs and zeros, weights, the accuracy contract, merges, and the empty and out-of-range contracts |

The suite is not decorative. Injecting a one-key offset into the rank search in
`src/kernels.mojo` fails 34 of the tests; inverting the direction of the
store's re-centring shift aborts the run.

## Performance

Best-of-five wall clock, alternating the two candidates, same process, on the
shared 36-core box. Every case verifies numerical agreement before timing.
Because the box is shared, run-to-run variation on the memory-bound cases is
around 10%.

| case | reference | mojo-ddsketch | result |
| --- | ---: | ---: | ---: |
| keys, logarithmic map, n=1048576 | 22.72 ms | 27.17 ms | 0.84x |
| keys, linear map, n=1048576 | 34.60 ms | 11.70 ms | 2.96x |
| add_many n=1048576 | 2413.49 ms | 131.79 ms | 18.31x |
| store add n=1048576 | 9.23 ms | 12.02 ms | 0.77x |
| rank select, 4096 bins x 20000 ranks | 0.94 ms | 1.66 ms | 0.57x |
| get_quantiles, n=1048576 x 1000 | 101.89 ms | 0.38 ms | 271.49x |
| scalar add n=200000 | 317.75 ms | 1247.85 ms | 0.25x |

Reading the losses honestly:

- **keys, logarithmic map (0.84x).** The kernel is one scalar `log2` per
  element, and Mojo's `log2` is a libm call that does not vectorise here.
  NumPy's `np.log2` has its own vectorised implementation, so it wins. The
  linear-mapping row is the control: same job, `frexp` plus three flops
  instead of a `log2`, and it runs 3x faster than NumPy. The cost is
  `log2` itself, not the loop.
- **store add (0.77x) and rank select (0.57x).** Both are within a small
  constant of NumPy's `np.add.at` and `np.searchsorted`, and the shim's
  Python-side work (range check, weight reduction) is a visible share of a
  kernel this short. The rank select in particular is a branchy binary search
  with random access into the scan, which is exactly what `searchsorted` is
  tuned for.
- **scalar add (0.25x).** `add` costs one ctypes crossing per value, roughly
  5 us against upstream's 1.6 us of pure Python. This path exists for API
  parity; `add_many` is the ported one, and it is 18x faster than the
  equivalent upstream loop.

The wins are where the API shape differs, not where the kernel is cleverer:
ingesting a batch (18x) and reading a batch of quantiles (271x, because
upstream walks the bins from the start for every single quantile) both come
from replacing a per-value Python call with one array pass.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-ddsketch.so`.

The `python/mojo_ddsketch` layer owns every array. Buffers cross the C ABI as
64-bit addresses and are reconstructed in Mojo as
`Pointer[Float64, AnyOrigin[mut=True]]` and `Pointer[Int64, ...]`, which keeps
the exported symbols non-parametric.

The scalar and array entry points share one implementation (`_key_of` and
`_value_of` in the kernel file), so the per-value and whole-array paths cannot
drift; the mapping tests assert exactly that.

Non-indexable inputs -- zero, negatives, infinities, NaN -- are not the
mapping's business: the sketch routes them before calling a kernel, and the
kernels return `Int64.MIN` for them, which can never be a bucket key.

## Numerics

Mojo emits FMA and its libm is not bit-identical to CPython's, so results match
`ddsketch` to a tolerance rather than bit for bit. The parity tests measure the
spread: the linear mapping's values are bit-identical, the logarithmic
mapping's agree to 3e-13 relative, and the cubic mapping's to 4e-10 (its
cube root goes through `pow`, which is itself an approximation). Bucket
*keys* are compared for exact equality, which the tests show is achievable
here: the Mojo expressions are the upstream expressions in the same order.

The `sum` of a sketch is accumulated with NumPy reductions in `add_many` and
sequentially in `add`, so it agrees with upstream to 1e-14 relative rather than
exactly. Quantile estimates, bin counts, offsets, store lengths and the
collapse flag all match exactly with unit weights.

## License

MIT
