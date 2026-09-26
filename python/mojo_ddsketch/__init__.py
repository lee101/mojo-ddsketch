"""mojo-ddsketch: the DDSketch quantile sketch with Mojo kernels.

Installable alongside the real `ddsketch` package, which it is tested against
for parity. The bucket mapping (value -> log index -> value) and the dense
store arithmetic run in `src/kernels.mojo`; this layer owns the arrays and the
range bookkeeping.
"""

from .mapping import (
    CubicallyInterpolatedMapping,
    KeyMapping,
    LinearlyInterpolatedMapping,
    LogarithmicMapping,
)
from .sketch import (
    DEFAULT_BIN_LIMIT,
    DEFAULT_REL_ACC,
    BaseDDSketch,
    DDSketch,
    LogCollapsingHighestDenseDDSketch,
    LogCollapsingLowestDenseDDSketch,
)
from .store import (
    CHUNK_SIZE,
    CollapsingHighestDenseStore,
    CollapsingLowestDenseStore,
    DenseStore,
)

__all__ = [
    "BaseDDSketch",
    "CHUNK_SIZE",
    "CollapsingHighestDenseStore",
    "CollapsingLowestDenseStore",
    "CubicallyInterpolatedMapping",
    "DEFAULT_BIN_LIMIT",
    "DEFAULT_REL_ACC",
    "DDSketch",
    "DenseStore",
    "KeyMapping",
    "LinearlyInterpolatedMapping",
    "LogCollapsingHighestDenseDDSketch",
    "LogCollapsingLowestDenseDDSketch",
    "LogarithmicMapping",
]
__version__ = "0.1.0"
