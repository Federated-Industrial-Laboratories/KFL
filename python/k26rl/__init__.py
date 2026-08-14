"""k26rl: Gymnasium environments over compiled KFL simulation
artifacts.

A KFL program that declares reinforcement-learning constructs
compiles to a batch executable plus a companion shared object
exporting the frozen k26rl_ stepping surface. This package dlopens
that shared object, reads its spec blob, and exposes the result as a
gymnasium.Env (:class:`K26RlEnv`) or a gymnasium.vector.VectorEnv
(:class:`K26RlVectorEnv`). It is a marshalling layer only: no
physics, no per-environment code, and no arithmetic on stream
values, which cross the boundary as IEEE-754 binary64 bit patterns.

The package is versioned against the stepping ABI, not against the
grammar that produced an artifact: it serves every artifact whose
ABI major equals :data:`ABI_MAJOR`.
"""

__version__ = "0.1.0"

from ._abi import ABI_MAJOR, ABI_MINOR_MIN
from ._errors import (
    K26RlError,
    K26RlFaultError,
    K26RlOutputError,
    K26RlSeedReuseError,
)

__all__ = [
    "ABI_MAJOR",
    "ABI_MINOR_MIN",
    "K26RlEnv",
    "K26RlError",
    "K26RlFaultError",
    "K26RlOutputError",
    "K26RlSeedReuseError",
    "K26RlVectorEnv",
    "__version__",
]


def __getattr__(name):
    # The environment classes need gymnasium and numpy; the binding,
    # spec, and error layers need neither. Loading the classes on
    # first use keeps the rest of the package importable where only
    # the binding is wanted.
    if name == "K26RlEnv":
        from .env import K26RlEnv
        return K26RlEnv
    if name == "K26RlVectorEnv":
        from .vector import K26RlVectorEnv
        return K26RlVectorEnv
    raise AttributeError("module %r has no attribute %r"
                         % (__name__, name))
