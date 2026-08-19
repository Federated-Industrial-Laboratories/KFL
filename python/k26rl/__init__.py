"""k26rl: Gymnasium environments over compiled KFL simulation
artifacts.

A KFL program that declares reinforcement-learning constructs
compiles to a batch executable plus a companion shared object
exporting the frozen k26rl_ stepping surface. This package dlopens
that shared object, reads its spec blob, and exposes the result as a
gymnasium.Env (:class:`K26RlEnv`), a gymnasium.vector.VectorEnv
(:class:`K26RlVectorEnv`), or, for an artifact declaring more than one
agent, a PettingZoo ParallelEnv (:class:`K26RlParallelEnv`). It is a
marshalling layer only: no physics, no per-environment code, and no
arithmetic on stream values, which cross the boundary as IEEE-754
binary64 bit patterns.

It also writes trained policies. :mod:`k26rl.policy` encodes the
``.k26pol`` file the visibility and inference tiers load, and knows
nothing about any training framework; :func:`export_policy`, in
:mod:`k26rl.sb3`, is the translation from one such framework and is
the only place that names it.

Gymnasium and numpy are required; pettingzoo is optional and is
imported only when :class:`K26RlParallelEnv` is asked for, as are the
training packages :func:`export_policy` reads a model through.

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
    "K26RlParallelEnv",
    "K26RlSeedReuseError",
    "K26RlVectorEnv",
    "__version__",
    "export_policy",
    "policy",
]


def __getattr__(name):
    # The environment classes need gymnasium and numpy, and the
    # multi-agent one needs pettingzoo besides; the binding, spec, and
    # error layers need none of them. Loading the classes on first use
    # keeps the rest of the package importable where only the binding
    # is wanted, and keeps an optional dependency optional.
    if name == "K26RlEnv":
        from .env import K26RlEnv
        return K26RlEnv
    if name == "K26RlVectorEnv":
        from .vector import K26RlVectorEnv
        return K26RlVectorEnv
    if name == "K26RlParallelEnv":
        from .parallel import K26RlParallelEnv
        return K26RlParallelEnv
    # The exporter reads a trained model through the package that
    # trained it, which nothing else here needs; the format writer it
    # calls needs only the standard library.
    if name == "export_policy":
        from .sb3 import export_policy
        return export_policy
    if name == "policy":
        # Through importlib rather than `from . import policy`: the
        # latter re-enters this hook and never terminates.
        import importlib

        return importlib.import_module(".policy", __name__)
    raise AttributeError("module %r has no attribute %r"
                         % (__name__, name))
