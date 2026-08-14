"""Gate 4: status decode through the artifact, never a local table.

A stub artifact mints a status value this package was never taught
(300) and alters the decode text of a known value (the seed-reuse
status). The package must raise the base type for the unknown value
and the typed error for the known one, each with a message equal to
the loaded artifact's own k26rl_status_str output, so a status table
cached at package build time fails here even for known values.

A second stub variant decodes the minted value to NULL; the raise
must carry the bare numeric rendering, the package inventing no name.

Needs a C compiler for the stubs and gymnasium for the environment
classes; skips (77) when gymnasium is absent.
"""

import ctypes
import sys

import _gateutil as g

GATE = "gate04_status_decode"
UNKNOWN_STATUS = 300
E_SEED_REUSE = 4


def main():
    g.require_gymnasium(GATE)

    import numpy as np
    from k26rl import K26RlError, K26RlSeedReuseError
    from k26rl.vector import K26RlVectorEnv

    stub = g.build_stub("default")

    # The decode reference comes from the stub itself, reached
    # directly rather than through the package.
    ref = ctypes.CDLL(str(stub))
    ref.k26rl_status_str.restype = ctypes.c_char_p
    ref.k26rl_status_str.argtypes = [ctypes.c_int]
    unknown_text = ref.k26rl_status_str(UNKNOWN_STATUS).decode()
    altered_text = ref.k26rl_status_str(E_SEED_REUSE).decode()

    env = K26RlVectorEnv(stub, seed=1, n_envs=2)

    # A value outside the known set raises the base type with the
    # artifact's decode.
    actions = (np.zeros((2, 1)), np.zeros(2, dtype=np.int64))
    try:
        env.step(actions)
    except K26RlError as exc:
        g.check(type(exc) is K26RlError,
                "unknown status raised %s, expected the base type"
                % type(exc).__name__)
        g.check(exc.status == UNKNOWN_STATUS,
                "unknown status value %r" % exc.status)
        g.check(exc.message == unknown_text,
                "unknown status message %r, artifact says %r"
                % (exc.message, unknown_text))
    else:
        g.check(False, "unknown status was discarded")

    # A known value whose stub message is altered surfaces the
    # altered text through the typed error.
    try:
        env.reset(seed=99)
    except K26RlSeedReuseError as exc:
        g.check(exc.status == E_SEED_REUSE,
                "seed-reuse status value %r" % exc.status)
        g.check(exc.message == altered_text,
                "seed-reuse message %r, artifact says %r"
                % (exc.message, altered_text))
    else:
        g.check(False, "the stub's seed-reuse refusal was discarded")

    env.close()

    # A decoder returning NULL for a value yields the bare numeric
    # rendering; no name is invented for it.
    nulldec = g.build_stub("nulldecode", ["STUB_NULL_DECODE"])
    env_n = K26RlVectorEnv(nulldec, seed=1, n_envs=2)
    try:
        env_n.step(actions)
    except K26RlError as exc:
        g.check(type(exc) is K26RlError,
                "NULL-decoded status raised %s, expected the base "
                "type" % type(exc).__name__)
        g.check(exc.status == UNKNOWN_STATUS,
                "NULL-decoded status value %r" % exc.status)
        g.check(exc.message == "status %d" % UNKNOWN_STATUS,
                "NULL-decoded message %r, expected the bare numeric "
                "rendering" % exc.message)
    else:
        g.check(False, "the NULL-decoded status was discarded")
    env_n.close()

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
