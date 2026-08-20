"""Gate 17: absence said, not hidden.

Four getters postdate the surface's first version, and an artifact
compiled before one of them arrived does not carry it. This package
binds all four, resolves each by the minor the loaded artifact
reports, and keeps the Python method present whatever that minor is,
so a consumer meets a refusal that says what is missing rather than an
attribute that is not there.

Against a stub artifact reporting version 1.0, which carries none of
the four: every one of the four calls, on both environment shapes,
raises the package's error naming the symbol, the minor that symbol
arrived at, and the minor the artifact reports. Beside them the
frozen surface is unchanged, which is the half that stops this from
being satisfied by a package that refused the artifact outright:
version 1.0 loads, its spaces are built, its spec is read, its
episode-output switch works, its reset delivers an observation, and
its status values still decode through the artifact itself.

The last arm is the one an artifact can fail rather than a consumer:
a stub reporting a minor high enough to carry all four while
exporting none is refused at load, naming the first symbol its claim
did not cover, because a consumer that believed the claim would meet
the absence as an attribute error at the far end of a training run.

Needs a C compiler for the stubs and gymnasium for the environment
classes; skips (77) when gymnasium is absent.
"""

import sys

import _gateutil as g

GATE = "gate17_abi_absence"
STUB_MINOR = 0
UNKNOWN_STATUS = 300


def main():
    g.require_gymnasium(GATE)

    import numpy as np
    from k26rl import (
        ABI_MINOR_ACTUATORS,
        ABI_MINOR_BODIES,
        ABI_MINOR_DATALINKS,
        ABI_MINOR_TAP,
        BODY_REF_ORIGIN,
        K26RlError,
    )
    from k26rl.env import K26RlEnv
    from k26rl.vector import K26RlVectorEnv

    stub = g.build_stub("default")

    # ---- the four absences, on both shapes ---------------------------
    vector = K26RlVectorEnv(stub, seed=1, n_envs=2)
    single = K26RlEnv(stub, seed=1)
    calls = (
        ("k26rl_env_tap", ABI_MINOR_TAP, lambda e: e.tap("gate17")),
        ("k26rl_env_bodies", ABI_MINOR_BODIES,
         lambda e: e.bodies(BODY_REF_ORIGIN)),
        ("k26rl_env_actuators", ABI_MINOR_ACTUATORS,
         lambda e: e.actuators()),
        ("k26rl_env_datalinks", ABI_MINOR_DATALINKS,
         lambda e: e.datalinks()),
    )
    for shape, env in (("vectorised", vector), ("single", single)):
        for symbol, minor, call in calls:
            g.check(callable(getattr(env, symbol.split("_")[-1], None)),
                    "the %s shape has no method for %s"
                    % (shape, symbol))
            try:
                call(env)
            except K26RlError as exc:
                text = str(exc)
                g.check(type(exc) is K26RlError,
                        "%s on the %s shape raised %s, expected the "
                        "base type" % (symbol, shape,
                                       type(exc).__name__))
                g.check(exc.status is None,
                        "%s carries an artifact status it never "
                        "obtained: %r" % (symbol, exc.status))
                g.check(symbol in text,
                        "the refusal does not name the symbol: %s"
                        % text)
                g.check("minor %d" % minor in text,
                        "the refusal does not name the minor %s "
                        "arrived at: %s" % (symbol, text))
                g.check("minor %d" % STUB_MINOR in text,
                        "the refusal does not name the minor the "
                        "artifact reports: %s" % text)
                print("%s: %s shape, %s" % (GATE, shape, text))
            else:
                g.check(False, "%s succeeded against an artifact "
                        "reporting version 1.%d" % (symbol, STUB_MINOR))

    # ---- the frozen surface, unchanged beside them -------------------
    g.check(vector.env_spec.abi_version == 0x00010000 | STUB_MINOR,
            "the spec blob reports %r" % (vector.env_spec.abi_version,))
    g.check(vector.single_observation_space.shape == (3,),
            "observation space %r" % (vector.single_observation_space,))
    g.check(vector.num_envs == 2, "num_envs %r" % (vector.num_envs,))
    g.check(vector.body_names == {},
            "an artifact publishing no body names produced %r"
            % (vector.body_names,))

    obs, info = vector.reset()
    g.check(obs.shape == (2, 3), "reset observation shape %r"
            % (obs.shape,))
    g.check(info == {}, "reset info %r" % (info,))
    output = g.WORK / "gate17_never_written.episode"
    vector.set_output(str(output))
    g.check(vector.output_path == str(output),
            "the output switch did not record its path: %r"
            % (vector.output_path,))
    vector.set_output(None)

    # Status decode still routes through the artifact: the stub's step
    # mints a value this package was never taught, and the message is
    # the stub's own.
    import ctypes

    reference = ctypes.CDLL(str(stub))
    reference.k26rl_status_str.restype = ctypes.c_char_p
    reference.k26rl_status_str.argtypes = [ctypes.c_int]
    minted = reference.k26rl_status_str(UNKNOWN_STATUS).decode()
    try:
        vector.step((np.zeros((2, 1)), np.zeros(2, dtype=np.int64)))
    except K26RlError as exc:
        g.check(exc.status == UNKNOWN_STATUS and exc.message == minted,
                "the frozen surface's status decode changed: %r"
                % (exc.message,))
    else:
        g.check(False, "the stub's minted status was discarded")

    vector.close()
    single.close()
    # The refusals are the artifact's absence, not this package's
    # closed guard, so a closed environment still refuses first.
    try:
        vector.bodies(BODY_REF_ORIGIN)
    except K26RlError as exc:
        g.check("closed" in str(exc),
                "a closed environment refused with the absence "
                "message instead of the closed one: %s" % exc)
    else:
        g.check(False, "a closed environment served the body getter")
    print("%s: version 1.%d loads, builds its spaces, resets, switches "
          "output, and decodes its own status values" % (GATE,
                                                         STUB_MINOR))

    # ---- an artifact whose claim outruns its exports -----------------
    liar = g.build_stub("minor7",
                        ["STUB_ABI_MINOR=%d" % ABI_MINOR_DATALINKS])
    try:
        K26RlVectorEnv(liar, seed=1, n_envs=1)
    except K26RlError as exc:
        text = str(exc)
        g.check("k26rl_env_tap" in text,
                "the refusal does not name the symbol the claimed "
                "minor carries: %s" % text)
        g.check("minor %d" % ABI_MINOR_DATALINKS in text,
                "the refusal does not name the claimed minor: %s"
                % text)
        print("%s: an artifact claiming minor %d without the symbols "
              "is refused: %s" % (GATE, ABI_MINOR_DATALINKS, text))
    else:
        g.check(False, "an artifact claiming a minor it does not "
                "carry was accepted")

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
