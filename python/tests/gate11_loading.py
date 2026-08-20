"""Gate 11: loading discipline.

A stub artifact reporting ABI major 2 is refused naming both majors;
a stub with one symbol removed is refused naming the missing symbol;
a missing artifact file is refused as the package's typed error with
the loader's OSError chained.

The isolation pin: two stubs built without export maps both export a
global double named k26rl_stub_pair_scale, with different values,
read by their observation getters through the GOT. Loaded RTLD_GLOBAL
in this order, the second stub's reads would resolve against the
first's definition and its observations take the first's scale;
loaded RTLD_LOCAL, as the binding does, each stub sees only its own
value, and this gate asserts those isolated values. Beside it, two
compiled artifacts loaded together each produce streams bit-identical
to their solo runs (coexistence, not a loader-mode pin: their export
maps confine their globals under either mode).

Use after close: every stateful public method and every spec-reading
property of a closed environment raises the Python-side refusal
before any artifact call, the getters that postdate the frozen set
included, so a closed environment reports being closed rather than
reporting whatever the artifact does or does not carry; the
seeds_held, output_path and tap_name records stay readable, because
they record where a finished run's output and telemetry went; close
itself is idempotent.

The stub arms need only a C compiler and gymnasium; the coexistence
arm also needs the built compiler and stack archives.
"""

import sys

import _gateutil as g

GATE = "gate11_loading"


def main():
    g.require_gymnasium(GATE)
    g.require_stack(GATE)

    import numpy as np
    from k26rl import K26RlError
    from k26rl.env import K26RlEnv
    from k26rl.vector import K26RlVectorEnv

    # ---- ABI major mismatch, naming both majors ----------------------
    major2 = g.build_stub("major2", ["STUB_ABI_MAJOR2"])
    try:
        K26RlVectorEnv(major2, seed=1, n_envs=1)
    except K26RlError as exc:
        g.check("major 2" in str(exc) and "major 1" in str(exc),
                "major-mismatch refusal does not name both majors: %s"
                % exc)
    else:
        g.check(False, "ABI major 2 was accepted")

    # ---- missing symbol, named ---------------------------------------
    nosym = g.build_stub("nosym", ["STUB_OMIT_SYMBOL"])
    try:
        K26RlVectorEnv(nosym, seed=1, n_envs=1)
    except K26RlError as exc:
        g.check("k26rl_env_reset_seeded" in str(exc),
                "missing-symbol refusal does not name the symbol: %s"
                % exc)
    else:
        g.check(False, "a missing symbol was accepted")

    # ---- missing file, typed with the OSError chained ----------------
    absent = str(g.WORK / "no_such_artifact.so")
    try:
        K26RlVectorEnv(absent, seed=1, n_envs=1)
    except K26RlError as exc:
        g.check(absent in str(exc),
                "missing-file refusal does not name the path: %s"
                % exc)
        g.check(isinstance(exc.__cause__, OSError),
                "the loader's OSError is not chained: %r"
                % (exc.__cause__,))
    else:
        g.check(False, "a missing artifact file was accepted")

    # ---- the RTLD_LOCAL pin: an unconfined shared symbol name --------
    # Both pair stubs export the global k26rl_stub_pair_scale (no
    # export map confines it) and scale their observations by it.
    # Loaded RTLD_GLOBAL in this order, the second's reads would
    # resolve against the first's definition; RTLD_LOCAL keeps each
    # stub on its own value, asserted here.
    pair_a = g.build_stub("pair_a", ["STUB_PAIR_SCALE=2.0"])
    pair_b = g.build_stub("pair_b", ["STUB_PAIR_SCALE=3.0"])
    base = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    env_a = K26RlVectorEnv(pair_a, seed=1, n_envs=1)
    env_b = K26RlVectorEnv(pair_b, seed=1, n_envs=1)
    obs_b, _ = env_b.reset()
    g.check(obs_b.tobytes() == (base * 3.0).tobytes(),
            "the second pair stub does not read its own scale "
            "symbol: %r" % (obs_b,))
    obs_a, _ = env_a.reset()
    g.check(obs_a.tobytes() == (base * 2.0).tobytes(),
            "the first pair stub does not read its own scale "
            "symbol: %r" % (obs_a,))
    env_a.close()
    env_b.close()

    # ---- two compiled artifacts in one process, coexisting -----------
    point_so = g.compile_fixture("rl_pointing")
    fault_so = g.compile_fixture("rl_shimfault", g.FAULT_KFL)
    n = 2
    steps = 4

    def run_point(env, t):
        thrust = np.full((n, 1), 0.25, dtype=np.float64)
        gear = np.full(n, 1, dtype=np.int64)
        obs, rew, _, _, _ = env.step((thrust, gear))
        return obs.tobytes() + rew.tobytes()

    def run_fault(env, t):
        obs, rew, _, _, _ = env.step(
            np.full((n, 1), 0.2, dtype=np.float64))
        return obs.tobytes() + rew.tobytes()

    env_p = K26RlVectorEnv(point_so, seed=9, n_envs=n)
    solo_p = b"".join(run_point(env_p, t) for t in range(steps))
    env_p.close()
    env_f = K26RlVectorEnv(fault_so, seed=9, n_envs=n)
    solo_f = b"".join(run_fault(env_f, t) for t in range(steps))
    env_f.close()

    env_p = K26RlVectorEnv(point_so, seed=9, n_envs=n)
    env_f = K26RlVectorEnv(fault_so, seed=9, n_envs=n)
    both_p = []
    both_f = []
    for t in range(steps):
        both_p.append(run_point(env_p, t))
        both_f.append(run_fault(env_f, t))
    env_p.close()
    env_f.close()
    g.check(b"".join(both_p) == solo_p,
            "the first artifact's streams changed beside a second "
            "artifact")
    g.check(b"".join(both_f) == solo_f,
            "the second artifact's streams changed beside the first")

    # ---- use-after-close, both shapes --------------------------------
    def expect_closed(fn, what):
        try:
            fn()
        except K26RlError as exc:
            g.check("closed" in str(exc),
                    "%s after close: %s" % (what, exc))
        else:
            g.check(False, "%s on a closed environment succeeded"
                    % what)

    refusing_properties = ("env_spec", "control_dt",
                           "obs_channel_names", "obs_channel_kinds",
                           "body_names", "on_fault")

    venv = K26RlVectorEnv(point_so, seed=11, n_envs=n)
    venv.close()
    venv.close()
    acts = (np.zeros((n, 1)), np.zeros(n, dtype=np.int64))
    expect_closed(lambda: venv.step(acts), "vector step")
    expect_closed(lambda: venv.reset(), "vector reset")
    expect_closed(lambda: venv.reset(seed=12), "vector seeded reset")
    expect_closed(lambda: venv.set_output("/tmp/never.episode"),
                  "vector set_output")
    expect_closed(lambda: venv.tap("k26rl_gate11_never"), "vector tap")
    expect_closed(lambda: venv.bodies(0), "vector bodies")
    expect_closed(lambda: venv.actuators(), "vector actuators")
    for prop in refusing_properties:
        expect_closed(lambda prop=prop: getattr(venv, prop),
                      "vector property %s" % prop)
    # The record properties stay readable after close: they say where
    # the finished run's output and telemetry went.
    g.check(venv.seeds_held == frozenset({11}),
            "seeds_held unreadable or wrong after close: %r"
            % (venv.seeds_held,))
    g.check(venv.output_path is None,
            "output_path unreadable or wrong after close: %r"
            % (venv.output_path,))
    g.check(venv.tap_name is None,
            "tap_name unreadable or wrong after close: %r"
            % (venv.tap_name,))

    senv = K26RlEnv(point_so, seed=13)
    senv.close()
    senv.close()
    expect_closed(lambda: senv.step((np.zeros(1), 0)), "single step")
    expect_closed(lambda: senv.reset(), "single reset")
    expect_closed(lambda: senv.set_output("/tmp/never.episode"),
                  "single set_output")
    expect_closed(lambda: senv.tap("k26rl_gate11_never"), "single tap")
    expect_closed(lambda: senv.bodies(0), "single bodies")
    expect_closed(lambda: senv.actuators(), "single actuators")
    for prop in refusing_properties:
        expect_closed(lambda prop=prop: getattr(senv, prop),
                      "single property %s" % prop)
    g.check(senv.seeds_held == frozenset({13}),
            "seeds_held unreadable or wrong after close: %r"
            % (senv.seeds_held,))
    g.check(senv.output_path is None,
            "output_path unreadable or wrong after close: %r"
            % (senv.output_path,))
    g.check(senv.tap_name is None,
            "tap_name unreadable or wrong after close: %r"
            % (senv.tap_name,))

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
