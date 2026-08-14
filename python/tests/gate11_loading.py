"""Gate 11: loading discipline.

A stub artifact reporting ABI major 2 is refused naming both majors;
a stub with one symbol removed is refused naming the missing symbol;
two artifacts loaded in one process each produce streams
bit-identical to their solo runs, pinning the RTLD_LOCAL isolation;
and every stateful public method of a closed environment raises the
Python-side use-after-close refusal before any artifact call, with
close itself idempotent.

The stub halves need only a C compiler and gymnasium; the isolation
half also needs the built compiler and stack archives.
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

    # ---- two artifacts in one process, isolated ----------------------
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

    venv = K26RlVectorEnv(point_so, seed=11, n_envs=n)
    venv.close()
    venv.close()
    acts = (np.zeros((n, 1)), np.zeros(n, dtype=np.int64))
    expect_closed(lambda: venv.step(acts), "vector step")
    expect_closed(lambda: venv.reset(), "vector reset")
    expect_closed(lambda: venv.reset(seed=12), "vector seeded reset")
    expect_closed(lambda: venv.set_output("/tmp/never.episode"),
                  "vector set_output")

    senv = K26RlEnv(point_so, seed=13)
    senv.close()
    senv.close()
    expect_closed(lambda: senv.step((np.zeros(1), 0)), "single step")
    expect_closed(lambda: senv.reset(), "single reset")
    expect_closed(lambda: senv.set_output("/tmp/never.episode"),
                  "single set_output")

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
