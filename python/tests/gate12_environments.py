"""Gate 12: the validation environments through both drivers.

The two environments whose actions move the dynamics, driven once
through the package's vector shape and once through the C reference
driver with the same seed and the same scripted action stream, episode
output enabled in both: the two episode files must be byte-identical.
Beside that, two package-driven runs at one seed must match each other,
and a run at a different seed must differ, so the byte identity is
determinism rather than degeneracy.

The last check is the ordering pin from the package's side: two runs
whose action streams first differ at step t must produce observation
streams that first differ at step t itself. An implementation that
applied the action one step late would pass every comparison above and
fail this one.

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent.
"""

import sys

import _gateutil as g

GATE = "gate12_environments"
SEED = 20260814
ALT_SEED = 20260815
N_ENVS = 2
T = 60

ENVIRONMENTS = ("orbit_transfer", "stationkeeping")


def drive_python(so, seed, out_path, first_diff_at=None):
    """Drive the vector shape for T steps. With first_diff_at set, the
    action stream switches to a second value from that step, and the
    per-step observations are returned instead of an episode file."""
    import numpy as np
    from k26rl.vector import K26RlVectorEnv

    env = K26RlVectorEnv(so, seed=seed, n_envs=N_ENVS)
    if out_path is not None:
        env.set_output(out_path)
    seen = []
    for t in range(T):
        if first_diff_at is not None and t >= first_diff_at:
            values = [[3.0] for _ in range(N_ENVS)]
        else:
            values = [[g.act_thrust(t, e)] for e in range(N_ENVS)]
        obs, _rew, _term, _trunc, _info = env.step(
            np.array(values, dtype=np.float64))
        seen.append(obs.copy())
    env.close()
    return seen


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    cdriver = g.build_cdriver()
    for name in ENVIRONMENTS:
        so = g.compile_fixture(name)

        c_file = g.WORK / ("%s_c.episode" % name)
        py_file = g.WORK / ("%s_py.episode" % name)
        py2_file = g.WORK / ("%s_py2.episode" % name)
        alt_file = g.WORK / ("%s_alt.episode" % name)
        for f in (c_file, py_file, py2_file, alt_file):
            if f.exists():
                f.unlink()

        g.run([str(cdriver), "emit", str(so), str(SEED), str(N_ENVS),
               str(T), str(c_file)])
        drive_python(so, SEED, str(py_file))
        drive_python(so, SEED, str(py2_file))
        drive_python(so, ALT_SEED, str(alt_file))

        c_bytes = c_file.read_bytes()
        py_bytes = py_file.read_bytes()
        py2_bytes = py2_file.read_bytes()
        alt_bytes = alt_file.read_bytes()
        g.check(len(c_bytes) > 0, "%s: C-driven episode file is empty"
                % name)
        print("%s: episode file sizes: C %d bytes, package %d bytes"
              % (name, len(c_bytes), len(py_bytes)))
        g.check(c_bytes == py_bytes,
                "%s: package-driven episode file differs from the "
                "C-driven one" % name)
        g.check(py_bytes == py2_bytes,
                "%s: two package-driven episode files differ" % name)
        g.check(py_bytes != alt_bytes,
                "%s: a different seed produced the same bytes" % name)

        # The ordering pin. The streams agree up to step 20 and differ
        # from it, so the observations must first differ at step 20.
        at = 20
        base = drive_python(so, SEED, None)
        alt = drive_python(so, SEED, None, first_diff_at=at)
        first = None
        for t in range(T):
            if base[t].tobytes() != alt[t].tobytes():
                first = t
                break
        g.check(first == at,
                "%s: observations first differ at step %s, expected %d"
                % (name, first, at))
        print("%s: action changed at step %d, observation changed at "
              "step %s" % (name, at, first))

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
