"""Gate 1: cross-driver determinism, the load-bearing check.

The same artifact, seed, and scripted action stream driven once
through the package's vector shape and once through the C reference
driver, episode output enabled in both; the two episode files must be
byte-identical. The scripted stream includes out-of-bounds box values
and out-of-arity choice indices, so a helpful clip anywhere in the
package breaks the comparison. Two package-driven runs are compared
the same way beside it (repeatability).

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent.
"""

import sys

import _gateutil as g

GATE = "gate01_cross_driver"
SEED = 20260814
N_ENVS = 3
# Enough steps to cover a full episode of the fixture (termination at
# transition 901) plus boundary resets and part of the next episode.
T = 950


def drive_python(so, out_path):
    import numpy as np
    from k26rl.vector import K26RlVectorEnv

    env = K26RlVectorEnv(so, seed=SEED, n_envs=N_ENVS)
    env.set_output(out_path)
    for t in range(T):
        thrust = np.array([[g.act_thrust(t, e)] for e in range(N_ENVS)],
                          dtype=np.float64)
        gear = np.array([g.act_gear(t, e) for e in range(N_ENVS)],
                        dtype=np.int64)
        env.step((thrust, gear))
    env.close()


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    so = g.compile_fixture("rl_pointing")
    cdriver = g.build_cdriver()

    c_file = g.WORK / "gate01_c.episode"
    py_file = g.WORK / "gate01_py.episode"
    py2_file = g.WORK / "gate01_py2.episode"
    for f in (c_file, py_file, py2_file):
        if f.exists():
            f.unlink()

    g.run([str(cdriver), "emit", str(so), str(SEED), str(N_ENVS),
           str(T), str(c_file)])
    drive_python(so, str(py_file))
    drive_python(so, str(py2_file))

    c_bytes = c_file.read_bytes()
    py_bytes = py_file.read_bytes()
    py2_bytes = py2_file.read_bytes()
    g.check(len(c_bytes) > 0, "C-driven episode file is empty")
    print("episode file sizes: C %d bytes, package %d bytes"
          % (len(c_bytes), len(py_bytes)))
    g.check(c_bytes == py_bytes,
            "package-driven episode file differs from the C-driven one")
    g.check(py_bytes == py2_bytes,
            "two package-driven episode files differ")
    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
