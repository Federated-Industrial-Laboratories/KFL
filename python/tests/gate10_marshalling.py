"""Gate 10: bit-preserving marshalling.

The C reference driver dumps the raw getter outputs per step for a
fixed seed and the shared scripted action stream; a package-driven
session over the same artifact, seed, and stream retains every
returned observation and reward array and every flag decode to the
end of the run, then compares bitwise. Retaining every array to the
end also fails any array returned as a view over a reused buffer,
because a later step's overwrite would corrupt the retained value.
The run covers an episode ending and its boundary reset, so the
flag-decode comparison includes the boundary step's rendering.

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent.
"""

import struct
import sys

import _gateutil as g

GATE = "gate10_marshalling"
SEED = 777
N_ENVS = 2
# Covers the fixture's termination at transition 901 and the boundary
# reset that follows it.
T = 905

FLAG_TERMINATED = 1
FLAG_TRUNCATED = 2
FLAG_FAULT = 4
FLAG_RESET_BOUNDARY = 8


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    import numpy as np
    from k26rl.vector import K26RlVectorEnv

    so = g.compile_fixture("rl_pointing")
    cdriver = g.build_cdriver()
    dump_file = g.WORK / "gate10_c.dump"
    if dump_file.exists():
        dump_file.unlink()
    g.run([str(cdriver), "dump", str(so), str(SEED), str(N_ENVS),
           str(T), str(dump_file)])

    env = K26RlVectorEnv(so, seed=SEED, n_envs=N_ENVS)
    obs_total = env.env_spec.obs_total

    kept_obs = []
    kept_rew = []
    kept_term = []
    kept_trunc = []
    for t in range(T):
        thrust = np.array([[g.act_thrust(t, e)] for e in range(N_ENVS)],
                          dtype=np.float64)
        gear = np.array([g.act_gear(t, e) for e in range(N_ENVS)],
                        dtype=np.int64)
        obs, rew, term, trunc, _ = env.step((thrust, gear))
        kept_obs.append(obs)
        kept_rew.append(rew)
        kept_term.append(term)
        kept_trunc.append(trunc)
    env.close()

    raw = dump_file.read_bytes()
    step_bytes = 8 * N_ENVS * obs_total + 8 * N_ENVS + 4 * N_ENVS
    g.check(len(raw) == T * step_bytes,
            "dump is %d bytes, expected %d" % (len(raw), T * step_bytes))

    saw_ending = False
    saw_boundary = False
    off = 0
    for t in range(T):
        c_obs = raw[off:off + 8 * N_ENVS * obs_total]
        off += 8 * N_ENVS * obs_total
        c_rew = raw[off:off + 8 * N_ENVS]
        off += 8 * N_ENVS
        c_flags = struct.unpack_from("<%dI" % N_ENVS, raw, off)
        off += 4 * N_ENVS

        g.check(kept_obs[t].tobytes() == c_obs,
                "observation bytes differ at step %d" % t)
        g.check(kept_rew[t].tobytes() == c_rew,
                "reward bytes differ at step %d" % t)
        for e, word in enumerate(c_flags):
            g.check(word & FLAG_FAULT == 0,
                    "unexpected fault at step %d" % t)
            g.check(bool(kept_term[t][e]) ==
                    bool(word & FLAG_TERMINATED),
                    "terminated decode differs at step %d env %d"
                    % (t, e))
            g.check(bool(kept_trunc[t][e]) ==
                    bool(word & FLAG_TRUNCATED),
                    "truncated decode differs at step %d env %d"
                    % (t, e))
            if word & (FLAG_TERMINATED | FLAG_TRUNCATED):
                saw_ending = True
            if word & FLAG_RESET_BOUNDARY:
                saw_boundary = True
                g.check(not kept_term[t][e] and not kept_trunc[t][e],
                        "boundary step decoded as an ending at step "
                        "%d" % t)
                g.check(kept_rew[t][e] == 0.0,
                        "boundary step reward at step %d" % t)

    g.check(saw_ending, "the run covered no episode ending")
    g.check(saw_boundary, "the run covered no boundary reset")
    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
