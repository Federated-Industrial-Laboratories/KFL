"""Gate 6: seed semantics.

reset(seed=S) delivers episode 0 of key S by three routes chosen from
handle state (no-op on an untouched handle at S, seeded reset for a
never-held key, recreate for a held one), and all three yield
bit-identical streams. Route 2 runs on a stepped handle on purpose: a
seeded reset must restore the create-time world baseline whole before
applying the new episode's draws, and a handle nothing has perturbed
would pin nothing, so this is the first behavioural check of that
restore (the artifact's reset baseline contract) anywhere in the
suite. Also pinned: plain reset() on an untouched handle preserves
episode 0 (construct-reset-run bit-identical to construct-run) and on
a touched handle advances episodes; the held-seed record is exposed
as read-only data, accumulates across a recreate, and a seed held
before the recreate still routes as held rather than being fed to the
artifact's seeded reset as if fresh; the vector seed-list refusal;
the non-empty options refusal; the out-of-range seed refusals; and
the recreate-while-output-enabled refusal, tested beside a repeated
reset(seed=S) to the current key with output enabled succeeding as a
no-op.

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent.
"""

import sys

import _gateutil as g

GATE = "gate06_seeding"
K = 5
SEED_S = 1001
SEED_A = 2002


class Spy:
    """Counts the artifact calls the routes are allowed to make.
    Wraps the bound methods of one session's artifact object; the
    private attribute is deliberate, because which route ran is
    otherwise invisible when all routes yield identical streams."""

    def __init__(self, env):
        art = env._session.artifact
        self.counts = {}
        for name in ("reset", "reset_seeded", "create", "destroy"):
            self.counts[name] = 0
            setattr(art, name, self._wrap(art, name))

    def _wrap(self, art, name):
        original = getattr(art, name)

        def wrapper(*args, **kwargs):
            self.counts[name] += 1
            return original(*args, **kwargs)
        return wrapper


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    import numpy as np
    from k26rl import K26RlError
    from k26rl.vector import K26RlVectorEnv

    so = g.compile_fixture("rl_pointing")
    n = 2

    def V(seed):
        return K26RlVectorEnv(so, seed=seed, n_envs=n)

    def run(env, steps=K):
        """A fixed benign action stream; the returned bytes are the
        retained observation and reward streams."""
        chunks = []
        thrust = np.full((n, 1), 0.25, dtype=np.float64)
        gear = np.ones(n, dtype=np.int64)
        for _ in range(steps):
            obs, rew, _, _, _ = env.step((thrust, gear))
            chunks.append(obs.tobytes())
            chunks.append(rew.tobytes())
        return b"".join(chunks)

    # ---- construct-run and construct-reset-run are bit-identical -----
    env_a = V(SEED_S)
    stream_construct = run(env_a)
    env_a.close()

    env_b = V(SEED_S)
    obs0_s, _ = env_b.reset()
    stream_reset = run(env_b)
    g.check(stream_reset == stream_construct,
            "plain reset() on an untouched handle did not preserve "
            "episode 0")
    env_b.close()

    # ---- the three routes yield bit-identical streams ----------------
    env_c = V(SEED_A)
    run(env_c, 3)
    spy = Spy(env_c)

    # Route 2 on a stepped handle: the baseline-restore pin.
    obs, _ = env_c.reset(seed=SEED_S)
    g.check(spy.counts["reset_seeded"] == 1
            and spy.counts["create"] == 0,
            "route 2 did not use the artifact's seeded reset")
    g.check(obs.tobytes() == obs0_s.tobytes(),
            "route 2 initial observation differs")

    # Route 1: repeated seeded reset to the current key is a no-op.
    obs, _ = env_c.reset(seed=SEED_S)
    g.check(spy.counts["reset_seeded"] == 1
            and spy.counts["reset"] == 0
            and spy.counts["create"] == 0,
            "route 1 made an artifact call")
    g.check(obs.tobytes() == obs0_s.tobytes(),
            "route 1 initial observation differs")
    g.check(run(env_c) == stream_construct,
            "route 2 stream differs from the construct-run stream")

    # Route 3: a held seed on a touched handle recreates.
    obs, _ = env_c.reset(seed=SEED_S)
    g.check(spy.counts["destroy"] == 1 and spy.counts["create"] == 1
            and spy.counts["reset_seeded"] == 1,
            "route 3 did not recreate (calls: %r)" % spy.counts)
    g.check(obs.tobytes() == obs0_s.tobytes(),
            "route 3 initial observation differs")
    g.check(run(env_c) == stream_construct,
            "route 3 stream differs from the construct-run stream")

    # ---- the held-seed record and its carry across recreates ---------
    held = env_c.seeds_held
    g.check(isinstance(held, frozenset), "seeds_held is %s"
            % type(held).__name__)
    g.check(held == {SEED_A, SEED_S}, "seeds_held %r" % held)

    # SEED_A was held only by the handle destroyed above; it must
    # still route as held (recreate), never reach the artifact's
    # seeded reset as if fresh.
    env_ref = V(SEED_A)
    obs0_a, _ = env_ref.reset()
    stream_a = run(env_ref)
    env_ref.close()

    obs, _ = env_c.reset(seed=SEED_A)
    g.check(spy.counts["reset_seeded"] == 1
            and spy.counts["create"] == 2,
            "a seed held before the recreate was fed to the seeded "
            "reset (calls: %r)" % spy.counts)
    g.check(obs.tobytes() == obs0_a.tobytes()
            and run(env_c) == stream_a,
            "the recreate for a pre-recreate seed diverged")
    g.check(env_c.seeds_held == {SEED_A, SEED_S},
            "seeds_held after recreates %r" % env_c.seeds_held)
    env_c.close()

    # ---- plain reset() on a touched handle advances episodes ---------
    env_d = V(SEED_S)
    run(env_d, 1)
    spy_d = Spy(env_d)
    obs1, _ = env_d.reset()
    g.check(spy_d.counts["reset"] == 1, "touched plain reset route")
    g.check(obs1.tobytes() != obs0_s.tobytes(),
            "plain reset on a touched handle did not advance the "
            "episode")
    env_d.close()
    env_e = V(SEED_S)
    run(env_e, 1)
    obs1b, _ = env_e.reset()
    g.check(obs1b.tobytes() == obs1.tobytes(),
            "the advanced episode is not deterministic")
    env_e.close()

    # ---- refusals ----------------------------------------------------
    env_f = V(77)

    def expect(exc_type, needle, what, fn):
        try:
            fn()
        except exc_type as exc:
            g.check(needle in str(exc),
                    "%s refusal message: %s" % (what, exc))
        except Exception as exc:
            g.check(False, "%s raised %s: %s"
                    % (what, type(exc).__name__, exc))
        else:
            g.check(False, "%s was not refused" % what)

    expect(ValueError, "2**64", "seed 2**64",
           lambda: env_f.reset(seed=2 ** 64))
    expect(ValueError, "2**64", "seed -1",
           lambda: env_f.reset(seed=-1))
    expect(TypeError, "integer", "float seed",
           lambda: env_f.reset(seed=1.5))
    expect(TypeError, "one integer seed", "seed list",
           lambda: env_f.reset(seed=[1, 2]))
    expect(ValueError, "alpha", "non-empty options",
           lambda: env_f.reset(options={"alpha": 1}))
    env_f.reset(options={})
    env_f.reset(options=None)
    env_f.close()

    # ---- output-enabled interplay ------------------------------------
    out = g.WORK / "gate06.episode"
    if out.exists():
        out.unlink()
    env_h = V(88)
    env_h.set_output(str(out))
    spy_h = Spy(env_h)
    # A repeated seeded reset to the current key succeeds as a no-op
    # with output enabled: no call, nothing recorded twice.
    obs, _ = env_h.reset(seed=88)
    g.check(all(v == 0 for v in spy_h.counts.values()),
            "the current-key no-op made artifact calls with output "
            "enabled (calls: %r)" % spy_h.counts)
    run(env_h, 1)
    # A held seed now needs a recreate, which would close the file
    # mid-recording; refused with directions.
    expect(K26RlError, "output", "recreate while output is enabled",
           lambda: env_h.reset(seed=88))
    # Following the message's directions: reach a boundary, disable
    # output, and the seeded reset proceeds by recreate.
    env_h.reset()
    env_h.set_output(None)
    obs, _ = env_h.reset(seed=88)
    g.check(spy_h.counts["create"] == 1 and spy_h.counts["destroy"] == 1,
            "the post-disable seeded reset did not recreate "
            "(calls: %r)" % spy_h.counts)
    g.check(out.exists() and out.stat().st_size > 0,
            "no episode file was recorded")
    env_h.close()

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
