"""Gate 13: the multi-agent parallel shape.

The two-agent fixture declares agents of deliberately different widths,
five observation components and two action channels against ten and
one, so a right slicing and a swapped one cannot agree on either
vector.

The arms: a two-agent artifact loads where the single-agent shapes
refuse it, and refuses in the other direction with a message naming
the count found; the per-agent spaces come out of the published
slices; every per-agent array, reward, flag and info key is compared
bitwise against a raw handle driven with the same seed and the same
concatenated action vector, with the two agents' values asserted to
differ at every step so a mis-keyed dictionary cannot look correct;
the action dictionary is shown to be load-bearing by a run whose
agents' actions are exchanged producing a different stream; ending is
per environment and identical for every agent, on termination, on
truncation and on fault, with the roster emptying and step-after-end
refused; an artifact whose slices do not partition its vectors is
refused naming what it found; an agent that publishes no name is
refused rather than named by invention; a single-agent artifact is
served by the single-agent shape exactly as before; importing the
package does not import the optional dependency; and PettingZoo's own
parallel API conformance check runs against the fixture.

Skips (77) when the built compiler, the stack archives, gymnasium or
pettingzoo are absent.
"""

import ctypes
import struct
import subprocess
import sys

import _gateutil as g

GATE = "gate13_parallel"
SEED = 20260816
T = 24

# Two agents of different widths that terminate on the step count, so
# every agent's terminated flag is exercised rather than only the
# horizon's truncation. `alpha` owns one observation statement and two
# actions, `beta` two observation statements and one action.
PAIR_TERM_KFL = """\
form RL_PAIRTERM
fn world pair_term_world
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body a_craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7546.0
    astro_body b_craft gm=1.0 parent=earth pos_x=7.4e6 vel_y=7340.0
    episode
        control_dt 1.0
        horizon 100
        terminated when episode.steps > 2
    end
    agent alpha
        action push box -1.0 1.0 default 0.0
        action bank box -1.0 1.0 default 0.0
        observe a_craft from earth mode=geometric as trk
        objective
            reward 1.0
            terminal 5.0
        end
    end
    agent beta
        action push box -1.0 1.0 default 0.0
        observe b_craft from earth mode=geometric as trk
        observe b_craft from a_craft mode=geometric as rel
        objective
            reward 2.0
            terminal 7.0
        end
    end
    on_step
        a_craft.vel_x = a_craft.vel_x + alpha.push
        a_craft.vel_z = a_craft.vel_z + bank
        b_craft.vel_x = b_craft.vel_x + beta.push
    end
end
end
"""

# `beta`'s objective divides by (1.0 - push): finite at the default
# action, a division by zero when a caller drives beta's push to
# exactly 1.0. The fault is the environment's, not one agent's.
PAIR_FAULT_KFL = """\
form RL_PAIRFAULT
fn world pair_fault_world
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body a_craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7546.0
    astro_body b_craft gm=1.0 parent=earth pos_x=7.4e6 vel_y=7340.0
    episode
        control_dt 1.0
        horizon 8
    end
    agent alpha
        action push box -1.0 1.0 default 0.0
        action bank box -1.0 1.0 default 0.0
        observe a_craft from earth mode=geometric as trk
        objective
            reward 1.0
        end
    end
    agent beta
        action push box 0.0 4.0 default 0.0
        observe b_craft from earth mode=geometric as trk
        observe b_craft from a_craft mode=geometric as rel
        objective
            reward 1.0 / (1.0 - push)
        end
    end
    on_step
        a_craft.vel_x = a_craft.vel_x + alpha.push
        a_craft.vel_z = a_craft.vel_z + bank
        b_craft.vel_x = b_craft.vel_x + beta.push
    end
end
end
"""

# `alpha` declares actions and an objective but no observation, so
# nothing in the spec publishes its name: agent names reach a consumer
# only as the qualifier on an observation channel name.
UNNAMED_KFL = """\
form RL_UNNAMED
fn world unnamed_world
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body a_craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7546.0
    astro_body b_craft gm=1.0 parent=earth pos_x=7.4e6 vel_y=7340.0
    episode
        control_dt 1.0
        horizon 8
    end
    agent alpha
        action push box -1.0 1.0 default 0.0
        objective
            reward 1.0
        end
    end
    agent beta
        action push box -1.0 1.0 default 0.0
        observe b_craft from earth mode=geometric as trk
        objective
            reward 2.0
        end
    end
    on_step
        a_craft.vel_x = a_craft.vel_x + alpha.push
        b_craft.vel_x = b_craft.vel_x + beta.push
    end
end
end
"""

# `alpha` declares no action, which is a legal observation-only agent
# and gives it a zero-width action slice.
WATCHER_KFL = """\
form RL_WATCHER
fn world watcher_world
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body a_craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7546.0
    astro_body b_craft gm=1.0 parent=earth pos_x=7.4e6 vel_y=7340.0
    episode
        control_dt 1.0
        horizon 8
    end
    agent alpha
        observe a_craft from earth mode=geometric as trk
        objective
            reward 1.0
        end
    end
    agent beta
        action push box -1.0 1.0 default 0.0
        observe b_craft from earth mode=geometric as trk
        observe b_craft from a_craft mode=geometric as rel
        objective
            reward 2.0
        end
    end
    on_step
        b_craft.vel_x = b_craft.vel_x + beta.push
    end
end
end
"""

# The import-isolation probe: importing the package, and reaching the
# single-agent class through it, must not pull in the optional
# dependency.
IMPORT_PROBE = """\
import sys
sys.path.insert(0, %r)
import k26rl
assert "pettingzoo" not in sys.modules, "import k26rl imported pettingzoo"
k26rl.K26RlEnv
assert "pettingzoo" not in sys.modules, "K26RlEnv imported pettingzoo"
k26rl.K26RlParallelEnv
assert "pettingzoo" in sys.modules, \\
    "K26RlParallelEnv did not import pettingzoo"
print("ok")
"""


# ---- the scripted action stream -------------------------------------
#
# Pure functions of the step index, distinct per agent and per channel
# and different at every step, so an action written onto another
# agent's channels changes the trajectory and the bitwise comparison
# against the raw handle sees it.

def act_alpha(t):
    return (((t * 5) % 11) / 11.0 - 0.5, ((t * 3) % 7) / 7.0 - 0.5)


def act_beta(t):
    return (((t * 9) % 13) / 13.0 - 0.5,)


class Arms:
    """Counts the arms so the gate reports what it checked."""

    def __init__(self):
        self.n = 0

    def check(self, cond, what):
        self.n += 1
        g.check(cond, what)

    def completed(self, what):
        """An arm whose assertion is that the call before it returned
        rather than raising; counted so the tally matches the arms."""
        self.n += 1
        del what


def expect_refusal(arms, call, needles, what, exc_type=Exception):
    """Run call and require it to raise ``exc_type`` naming each
    needle. The type matters as much as the text: a bare lookup
    failure inside the marshalling would carry the agent's name too,
    and an arm that accepted it could not tell a refusal that explains
    itself from one that fell over."""
    try:
        call()
    except Exception as exc:
        arms.check(isinstance(exc, exc_type),
                   "%s raised %s, expected %s: %s"
                   % (what, type(exc).__name__, exc_type.__name__, exc))
        for needle in needles:
            arms.check(needle in str(exc),
                       "%s refusal does not name %r: %s"
                       % (what, needle, exc))
        return
    arms.check(False, "%s was not refused" % what)


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)
    g.require_pettingzoo(GATE)

    import numpy as np
    from gymnasium import spaces as gym_spaces
    import gymnasium

    from k26rl import _abi, _spec
    from k26rl.env import K26RlEnv
    from k26rl.parallel import K26RlParallelEnv
    from k26rl.vector import K26RlVectorEnv
    from k26rl._errors import K26RlFaultError

    arms = Arms()
    pair = g.compile_fixture("rl_multi_agent")
    single = g.compile_fixture("rl_pointing")

    # ---- the artifact the single-agent shapes refused ----------------
    for shape, build in (
            ("K26RlEnv", lambda: K26RlEnv(pair, seed=SEED)),
            ("K26RlVectorEnv",
             lambda: K26RlVectorEnv(pair, seed=SEED, n_envs=1))):
        expect_refusal(arms, build,
                       ["agent count 2", "K26RlParallelEnv"],
                       "%s on a two-agent artifact" % shape)

    expect_refusal(arms, lambda: K26RlParallelEnv(single, seed=SEED),
                   ["agent count 1", "K26RlEnv"],
                   "the parallel shape on a single-agent artifact")

    env = K26RlParallelEnv(pair, seed=SEED)
    arms.check(env.possible_agents == ["leader", "follower"],
               "published agents are %s" % (env.possible_agents,))
    arms.check(env.agents == env.possible_agents,
               "the roster starts short of the possible agents")
    arms.check(env.num_agents == 2 and env.max_num_agents == 2,
               "agent counts are %d and %d"
               % (env.num_agents, env.max_num_agents))

    # ---- the per-agent spaces come out of the published slices -------
    widths = {"leader": (5, 2), "follower": (10, 1)}
    for name, (obs_n, act_n) in widths.items():
        arms.check(env.observation_space(name).shape == (obs_n,),
                   "%s observation space is %s"
                   % (name, env.observation_space(name).shape))
        space = env.action_space(name)
        arms.check(isinstance(space, gym_spaces.Box)
                   and space.shape == (act_n,),
                   "%s action space is %r" % (name, space))
        arms.check(np.all(space.low == -1.0)
                   and np.all(space.high == 1.0),
                   "%s action bounds are %r, %r"
                   % (name, space.low, space.high))
        arms.check(env.observation_space(name)
                   is env.observation_space(name),
                   "%s observation space is rebuilt per call" % name)
        arms.check(env.action_space(name) is env.action_space(name),
                   "%s action space is rebuilt per call" % name)
    arms.check(all(text.startswith(name + ".")
                   for name, texts in env.agent_obs_channel_names.items()
                   for text in texts),
               "an agent's channels do not all carry its own qualifier: "
               "%s" % (env.agent_obs_channel_names,))

    # ---- every returned value against a raw handle -------------------
    # The oracle is the frozen getter surface driven with the same
    # seed and the same concatenated action vector; the slice offsets
    # are re-read here from the artifact's own blob rather than taken
    # from the wrapper under test.
    art = _abi.Artifact(pair)
    handle = art.create(SEED, 1)
    raw_spec = _spec.parse(art.spec_blob(handle))
    obs_at = dict(zip(["leader", "follower"],
                      _spec.slices_by_agent(raw_spec.agent_obs_slices,
                                            raw_spec.agent_count)))
    obs_buf = np.empty(raw_spec.obs_total, dtype=np.float64)
    rew_buf = np.empty(raw_spec.agent_count, dtype=np.float64)
    flag_buf = np.empty(1, dtype=np.uint32)

    def raw_step(flat):
        art.step(handle, flat.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)))
        art.obs(handle, obs_buf.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)))
        art.reward(handle, rew_buf.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)))
        art.flags(handle, flag_buf.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint32)))
        return obs_buf.copy(), rew_buf.copy(), int(flag_buf[0])

    env.reset()
    retained = []
    for t in range(T):
        alpha = act_alpha(t)
        beta = act_beta(t)
        flat = np.array([alpha[0], alpha[1], beta[0]], dtype=np.float64)
        obs, rew, term, trunc, infos = env.step(
            {"leader": np.array(alpha, dtype=np.float64),
             "follower": np.array(beta, dtype=np.float64)})
        want_obs, want_rew, word = raw_step(flat)

        for agent, (offset, count) in obs_at.items():
            arms.check(
                obs[agent].tobytes()
                == want_obs[offset:offset + count].tobytes(),
                "step %d: %s observation differs from the raw handle's "
                "channels [%d, %d)" % (t, agent, offset, offset + count))
        # The two agents' observations must differ everywhere they can
        # be compared, or a mis-keyed dictionary could agree with a
        # right one.
        arms.check(
            not np.any(obs["leader"] == obs["follower"][:5]),
            "step %d: the two agents' first five observation "
            "components agree somewhere, so a swap could pass" % t)
        arms.check(rew["leader"] == want_rew[0]
                   and rew["follower"] == want_rew[1],
                   "step %d: rewards %s against the raw handle's %s"
                   % (t, rew, want_rew))
        arms.check(rew["leader"] != 0.0
                   and rew["leader"] != rew["follower"],
                   "step %d: the two agents' rewards are %s, which a "
                   "swapped index could reproduce" % (t, rew))
        arms.check(rew["leader"] == obs["follower"][8],
                   "step %d: the leader's reward %r is not the "
                   "follower's published rel_range %r"
                   % (t, rew["leader"], obs["follower"][8]))
        arms.check(term == {"leader": bool(word & _abi.FLAG_TERMINATED),
                            "follower": bool(word & _abi.FLAG_TERMINATED)},
                   "step %d: terminations %s against flag word 0x%x"
                   % (t, term, word))
        arms.check(trunc == {"leader": bool(word & _abi.FLAG_TRUNCATED),
                             "follower": bool(word & _abi.FLAG_TRUNCATED)},
                   "step %d: truncations %s against flag word 0x%x"
                   % (t, trunc, word))
        arms.check(infos == {"leader": {}, "follower": {}},
                   "step %d: info is %s on an ordinary step"
                   % (t, infos))
        retained.append((obs, {a: want_obs[o:o + c].tobytes()
                               for a, (o, c) in obs_at.items()}))
    art.destroy(handle)

    # Every array returned above is still what it was: an array
    # returned as a view over a buffer the next step overwrites would
    # have been corrupted by the steps that followed it.
    for t, (obs, want) in enumerate(retained):
        for agent, wanted in want.items():
            arms.check(obs[agent].tobytes() == wanted,
                       "step %d: the %s observation changed after the "
                       "steps that followed it" % (t, agent))

    # ---- the recorded stream is the artifact's, not this shape's -----
    # The same seed and the same actions, recorded once through the
    # dictionary interface and once through a raw handle fed the
    # concatenated vector directly: the two episode files must be
    # byte-identical, or this shape is adding something to or taking
    # something from the stream it marshals.
    py_file = g.WORK / "gate13_py.episode"
    raw_file = g.WORK / "gate13_raw.episode"
    for path in (py_file, raw_file):
        if path.exists():
            path.unlink()

    recorded = K26RlParallelEnv(pair, seed=SEED)
    recorded.set_output(str(py_file))
    arms.check(recorded.output_path == str(py_file),
               "the recorded run reports output path %r"
               % (recorded.output_path,))
    recorded.reset()
    for t in range(T):
        recorded.step(
            {"leader": np.array(act_alpha(t), dtype=np.float64),
             "follower": np.array(act_beta(t), dtype=np.float64)})
    recorded.close()

    handle = art.create(SEED, 1)
    art.output(handle, str(raw_file))
    for t in range(T):
        alpha = act_alpha(t)
        flat = np.array([alpha[0], alpha[1], act_beta(t)[0]],
                        dtype=np.float64)
        art.step(handle, flat.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)))
    art.destroy(handle)

    arms.check(py_file.exists() and py_file.stat().st_size > 0,
               "the recorded episode file is missing or empty")
    arms.check(py_file.read_bytes() == raw_file.read_bytes(),
               "the recorded episode files differ: %d bytes through "
               "this shape against %d through the raw handle"
               % (py_file.stat().st_size, raw_file.stat().st_size))
    print("%s: episode file %d bytes, identical through both drivers"
          % (GATE, py_file.stat().st_size))

    # ---- the action dictionary is load-bearing -----------------------
    # The same values with the agents exchanged must produce a
    # different stream; if it did not, the comparison above would be
    # passing on a mapping that does not matter.
    def drive(swap):
        run = K26RlParallelEnv(pair, seed=SEED)
        run.reset()
        seen = []
        for t in range(T):
            alpha = np.array(act_alpha(t), dtype=np.float64)
            beta = np.array(act_beta(t), dtype=np.float64)
            if swap:
                # Exchange the one channel the two agents share: the
                # widths differ, so only the shared channel can swap.
                alpha = np.array([beta[0], alpha[1]], dtype=np.float64)
                beta = np.array([act_alpha(t)[0]], dtype=np.float64)
            obs = run.step({"leader": alpha, "follower": beta})[0]
            seen.append(obs["leader"].tobytes()
                        + obs["follower"].tobytes())
        run.close()
        return b"".join(seen)

    arms.check(drive(False) != drive(True),
               "exchanging the two agents' actions changed nothing, so "
               "the dictionary keys are not reaching the action vector")

    # ---- ending is per environment, identical for every agent --------
    term_so = g.compile_fixture("rl_pair_term", PAIR_TERM_KFL)
    run = K26RlParallelEnv(term_so, seed=SEED)
    obs, infos = run.reset()
    arms.check(sorted(obs) == ["alpha", "beta"]
               and sorted(infos) == ["alpha", "beta"],
               "reset returned %s and %s" % (sorted(obs), sorted(infos)))
    endings = []
    for t in range(4):
        obs, rew, term, trunc, infos = run.step(
            {"alpha": np.zeros(2), "beta": np.zeros(1)})
        endings.append((term["alpha"], term["beta"],
                        trunc["alpha"], trunc["beta"]))
        if term["alpha"] or trunc["alpha"]:
            break
    arms.check(endings[-1][:2] == (True, True),
               "the ending step reported terminations %s" % (endings,))
    arms.check(len(endings) == 3,
               "the episode ended at step %d, expected 3"
               % len(endings))
    arms.check(all(not any(e) for e in endings[:-1]),
               "an earlier step already reported an ending: %s"
               % (endings,))
    arms.check(rew["alpha"] == 6.0 and rew["beta"] == 9.0,
               "the terminal step's rewards are %s, expected the step "
               "rewards plus each agent's own terminal adjustment"
               % (rew,))
    arms.check(run.agents == [],
               "the roster is %s after an ending" % (run.agents,))
    expect_refusal(arms,
                   lambda: run.step({"alpha": np.zeros(2),
                                     "beta": np.zeros(1)}),
                   ["reset()"], "stepping a finished episode")
    arms.check(isinstance(
        _raised(lambda: run.step({"alpha": np.zeros(2),
                                  "beta": np.zeros(1)})),
        gymnasium.error.ResetNeeded),
        "step-after-end raised the wrong type")
    run.reset()
    arms.check(run.agents == ["alpha", "beta"],
               "reset left the roster at %s" % (run.agents,))

    # An action dictionary that does not carry every acting agent, or
    # carries one that is not an agent, is refused naming it.
    expect_refusal(arms, lambda: run.step({"alpha": np.zeros(2)}),
                   ["no action for", "beta"],
                   "an action dictionary missing an agent",
                   exc_type=ValueError)
    expect_refusal(arms,
                   lambda: run.step({"alpha": np.zeros(2),
                                     "beta": np.zeros(1),
                                     "gamma": np.zeros(1)}),
                   ["gamma", "not an agent of this environment"],
                   "an action dictionary naming a stranger",
                   exc_type=ValueError)
    expect_refusal(arms, lambda: run.step([np.zeros(2), np.zeros(1)]),
                   ["mapping"], "an action sequence rather than a map",
                   exc_type=TypeError)
    run.close()

    # Truncation at the horizon reaches every agent the same way.
    run = K26RlParallelEnv(pair, seed=SEED)
    run.reset()
    horizon = run.env_spec.horizon
    for t in range(horizon):
        obs, rew, term, trunc, infos = run.step(
            {"leader": np.zeros(2), "follower": np.zeros(1)})
        if term["leader"] or trunc["leader"]:
            break
    arms.check(trunc == {"leader": True, "follower": True}
               and term == {"leader": False, "follower": False},
               "the horizon step reported %s and %s" % (term, trunc))
    arms.check(t + 1 == horizon,
               "the horizon step was %d, expected %d"
               % (t + 1, horizon))
    arms.check(run.agents == [],
               "the roster is %s after truncation" % (run.agents,))
    run.close()

    # ---- a fault is the environment's, so every agent carries it -----
    fault_so = g.compile_fixture("rl_pair_fault", PAIR_FAULT_KFL)
    decoder = _abi.Artifact(fault_so)
    for mode in ("raise", "truncate"):
        run = K26RlParallelEnv(fault_so, seed=SEED, on_fault=mode)
        run.reset()
        actions = {"alpha": np.zeros(2), "beta": np.array([1.0])}
        if mode == "raise":
            raised = _raised(lambda: run.step(actions))
            arms.check(isinstance(raised, K26RlFaultError),
                       "raise mode raised %r" % (raised,))
            results = raised.results
        else:
            results = run.step(actions)
        _obs, _rew, term, trunc, infos = results
        arms.check(trunc == {"alpha": True, "beta": True},
                   "%s mode reported truncations %s" % (mode, trunc))
        arms.check(term == {"alpha": False, "beta": False},
                   "%s mode reported terminations %s" % (mode, term))
        for name in ("alpha", "beta"):
            arms.check("fault_code" in infos[name]
                       and "fault_reason" in infos[name],
                       "%s mode left %s without the fault keys: %s"
                       % (mode, name, infos))
        arms.check(infos["alpha"] == infos["beta"],
                   "%s mode gave the two agents different fault keys: "
                   "%s" % (mode, infos))
        code = infos["alpha"]["fault_code"]
        arms.check(code != 0,
                   "%s mode reported fault code %r" % (mode, code))
        arms.check(infos["alpha"]["fault_reason"]
                   == decoder.status_message(code),
                   "%s mode reported reason %r, the artifact decodes "
                   "%d as %r" % (mode, infos["alpha"]["fault_reason"],
                                 code, decoder.status_message(code)))
        arms.check(run.agents == [],
                   "%s mode left the roster at %s" % (mode, run.agents))
        run.close()

    # ---- slices that do not partition are still refused --------------
    def blob(agent_count=2, obs_total=15, act_total=3,
             obs_slices=((0, 0, 5), (1, 5, 10)),
             act_slices=((0, 0, 2), (1, 2, 1))):
        parts = [
            _tlv(_spec.TAG_ABI_VERSION, _u32(0x00010000)),
            _tlv(_spec.TAG_ENDIAN_PROBE, _u32(0x01020304)),
            _tlv(_spec.TAG_AGENT_COUNT, _u32(agent_count)),
            _tlv(_spec.TAG_N_ENVS, _u32(1)),
            _tlv(_spec.TAG_CONTROL_DT, struct.pack("<d", 1.0)),
            _tlv(_spec.TAG_HORIZON, _u32(10)),
            _tlv(_spec.TAG_OBS_TOTAL, _u32(obs_total)),
            _tlv(_spec.TAG_ACT_TOTAL, _u32(act_total)),
            _tlv(_spec.TAG_EPISODE_FLAGS, _u32(1)),
        ]
        for agent, offset, count in obs_slices:
            parts.append(_tlv(_spec.TAG_AGENT_OBS_SLICE,
                              _u32(agent) + _u32(offset) + _u32(count)))
        for agent, offset, count in act_slices:
            parts.append(_tlv(_spec.TAG_AGENT_ACT_SLICE,
                              _u32(agent) + _u32(offset) + _u32(count)))
        for offset in range(act_total):
            parts.append(_tlv(_spec.TAG_ACT_BOUNDS,
                              _u32(offset) + struct.pack("<dd", -1.0, 1.0)))
            parts.append(_tlv(_spec.TAG_ACT_KIND,
                              _u32(offset) + struct.pack("<H", 0)))
        return b"".join(parts)

    def validate(**kwargs):
        _spec.validate(_spec.parse(blob(**kwargs)), 0x00010000, 1)

    # Exchanging the two agents' offsets: only detectable because the
    # widths differ, which is why the fixture's widths differ.
    expect_refusal(arms,
                   lambda: validate(obs_slices=((0, 10, 5), (1, 0, 10))),
                   ["agent 0", "contiguous"],
                   "observation slices with the agents' offsets "
                   "exchanged")
    expect_refusal(arms,
                   lambda: validate(act_slices=((0, 1, 2), (1, 0, 1))),
                   ["agent 0", "contiguous"],
                   "action slices with the agents' offsets exchanged")
    expect_refusal(arms,
                   lambda: validate(obs_slices=((0, 0, 5), (1, 5, 9))),
                   ["cover 14", "15"],
                   "observation slices leaving a component uncovered")
    expect_refusal(arms,
                   lambda: validate(obs_slices=((0, 0, 5), (1, 5, 11))),
                   ["runs past"],
                   "an observation slice running past the total")
    expect_refusal(arms,
                   lambda: validate(obs_slices=((0, 0, 15),)),
                   ["agent count 2", "1 per-agent observation slices"],
                   "one observation slice for two declared agents")
    expect_refusal(arms,
                   lambda: validate(obs_slices=((0, 0, 5), (0, 5, 10))),
                   ["two observation slices name agent 0"],
                   "two observation slices for one agent")
    expect_refusal(arms,
                   lambda: validate(obs_slices=((0, 0, 5), (2, 5, 10))),
                   ["agent 2", "agent count 2"],
                   "an observation slice naming an undeclared agent")
    expect_refusal(arms, lambda: validate(obs_slices=(), act_slices=()),
                   ["no per-agent observation slice"],
                   "two declared agents with no slices at all")
    # The same blob without the defect must validate, or the refusals
    # above would be measuring something else.
    validate()
    arms.completed("the undefective two-agent blob validates")

    # ---- an agent that publishes no name -----------------------------
    unnamed = g.compile_fixture("rl_unnamed", UNNAMED_KFL)
    # The needles are the cause rather than the symptom: the checks
    # that follow would refuse this artifact too, for a reason that
    # does not tell the reader what is missing.
    expect_refusal(arms, lambda: K26RlParallelEnv(unnamed, seed=SEED),
                   ["agent 0 declares no observation channel",
                    "no tag carries an action channel name"],
                   "an agent with no observation channel")
    # A qualifier-free name above one agent, and a slice whose channels
    # carry two different qualifiers.
    def names_from(mapping):
        spec = _spec.parse(blob())
        spec.obs_channel_names = mapping
        return _spec.agent_names(spec)

    plain = {c: ("leader.c%d" % c if c < 5 else "follower.c%d" % c)
             for c in range(15)}
    derived = names_from(plain)
    arms.check(derived == ["leader", "follower"],
               "names came out as %s" % (derived,))
    unqualified = dict(plain)
    unqualified[2] = "bare"
    expect_refusal(arms, lambda: names_from(unqualified),
                   ["'bare'", "carries no <agent>.<channel> qualifier"],
                   "an unqualified observation channel name")
    # An agent holding one channel is the case the qualifier-count
    # check cannot reach: with one name there is nothing to disagree
    # with, so only the qualifier check itself stands between an
    # unqualified name and an agent named by invention.
    def lone(mapping, obs_slices):
        spec = _spec.parse(blob(obs_slices=obs_slices))
        spec.obs_channel_names = mapping
        return _spec.agent_names(spec)

    lone_names = {0: "solo"}
    lone_names.update({c: "follower.c%d" % c for c in range(1, 15)})
    expect_refusal(arms,
                   lambda: lone(lone_names,
                                ((0, 0, 1), (1, 1, 14))),
                   ["'solo'", "carries no <agent>.<channel> qualifier"],
                   "an agent whose one channel is unqualified")
    mixed = dict(plain)
    mixed[2] = "other.c2"
    expect_refusal(arms, lambda: names_from(mixed),
                   ["agent 0", "2 different qualifiers"],
                   "one agent's channels under two qualifiers")
    missing = dict(plain)
    del missing[2]
    expect_refusal(arms, lambda: names_from(missing),
                   ["channel 2", "no published name"],
                   "an observation channel with no published name")
    collided = {c: "same.c%d" % c for c in range(15)}
    expect_refusal(arms, lambda: names_from(collided),
                   ["one name"], "two agents under one name")

    # ---- an observation-only agent -----------------------------------
    watcher = g.compile_fixture("rl_watcher", WATCHER_KFL)
    run = K26RlParallelEnv(watcher, seed=SEED)
    arms.check(run.action_space("alpha").shape == (0,),
               "the observation-only agent's action space is %r"
               % (run.action_space("alpha"),))
    run.reset()
    obs, rew, _term, _trunc, _infos = run.step(
        {"alpha": np.zeros(0), "beta": np.array([0.5])})
    arms.check(obs["alpha"].shape == (5,)
               and obs["beta"].shape == (10,),
               "the observation-only agent's arrays are %s"
               % ({k: v.shape for k, v in obs.items()},))
    arms.check(rew == {"alpha": 1.0, "beta": 2.0},
               "the observation-only run's rewards are %s" % (rew,))
    run.close()

    # ---- the single-agent shape is untouched -------------------------
    plain_env = K26RlEnv(single, seed=SEED)
    obs, _info = plain_env.reset()
    arms.check(obs.shape == (plain_env.env_spec.obs_total,),
               "the single-agent reset observation is %s" % (obs.shape,))
    obs, reward, term, trunc, info = plain_env.step((0.25, 1))
    arms.check(isinstance(reward, float) and not term and not trunc
               and info == {},
               "the single-agent step returned %r, %r, %r, %r"
               % (reward, term, trunc, info))
    expect_refusal(arms,
                   lambda: plain_env.reset(options={"scenario": "hard"}),
                   ["scenario"],
                   "a non-empty options mapping on the single-agent "
                   "shape")
    plain_env.close()

    # ---- the optional dependency stays optional ----------------------
    probe = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE % str(g.PKG_DIR)],
        capture_output=True, text=True)
    arms.check(probe.returncode == 0 and probe.stdout.strip() == "ok",
               "the import probe failed (rc=%d): %s%s"
               % (probe.returncode, probe.stdout, probe.stderr))

    # ---- the framework's own conformance check -----------------------
    from pettingzoo.test import parallel_api_test
    conformance = K26RlParallelEnv(pair, seed=SEED)
    parallel_api_test(conformance, num_cycles=250)
    conformance.close()
    arms.completed("pettingzoo.test.parallel_api_test passed")

    env.close()
    print("%s: %d arms" % (GATE, arms.n))
    g.ok(GATE)


def _raised(call):
    try:
        call()
    except Exception as exc:
        return exc
    return None


def _tlv(tag, payload):
    return struct.pack("<HI", tag, len(payload)) + payload


def _u32(value):
    return struct.pack("<I", value)


if __name__ == "__main__":
    sys.exit(main())
