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
refused rather than named by invention; every action-space branch is
driven against a raw handle over a three-agent mixed-kind artifact; a
reset option this version does not define comes back named in the info
mapping as well as in a warning, so the record survives a consumer's
warning filters; a single-agent artifact is served by the single-agent
shape exactly as before; close is idempotent and everything on a
closed environment is refused before any artifact call, while the two
run records stay readable; importing the package does not import the
optional dependency; and PettingZoo's own parallel API conformance
check runs against the fixture.

Skips (77) when the built compiler, the stack archives, gymnasium or
pettingzoo are absent.
"""

import ctypes
import struct
import subprocess
import sys
import warnings

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

# Three agents whose action spaces take three different shapes, so the
# marshalling's every branch is driven: alpha's box-and-discrete
# mixture is a Tuple, beta's two discretes are a MultiDiscrete, and
# gamma's one discrete is a Discrete. Their widths are three, two and
# one against five, ten and fifteen observation components, so no two
# agents' slices can be exchanged without changing a shape. Every
# action channel reaches a body's velocity, because a channel nothing
# reads cannot show that it was marshalled to the right offset.
PAIR_MIX_KFL = """\
form RL_PAIRMIX
fn world mix_world
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body a_craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7546.0
    astro_body b_craft gm=1.0 parent=earth pos_x=7.4e6 vel_y=7340.0
    astro_body c_craft gm=1.0 parent=earth pos_x=7.8e6 vel_y=7150.0
    episode
        control_dt 1.0
        horizon 60
    end
    agent alpha
        action push box -1.0 1.0 default 0.0
        action bank box -2.0 2.0 default 0.0
        action stage discrete 3 default 0
        observe a_craft from earth mode=geometric as trk
        objective
            reward 1.0
        end
    end
    agent beta
        action gearsel discrete 4 default 0
        action trim discrete 2 default 0
        observe b_craft from earth mode=geometric as trk
        observe b_craft from a_craft mode=geometric as rel
        objective
            reward 2.0
        end
    end
    agent gamma
        action sel discrete 5 default 0
        observe c_craft from earth mode=geometric as trk
        observe c_craft from a_craft mode=geometric as rela
        observe c_craft from b_craft mode=geometric as relb
        objective
            reward 3.0
        end
    end
    on_step
        a_craft.vel_x = a_craft.vel_x + push
        a_craft.vel_z = a_craft.vel_z + bank
        a_craft.vel_y = a_craft.vel_y + stage
        b_craft.vel_x = b_craft.vel_x + gearsel
        b_craft.vel_z = b_craft.vel_z + trim
        c_craft.vel_x = c_craft.vel_x + sel
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

# The suppressed-warning probe: with warnings off at the interpreter,
# the record of what an ignored reset option discarded must still
# reach the consumer through the info mapping.
OPTIONS_PROBE = """\
import sys
import warnings
sys.path.insert(0, %r)
from k26rl.parallel import K26RlParallelEnv, INFO_IGNORED_OPTIONS
env = K26RlParallelEnv(%r, seed=%d)
with warnings.catch_warnings(record=True) as caught:
    obs, infos = env.reset(options={"scenario": "hard", "b": 1})
    seen = [str(w.message) for w in caught]
env.close()
first = infos[sorted(infos)[0]]
print(len(seen), first[INFO_IGNORED_OPTIONS])
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


# The mixed-kind stream: one function per channel, each inside its
# declared bounds or arity and each moving at every step, so a value
# marshalled to the wrong offset lands on a channel that was carrying
# something else.
def mix_channels(t):
    return (((t * 5) % 11) / 11.0 - 0.5,      # alpha push, box
            ((t * 3) % 7) / 7.0 - 0.5,        # alpha bank, box
            (t * 2) % 3,                      # alpha stage, arity 3
            (t + 1) % 4,                      # beta gearsel, arity 4
            t % 2,                            # beta trim, arity 2
            (t * 3) % 5)                      # gamma sel, arity 5


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
    from k26rl.parallel import K26RlParallelEnv, INFO_IGNORED_OPTIONS
    from k26rl.vector import K26RlVectorEnv
    from k26rl._errors import K26RlError, K26RlFaultError

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
        # Each agent's array owns its buffer. A view would carry the
        # right values, so the retention arm below cannot see this;
        # what it would carry with them is every other agent's
        # components, once for every step a consumer stores.
        for agent in obs:
            arms.check(obs[agent].base is None,
                       "step %d: the %s observation is a view over a "
                       "%d-element parent, not its own array"
                       % (t, agent, obs[agent].base.size
                          if obs[agent].base is not None else 0))
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

    # ---- every action-space branch, against the raw handle -----------
    # The fixture above drives all four marshalling branches: a Tuple
    # carrying box and discrete elements, a MultiDiscrete, and a
    # Discrete, beside the all-box case the two-agent fixture already
    # covers. Each is compared bitwise against a raw handle fed the
    # concatenated vector, so a value written to the wrong offset, or
    # transformed on the way, shows up as a different trajectory.
    mix_so = g.compile_fixture("rl_pair_mix", PAIR_MIX_KFL)
    mix = K26RlParallelEnv(mix_so, seed=SEED)
    arms.check(mix.possible_agents == ["alpha", "beta", "gamma"],
               "the mixed fixture publishes %s"
               % (mix.possible_agents,))
    alpha_space = mix.action_space("alpha")
    arms.check(isinstance(alpha_space, gym_spaces.Tuple)
               and len(alpha_space.spaces) == 3
               and isinstance(alpha_space.spaces[0], gym_spaces.Box)
               and isinstance(alpha_space.spaces[2],
                              gym_spaces.Discrete),
               "alpha's action space is %r" % (alpha_space,))
    beta_space = mix.action_space("beta")
    arms.check(isinstance(beta_space, gym_spaces.MultiDiscrete)
               and list(beta_space.nvec) == [4, 2],
               "beta's action space is %r" % (beta_space,))
    gamma_space = mix.action_space("gamma")
    arms.check(isinstance(gamma_space, gym_spaces.Discrete)
               and gamma_space.n == 5,
               "gamma's action space is %r" % (gamma_space,))

    mix_art = _abi.Artifact(mix_so)
    mix_handle = mix_art.create(SEED, 1)
    mix_spec = _spec.parse(mix_art.spec_blob(mix_handle))
    mix_at = dict(zip(["alpha", "beta", "gamma"],
                      _spec.slices_by_agent(mix_spec.agent_obs_slices,
                                            mix_spec.agent_count)))
    mix_obs = np.empty(mix_spec.obs_total, dtype=np.float64)
    mix_rew = np.empty(mix_spec.agent_count, dtype=np.float64)
    mix.reset()
    for t in range(T):
        push, bank, stage, gearsel, trim, sel = mix_channels(t)
        obs, rew, _term, _trunc, _infos = mix.step({
            "alpha": (push, bank, stage),
            "beta": np.array([gearsel, trim], dtype=np.int64),
            "gamma": sel})
        flat = np.array([push, bank, stage, gearsel, trim, sel],
                        dtype=np.float64)
        mix_art.step(mix_handle, flat.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)))
        mix_art.obs(mix_handle, mix_obs.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)))
        mix_art.reward(mix_handle, mix_rew.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)))
        for agent, (offset, count) in mix_at.items():
            arms.check(
                obs[agent].tobytes()
                == mix_obs[offset:offset + count].tobytes(),
                "mixed step %d: %s observation differs from the raw "
                "handle's channels [%d, %d)"
                % (t, agent, offset, offset + count))
        arms.check(
            [rew[n] for n in ("alpha", "beta", "gamma")]
            == [mix_rew[0], mix_rew[1], mix_rew[2]],
            "mixed step %d: rewards %s against the raw handle's %s"
            % (t, rew, mix_rew))
    mix_art.destroy(mix_handle)
    mix.close()

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
    # The compiler refuses this shape outright, so no artifact of it
    # reaches this package: an agent's name arrives only as the
    # qualifier on its own observation channel names, and a block that
    # declares none publishes a name nothing can read.
    refusal = g.refuse_fixture("rl_unnamed", UNNAMED_KFL)
    arms.check("declares no `observe ... as`" in refusal,
               "the compiler's refusal names the missing observation")
    arms.completed("the compiler refuses an agent with no observation")
    # The package's own guard is still its contract and still fires. It
    # is driven against a spec built here rather than a compiled one,
    # because the compiler no longer produces the artifact it refuses.
    def silent():
        spec = _spec.parse(blob(obs_slices=((0, 0, 0), (1, 0, 15))))
        spec.obs_channel_names = {c: "follower.c%d" % c
                                  for c in range(15)}
        return _spec.agent_names(spec)

    expect_refusal(arms, silent,
                   ["agent 0 declares no observation channel",
                    "no tag carries an action channel name"],
                   "a spec whose agent 0 has no observation channel")
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

    # ---- an ignored reset option is recoverable ----------------------
    # The warning is one half and the filterable half; the info payload
    # is the half that survives a consumer's warning filters, and the
    # arms are separate because deleting either leaves the other
    # passing.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = K26RlParallelEnv(pair, seed=SEED)
        _obs, infos = run.reset(options={"scenario": "hard", "b": 1})
        run.close()
    arms.check(len(caught) == 1,
               "a non-empty options mapping raised %d warnings, "
               "expected 1" % len(caught))
    arms.check("'b', 'scenario'" in str(caught[0].message),
               "the warning does not name the keys: %s"
               % caught[0].message)
    arms.check(all(info.get(INFO_IGNORED_OPTIONS) == ["b", "scenario"]
                   for info in infos.values()),
               "the reset info does not carry the ignored keys: %s"
               % (infos,))
    run = K26RlParallelEnv(pair, seed=SEED)
    _obs, infos = run.reset()
    arms.check(all(INFO_IGNORED_OPTIONS not in info
                   for info in infos.values()),
               "a reset that ignored nothing still reported keys: %s"
               % (infos,))
    run.close()

    # Measured rather than argued: with warnings suppressed at the
    # interpreter, the warning is gone and the payload is not.
    probe = subprocess.run(
        [sys.executable, "-W", "ignore", "-c",
         OPTIONS_PROBE % (str(g.PKG_DIR), str(pair), SEED)],
        capture_output=True, text=True)
    arms.check(probe.returncode == 0,
               "the suppressed-warning probe failed (rc=%d): %s%s"
               % (probe.returncode, probe.stdout, probe.stderr))
    arms.check(probe.stdout.strip() == "0 ['b', 'scenario']",
               "under -W ignore the probe reported %r, expected no "
               "warning and the ignored keys"
               % probe.stdout.strip())

    # ---- close, on the same terms as the other two shapes ------------
    # The loading gate pins this for the single and vectorised shapes
    # and stays free of this optional dependency; the same discipline
    # for this shape is pinned here, where the dependency already is.
    def expect_closed(call, what):
        try:
            call()
        except K26RlError as exc:
            arms.check("closed" in str(exc),
                       "%s after close: %s" % (what, exc))
        else:
            arms.check(False,
                       "%s on a closed environment succeeded" % what)

    closing = K26RlParallelEnv(pair, seed=17)
    roster = list(closing.possible_agents)
    closing.close()
    closing.close()
    shut = {"leader": np.zeros(2), "follower": np.zeros(1)}
    expect_closed(lambda: closing.step(shut), "parallel step")
    expect_closed(lambda: closing.reset(), "parallel reset")
    expect_closed(lambda: closing.reset(seed=18),
                  "parallel seeded reset")
    expect_closed(lambda: closing.set_output("/tmp/never.episode"),
                  "parallel set_output")
    for prop in ("env_spec", "control_dt", "obs_channel_names",
                 "obs_channel_kinds", "agent_obs_channel_names",
                 "on_fault"):
        expect_closed(lambda prop=prop: getattr(closing, prop),
                      "parallel property %s" % prop)
    arms.check(closing.seeds_held == frozenset({17}),
               "seeds_held unreadable or wrong after close: %r"
               % (closing.seeds_held,))
    arms.check(closing.output_path is None,
               "output_path unreadable or wrong after close: %r"
               % (closing.output_path,))
    # The roster is this object's own record and reaches no artifact,
    # so it survives close as the other two records do.
    arms.check(closing.possible_agents == roster,
               "the agent roster changed across close: %r"
               % (closing.possible_agents,))

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
