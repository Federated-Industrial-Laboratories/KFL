"""Gate 14: the measured half of a paired observation, and the ground
truth beside it.

A policy reads its craft and its surroundings through the sensors the
program declared, imperfections and all. Where an observe is routed
through a sensor the artifact publishes the uncorrupted values beside
the measured ones as paired ground-truth channels, and those are for a
privileged critic and for the episode record; a policy never sees
them. This gate holds the package to that.

The fixture publishes two observes, one sensed and one not, so its
measured channels are two runs with the truth ones between them: a
shape that returned a prefix, a suffix or the whole vector fails here,
and one that happened to slice a single-observe world correctly could
not.

Against the compiled fixtures:

* the pairing the artifact publishes, read from the source tags and
  cross-checked against the published names;
* the single shape's observation space, observation and info entry,
  each held bit for bit against a raw handle stepped in lockstep;
* the enforcement arm, that no component handed to a policy is a
  ground-truth component of the same observation, with two controls
  behind it: the same predicate over the whole declared vector, and
  the same arms run once more against a session whose split has been
  disabled, since an arm that cannot fail on the defect it names
  enforces nothing;
* the vectorised shape over two environments, its info entry and the
  presence mask beside it;
* the multi-agent shape, where one agent declares a sensor and the
  other declares none;
* a world with no sensor, whose observation is still its whole
  declared vector and which carries no info entry at all.

From synthetic blobs: the pairing refusals (a source kind this version
does not know, a source tag naming a channel outside the declared
total, a partner outside it, a channel paired with itself, a pair of
two channels of one kind, a pairing named from one end only, and
ground truth naming no measurement), and the untagged spec an older
artifact publishes, whose channels are all measurements.

Against the policy exporter: the channels it declares are the agent's
measured ones and no others, interleaved with ground-truth channels
or not, and an agent with no measured channel at all is refused by
name rather than exported with a list that would feed the inference
tier the truth.

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent. The multi-agent arms need pettingzoo, which is optional;
without it they report as not run and the rest of the gate still
enforces.
"""

import ctypes
import struct
import sys

import _gateutil as g

GATE = "gate14_sensor_split"

SEED = 20260819

# The stepping ABI version the synthetic blobs echo, as the artifact
# reports it: major in the high half, minor in the low.
ABI_VERSION = 0x00010000

# Two observes, one through a sensor and one not, so the measured
# channels are 0 to 4 and 10 to 14 with the ground-truth ones at 5 to
# 9 between them. The noise is large and the quantiser coarse, so
# every measured component differs from its truth once a transition
# has been taken, which is what makes the enforcement arm able to see
# a truth value that reached the policy.
#
# Both bodies are off the reference plane, and on different sides of
# it. Left in the plane, the two observes would both publish a
# direction component of exactly zero, and a component the policy is
# entitled to hold would carry the same bytes as a ground-truth one:
# the arm below would then be unable to tell a leaked value from a
# geometric coincidence, and it says so rather than reporting either.
SPLIT_KFL = """\
form RL_SPLIT
fn world split_world
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body craft mass=1.0 parent=earth pos_x=7.0e6\
 pos_z=3.1e5 vel_y=7546.049108166324 vel_z=113.0
    astro_body probe mass=1.0 parent=earth pos_x=7.05e6\
 pos_z=-2.3e5 vel_y=7520.0 vel_z=-87.0
    sensor rangefinder
        noise normal 0.0 40.0
        quantise 5.0
    end
    episode
        control_dt 1.0
        horizon 50
        substeps 1
    end
    action a box -1.0 1.0 default 0.0
    on_step
        craft.vel_x = craft.vel_x + a * 0.0
    end
    observe craft from earth mode=geometric through rangefinder\
 with truth as look
    observe probe from earth mode=geometric as plain
    objective
        reward look_range + a * 0.0
    end
end
end
"""

# Two agents, one of which declares a sensor: the split is per agent
# because the slices are, and an agent that declares no sensor must
# come through untouched.
SPLIT_MA_KFL = """\
form RL_SPLITMA
fn world ma_world
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body leader_craft mass=1.0 parent=earth pos_x=7.0e6\
 vel_y=7546.049108166324
    astro_body follower_craft mass=1.0 parent=earth pos_x=7.4e6\
 vel_y=7340.0
    sensor rf
        noise normal 0.0 40.0
        quantise 5.0
    end
    episode
        control_dt 1.0
        horizon 50
        substeps 1
    end
    agent leader
        action push box -1.0 1.0 default 0.0
        observe leader_craft from earth mode=geometric through rf\
 with truth as trk
        objective
            reward trk_range + push * 0.0
        end
    end
    agent follower
        action push box -1.0 1.0 default 0.0
        observe follower_craft from earth mode=geometric as trk
        objective
            reward trk_range + push * 0.0
        end
    end
    on_step
        leader_craft.vel_x = leader_craft.vel_x + leader.push
        follower_craft.vel_x = follower_craft.vel_x + follower.push
    end
end
end
"""

MEASURED = tuple(range(0, 5)) + tuple(range(10, 15))
TRUTH = tuple(range(5, 10))


def tlv(tag, payload):
    return struct.pack("<HI", tag, len(payload)) + payload


def source_tlv(channel, source, pair):
    from k26rl import _spec

    return tlv(_spec.TAG_OBS_CHANNEL_SOURCE,
               struct.pack("<IHI", channel, source, pair))


def truth_reaching(handed, whole, truth_channels):
    """The components of ``handed`` that are a ground-truth component
    of ``whole``, by value. The values are compared as bytes and not
    the indices, so this answers the question a policy asks, what
    numbers did I receive, without trusting the same tag parse the
    package under test used to answer it."""
    wanted = {whole[c].tobytes() for c in truth_channels}
    return [i for i, v in enumerate(handed) if v.tobytes() in wanted]


def action_of(t):
    """A scripted action stream. The world's own body is unmoved by
    it, so both drivers see the same trajectory whatever it is; it
    varies so that a comparison is not of one repeated number."""
    return ((t * 5) % 11) / 11.0 * 2.0 - 1.0


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    import numpy as np
    from k26rl import K26RlError, _abi, _spec
    from k26rl.env import INFO_TRUTH_OBS, K26RlEnv
    from k26rl.vector import K26RlVectorEnv

    so = g.compile_fixture("rl_split", SPLIT_KFL)
    arms = 0

    # ---- 1. what the artifact publishes ------------------------------
    art = _abi.Artifact(so)
    handle = art.create(SEED, 1)
    spec = _spec.parse(art.spec_blob(handle))
    g.check(spec.obs_total == 15,
            "the fixture declares %s observation channels, expected 15"
            % spec.obs_total)
    measured, truth = _spec.split_channels(spec, 0, spec.obs_total)
    g.check(tuple(measured) == MEASURED and tuple(truth) == TRUTH,
            "the source tags split the fixture into measured %s and "
            "ground truth %s, expected %s and %s"
            % (measured, truth, list(MEASURED), list(TRUTH)))
    # The names are the independent reading: a paired truth channel
    # publishes with _truth before the component, so a tag parse that
    # had the two kinds the wrong way round would disagree here.
    by_name = sorted(c for c, name in spec.obs_channel_names.items()
                     if "_truth" in name)
    g.check(by_name == list(truth),
            "the source tags call %s ground truth and the published "
            "names call %s ground truth" % (truth, by_name))
    for m in truth:
        g.check(_spec.channel_pair(spec, _spec.channel_pair(spec, m)) == m,
                "channel %d's pairing does not name it back" % m)
    arms += 4
    print("%s: the fixture pairs %d measured channels with %d "
          "ground-truth ones" % (GATE, len(measured), len(truth)))

    # ---- 2 and 3. the single shape and the enforcement arm ----------
    env = K26RlEnv(so, seed=SEED)
    g.check(env.observation_space.shape == (len(MEASURED),),
            "the observation space is %s, expected (%d,)"
            % (env.observation_space.shape, len(MEASURED)))
    g.check(env.truth_observation_space.shape == (len(TRUTH),),
            "the privileged space is %s, expected (%d,)"
            % (env.truth_observation_space.shape, len(TRUTH)))
    g.check(tuple(env.policy_channels) == MEASURED
            and tuple(env.truth_channels) == TRUTH,
            "the shape reports measured %s and ground truth %s"
            % (env.policy_channels, env.truth_channels))
    arms += 3

    whole = np.empty(spec.obs_total, dtype=np.float64)
    obs, info = env.reset()
    art.obs(handle, whole.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    corrupted = 0
    controls = 0
    for t in range(12):
        g.check(obs.shape == (len(MEASURED),) and obs.dtype == np.float64,
                "step %d handed the policy %s of %s"
                % (t, obs.shape, obs.dtype))
        g.check(obs in env.observation_space,
                "step %d handed the policy a vector outside its own "
                "observation space" % t)
        g.check(obs.tobytes() == whole[list(MEASURED)].tobytes(),
                "step %d: the observation is not the measured channels "
                "of the raw handle's vector" % t)
        g.check(info[INFO_TRUTH_OBS].tobytes()
                == whole[list(TRUTH)].tobytes(),
                "step %d: the info entry is not the ground-truth "
                "channels of the raw handle's vector" % t)

        # The enforcement arm, and its control. The arm can only see a
        # truth value that reached the policy where the sensor has
        # moved the measurement off it, so how many of the five pairs
        # differ is counted and held below.
        differing = [c for c in TRUTH
                     if whole[c].tobytes()
                     != whole[_spec.channel_pair(spec, c)].tobytes()]
        corrupted += len(differing)
        # A hit is a component of the observation carrying a
        # ground-truth value. Where the same bytes also sit at a
        # measured channel of the raw vector the two cannot be told
        # apart by value, and that is a fixture this arm can no longer
        # enforce with, so it is reported as itself rather than as a
        # leak.
        hits = truth_reaching(obs, whole, differing)
        held = {whole[c].tobytes() for c in MEASURED}
        coincident = [i for i in hits if obs[i].tobytes() in held]
        g.check(not coincident,
                "step %d: components %s carry bytes that are both a "
                "measured channel's value and a ground-truth one, so "
                "this fixture can no longer tell a leaked value from a "
                "coincidence" % (t, coincident))
        g.check(not hits,
                "step %d handed the policy components %s, which are "
                "ground-truth values of the same observation"
                % (t, hits))
        if differing:
            # The same predicate over the whole declared vector, which
            # is what a shape that split nothing would hand over. It
            # must report every differing truth channel; an arm that
            # passed here would be measuring nothing.
            found = truth_reaching(whole, whole, differing)
            g.check(set(found) >= set(differing),
                    "step %d: the check reports %s on the whole "
                    "declared vector, expected at least %s; an arm "
                    "that passes on the unsplit vector enforces "
                    "nothing" % (t, found, differing))
            controls += 1
        arms += 6

        flat = np.array([action_of(t)], dtype=np.float64)
        obs, _rew, _term, _trunc, info = env.step(flat)
        art.step(handle, flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        art.obs(handle, whole.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    env.close()
    art.destroy(handle)
    g.check(corrupted >= 40,
            "only %d of the 60 paired components differed from their "
            "measurement over the run; with too few differing the "
            "enforcement arm compares numbers that are equal either "
            "way" % corrupted)
    g.check(controls >= 10,
            "the control ran on %d steps, too few to establish that "
            "the enforcement arm can fail" % controls)
    arms += 2
    print("%s: the single shape hands over %d measured components and "
          "no ground-truth one; %d differing pairs seen, control ran "
          "on %d steps" % (GATE, len(MEASURED), corrupted, controls))

    # ---- 3b. the same arms against the defect they name ---------------
    #
    # The split is disabled in the session and the single shape driven
    # again, which is a binding that hands a policy the whole declared
    # vector. The space membership and the enforcement arm must both
    # reject what comes back: an arm that cannot fail on the defect it
    # names measures nothing, and this one is the reason the gate
    # exists.
    from k26rl.env import _Session

    whole_vector = _Session.policy_obs
    _Session.policy_obs = lambda session, obs: obs
    try:
        red = K26RlEnv(so, seed=SEED)
        red_obs, _red_info = red.reset()
        for t in range(3):
            red_obs, _r, _te, _tr, _i = red.step(
                np.array([action_of(t)], dtype=np.float64))
        red.close()
    finally:
        _Session.policy_obs = whole_vector
    g.check(red_obs.shape == (spec.obs_total,),
            "the defect control handed over %s, and the defect it "
            "stands for is the whole declared vector %s"
            % (red_obs.shape, (spec.obs_total,)))
    g.check(red_obs not in env.observation_space,
            "the observation space accepted the whole declared "
            "vector, so its width enforces nothing")
    red_differing = [c for c in TRUTH
                     if red_obs[c].tobytes()
                     != red_obs[_spec.channel_pair(spec, c)].tobytes()]
    g.check(red_differing,
            "the defect control found no pair whose halves differ, so "
            "it establishes nothing")
    g.check(sorted(truth_reaching(red_obs, red_obs, red_differing))
            == sorted(red_differing),
            "the enforcement arm reports %s against a binding that "
            "hands over the whole vector, expected %s"
            % (truth_reaching(red_obs, red_obs, red_differing),
               red_differing))
    arms += 4
    print("%s: with the split disabled the observation is %d wide and "
          "carries %d ground-truth components, and the arms above "
          "reject it" % (GATE, red_obs.shape[0], len(red_differing)))

    # ---- 4. the vectorised shape -------------------------------------
    n_envs = 2
    art = _abi.Artifact(so)
    handle = art.create(SEED, n_envs)
    venv = K26RlVectorEnv(so, seed=SEED, n_envs=n_envs)
    g.check(venv.single_observation_space.shape == (len(MEASURED),)
            and venv.observation_space.shape == (n_envs, len(MEASURED)),
            "the vectorised spaces are %s and %s"
            % (venv.single_observation_space.shape,
               venv.observation_space.shape))
    g.check(venv.truth_observation_space.shape == (n_envs, len(TRUTH)),
            "the vectorised privileged space is %s"
            % (venv.truth_observation_space.shape,))
    whole = np.empty(n_envs * spec.obs_total, dtype=np.float64)
    obs, infos = venv.reset()
    art.obs(handle, whole.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    arms += 2
    for t in range(6):
        rows = whole.reshape(n_envs, spec.obs_total)
        g.check(obs.tobytes() == rows[:, list(MEASURED)].tobytes(),
                "vectorised step %d: the observations are not the "
                "measured channels of the raw handle's vectors" % t)
        g.check(infos[INFO_TRUTH_OBS].tobytes()
                == rows[:, list(TRUTH)].tobytes(),
                "vectorised step %d: the info entry is not the "
                "ground-truth channels" % t)
        g.check(infos["_" + INFO_TRUTH_OBS].all(),
                "vectorised step %d: the presence mask beside the "
                "info entry is not set for every environment" % t)
        for e in range(n_envs):
            differing = [c for c in TRUTH
                         if rows[e][c].tobytes()
                         != rows[e][_spec.channel_pair(spec, c)].tobytes()]
            g.check(truth_reaching(obs[e], rows[e], differing) == [],
                    "vectorised step %d handed environment %d a "
                    "ground-truth value" % (t, e))
        arms += 3 + n_envs
        flat = np.array([[action_of(t + e)] for e in range(n_envs)],
                        dtype=np.float64)
        obs, _rew, _term, _trunc, infos = venv.step(flat)
        art.step(handle, np.ascontiguousarray(flat.reshape(-1)).ctypes
                 .data_as(ctypes.POINTER(ctypes.c_double)))
        art.obs(handle, whole.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    venv.close()
    art.destroy(handle)
    print("%s: the vectorised shape hands over the measured channels "
          "of both environments" % GATE)

    # ---- 5. the multi-agent shape ------------------------------------
    try:
        import pettingzoo  # noqa: F401  (presence probe only)
    except ImportError:
        print("%s: the multi-agent arms did not run: the pettingzoo "
              "package is not importable with %s" % (GATE, sys.executable))
    else:
        from k26rl.parallel import K26RlParallelEnv

        ma_so = g.compile_fixture("rl_split_ma", SPLIT_MA_KFL)
        art = _abi.Artifact(ma_so)
        handle = art.create(SEED, 1)
        ma_spec = _spec.parse(art.spec_blob(handle))
        penv = K26RlParallelEnv(ma_so, seed=SEED)
        g.check(penv.agent_policy_channels["leader"] == tuple(range(0, 5))
                and penv.agent_truth_channels["leader"] == tuple(range(5, 10)),
                "the sensed agent's channels split as %s and %s"
                % (penv.agent_policy_channels["leader"],
                   penv.agent_truth_channels["leader"]))
        g.check(penv.agent_policy_channels["follower"] == tuple(range(10, 15))
                and penv.agent_truth_channels["follower"] == (),
                "the unsensed agent's channels split as %s and %s"
                % (penv.agent_policy_channels["follower"],
                   penv.agent_truth_channels["follower"]))
        g.check(penv.observation_spaces["leader"].shape == (5,)
                and penv.observation_spaces["follower"].shape == (5,),
                "the per-agent observation spaces are %s and %s"
                % (penv.observation_spaces["leader"].shape,
                   penv.observation_spaces["follower"].shape))
        g.check(penv.truth_observation_spaces["leader"].shape == (5,)
                and penv.truth_observation_spaces["follower"] is None,
                "the per-agent privileged spaces are %s and %s"
                % (penv.truth_observation_spaces["leader"],
                   penv.truth_observation_spaces["follower"]))
        arms += 4

        whole = np.empty(ma_spec.obs_total, dtype=np.float64)
        obs, infos = penv.reset()
        art.obs(handle, whole.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        for t in range(6):
            g.check(obs["leader"].tobytes()
                    == whole[0:5].tobytes()
                    and obs["follower"].tobytes()
                    == whole[10:15].tobytes(),
                    "multi-agent step %d: an agent's observation is not "
                    "its own measured channels" % t)
            g.check(infos["leader"][INFO_TRUTH_OBS].tobytes()
                    == whole[5:10].tobytes(),
                    "multi-agent step %d: the sensed agent's info entry "
                    "is not its ground-truth channels" % t)
            g.check(INFO_TRUTH_OBS not in infos["follower"],
                    "multi-agent step %d: the agent that declares no "
                    "sensor carries a ground-truth entry" % t)
            differing = [c for c in range(5, 10)
                         if whole[c].tobytes() != whole[c - 5].tobytes()]
            g.check(truth_reaching(obs["leader"], whole, differing) == [],
                    "multi-agent step %d handed the sensed agent a "
                    "ground-truth value" % t)
            arms += 4
            act = {"leader": np.array([action_of(t)], dtype=np.float64),
                   "follower": np.array([action_of(t + 1)],
                                        dtype=np.float64)}
            obs, _rew, _term, _trunc, infos = penv.step(act)
            flat = np.array([action_of(t), action_of(t + 1)],
                            dtype=np.float64)
            art.step(handle, flat.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)))
            art.obs(handle,
                    whole.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        penv.close()
        art.destroy(handle)
        print("%s: the multi-agent shape splits each agent's own slice"
              % GATE)

    # ---- 6. a world that declares no sensor --------------------------
    plain_so = g.compile_fixture("rl_pointing")
    plain = K26RlEnv(plain_so, seed=SEED)
    plain_total = plain.env_spec.obs_total
    obs, info = plain.reset()
    g.check(obs.shape == (plain_total,) and plain.truth_channels == (),
            "a world with no sensor observes %s of a declared %d"
            % (obs.shape, plain_total))
    g.check(plain.truth_observation_space is None
            and INFO_TRUTH_OBS not in info,
            "a world with no sensor carries a privileged space %s and "
            "info keys %s" % (plain.truth_observation_space,
                              sorted(info)))
    plain.close()
    arms += 2
    print("%s: a world with no sensor is unchanged: %d channels, no "
          "privileged entry" % (GATE, plain_total))

    # ---- 7. the pairing refusals -------------------------------------
    def blob(entries):
        """A minimal valid spec with the given source tags appended."""
        parts = [
            tlv(_spec.TAG_ABI_VERSION, struct.pack("<I",
                                                   ABI_VERSION)),
            tlv(_spec.TAG_ENDIAN_PROBE,
                struct.pack("<I", _spec.ENDIAN_PROBE_VALUE)),
            tlv(_spec.TAG_AGENT_COUNT, struct.pack("<I", 1)),
            tlv(_spec.TAG_N_ENVS, struct.pack("<I", 1)),
            tlv(_spec.TAG_OBS_TOTAL, struct.pack("<I", 4)),
            tlv(_spec.TAG_ACT_TOTAL, struct.pack("<I", 1)),
            tlv(_spec.TAG_EPISODE_FLAGS,
                struct.pack("<I", _spec.EPISODE_FLAG_AUTO_RESET)),
        ]
        return b"".join(parts + list(entries))

    def refused(entries, wanted):
        parsed = _spec.parse(blob(entries))
        try:
            _spec.validate(parsed, ABI_VERSION, 1)
        except K26RlError as exc:
            g.check(wanted in str(exc),
                    "the refusal reads %r, which does not name %r"
                    % (str(exc), wanted))
            return
        g.check(False, "a spec whose pairing is %s was accepted" % wanted)

    M = _spec.OBS_SOURCE_MEASURED
    T = _spec.OBS_SOURCE_TRUTH
    NONE = _spec.OBS_PAIR_NONE
    refused([source_tlv(0, 7, NONE)], "neither measured")
    refused([source_tlv(0, M, 9)], "outside the declared observation")
    refused([source_tlv(9, M, NONE)], "outside the declared observation")
    refused([source_tlv(0, M, 0)], "paired with itself")
    refused([source_tlv(0, M, 1), source_tlv(1, M, 0)],
            "both publish source kind")
    refused([source_tlv(0, M, 1), source_tlv(1, T, 2),
             source_tlv(2, M, 1)], "names itself from both ends")
    refused([source_tlv(0, T, NONE)], "no measured channel paired")
    arms += 7

    # An artifact that publishes no source tag at all predates them,
    # and every one of its channels is a measurement with nothing
    # beside it, which is also what a program declaring no sensor
    # publishes explicitly.
    untagged = _spec.parse(blob([]))
    _spec.validate(untagged, ABI_VERSION, 1)
    g.check(_spec.split_channels(untagged, 0, 4) == ([0, 1, 2, 3], []),
            "an untagged spec splits as %s"
            % (_spec.split_channels(untagged, 0, 4),))
    arms += 1
    print("%s: seven malformed pairings refused, an untagged spec "
          "served whole" % GATE)

    # ---- 8. what the exporter declares -------------------------------
    from k26rl import sb3
    from k26rl import policy as k26policy

    g.check(sb3._policy_channels(spec, 0, 0, 5) == [0, 1, 2, 3, 4],
            "the exporter names %s for a slice whose measured "
            "channels lead it" % (sb3._policy_channels(spec, 0, 0, 5),))
    g.check(sb3._policy_channels(spec, 0, 10, 5) == [10, 11, 12, 13, 14],
            "the exporter names %s for a slice of measured "
            "channels" % (sb3._policy_channels(spec, 0, 10, 5),))
    # The whole vector: two runs of measured channels with the truth
    # between them, which no offset and width describes and which the
    # exporter now names channel by channel.
    interleaved = sb3._policy_channels(spec, 0, 0, 15)
    g.check(interleaved == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
            "the exporter names %s for an agent whose measured "
            "channels are interleaved with ground-truth ones"
            % (interleaved,))
    arms += 3

    # And a file carrying that list says so: the header field counts
    # the channels and the body names them, so what a reader gathers
    # is the measured half and nothing beside it.
    written = k26policy.encode(
        [k26policy.Layer([[0.5] * len(interleaved)], [0.0], "tanh")],
        obs_total=15, act_total=1, obs_offset=0, obs_width=15,
        obs_channels=interleaved, provenance="sensor gate")
    count, = struct.unpack_from("<I", written, 76)
    g.check(count == len(interleaved),
            "the written policy declares %d channels for a list of %d"
            % (count, len(interleaved)))
    body = 100 + len("sensor gate")
    listed = list(struct.unpack_from("<%dI" % count, written, body))
    g.check(listed == interleaved,
            "the written policy names %s where the exporter named %s"
            % (listed, interleaved))
    arms += 2

    # An agent with nothing measured has nothing a policy may read,
    # and that is a refusal rather than an empty list.
    try:
        sb3._policy_channels(spec, 0, 5, 5)
    except K26RlError as exc:
        g.check("no measured observation channel" in str(exc),
                "the exporter's refusal reads %r" % str(exc))
    else:
        g.check(False,
                "the exporter named channels for an agent whose slice "
                "carries none a policy may read")
    arms += 1
    print("%s: the exporter names the measured channels, interleaved "
          "or not, and writes them into the file" % GATE)

    print("%s: %d arms" % (GATE, arms))
    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
