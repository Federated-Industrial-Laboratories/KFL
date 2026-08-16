"""Gate 2: spec round trip from Python.

Against the compiled fixture: the sizing protocol (capacity 0 sizes,
undersized capacity writes nothing, exact capacity fills), the TLV
parse, spaces and buffer geometry equal to the declared totals, the
endian probe validated, and an unknown-tag injection skipped cleanly
with every recovered value unchanged.

Against a stub artifact whose sizing call returns a negated status:
the typed raise with the negation undone and the message equal to the
artifact's own decode.

From synthetic blobs: the validation refusals (endian probe, ABI
version echo, environment-count echo, agent count below 1, slice
overrun, slice under-coverage, unknown action kind, action total 0,
episode auto-reset bit clear), each naming what it found; the
single-agent shapes' own refusal of an artifact declaring more than
one agent, which is the shape's question rather than the blob's; and
the four action-space branches: all box, single discrete, all
discrete, and mixed declarations yield Box, Discrete, MultiDiscrete,
and Tuple.

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent.
"""

import ctypes
import struct
import sys

import _gateutil as g

GATE = "gate02_spec"


def tlv(tag, payload):
    return struct.pack("<HI", tag, len(payload)) + payload


def u32(v):
    return struct.pack("<I", v)


def f64(v):
    return struct.pack("<d", v)


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    import numpy as np
    from gymnasium import spaces as gym_spaces
    from k26rl import _abi, _spec, K26RlSeedReuseError
    from k26rl.env import build_action_space, build_observation_space
    from k26rl.vector import K26RlVectorEnv

    so = g.compile_fixture("rl_pointing")
    n_envs = 2

    # ---- sizing protocol against the real artifact -------------------
    art = _abi.Artifact(so)
    handle = art.create(1234, n_envs)
    blob = art.spec_blob(handle)
    g.check(len(blob) > 0, "spec blob is empty")

    # An undersized capacity must write nothing and still return the
    # requirement. The raw symbol is reached through the binding's
    # resolved table because the public wrapper hides capacities.
    raw_spec = art._fn["k26rl_env_spec"]
    sentinel = (ctypes.c_uint8 * 1)(0xAB)
    need = raw_spec(handle, sentinel, 1)
    g.check(need == len(blob),
            "undersized capacity returned %d, expected %d"
            % (need, len(blob)))
    g.check(sentinel[0] == 0xAB,
            "undersized capacity wrote into the buffer")

    # ---- parse and geometry ------------------------------------------
    spec = _spec.parse(blob)
    _spec.validate(spec, art.abi_version, n_envs)
    g.check(spec.endian_probe == 0x01020304, "endian probe value")
    # One observe-as statement, five components each.
    g.check(spec.obs_total == 5, "fixture obs_total is %s"
            % spec.obs_total)
    g.check(spec.obs_channel_names.get(4) == "track_range_rate",
            "fifth component name is %s"
            % spec.obs_channel_names.get(4))
    g.check(spec.act_total == 2, "fixture act_total is %s"
            % spec.act_total)
    g.check(spec.agent_count == 1, "fixture agent count")
    g.check(len(spec.act_channels) == 2, "channel count")

    obs_space = build_observation_space(spec)
    g.check(obs_space.shape == (spec.obs_total,),
            "observation space shape %s" % (obs_space.shape,))
    act_space = build_action_space(spec)
    g.check(isinstance(act_space, gym_spaces.Tuple),
            "fixture action space is %s, expected Tuple"
            % type(act_space).__name__)

    env = K26RlVectorEnv(so, seed=1234, n_envs=n_envs)
    g.check(env.single_observation_space.shape == (spec.obs_total,),
            "environment observation space shape")
    obs, _ = env.reset()
    g.check(obs.shape == (n_envs, spec.obs_total),
            "reset observation geometry %s" % (obs.shape,))
    g.check(obs.dtype == np.float64, "observation dtype")
    env.close()

    # ---- unknown-tag injection ---------------------------------------
    # Splice an unknown tag after the first entry; the parse must
    # skip it by length and recover every value unchanged.
    first_len = 6 + struct.unpack_from("<I", blob, 2)[0]
    unknown = tlv(0x7FFF, b"\x01\x02\x03\x04\x05")
    spliced = blob[:first_len] + unknown + blob[first_len:]
    spec2 = _spec.parse(spliced)
    for attr in ("abi_version", "endian_probe", "agent_count",
                 "n_envs", "control_dt", "horizon", "obs_total",
                 "act_total", "agent_obs_slices", "agent_act_slices",
                 "act_bounds", "act_kinds", "obs_channel_names",
                 "obs_channel_kinds", "episode_flags"):
        g.check(getattr(spec, attr) == getattr(spec2, attr),
                "unknown-tag injection changed %s" % attr)

    art.destroy(handle)

    # ---- negated status on the sizing call ---------------------------
    # A stub whose sizing call returns the negated seed-reuse status:
    # the negation is undone, the raise is the typed one, and the
    # message is the artifact's own decode of the value.
    specneg = g.build_stub("specneg", ["STUB_SPEC_SIZING_NEGATIVE"])
    ref = ctypes.CDLL(str(specneg))
    ref.k26rl_status_str.restype = ctypes.c_char_p
    ref.k26rl_status_str.argtypes = [ctypes.c_int]
    reuse_text = ref.k26rl_status_str(_abi.E_SEED_REUSE).decode()

    art_neg = _abi.Artifact(specneg)
    h_neg = art_neg.create(1, 1)
    try:
        art_neg.spec_blob(h_neg)
    except K26RlSeedReuseError as exc:
        g.check(exc.status == _abi.E_SEED_REUSE,
                "negated sizing status decoded to %r" % exc.status)
        g.check(exc.message == reuse_text,
                "negated sizing message %r, artifact says %r"
                % (exc.message, reuse_text))
    except Exception as exc:
        g.check(False, "negated sizing status raised %s: %s"
                % (type(exc).__name__, exc))
    else:
        g.check(False, "a negated sizing status was not raised")
    art_neg.destroy(h_neg)

    # ---- synthetic blobs: refusals -----------------------------------
    def make_blob(agent_count=1, episode_flags=1, endian=0x01020304,
                  acts=("box", "discrete"), abi=0x00010000, n_envs=1,
                  obs_slice=(0, 0, 3), act_slice=None):
        if act_slice is None:
            act_slice = (0, 0, len(acts))
        parts = [
            tlv(_spec.TAG_ABI_VERSION, u32(abi)),
            tlv(_spec.TAG_ENDIAN_PROBE, u32(endian)),
            tlv(_spec.TAG_AGENT_COUNT, u32(agent_count)),
            tlv(_spec.TAG_N_ENVS, u32(n_envs)),
            tlv(_spec.TAG_CONTROL_DT, f64(0.5)),
            tlv(_spec.TAG_HORIZON, u32(10)),
            tlv(_spec.TAG_OBS_TOTAL, u32(3)),
            tlv(_spec.TAG_ACT_TOTAL, u32(len(acts))),
            tlv(_spec.TAG_AGENT_OBS_SLICE,
                u32(obs_slice[0]) + u32(obs_slice[1])
                + u32(obs_slice[2])),
            tlv(_spec.TAG_AGENT_ACT_SLICE,
                u32(act_slice[0]) + u32(act_slice[1])
                + u32(act_slice[2])),
            tlv(_spec.TAG_EPISODE_FLAGS, u32(episode_flags)),
        ]
        for i, kind in enumerate(acts):
            if kind == "box":
                parts.append(tlv(_spec.TAG_ACT_BOUNDS,
                                 u32(i) + f64(-1.0) + f64(1.0)))
                parts.append(tlv(_spec.TAG_ACT_KIND,
                                 u32(i) + struct.pack("<H", 0)))
            elif kind == "unknown":
                parts.append(tlv(_spec.TAG_ACT_KIND,
                                 u32(i) + struct.pack("<H", 7)))
            else:
                arity = 3 + i
                parts.append(tlv(_spec.TAG_ACT_KIND,
                                 u32(i) + struct.pack("<H", 1)
                                 + u32(arity)))
        return b"".join(parts)

    def expect_refusal(blob_bytes, needle, what):
        try:
            _spec.validate(_spec.parse(blob_bytes), 0x00010000, 1)
        except Exception as exc:
            g.check(needle in str(exc),
                    "%s refusal does not name what it found: %s"
                    % (what, exc))
            return
        g.check(False, "%s was not refused" % what)

    expect_refusal(make_blob(episode_flags=0), "auto-reset",
                   "auto-reset bit clear")
    expect_refusal(make_blob(agent_count=0), "agent count 0",
                   "agent count 0")

    # Above one agent the refusal belongs to the API shape rather than
    # to the blob: a Gymnasium environment reports one observation,
    # one action and one reward per step, so it cannot serve two
    # agents' streams, and the message names the count found and the
    # shape that does serve it.
    try:
        _spec.require_single_agent(_spec.parse(make_blob(agent_count=2)))
    except Exception as exc:
        g.check("agent count 2" in str(exc)
                and "K26RlParallelEnv" in str(exc),
                "the single-agent refusal does not name what it found "
                "and where the artifact goes: %s" % exc)
    else:
        g.check(False, "a two-agent artifact was not refused by the "
                       "single-agent shapes")
    expect_refusal(make_blob(endian=0x04030201), "0x04030201",
                   "byte-swapped endian probe")
    expect_refusal(make_blob(abi=0x00020000), "0x00020000",
                   "ABI version echo mismatch")
    expect_refusal(make_blob(n_envs=3), "3 environments",
                   "environment-count echo mismatch")
    expect_refusal(make_blob(obs_slice=(0, 0, 4)), "runs past",
                   "slice overrun")
    expect_refusal(make_blob(obs_slice=(0, 0, 2)), "cover 2",
                   "slice under-coverage")
    expect_refusal(make_blob(acts=("unknown",)), "unknown kind 7",
                   "unknown action kind")
    expect_refusal(make_blob(acts=()), "action total 0",
                   "action total 0")

    # ---- synthetic blobs: the four action-space branches -------------
    def space_for(acts):
        return build_action_space(_spec.parse(make_blob(acts=acts)))

    s = space_for(("box", "box"))
    g.check(isinstance(s, gym_spaces.Box) and s.shape == (2,),
            "all-box branch built %r" % s)
    g.check(np.all(s.low == -1.0) and np.all(s.high == 1.0),
            "all-box bounds %r" % s)
    s = space_for(("discrete",))
    g.check(isinstance(s, gym_spaces.Discrete) and s.n == 3,
            "single-discrete branch built %r" % s)
    s = space_for(("discrete", "discrete"))
    g.check(isinstance(s, gym_spaces.MultiDiscrete)
            and list(s.nvec) == [3, 4],
            "all-discrete branch built %r" % s)
    s = space_for(("box", "discrete"))
    g.check(isinstance(s, gym_spaces.Tuple)
            and isinstance(s.spaces[0], gym_spaces.Box)
            and isinstance(s.spaces[1], gym_spaces.Discrete),
            "mixed branch built %r" % s)

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
