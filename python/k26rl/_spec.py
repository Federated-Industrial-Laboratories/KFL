"""Decoder for the artifact's env_spec TLV blob.

The blob is a sequence of (tag uint16, length uint32, value) triples,
little-endian, no alignment padding; unknown tags are skipped by
length. Everything this package knows about an environment it learns
here, so the parse is strict about what it reads and silent about
what it does not know.

No numpy, no gymnasium: the parsed value object is plain Python, and
the spaces built from it live with the environment classes.
"""

import struct

from ._errors import K26RlError

TAG_ABI_VERSION = 0x0001
TAG_ENDIAN_PROBE = 0x0002
TAG_AGENT_COUNT = 0x0003
TAG_N_ENVS = 0x0004
TAG_CONTROL_DT = 0x0005
TAG_HORIZON = 0x0006
TAG_OBS_TOTAL = 0x0007
TAG_ACT_TOTAL = 0x0008
TAG_AGENT_OBS_SLICE = 0x0009
TAG_AGENT_ACT_SLICE = 0x000A
TAG_ACT_BOUNDS = 0x000B
TAG_ACT_KIND = 0x000C
TAG_OBS_CHANNEL_NAME = 0x000D
TAG_OBS_CHANNEL_KIND = 0x000E
TAG_EPISODE_FLAGS = 0x000F
TAG_REWARD_COMPONENTS = 0x0010

ACT_KIND_BOX = 0
ACT_KIND_DISCRETE = 1

OBS_KIND_VECTOR = 0

ENDIAN_PROBE_VALUE = 0x01020304

EPISODE_FLAG_AUTO_RESET = 1 << 0

_INF = float("inf")


class ActChannel:
    """One action scalar: its offset in the flat action vector, its
    kind, its bounds (box), and its arity (discrete)."""

    __slots__ = ("offset", "kind", "arity", "lo", "hi")

    def __init__(self, offset, kind, arity, lo, hi):
        self.offset = offset
        self.kind = kind
        self.arity = arity
        self.lo = lo
        self.hi = hi


class Spec:
    """The parsed spec: scalar tags as attributes, per-channel and
    per-agent tags as lists and mappings. Values default to None (or
    empty containers) until their tag is seen."""

    def __init__(self):
        self.abi_version = None
        self.endian_probe = None
        self.agent_count = None
        self.n_envs = None
        self.control_dt = None
        self.horizon = None
        self.obs_total = None
        self.act_total = None
        self.agent_obs_slices = []
        self.agent_act_slices = []
        self.act_bounds = {}
        self.act_kinds = {}
        self.obs_channel_names = {}
        self.obs_channel_kinds = {}
        self.episode_flags = None
        self.act_channels = []


def _malformed(detail):
    raise K26RlError(None, "spec blob is malformed: " + detail)


def _need_len(tag, length, expect):
    if length != expect:
        _malformed("tag 0x%04x carries %d bytes, expected %d"
                   % (tag, length, expect))


def parse(blob):
    """Parse a spec blob into a :class:`Spec`. Unknown tags are
    skipped by length; a truncated entry is refused."""
    spec = Spec()
    off = 0
    end = len(blob)
    while off < end:
        if off + 6 > end:
            _malformed("entry header at byte %d runs past the end" % off)
        tag, length = struct.unpack_from("<HI", blob, off)
        value_off = off + 6
        if value_off + length > end:
            _malformed("tag 0x%04x at byte %d runs past the end"
                       % (tag, off))

        if tag == TAG_ABI_VERSION:
            _need_len(tag, length, 4)
            (spec.abi_version,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_ENDIAN_PROBE:
            _need_len(tag, length, 4)
            (spec.endian_probe,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_AGENT_COUNT:
            _need_len(tag, length, 4)
            (spec.agent_count,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_N_ENVS:
            _need_len(tag, length, 4)
            (spec.n_envs,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_CONTROL_DT:
            _need_len(tag, length, 8)
            # The payload is the binary64 bit pattern; unpacking it as
            # a little-endian double reads those bits exactly.
            (spec.control_dt,) = struct.unpack_from("<d", blob, value_off)
        elif tag == TAG_HORIZON:
            _need_len(tag, length, 4)
            (spec.horizon,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_OBS_TOTAL:
            _need_len(tag, length, 4)
            (spec.obs_total,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_ACT_TOTAL:
            _need_len(tag, length, 4)
            (spec.act_total,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_AGENT_OBS_SLICE:
            _need_len(tag, length, 12)
            spec.agent_obs_slices.append(
                struct.unpack_from("<III", blob, value_off))
        elif tag == TAG_AGENT_ACT_SLICE:
            _need_len(tag, length, 12)
            spec.agent_act_slices.append(
                struct.unpack_from("<III", blob, value_off))
        elif tag == TAG_ACT_BOUNDS:
            _need_len(tag, length, 20)
            offset, lo, hi = struct.unpack_from("<Idd", blob, value_off)
            spec.act_bounds[offset] = (lo, hi)
        elif tag == TAG_ACT_KIND:
            # 6 bytes for a box channel; 10 with the trailing arity
            # for a discrete one.
            if length == 6:
                offset, kind = struct.unpack_from("<IH", blob, value_off)
                arity = 0
            elif length == 10:
                offset, kind, arity = struct.unpack_from(
                    "<IHI", blob, value_off)
            else:
                _malformed("tag 0x%04x carries %d bytes, expected 6 or 10"
                           % (tag, length))
            spec.act_kinds[offset] = (kind, arity)
        elif tag == TAG_OBS_CHANNEL_NAME:
            if length < 4:
                _malformed("tag 0x%04x carries %d bytes, expected at "
                           "least 4" % (tag, length))
            (channel,) = struct.unpack_from("<I", blob, value_off)
            name = blob[value_off + 4:value_off + length]
            spec.obs_channel_names[channel] = name.decode("utf-8",
                                                          "replace")
        elif tag == TAG_OBS_CHANNEL_KIND:
            _need_len(tag, length, 6)
            channel, kind = struct.unpack_from("<IH", blob, value_off)
            spec.obs_channel_kinds[channel] = kind
        elif tag == TAG_EPISODE_FLAGS:
            _need_len(tag, length, 4)
            (spec.episode_flags,) = struct.unpack_from("<I", blob,
                                                       value_off)
        # Any other tag, the reserved reward-components tag included,
        # is skipped by length: additions arrive as new tags under the
        # minor version rule and older consumers step over them.

        off = value_off + length

    if spec.act_total is not None:
        for offset in range(spec.act_total):
            kind, arity = spec.act_kinds.get(offset, (ACT_KIND_BOX, 0))
            lo, hi = spec.act_bounds.get(offset, (-_INF, _INF))
            spec.act_channels.append(
                ActChannel(offset, kind, arity, lo, hi))
    return spec


def validate(spec, abi_version, n_envs):
    """Refuse a spec this package cannot honestly serve. The checks
    are the load-time contract: probe, version echo, geometry echo,
    slice arithmetic, single-agent, and auto-reset stepping."""
    if spec.endian_probe != ENDIAN_PROBE_VALUE:
        raise K26RlError(
            None,
            "spec endian probe reads 0x%08x, expected 0x%08x"
            % (spec.endian_probe or 0, ENDIAN_PROBE_VALUE))
    if spec.abi_version != abi_version:
        raise K26RlError(
            None,
            "spec declares ABI version 0x%08x but the artifact "
            "reports 0x%08x" % (spec.abi_version or 0, abi_version))
    if spec.n_envs != n_envs:
        raise K26RlError(
            None,
            "spec declares %s environments but the handle was created "
            "with %d" % (spec.n_envs, n_envs))
    if spec.obs_total is None or spec.act_total is None:
        raise K26RlError(
            None, "spec declares no observation or action totals")
    if spec.agent_count != 1:
        raise K26RlError(
            None,
            "artifact declares agent count %s; version 1 of this "
            "package serves single-agent artifacts only"
            % spec.agent_count)
    _check_slices(spec.agent_obs_slices, spec.obs_total, "observation")
    _check_slices(spec.agent_act_slices, spec.act_total, "action")
    for chan in spec.act_channels:
        if chan.kind == ACT_KIND_DISCRETE and chan.arity < 1:
            raise K26RlError(
                None,
                "discrete action channel at offset %d declares arity "
                "%d" % (chan.offset, chan.arity))
        elif chan.kind not in (ACT_KIND_BOX, ACT_KIND_DISCRETE):
            raise K26RlError(
                None,
                "action channel at offset %d declares unknown kind %d"
                % (chan.offset, chan.kind))
    if spec.episode_flags is None or \
            not spec.episode_flags & EPISODE_FLAG_AUTO_RESET:
        raise K26RlError(
            None,
            "artifact declares episode flags 0x%x with the auto-reset "
            "bit clear; this package maps auto-reset stepping only "
            "and has no non-auto-reset mapping in version 1"
            % (spec.episode_flags or 0))


def _check_slices(slices, total, what):
    covered = 0
    for agent, offset, count in slices:
        if offset + count > total:
            raise K26RlError(
                None,
                "agent %d %s slice [%d, %d) runs past the declared "
                "total %d" % (agent, what, offset, offset + count,
                              total))
        covered += count
    if slices and covered != total:
        raise K26RlError(
            None,
            "agent %s slices cover %d of the declared total %d"
            % (what, covered, total))
