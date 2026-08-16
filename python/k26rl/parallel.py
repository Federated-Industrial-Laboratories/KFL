"""Multi-agent PettingZoo mapping over a compiled artifact.

An artifact whose spec declares more than one agent publishes, per
agent, an observation slice, an action slice and a reward entry; the
observation channel names publish qualified as ``<agent>.<channel>``.
Everything a per-agent API needs is therefore already in the blob, and
this module is marshalling on top of it: dictionary keys from the
published names, per-agent arrays from the published slices, and no
arithmetic on any stream value.

**The parallel API, not the agent-environment cycle.** PettingZoo has
two APIs: the turn-based agent-environment cycle, in which agents act
one at a time, and the parallel API, in which every agent acts
simultaneously. The stepping surface underneath takes one concatenated
action vector and advances every agent together, and an episode ends
for every agent at once, so the environment is a simultaneous-move
one. The parallel API is what it is; the cycle API would need a turn
order that the simulation does not have, and inventing one would put
behaviour in a wrapper that the environment cannot support.

**One environment per instance.** This shape creates its handle with
one environment. The parallel API has no vectorised form for a wrapper
to implement honestly, and the package's vectorised Gymnasium shape
(:class:`k26rl.vector.K26RlVectorEnv`) serves single-agent artifacts
only; running many multi-agent environments at once is not something
this package offers today.

**PettingZoo is an optional dependency**, imported here and nowhere
else, so a consumer who has Gymnasium and not PettingZoo can import
and use the rest of the package unchanged.
"""

import numpy as np
import gymnasium

try:
    from pettingzoo.utils.env import ParallelEnv
except ImportError as exc:  # pragma: no cover - dependency probe
    raise ImportError(
        "k26rl's multi-agent shape needs the optional pettingzoo "
        "package, which is not installed; the single-agent shapes "
        "k26rl.K26RlEnv and k26rl.K26RlVectorEnv need only gymnasium"
    ) from exc

from . import _abi, _spec
from ._errors import K26RlFaultError
from .env import (
    _Session,
    build_action_space_of,
    build_observation_space_of,
    check_options,
    check_seed,
    flatten_action_into,
)

# This shape holds one environment, so every env-major index in the
# buffers is taken at environment 0: agent ``a``'s reward for
# environment ``e`` lives at ``e * agent_count + a``, and the flag word
# is one per environment.
_ENV = 0


class K26RlParallelEnv(ParallelEnv):
    """A multi-agent artifact behind PettingZoo's ``ParallelEnv``.

    Construction loads the artifact at ``artifact_path``, checks its
    ABI version, creates a one-environment handle with the required
    explicit ``seed``, and builds the agent list and the per-agent
    spaces from the artifact's spec blob alone. The environment is live
    from birth: construction completes episode 0's initial reset.

    ``agents`` and ``possible_agents`` carry the published agent names,
    in agent-index order. ``reset`` and ``step`` take and return
    dictionaries keyed by those names.

    Termination is per environment rather than per agent: the episode
    ends for every agent at once, on the program's termination
    predicate, on horizon truncation, or on fault. Both returned
    dictionaries therefore report the same value for every agent,
    because that is what the environment does; a wrapper that reported
    otherwise would be inventing a per-agent ending the artifact does
    not have. On an ending step ``agents`` empties, as the API's
    convention has it, and ``reset`` repopulates it.

    ``on_fault`` selects the fault mode, as on the single-agent
    shapes: ``"raise"`` (the default) raises :class:`K26RlFaultError`
    after the faulting step's results are formed, ``"truncate"``
    returns the same results without raising, and no mode drops a
    fault.

    After any episode ending, a caught fault included, the next call is
    ``reset()``; stepping a finished episode raises
    ``gymnasium.error.ResetNeeded`` rather than letting the artifact's
    auto-reset engage unannounced.

    Not thread-safe; any number of instances per process.
    """

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, artifact_path, seed, on_fault="raise"):
        self._session = _Session(artifact_path, seed, 1, on_fault,
                                 multi_agent=True)
        try:
            spec = self._session.spec
            names = _spec.agent_names(spec)
            obs_slices = _spec.slices_by_agent(spec.agent_obs_slices,
                                               spec.agent_count)
            act_slices = _spec.slices_by_agent(spec.agent_act_slices,
                                               spec.agent_count)
        except Exception:
            self._session.close()
            raise

        self.possible_agents = list(names)
        self.agents = list(names)
        self._obs_slice = dict(zip(names, obs_slices))
        self._act_slice = dict(zip(names, act_slices))
        self._act_channels = {
            name: spec.act_channels[offset:offset + count]
            for name, (offset, count) in self._act_slice.items()}
        self._reward_index = {name: agent
                              for agent, name in enumerate(names)}
        # Built once and returned by identity: the API requires
        # observation_space(agent) and action_space(agent) to return
        # the same object every call, so that seeding a sampled space
        # sticks.
        self.observation_spaces = {
            name: build_observation_space_of(count)
            for name, (_offset, count) in self._obs_slice.items()}
        self.action_spaces = {
            name: build_action_space_of(self._act_channels[name])
            for name in names}
        self._needs_reset = False

    # ---- spec data exposed for consumers and tooling ------------------
    #
    # Like the methods, the spec-reading properties refuse after
    # close(). The two record properties, seeds_held and output_path,
    # stay readable: they are plain Python records of what the run
    # did, wanted precisely after it, and reading them touches no
    # artifact state.

    @property
    def env_spec(self):
        """The parsed spec value object."""
        self._session.ensure_open()
        return self._session.spec

    @property
    def control_dt(self):
        """Simulated seconds per external step."""
        self._session.ensure_open()
        return self._session.spec.control_dt

    @property
    def obs_channel_names(self):
        """Every observation channel's published name, by its offset in
        the whole observation vector. Above one agent the names are
        qualified, which is where the agent names come from."""
        self._session.ensure_open()
        return dict(self._session.spec.obs_channel_names)

    @property
    def obs_channel_kinds(self):
        self._session.ensure_open()
        return dict(self._session.spec.obs_channel_kinds)

    @property
    def agent_obs_channel_names(self):
        """Each agent's observation channel names, in the order its own
        observation array carries them."""
        self._session.ensure_open()
        published = self._session.spec.obs_channel_names
        return {name: [published[c]
                       for c in range(offset, offset + count)]
                for name, (offset, count) in self._obs_slice.items()}

    @property
    def on_fault(self):
        self._session.ensure_open()
        return self._session.on_fault

    @property
    def seeds_held(self):
        """Every seed this environment object's handles have held,
        cumulative across recreates. Read-only, and still readable
        after close(): the seed record of a finished run is exactly
        what reproduction needs."""
        return self._session.seeds_held

    # ---- the API ------------------------------------------------------

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self._session.ensure_open()
        check_options(options, refuse=False)
        if seed is not None:
            seed = check_seed(seed)
        # Nothing seeds a Python-side generator here: every draw in an
        # environment's life is artifact-side, and this API defines no
        # per-environment random number generator to keep available.
        obs = self._session.reset_routed(seed)
        self.agents = list(self.possible_agents)
        self._needs_reset = False
        return (self._split_obs(obs),
                {name: {} for name in self.agents})

    def step(self, actions):
        self._session.ensure_open()
        if self._needs_reset:
            raise gymnasium.error.ResetNeeded(
                "the episode has ended for every agent; call reset() "
                "before stepping")
        acting = list(self.agents)
        flat = self._flatten(actions, acting)
        obs, rew, flags, fault_mask, codes = \
            self._session.step_flat(flat)

        word = int(flags[_ENV])
        terminated = bool(word & _abi.FLAG_TERMINATED)
        truncated = bool(word & _abi.FLAG_TRUNCATED)
        infos = {name: {} for name in acting}
        faulted = bool(word & _abi.FLAG_FAULT)
        if faulted:
            # A fault is not termination, and the episode ended for a
            # reason outside the task's own dynamics, so truncated is
            # the rendering; the faulting step completed no
            # transition, so the observations are the pre-step ones
            # with reward zero, as the artifact delivers them. The
            # fault is the environment's, so every agent carries it.
            truncated = True
            indices, picked, reasons = \
                self._session.fault_details(fault_mask, codes)
            # One environment, so the one faulted entry is the first.
            for name in acting:
                infos[name] = {"fault_code": picked[0],
                               "fault_reason": reasons[0]}

        agent_count = self._session.spec.agent_count
        rewards = {
            name: float(rew[_ENV * agent_count + self._reward_index[name]])
            for name in acting}
        terminations = {name: terminated for name in acting}
        truncations = {name: truncated for name in acting}
        if terminated or truncated:
            # The convention is that an agent whose episode has ended
            # leaves the roster; here they all leave together.
            self.agents = []
            self._needs_reset = True

        results = (self._split_obs(obs), rewards, terminations,
                   truncations, infos)
        if faulted and self._session.on_fault == "raise":
            raise K26RlFaultError(indices, picked, reasons, results)
        return results

    def set_output(self, path):
        """Enable episode-file emission to path, or disable it with
        None."""
        self._session.ensure_open()
        self._session.set_output(path)

    @property
    def output_path(self):
        """The enabled episode-output path, or None. Still readable
        after close(): it locates the recorded file."""
        return self._session.output_path

    def close(self):
        # Idempotent, as the ecosystem expects; everything else on a
        # closed environment is refused before any artifact call,
        # except the seeds_held and output_path records.
        self._session.close()

    # ---- marshalling --------------------------------------------------

    def _split_obs(self, obs):
        """The one environment's flat observation vector cut into one
        array per agent at the published offsets. Each agent's array is
        its own copy, so no consumer's stored observation is a view
        over another's."""
        row = obs[_ENV]
        return {name: row[offset:offset + count].copy()
                for name, (offset, count) in self._obs_slice.items()}

    def _flatten(self, actions, acting):
        """The per-agent actions onto the one concatenated action
        vector the stepping surface takes, each at its own published
        offset."""
        if not isinstance(actions, dict):
            raise TypeError(
                "actions must be a mapping keyed by agent name, not %s"
                % type(actions).__name__)
        missing = [name for name in acting if name not in actions]
        if missing:
            raise ValueError(
                "no action for %s; every agent still in the episode "
                "acts on every step" % ", ".join(missing))
        unknown = [str(key) for key in actions if key not in acting]
        if unknown:
            raise ValueError(
                "actions carry %s, which %s not an agent of this "
                "environment; its agents are %s"
                % (", ".join(sorted(unknown)),
                   "are" if len(unknown) > 1 else "is",
                   ", ".join(acting)))
        # Zero-filled rather than uninitialised: the load-time
        # partition check has established that the agents' slices cover
        # the vector, and a deterministic value is what a determinism
        # claim wants underneath it if that ever ceases to hold.
        flat = np.zeros(self._session.spec.act_total, dtype=np.float64)
        for name in acting:
            offset, _count = self._act_slice[name]
            flatten_action_into(flat, offset, actions[name],
                                self.action_spaces[name],
                                self._act_channels[name],
                                "action for agent %r" % name)
        return flat
