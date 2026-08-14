"""Vectorised Gymnasium mapping over a compiled artifact.

The artifact's auto-reset stepping matches gymnasium's
reset-on-next-step autoreset convention: after an environment's
episode ends, a faulted episode included, its next step performs the
boundary reset, its action slice ignored, its observation the new
episode's initial one, its reward zero, and neither ending flag set.
The declared autoreset mode says exactly that.
"""

import numpy as np
import gymnasium
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from . import _abi
from .env import (
    _Session,
    build_action_space,
    build_observation_space,
    check_options,
    check_seed,
    flatten_action_batch,
)
from ._errors import K26RlFaultError


class K26RlVectorEnv(VectorEnv):
    """``n_envs`` environments in one handle behind
    gymnasium.vector.VectorEnv.

    Construction loads the artifact at ``artifact_path``, checks its
    ABI version, creates the handle with the required explicit
    ``seed``, and builds the spaces from the artifact's spec blob
    alone. Buffer geometry is env-major and spec-driven; per-
    environment slicing is arithmetic.

    One integer seed governs the whole handle: per-environment
    independence comes from the artifact's draw coordinates, not from
    per-environment seeds, so seed lists are refused.

    ``on_fault`` selects the fault mode, as on the single shape;
    stepping continues through the boundary reset either way.

    Not thread-safe; any number of instances per process.
    """

    metadata = {
        "render_modes": [],
        "autoreset_mode": gymnasium.vector.AutoresetMode.NEXT_STEP,
    }
    render_mode = None

    def __init__(self, artifact_path, seed, n_envs, on_fault="raise"):
        self._session = _Session(artifact_path, seed, n_envs, on_fault)
        spec = self._session.spec
        self.num_envs = self._session.n_envs
        self.single_observation_space = build_observation_space(spec)
        self.single_action_space = build_action_space(spec)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs)
        self.action_space = batch_space(
            self.single_action_space, self.num_envs)

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
        self._session.ensure_open()
        return dict(self._session.spec.obs_channel_names)

    @property
    def obs_channel_kinds(self):
        self._session.ensure_open()
        return dict(self._session.spec.obs_channel_kinds)

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

    def reset(self, *, seed=None, options=None):
        self._session.ensure_open()
        check_options(options)
        if seed is not None:
            if isinstance(seed, (list, tuple, np.ndarray)):
                raise TypeError(
                    "one integer seed governs the whole handle: "
                    "per-environment independence comes from the "
                    "artifact's draw coordinates, not per-environment "
                    "seeds, so a seed list would misrepresent what "
                    "was seeded and is refused")
            seed = check_seed(seed)
        # The base reset keeps np_random available as the ecosystem
        # expects; nothing here ever draws from it, because every
        # draw in an environment's life is artifact-side.
        super().reset(seed=seed, options=options)
        obs = self._session.reset_routed(seed)
        return obs, {}

    def step(self, actions):
        self._session.ensure_open()
        flat = flatten_action_batch(actions, self.single_action_space,
                                    self._session.spec, self.num_envs)
        obs, rew, flags, fault_mask, codes = \
            self._session.step_flat(flat)

        terminations = (flags & _abi.FLAG_TERMINATED) != 0
        truncations = (flags & _abi.FLAG_TRUNCATED) != 0
        rewards = rew.astype(np.float64, copy=True)
        infos = {}
        faulted = bool(fault_mask.any())
        if faulted:
            # A fault is not termination, and the episode ended for a
            # reason outside the task's own dynamics, so truncated is
            # the rendering for exactly the faulted environments; the
            # faulting step completed no transition, so their
            # observations are the pre-step ones with reward zero, as
            # the artifact delivers them.
            truncations = truncations | fault_mask
            indices, picked, reasons = \
                self._session.fault_details(fault_mask, codes)
            for i, code, reason in zip(indices, picked, reasons):
                infos = self._add_info(
                    infos, {"fault_code": code, "fault_reason": reason},
                    i)

        results = (obs, rewards, terminations, truncations, infos)
        if faulted and self._session.on_fault == "raise":
            # Raised after the completed call, with the results that
            # already report the truncation: the faulted environment
            # did not abort its neighbours, which all advanced.
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

    def close_extras(self, **kwargs):
        self._session.close()
