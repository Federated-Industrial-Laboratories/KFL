"""ctypes binding for the frozen k26rl stepping surface.

Loads a compiled artifact's shared object by exact path, resolves the
thirteen k26rl_ symbols, and checks the ABI version before any handle
exists. This module is marshalling only, and it imports neither numpy
nor gymnasium, so the binding layer stays usable on its own.

The dlopen mode is RTLD_LOCAL, passed explicitly rather than left to a
platform default, so concurrent artifacts in one process stay
separate. RTLD_NOW matches the surface's C consumers: unresolved
references surface at load, beside the all-symbols-at-load rule below.
"""

import ctypes
import os

from ._errors import (
    K26RlError,
    K26RlOutputError,
    K26RlSeedReuseError,
)

# The one ABI major this package serves, and the least minor it needs.
# The binding uses only version 1.0 symbols and tags, so every 1.x
# artifact passes the minor check.
ABI_MAJOR = 1
ABI_MINOR_MIN = 0

# Version 1 status registry values, used for control flow only:
# exception typing below and fault-code recognition above. Message
# text is never taken from these names; every raised message is
# decoded through the loaded artifact's k26rl_status_str.
OK = 0
E_NULL = 1
E_GEOMETRY = 2
E_RNG_EXHAUSTED = 3
E_SEED_REUSE = 4
E_OUTPUT_TIMING = 5
E_OUTPUT_EXISTS = 6
E_FPU_RACE = 7
E_USE_AFTER_DESTROY = 8
E_INTERNAL = 9
E_DIVERGED = 10
E_ENV_INTERNAL = 11

# Flag word bits, as the surface defines them.
FLAG_TERMINATED = 1 << 0
FLAG_TRUNCATED = 1 << 1
FLAG_FAULT = 1 << 2
FLAG_RESET_BOUNDARY = 1 << 3

_C_STATUS = ctypes.c_int
_C_HANDLE = ctypes.c_void_p

# The thirteen frozen symbols with their signatures. Every one is
# resolved at load; a missing symbol is a refusal naming it.
_SIGNATURES = (
    ("k26rl_abi_version", ctypes.c_uint32, ()),
    ("k26rl_env_create", _C_STATUS,
     (ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(_C_HANDLE))),
    ("k26rl_env_output", _C_STATUS, (_C_HANDLE, ctypes.c_char_p)),
    ("k26rl_env_reset", _C_STATUS, (_C_HANDLE,)),
    ("k26rl_env_reset_seeded", _C_STATUS, (_C_HANDLE, ctypes.c_uint64)),
    ("k26rl_env_step", _C_STATUS,
     (_C_HANDLE, ctypes.POINTER(ctypes.c_double))),
    ("k26rl_env_obs", _C_STATUS,
     (_C_HANDLE, ctypes.POINTER(ctypes.c_double))),
    ("k26rl_env_reward", _C_STATUS,
     (_C_HANDLE, ctypes.POINTER(ctypes.c_double))),
    ("k26rl_env_flags", _C_STATUS,
     (_C_HANDLE, ctypes.POINTER(ctypes.c_uint32))),
    ("k26rl_env_fault_codes", _C_STATUS,
     (_C_HANDLE, ctypes.POINTER(ctypes.c_uint16))),
    ("k26rl_env_spec", ctypes.c_int32,
     (_C_HANDLE, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32)),
    ("k26rl_status_str", ctypes.c_char_p, (_C_STATUS,)),
    ("k26rl_env_destroy", None, (_C_HANDLE,)),
)


class Artifact:
    """One loaded artifact: the shared object, its resolved symbols,
    and the checked-call wrappers. Handles are opaque values created
    by :meth:`create` and owned by the caller."""

    def __init__(self, path):
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(
                "artifact path must be a str or os.PathLike, not %s"
                % type(path).__name__)
        self.path = os.fspath(path)
        try:
            self._lib = ctypes.CDLL(self.path,
                                    mode=os.RTLD_NOW | os.RTLD_LOCAL)
        except OSError as exc:
            # A missing or unloadable file surfaces as the package's
            # typed error with the loader's own report chained.
            raise K26RlError(
                None,
                "artifact %s could not be loaded: %s"
                % (self.path, exc)) from exc
        self._fn = {}
        for name, restype, argtypes in _SIGNATURES:
            try:
                fn = getattr(self._lib, name)
            except AttributeError:
                raise K26RlError(
                    None,
                    "artifact %s is missing required symbol %s"
                    % (self.path, name)) from None
            fn.restype = restype
            fn.argtypes = list(argtypes)
            self._fn[name] = fn

        word = int(self._fn["k26rl_abi_version"]())
        self.abi_version = word
        self.abi_major = word >> 16
        self.abi_minor = word & 0xFFFF
        if self.abi_major != ABI_MAJOR:
            raise K26RlError(
                None,
                "artifact %s reports ABI major %d; this package "
                "supports major %d"
                % (self.path, self.abi_major, ABI_MAJOR))
        if self.abi_minor < ABI_MINOR_MIN:
            raise K26RlError(
                None,
                "artifact %s reports ABI minor %d; this package "
                "needs at least minor %d"
                % (self.path, self.abi_minor, ABI_MINOR_MIN))

    # ---- status decode ------------------------------------------------

    def status_message(self, value):
        """The artifact's own decode of a status value, fetched at
        call time so values minted after this package was written
        still decode to their true names."""
        text = self._fn["k26rl_status_str"](int(value))
        if text is None:
            # The artifact returned no decode string; the numeric
            # value is all there is to report.
            return "status %d" % int(value)
        return text.decode("utf-8", "replace")

    def _raise(self, status, symbol):
        # Exception typing branches on the version 1 registry values;
        # anything outside them raises the base type. The message is
        # exactly the artifact's decode, never a local rewording.
        status = int(status)
        message = self.status_message(status)
        if status == E_SEED_REUSE:
            raise K26RlSeedReuseError(status, message)
        if status in (E_OUTPUT_TIMING, E_OUTPUT_EXISTS):
            raise K26RlOutputError(status, message)
        raise K26RlError(status, message)

    def _check(self, status, symbol):
        if int(status) != OK:
            self._raise(status, symbol)

    # ---- checked calls, one per symbol --------------------------------

    def create(self, seed, n_envs):
        handle = _C_HANDLE()
        st = self._fn["k26rl_env_create"](
            int(seed), int(n_envs), ctypes.byref(handle))
        self._check(st, "k26rl_env_create")
        if not handle.value:
            raise K26RlError(
                None, "artifact %s returned no handle from a "
                "successful create" % self.path)
        return handle

    def output(self, handle, path):
        encoded = None if path is None else os.fsencode(path)
        st = self._fn["k26rl_env_output"](handle, encoded)
        self._check(st, "k26rl_env_output")

    def reset(self, handle):
        self._check(self._fn["k26rl_env_reset"](handle),
                    "k26rl_env_reset")

    def reset_seeded(self, handle, seed):
        self._check(self._fn["k26rl_env_reset_seeded"](handle, int(seed)),
                    "k26rl_env_reset_seeded")

    def step(self, handle, actions_ptr):
        self._check(self._fn["k26rl_env_step"](handle, actions_ptr),
                    "k26rl_env_step")

    def obs(self, handle, out_ptr):
        self._check(self._fn["k26rl_env_obs"](handle, out_ptr),
                    "k26rl_env_obs")

    def reward(self, handle, out_ptr):
        self._check(self._fn["k26rl_env_reward"](handle, out_ptr),
                    "k26rl_env_reward")

    def flags(self, handle, out_ptr):
        self._check(self._fn["k26rl_env_flags"](handle, out_ptr),
                    "k26rl_env_flags")

    def fault_codes(self, handle, out_ptr):
        self._check(self._fn["k26rl_env_fault_codes"](handle, out_ptr),
                    "k26rl_env_fault_codes")

    def spec_blob(self, handle):
        """The spec TLV blob, fetched by the sizing protocol: a
        capacity-0 call sizes it, an exact-capacity call fills it. A
        negative return is the negated status, undone and decoded."""
        fn = self._fn["k26rl_env_spec"]
        need = int(fn(handle, None, 0))
        if need < 0:
            self._raise(-need, "k26rl_env_spec")
        buf = (ctypes.c_uint8 * need)()
        got = int(fn(handle, buf, need))
        if got < 0:
            self._raise(-got, "k26rl_env_spec")
        if got != need:
            # The blob never changes after create, so the two sizing
            # answers disagreeing is an artifact defect worth naming.
            raise K26RlError(
                None,
                "artifact %s sized its spec blob at %d bytes and then "
                "at %d bytes" % (self.path, need, got))
        return bytes(buf)

    def destroy(self, handle):
        # Void return; the one refusable condition (use after
        # destroy) is prevented by the owning object's closed guard.
        self._fn["k26rl_env_destroy"](handle)
