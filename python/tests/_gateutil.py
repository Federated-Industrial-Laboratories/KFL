"""Shared plumbing for the k26rl package gates.

Follows the compiled-gate pattern the compiler's test binaries use:
compile a .kfl source through the built kflc with the stack's include
and archive lists, then drive the produced shared object. Gates that
need the stack call require_stack() and exit 77 with a printed reason
when the compiler or a sibling archive is missing; gates that need
gymnasium call require_gymnasium() the same way. Exit codes follow
the harness convention: 0 pass, 77 skip, anything else fail.

Compiled fixtures and helper binaries are cached under
/tmp/k26rl_shim_gates and rebuilt when their inputs are newer.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = ROOT / "python"
KFLC = ROOT / "kflc" / "bin" / "kflc"
WORK = Path("/tmp/k26rl_shim_gates")

# Make the in-tree package importable without an install step.
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

_INCLUDE_DIRS = [
    "libk26astro_rt/include", "libk26astro_body/include",
    "libk26astro_core/include", "libk26astro_grav/include",
    "libk26astro_conics/include", "libk26astro_ephem/include",
    "libk26astro_vehicle/include", "libk26astro_att/include",
    "libk26astro_atmos/include",
    "libk26tick/include", "libk26compute/include",
    "libk26m3d/include", "libk26rl/include",
    "libk26rng/include",
    "libk26sense/include",
    # A vehicle assembly derives its mass properties from its
    # components' shapes, so a fixture that declares actuators on an
    # assembly reaches the collision library's headers.
    "libk26astro_coll/include",
]

_LINK_LIBS = [
    "libk26rl/libk26rl.a",
    "libk26rng/libk26rng.a",
    "libk26sense/libk26sense.a",
    "libk26astro_rt/libk26astro_rt.a",
    "libk26astro_att/libk26astro_att.a",
    "libk26astro_coll/libk26astro_coll.a",
    "libk26astro_vehicle/libk26astro_vehicle.a",
    "libk26astro_atmos/libk26astro_atmos.a",
    "libk26astro_grav/libk26astro_grav.a",
    "libk26astro_conics/libk26astro_conics.a",
    "libk26astro_body/libk26astro_body.a",
    "libk26astro_ephem/libk26astro_ephem.a",
    "libk26astro_core/libk26astro_core.a",
    "libk26compute/libk26compute.a",
    "libk26tick/libk26tick.a",
    "libk26m3d/libk26m3d.a",
]


def skip(reason):
    print("SKIP: " + reason)
    sys.exit(77)


def require_stack(gate_name):
    if not KFLC.exists():
        skip("%s: %s not built" % (gate_name, KFLC))
    for rel in _LINK_LIBS:
        if not (ROOT / rel).exists():
            skip("%s: %s not built" % (gate_name, rel))


def require_gymnasium(gate_name):
    try:
        import gymnasium  # noqa: F401  (presence probe only)
    except ImportError:
        skip("%s: the gymnasium package is not importable with %s"
             % (gate_name, sys.executable))


def require_pettingzoo(gate_name):
    try:
        import pettingzoo  # noqa: F401  (presence probe only)
    except ImportError:
        skip("%s: the pettingzoo package is not importable with %s"
             % (gate_name, sys.executable))


def run(cmd, **kwargs):
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        print("command failed (rc=%d): %s" % (proc.returncode, cmd))
        sys.exit(1)
    return proc


def _mtime(path):
    return path.stat().st_mtime


# The frozen ABI header, an input of every stub build and, through
# the generated C, of every compiled fixture.
_ABI_HEADER = ROOT / "libk26rl" / "include" / "k26rl_env.h"


def write_work_file(name, text):
    """Write a side file a fixture names (an assembly, say) into WORK,
    only when its content differs, so an unchanged file keeps its
    timestamp and the compile caches below stay usable."""
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / name
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return path


def compile_fixture(name, source_text=None, aux_files=None):
    """Compile a .kfl program into WORK and return the path of its
    companion shared object. With source_text None the source is the
    checked-in integration fixture of that name; otherwise the text
    is written into WORK first. aux_files maps the name of a side
    file the source refers to onto its content; each is written into
    WORK and counted as an input. Cached on the timestamps of every
    input: the source, its side files, the compiler, the linked
    archives, and the frozen ABI header."""
    WORK.mkdir(parents=True, exist_ok=True)
    if source_text is None:
        src = ROOT / "kflc" / "integration_tests" / (name + ".kfl")
    else:
        src = WORK / (name + ".kfl")
        stale = not src.exists() or src.read_text() != source_text
        if stale:
            src.write_text(source_text)
    aux = [write_work_file(n, t)
           for n, t in sorted((aux_files or {}).items())]
    out = WORK / name
    so = WORK / (name + ".rlenv.so")
    inputs = [src, KFLC, _ABI_HEADER] + aux + \
        [ROOT / l for l in _LINK_LIBS]
    if so.exists() and _mtime(so) >= max(_mtime(p) for p in inputs):
        return so

    cflags = ("-O2 -g -std=c++11 -Wno-format-truncation "
              "-ffp-contract=off -fexcess-precision=standard "
              + " ".join("-I%s" % (ROOT / d) for d in _INCLUDE_DIRS))
    ldlibs = " ".join(str(ROOT / l) for l in _LINK_LIBS) + \
        " -lgfortran -lm"
    env = dict(os.environ, KFLC_CFLAGS=cflags, KFLC_LDLIBS=ldlibs)
    log = WORK / (name + ".kflc.log")
    with open(log, "w") as lf:
        proc = subprocess.run(
            [str(KFLC), str(src), "-o", str(out)],
            env=env, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not so.exists():
        print(log.read_text())
        print("kflc failed (rc=%d) for %s" % (proc.returncode, src))
        sys.exit(1)
    return so


def refuse_fixture(name, source_text):
    """Run the compiler's check pass over a program that must not
    compile, and return its diagnostic. Exits non-zero when the
    program compiles, because an arm that cannot tell a refusal from
    an acceptance measures nothing."""
    WORK.mkdir(parents=True, exist_ok=True)
    src = WORK / (name + ".kfl")
    src.write_text(source_text)
    log = WORK / (name + ".refuse.log")
    with open(log, "w") as lf:
        proc = subprocess.run([str(KFLC), "--check", str(src)],
                              stdout=lf, stderr=subprocess.STDOUT)
    text = log.read_text()
    if proc.returncode == 0:
        print(text)
        print("kflc accepted %s, which this arm requires it to refuse"
              % src)
        sys.exit(1)
    return text


def build_c(source_name, out_name, extra_args=(), extra_inputs=()):
    """Compile one helper C source from this directory into WORK,
    cached on the timestamps of the source, the included ABI header,
    and any extra_inputs named, and on the argument set. Most helpers
    link no stack archive (the C driver dlopens its artifact), so the
    archives are inputs only where a helper names one."""
    WORK.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent / source_name
    out = WORK / out_name
    stamp = WORK / (out_name + ".args")
    args_text = " ".join(extra_args)
    inputs = [src, _ABI_HEADER] + [Path(p) for p in extra_inputs]
    fresh = (out.exists()
             and _mtime(out) >= max(_mtime(p) for p in inputs)
             and stamp.exists() and stamp.read_text() == args_text)
    if not fresh:
        cmd = ["cc", "-O2", "-Wall",
               "-I%s" % (ROOT / "libk26rl" / "include"),
               str(src), "-o", str(out)] + list(extra_args)
        run(cmd)
        stamp.write_text(args_text)
    return out


def build_stub(variant, defines=()):
    """Build one stub-artifact variant as a shared object."""
    return build_c(
        "stub_artifact.c", "stub_%s.so" % variant,
        ["-shared", "-fPIC"] + ["-D%s" % d for d in defines])


def build_cdriver():
    """Build the C reference driver for the cross-driver and
    marshalling gates."""
    return build_c("rl_shim_cdriver.c", "rl_shim_cdriver",
                   ["-ldl", "-lm"])


def build_tap_drain():
    """Build the telemetry-ring reader the tap gate drains with. It
    links the record library, whose reader it uses rather than
    reimplementing the ring's protocol in the gate that is meant to
    check it."""
    archive = ROOT / "libk26rl" / "libk26rl.a"
    return build_c("rl_tap_drain.c", "rl_tap_drain",
                   [str(archive), "-lrt", "-lm"], [archive])


# ---- shared fixture programs ----------------------------------------

# A world whose objective divides by (1.0 - push): finite at the
# default action, a division by zero when a caller drives push to
# exactly 1.0, so the fault lands in the environment whose action
# slice carries 1.0. Horizon 6 truncates environments that never
# fault.
FAULT_KFL = """\
form RL_SHIMFAULT
fn world fault_world
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7350.0
    episode
        control_dt 0.1
        horizon 6
    end
    action push box 0.0 4.0 default 0.0
    observe craft from earth mode=geometric as trk
    objective
        reward 1.0 / (1.0 - push)
    end
end
end
"""

# Every episode terminates at transition 3 with terminal adjustment
# 5.0 on top of the step reward 1.0, and no reset draws, so every
# episode's initial observation is the same and the boundary step is
# checkable bitwise against episode 0's initial observation.
TERM_KFL = """\
form RL_SHIMTERM
fn world term_world
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7350.0
    episode
        control_dt 0.1
        horizon 100
        terminated when episode.steps > 2
    end
    action push box -1.0 1.0 default 0.0
    observe craft from earth mode=geometric as trk
    objective
        reward 1.0
        terminal 5.0
    end
end
end
"""


# ---- the scripted action stream for the cross-driver gates ----------
#
# Pure functions of (step, environment) computed identically here and
# in rl_shim_cdriver.c, literal for literal, so both drivers feed
# bit-identical doubles. The box values leave the declared [-1, 1]
# bounds and the gear indices leave the declared arity 3: the package
# passes action values through unchecked, and a helpful clip anywhere
# breaks the file comparison.
#
# Both channels reach the fixture's world, so these values move the
# trajectory rather than sitting in the action buffer unread. That is
# what makes the comparisons that consume this stream able to fail on
# a value delivered to the wrong offset; a stream feeding a channel
# nothing consumes would compare two runs of the same inert number.

def act_thrust(t, e):
    return ((t * 7 + e * 3) % 13) / 13.0 * 4.0 - 2.0


def act_gear(t, e):
    return (t + e) % 5


def check(cond, what):
    if not cond:
        print("FAIL: " + what)
        sys.exit(1)


def ok(gate_name):
    print("%s: OK" % gate_name)
    sys.exit(0)
