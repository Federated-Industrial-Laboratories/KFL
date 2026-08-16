# KFL — the KFL programming language

KFL is a small, block-structured language for writing **artifact-producing
batch programs**: numerical computations, plots, and physical simulations
that run to completion, print their results, and write image files. It is
not a UI language — a KFL program has no window, no widgets, and no event
loop.

A source file (extension `.kfl`) is compiled by `kflc`:

```
kflc program.kfl -o program        # emit + compile to a native binary
./program                          # run it
```

`kflc` translates the `.kfl` into standalone C++, then invokes the system
C++ toolchain to produce a native executable. The generated program runs
any simulations, renders any plots, calls the program entry function, and
exits. Link libraries are supplied through the `KFLC_LDLIBS` environment
variable (see *Compilation and the generated program*).

KFL is hand-authored, block-structured, and indentation-tolerant. It is
not JSON, not YAML, not INI.

---

## Lexical structure

- **Comments** — `#` to end of line. No block-comment form.
- **Strings** — double-quoted, with C-style escapes (`\n`, `\t`, `\"`,
  `\\`). An embedded NUL is forbidden. UTF-8 inside a string is preserved
  verbatim.
- **Identifiers** — `[a-zA-Z_][a-zA-Z0-9_]*`, case-sensitive.
- **Integers** — optional leading `-`, then decimal digits. No hex,
  octal, or binary.
- **Floats** — integer `.` integer, e.g. `0.5`, `3.141592653589793`.
  Scientific notation (`1.32712440018e20`) is accepted for the
  named-argument values of simulation statements.

Whitespace separates tokens; newlines terminate statements. Indentation
is cosmetic and ignored by the parser. The lexer also recognises a few
legacy token forms (`WxH` sizes, `#rrggbb` colours, `Ctrl+Q`-style
shortcuts) retained for backward compatibility; the current language has
no construct that consumes them.

---

## Program structure

Every file is exactly one `form` block:

```
form <NAME>
    <form-level construct>...
end
```

`<NAME>` is an identifier that names the program. There is **no window
header** — the language carries no title, size, or configuration and
produces no graphical surface. (`title`, `size`, and `cfg` lines are still
accepted so that older files parse, but they are ignored.)

The constructs allowed directly inside a `form` are:

| Construct                    | Purpose                                                            |
|------------------------------|--------------------------------------------------------------------|
| `fn <type> <name>(<args>)`   | A typed function. `fn void run()` is the program entry point.       |
| `fn data <name>`             | A plot-data producer (builds series for a `plot`).                 |
| `fn world <name>`            | A simulation callback run against a fresh world.                  |
| `plot <name> data <fn>`      | A figure declaration; renders to `<name>.png` and `<name>.svg`.    |
| `arena <name> capacity ...`  | A form-lifetime bump allocator (see *Memory model*).              |
| `arg <name> [default <v>]`   | A command-line-settable global.                                   |
| `frame <name> ...`           | A named reference frame for the simulation surface.               |
| `epoch <name> "<ISO>" <sc>`  | A compile-time epoch constant.                                    |
| `tick <handler> interval_ms N` | A periodic simulation callback.                                 |

Order is free. A program need only contain what it uses: a pure-compute
program is one or more `fn`s and a `fn void run()`; a plotting program adds
`fn data` producers and `plot` declarations; a simulation adds a
`fn world`.

---

## Functions

```
fn <return-type> <name>(<arg>, <arg>, ...)
    <statement>...
end
```

Return and argument types are `void`, `double`, `int`, `bool`, `string`,
`vector`, `matrix`, or a registered **opaque** type (see *Simulation
surface*). Each argument is `<type> <name>`, optionally prefixed with an
ownership qualifier (`own` / `borrow` / `ptr`; see *Memory model*).

```
fn double period_years(double a_au)
    return sqrt(a_au * a_au * a_au)
end
```

### The `run` entry point

A parameterless `fn void run()` is the conventional entry point. After the
program has executed any `fn world` simulations and rendered any `plot`
figures, the generated `main()` calls `run()` — it is where a compute
program does its work and prints results.

```
fn void run()
    print "Earth period T = ", period_years(1.0), " yr"
end
```

### Statements

| Statement                         | Form                                                     |
|-----------------------------------|----------------------------------------------------------|
| Binding                           | `let <name>: <type> = <expr>`                             |
| Constant binding                  | `const <name>: <type> = <expr>`                          |
| Assignment                        | `<name> = <expr>`                                         |
| Indexed assignment                | `<name>[<expr>] = <expr>`                                 |
| Conditional                       | `if <expr>` … [`else`] … `end`                            |
| Loop                              | `while <expr>` … `end`                                    |
| Return                            | `return [<expr>]`                                         |
| Print                             | `print <arg>, <arg>, ...`                                 |
| Expression statement              | `<expr>`                                                  |

The type annotation on `let` / `const` is written after a colon; when
omitted the binding defaults to `double`. Blocks (`if`, `else`, `while`)
are closed with `end`.

```
let n: int = 240
let mu: double = 4 * 3.141592653589793 * 3.141592653589793
let i: int = 0
while i < n
    x[i] = cos(i)
    i = i + 1
end
```

### Expressions

The expression sub-language covers the usual arithmetic and comparison
operators over `int` and `double`, calls to user `fn`s, and a whitelist of
built-ins:

- **Scalar maths** — `sqrt`, `sin`, `cos`, `tan`, `atan2`, `exp`, `log`,
  `log10`, `pow`, `abs`, `fabs`, `floor`, `ceil`, `min`, `max`, `fmod`.
- **String helpers** — `strlen`, `streq`, `starts_with`, `ends_with`,
  `concat`.
- **Vector / matrix helpers** — statistics and linear algebra such as
  `mean`, `std`, `dot`, `norm`, `mat_mul`, `mat_vec`, `transpose`,
  `solve`. Vector/matrix arguments must be in-scope bindings referenced by
  name, not anonymous temporaries.

### Vectors

A `vector` binding is initialised with a constructor — `zeros(N)`,
`ones(N)`, `linspace(a, b, N)`, `arange(a, b, step)`, or the literal form
`[ ... ]` — then read and written by index (indexing is bounds-checked at
run time):

```
let xs: vector = linspace(0.3, 35, 200)
let ys: vector = zeros(200)
let i: int = 0
while i < 200
    ys[i] = sqrt(1.0 / xs[i])
    i = i + 1
end
```

### Matrices

A `matrix` binding is a row-major 2-D array of doubles. It is initialised
either with the runtime constructor `zeros(rows, cols)` or with a nested
literal `[[ ... ], [ ... ]]`, and its cells are read and written with two
indices (both bounds-checked at run time):

```
let grid: matrix = zeros(300, 240)
grid[i][j] = value
```

Matrices back the `series_heatmap` plotting statement (see *Plotting*).

---

## The `print` statement

`print` writes one line to standard output. Its arguments are
comma-separated; each argument is either a **string literal** (printed
verbatim) or a **numeric expression** (evaluated and printed as a number):

```
print "planet      a[AU]   T[yr]"
print "Mercury  ", 0.3871, "   ", period_years(0.3871)
```

Commas that occur inside a string literal, or inside parentheses or
brackets of a call such as `f(g(a, b))`, do **not** split arguments — only
top-level commas separate them. Each `print` statement emits exactly one
line, terminated by a newline.

---

## Plotting

A plotting program has two parts: `fn data` producers that fill series,
and top-level `plot` declarations that render them.

A `fn data <name>` producer builds series and hands each to a `series_*`
statement (a full producer appears in the plotting example below). Most
kinds take a pair of x/y vectors — `series_<kind> "<label>" <xs> <ys>`,
where `<xs>` and `<ys>` name in-scope `vector` bindings — and cover
`series_line`, `series_scatter`, `series_errorbar`, `series_histogram`,
and `series_box`. The `series_heatmap "<label>" <matrix>` statement
instead takes a single `matrix` binding and renders it as a 2-D color
field over the index-space extent `[0, cols] x [0, rows]`, with the color
scale auto-fitted to the data range.

A top-level `plot` binds a producer to a figure and its labels:

```
plot p_speed data speed_curve
    title    "Circular orbital speed vs distance"
    x_label  "r (AU)"
    y_label  "v (km/s)"
```

Each `plot <name>` renders to `<name>.png` and `<name>.svg` in the working
directory via `libk26plot`. A plotting program must therefore link the
plotting libraries, for example:

```
KFLC_LDLIBS="-lk26plot -lk26compute -lk26m3d -lcairo -lm" \
    kflc orbital_mechanics.kfl -o orbital
```

---

## Memory model

KFL bindings and function arguments carry optional ownership qualifiers,
enforced by the compiler:

| Qualifier | Meaning                                                                    |
|-----------|----------------------------------------------------------------------------|
| `own`     | Single ownership. Assigning one `own` binding to another requires `move()`. |
| `borrow`  | Read-only reference; emitted as `const`. It may not outlive its source.     |
| `ptr`     | Raw passthrough; user-managed lifetime, no compile-time lifetime check.     |

`move(x)` makes an ownership transfer explicit and invalidates the source;
reading a moved-from binding within the same block is a compile error, and
returning a `borrow` of a function-local binding is a compile error.

```
fn double demo_move(own body src)
    let dst: own body = move(src)   # ownership transfers; src invalidated
    return 1.0
end

fn double demo_borrow(borrow body src)
    let view: borrow body = src     # read-only view for the fn lifetime
    return 2.0
end
```

### Arenas

A form-level arena is a bump allocator that lives for the program's
lifetime:

```
arena world_arena  capacity 64 MB
arena scratch      capacity 16 KB
arena small_buffer capacity 4096 reset_mode manual
```

`capacity` is a byte count with an optional `KB` / `MB` / `GB` suffix. A
function opts into an arena with an `allocator` binding in its prologue;
allocations inside the function then draw from that arena:

```
fn double demo()
    allocator = scratch
    return 1.0
end
```

The default reset mode, `fn_exit`, resets the arena automatically on every
return path from a function that binds it. `reset_mode manual` leaves the
caller responsible for resetting the arena when its data becomes stale.

---

## Simulation surface

The compiler ships no physics of its own. Numerical libraries publish
opaque types and function bindings through `.kflbi` manifest files, which
`kflc` loads at start-up from its builtins directory (default
`/usr/share/kflc/builtins/`, overridable with `KFLC_BUILTINS_PATH`). Each
manifest registers:

- **Opaque types** — nominal handle types such as `world`, `body`, and
  `epoch`. The compiler tracks them by name and rejects cross-type
  confusion. Once registered, an opaque name may be used in any type
  position (`let e: body = ...`, `fn world`, function arguments).
- **Builtins** — named functions with a fixed arity, callable from KFL
  expressions and simulation statements. A builtin line may end with the
  qualifier `pure`, the library's declaration that the function's result
  depends on its arguments alone and that it touches no world state, no
  input or output and no allocation. A line without it registers an impure
  builtin, which is what the positions below refuse.

With the astronomy libraries installed, a program can drive a physical
simulation.

### `fn world` and simulation statements

A `fn world <name>` is a callback executed against a freshly created world.
Its body admits the ordinary statements plus a declarative simulation
surface:

- `astro_body <name> <key>=<value> ...` — declare a body from named
  parameters (`gm`, `mass`, `radius`, `parent`, …).
- `step <dt-expr>` — advance the whole world by `dt` seconds.
- `propagate <body> for <dt-expr>` — advance a single body.
- `for_each <name> in <world>` … `end` — read-only iteration over bodies.
- `observe <target> from <observer> [key=value ...]` — compute an
  observation and print it to standard output.

```
form APOPHIS2029

    frame ICRF inertial
    epoch J2025 "2025-01-01T00:00:00" UTC

    fn world apophis_demo
        astro_body sun   gm=1.32712440018e20 mass=1.989e30 radius=6.957e8
        astro_body earth gm=3.986004418e14   mass=5.972e24 radius=6.371e6 parent=sun
        astro_body apophis mass=2.7e10 gm=1.8e0 radius=185.0 parent=sun

        step 3600.0
        observe apophis from earth mode=apparent
    end

end
```

*(This is the shape of `examples/astro_apophis_2029.kfl`.)*

### Frames, epochs, ticks

- `frame <name> inertial` or `frame <name> body_fixed body <ident>`
  declares a named reference frame used by the runtime's coordinate
  transforms.
- `epoch <name> "<ISO-8601>" <scale>` binds a compile-time epoch constant.
  `<scale>` is one of `TAI`, `UTC`, `UT1`, `TT`, `TDB`. The ISO-8601 string
  is parsed at compile time, never at run time.
- `tick <handler> interval_ms <N>` registers a `fn` as a periodic callback,
  invoked every `N` milliseconds while a simulation runs.

A simulation program links the astronomy libraries via `KFLC_LDLIBS`, for
example:

```
KFLC_LDLIBS="-lk26astro_rt -lk26astro_grav -lk26astro_conics \
    -lk26astro_body -lk26astro_ephem -lk26astro_core \
    -lk26compute -lk26tick -lk26m3d -lgfortran -lm" \
    kflc astro_apophis_2029.kfl -o apophis
```

A `fn world` can also declare a stepped decision environment instead of a
scripted simulation; see *Reinforcement learning (Grammar 3.2)*.

---

## Reinforcement learning (Grammar 3.2)

Grammar 3.2 lets a `fn world` declare a stepped decision environment: the
program states what a control step is, which scalar inputs an external
caller sets each step, what that caller observes, and what the reward is.
The compiler then produces an artifact an outside training loop can drive
step by step (see *The compiled artifact* below).

Nine constructs carry the surface, all of them statements inside a
`fn world` body:

| Construct                     | Purpose                                                    |
|-------------------------------|-------------------------------------------------------------|
| `episode` ... `end`           | The episode frame: control period, length bound, termination, per-episode state draws. |
| `action <name> ...`           | A scalar input channel, set from outside before each step.  |
| `on_step` ... `end`           | A statement block run once per control step.                |
| `observe ... as <name>`       | A named observation channel, recomputed each step.          |
| `objective` ... `end`         | The reward, and an optional terminal adjustment.            |
| `agent <name>` ... `end`      | A scope owning actions, observation channels, and an objective. |
| `sensor <name>` ... `end`     | An imperfection model bound to observation channels with `through`. |
| `astro_payload <name> ...`    | A defense payload carried by one craft: a detection sensor, an information state, or an effector. |
| `engage <payload> at <target>`| Fire an effector at a body. Inside `on_step` only. |

These words bind as keywords only at statement position inside a
`fn world` body. Everywhere else they stay ordinary identifiers, so
existing programs that use them as names keep compiling. (`episode`,
`action`, and `objective` sit on the compiler's reserved-name list, so a
`let`, `const`, or `arg` that binds one of them draws a warning.)

`agent`, `sensor`, `astro_payload`, `engage`, and `on_step` are not
reserved, and inside a `fn world` body each opens its construct only
when what follows it is what that construct's form requires: an
identifier for `agent`, `sensor`, `astro_payload`, and `engage`, the end
of the line for `on_step`. Written any other way they stay ordinary
identifiers, so a program that binds one of them as a name keeps
compiling:

```
fn world w
    let sensor: double = 3.0
    sensor = sensor + 1.0              # the binding, not a block
    sensor rf                          # a sensor block
        noise normal 0.0 1.0
    end
end
```

No statement form in the language has the shape
`<identifier> <identifier>`, so the two readings never overlap for
`agent`, `sensor`, `astro_payload`, or `engage`. For `on_step` they
overlap on one shape, a bare `on_step` alone on a line, which the block
form takes. `at` and `effect` are read as connectives inside `engage`
and `observe effect` alone, and are ordinary identifiers everywhere
else, including as body names.

A program that uses any of these constructs, or a distribution-valued
`astro_body` attribute (below), is a reinforcement learning program. Such
a program must declare an `episode` block, admits exactly one `fn world`,
and gives up the statements the episode machinery replaces:

- `step` and `propagate` are rejected inside the world: the stepping
  belongs to the episode machinery.
- `plot` declarations and `fn data` producers are rejected: the compiled
  environment has no figure output.
- `arena` declarations are rejected.
- `astro_body` declarations must be top level in the world body, not
  inside an `if`, `while`, or `for_each` block: the body set is part of
  the compiled program's identity.

The world body's remaining statements are its prefix. The prefix runs
once per environment when the environment is created and builds the
initial world.

### The `episode` block

```
episode
    control_dt <expr>                  # required
    horizon <expr>
    terminated when <expr>
    reset <body>.<key> <distribution>
end
```

At most one `episode` block per world, and each line except `reset` at
most once. `control_dt` is required: the simulated seconds the world
advances on every control step. `horizon` bounds the episode at that many
steps; an episode that reaches it ends as truncated. `terminated when`
gives a termination predicate, evaluated after each step in the scope
described under *Expression scope*; an episode that satisfies it ends as
terminated. Each `reset` line redraws one body-state scalar at the start
of every episode: `<body>` names an `astro_body` declared in the same
world, `<key>` is one of the six state keys (below), and the value must
be a distribution expression.

The `horizon` is published in the compiled artifact's spec, so it must
be decidable when the program is compiled: a constant expression of
literals and arithmetic over them, evaluating to a whole number of
steps from 0 to 4294967295. A value that is negative, fractional, out
of range, or not compile-time constant is an error; in particular a
`horizon` read from an `arg` is rejected. `control_dt` stays an
ordinary expression, validated at run time (finite and positive) when
the environment is created.

With neither a positive `horizon` nor a `terminated when` condition the
episode can never end. The compiler warns, and the batch executable
refuses to run such a program; the environment library still serves it,
for consumers that reset externally.

### Actions

```
action <name> box <low> <high> [default <expr>]
action <name> discrete <count> [default <expr>]
```

Each `action` declares one scalar input channel that the outside caller
sets before every step. `box` declares a continuous channel with the
given bounds; `discrete` declares an integer-valued channel with
`<count>` choices. The bounds, count, and default are
whitespace-separated expressions (balanced parentheses and brackets keep
a spaced expression together). Action names must be unique within the
world, or, in a world that declares `agent` blocks, within each block.
The `default` value, 0 when omitted, is what the batch executable
drives at every step; the declared bounds and kinds are published in the
compiled artifact's spec.

### The `on_step` block

The `on_step` body runs once per control step, before the world advances,
identically in batch mode and through the environment library. The
declared action names are in scope as read-only scalars, every agent's
included (see *The `agent` block*). The body admits
ordinary statements only: world construction (`astro_body`), stepping
(`step`, `propagate`), `observe`, nested reinforcement learning
constructs, and `print` are all rejected; the stepping path performs no
I/O. A `print` inside a `fn` the body calls is rejected too, for the
same reason.

One construct runs the other way. `engage` is an act of a step, so this
block is the one place it is admissible and the world prefix is where it
is refused (see *Effectors*).

**Body state.** Inside `on_step`, and nowhere else, a body's state is
read and written by dotted name:

```
<body>.<key>
```

`<body>` names an `astro_body` declared in the same world and `<key>` is
one of the six state keys (below). The name reads as an ordinary double
and is assigned with the ordinary assignment statement, so an action
reaches the dynamics:

```
on_step
    craft.vel_x = craft.vel_x + thrust
end
```

A write lands on the state the following step integrates, since the
block runs before the world advances. Statements run in order, so a
later write to a key overrides an earlier one and a read sees the writes
before it. The block does not run on the boundary step that starts an
episode, so an episode's first observation is its reset state alone. A
position key means metres from the world origin, the same meaning it
carries as an `astro_body` attribute and as a `reset` target; the
velocity keys are metres per second.

**What the block may reach.** The block runs once per control step of
every environment, so nothing it reaches may allocate, perform input or
output, or change the world. The rule covers everything the block reaches,
not the assigned expression alone, and it follows calls into `fn` bodies
to any depth: a statement is judged where it runs, not where it is
written.

Every expression the block evaluates must be side-effect free, in every
position it can occupy: an assignment, a `let` or `const` initialiser, an
expression statement, an `if` or `while` condition, an index expression, a
returned expression, an argument to another call, and anything nested
inside a block. A side-effect-free expression may call the scalar maths
built-ins, the string helpers that only read their arguments (`strlen`,
`streq`, `starts_with`, `ends_with`), and any library built-in whose
manifest declares it `pure`. `concat` is rejected, not because it
allocates but because its result is a pointer into a per-callsite buffer
that the next call at that site overwrites, so a second evaluation
invalidates the first. A built-in whose manifest declares nothing is
rejected for the same practical reason: nothing has established that it
can be evaluated twice.

These statements are rejected wherever the block reaches them, including
inside a `fn` it calls: `astro_body`, which adds a body to the world;
`step` and `propagate`, which advance it; `observe`, which runs its
observer pipeline; and `print`, because the stepping path performs no
input or output.

A `vector` or `matrix` binding is rejected in the block, and so is a call
to a `fn` that returns one, because the storage behind those values is
heap allocated. A call the compiler can classify as neither a built-in nor
a `fn` in the same form is rejected too, since some of the forms it lowers
itself allocate.

The refusal names the position, the line, the thing it found, and the
chain of functions it went through to reach it. Because an initialiser is
a position like any other, a name bound to a rejected call is rejected
where it is bound, and no later read of that name can carry the call past
the check. The compiler follows a bounded number of nested calls; a chain
deeper than that is rejected rather than assumed clean.

Outside `on_step` the dotted form is not a name at all, and an objective or
termination expression that names body state is told to read it through an
`observe ... as` channel instead.

### Vehicle assemblies

```
astro_body chaser assembly="crew_vehicle.k26asm" pos_x=... vel_y=...
```

An assembly is a text description of a vehicle: components with
placements and masses, collision primitives, docking ports, thrusters
and momentum devices. The compiler reads it, derives the vehicle's
mass, centre of mass and inertia tensor from the geometry, and writes
those numbers into the artifact as constants, so a running simulation
never opens an asset file. The path resolves against the directory of
the source file that names it, and the asset's bytes are hashed into
the compiled program's identity, so a changed asset is a changed
program.

Every property an assembly declares carries a `provenance` line
saying where its number came from: `cited` read from a named source,
`computed` derived here from cited inputs, `modelled` an arrangement
chosen and disclosed rather than found, or `unverified` a working
figure whose source has not been checked. The compiler reports the
modelled and the unverified ones by name. Those lines are inside the
bytes the digest covers, so a changed citation is a changed program.

A body that binds one gains mass properties, colliders, attitude, and
whatever actuators and ports the assembly declares. Declaring `mass=`
or `gm=` beside `assembly=` is refused naming both lines, because the
derivation is the one source of that number. `examples/assets/` holds
worked assets; every key the format takes is documented in the
compiler's own assembly reader.

A docking port is declared in the assembly and named by the port
observe form below:

```
    port forward
        at 3.5 0.0 0.0
        axis 1.0 0.0 0.0
        roll_ref 0.0 1.0 0.0
        capture idss_e
    end
```

`at` is the centre of the mating plane in the body frame, `axis` its
outward normal, and `roll_ref` a direction in the plane from which
roll misalignment is measured; the axis must be a unit vector and the
roll reference must not be parallel to it. `capture` names the
envelope a contact at this port is judged against, and the compiler
builds the port's mating plane collider from the diameter that
envelope publishes, so the interface geometry is the envelope's and
not the author's. One envelope is defined, `idss_e`; any other name is
refused. A port declaring no envelope is geometry the program can
describe and carries no collider and no test.

The mating plane is a square plate circumscribing the published
circle, because the collider set has no round primitive. The square
contains the circle, so it reports every contact a disc would report
and some a disc would miss: an approach passing between the circle's
edge and the square's corner, which for the defined envelope is
between 0.60 m and 0.85 m off the axis, the second figure rounded up
from 0.8485 m, meets the plate where a disc would let it by. That
band is far outside any capture, whose lateral misalignment limit is
0.10 m, so it changes which approaches count as an impact and never
which count as a capture.

### Observation channels

```
observe <target> from <observer> [key=value ...] as <name>
```

The `observe` statement of the simulation surface takes a trailing
`as <name>` clause, which must be the last clause on the line. Instead of
printing, the observation becomes a named channel recomputed after every
step. A line-of-sight observe contributes five components, readable by
name in the objective and termination expressions:

| Component                                       | Value                                                        |
|-------------------------------------------------|--------------------------------------------------------------|
| `<name>_dir_x`, `<name>_dir_y`, `<name>_dir_z`  | Unit direction from observer to target, after the observation mode's corrections. |
| `<name>_range`                                  | Distance from the observer to the corrected target position, in metres. |
| `<name>_range_rate`                             | Rate of change of the geometric range, in metres per second, positive when the pair separates. |

The range rate is the dot product of the relative position and the
relative velocity over the separation, taken from the two bodies' state
as it stands. It is the rate of the geometric range rather than the
derivative of the corrected range beside it, because an observation mode
corrects a position and has no corrected velocity to differentiate;
under `mode=geometric` the two are the same thing. At exactly zero
separation it reads 0.0.

#### Relative state

```
observe relative <target> from <chief> as <name>
```

Publishes the target's position and velocity in the chief's own
local-vertical local-horizontal frame: the first axis points from the
body the chief orbits out to the chief, the third along the orbital
angular momentum, and the second completes the right-handed set along
the direction of motion. Six components:

| Component                                      | Value                                                       |
|------------------------------------------------|-------------------------------------------------------------|
| `<name>_r_x`, `<name>_r_y`, `<name>_r_z`       | Position of the target relative to the chief, in metres, on the three axes above. |
| `<name>_v_x`, `<name>_v_y`, `<name>_v_z`       | Velocity of the target as seen from the chief's rotating frame, in metres per second. |

The velocity is the rate an observer riding the frame sees, so two
craft holding station on one orbit read zero however their inertial
velocities differ. This is what separates an approach along the
direction of motion from one from below, which the line-of-sight
channels above cannot tell apart.

The chief must declare a `parent=`, since the frame is built from its
state relative to the body it orbits; a chief without one is refused.
The target's own `parent=`, if it declares one, has no part in the
frame. Both bodies must be declared in the world, they may not be the
same body, and this form requires its `as` clause: unlike the
line-of-sight form it has no printing spelling.

Two states have no frame and cannot be refused when the program is
compiled, because they are configurations rather than declarations: a
chief sitting exactly at the centre of the body it orbits, and a chief
moving straight towards or away from it. Neither names a direction of
motion. All six channels read 0.0 there, which a reader cannot tell
from a genuine zero relative state; a program that can reach either
configuration should test for it through the line-of-sight range to the
parent rather than through these channels.

#### Attitude

```
observe attitude of <body> as <name>
```

Publishes a body's own orientation and angular velocity. There is no
observer and no correction of any kind. Seven components:

| Component                                                            | Value                                                    |
|----------------------------------------------------------------------|----------------------------------------------------------|
| `<name>_quat_w`, `<name>_quat_x`, `<name>_quat_y`, `<name>_quat_z`   | Orientation quaternion, scalar part first.               |
| `<name>_omega_x`, `<name>_omega_y`, `<name>_omega_z`                 | Angular velocity in the body frame, radians per second.  |

The body must be declared in the world. Attitude is advanced only for a
body that binds an `assembly=`, since that is what gives it an inertia
tensor; a body without one carries no attitude state to publish.

#### Contact

```
observe contact of <body> as <name>
```

Publishes what the transition just taken did, rather than where a body
is. Three components:

| Component            | Value                                                                             |
|----------------------|-----------------------------------------------------------------------------------|
| `<name>_hit`         | 1.0 when the transition contained a contact for this body, 0.0 otherwise.         |
| `<name>_fraction`    | Where in the control period the first contact fell, from 0.0 to 1.0.              |
| `<name>_speed`       | Closing speed along the contact normal at that moment, metres per second.         |

The latch is cleared at the start of every transition and reports the
first contact within it. The body must declare an `assembly=`: its
colliders are the primitives that assembly declares, and a body without
one carries none and can report no contact, so it is refused rather
than given three channels that could never be anything but zero.

A contact ends an episode only if the program says so, through an
ordinary `terminated when` predicate over these channels. It is not a
fault.

**What a contact does to the craft** is declared once per episode:

```
episode
    contact arrest
    contact bounce restitution <expr> friction <expr>
end
```

At most one such line. Absent, the environment arrests: the pair is
placed at the configuration the sweep computed for the contact, their
relative velocity is removed by a momentum-conserving merge, and the
rest of the control period advances with the two moving together.

`bounce` applies one impulse at the contact point instead, with a
restitution coefficient in the closed interval from 0 to 1 and a
non-negative Coulomb friction coefficient. Both are compile-time
scalars and neither is optional: a restitution nobody declared would
be a number the compiler invented. The impulse uses the effective
mass at the contact point, so an off-centre hit spins the craft by
the amount its geometry gives.

Neither runs on a contact that captures at a docking port; see below.

#### Docking ports

```
observe port <port> of <body> as <name>
```

Publishes how far a named docking port on a body is from mated with
the port it faces, and whether the contact it just made was a capture.
Nine components:

| Component            | Value                                                                              |
|----------------------|------------------------------------------------------------------------------------|
| `<name>_captured`    | 1.0 when the transition's contact met every condition of the port's capture envelope, 0.0 otherwise. |
| `<name>_axial`       | Separation of the two mating planes along the other port's axis, in metres, positive while apart. |
| `<name>_lateral`     | Distance from this port's centre to the other port's axis, in metres.              |
| `<name>_pitchyaw`    | Vector sum of the pitch and yaw misalignment, in radians.                          |
| `<name>_roll`        | Roll misalignment, in radians.                                                     |
| `<name>_v_axial`     | Closing rate along the other port's axis, in metres per second, positive while closing. |
| `<name>_v_lateral`   | Lateral rate at the ports, in metres per second.                                   |
| `<name>_v_pitchyaw`  | Vector sum of the pitch and yaw rate, in radians per second.                        |
| `<name>_v_roll`      | Roll rate, in radians per second.                                                  |

Angles are the yaw, pitch and roll of this port's frame with respect
to the mated configuration, taken in that order about the other port's
third, second and first axes. Units are SI throughout, so a program
comparing against an envelope published in degrees converts once.

The eight residuals are recomputed every step from the state as it
stands, which is what an approach is flown on. On a step whose
transition ended in a contact between the two ports they instead carry
the values the capture test itself was given, taken at the instant of
contact: by the end of that transition the contact has been resolved
and the closing rate the test read is gone, so publishing the later
state would hide the test's own inputs.

The body must declare an `assembly=` carrying a port of that name with
a `capture` envelope, and exactly one port carrying an envelope must
be declared on some other body: that is the port this one is measured
against, and a world with none or with several is refused rather than
paired by declaration order.

A port must stand proud of the hull it is mounted on. Its mating
plane is an ordinary collider, so a hull that reaches past the plane
is what the other craft meets first: the contact reported is the
hull's, the capture channel stays clear, and the approach reads as an
impact. A hull that reaches exactly the same plane is worse than
either, because both pairs then touch at the same instant and which
one the sweep reports is decided by arithmetic at the last bit rather
than by anything an author can read off the geometry.

Capture is not a resolution the program declares. It is a consequence
of the geometry: when a contact between two ports meets every
condition, the capture channel is set for that step, and a program
ends the episode on it through an ordinary `terminated when`
predicate.

A capture takes precedence over whichever resolution the episode
declared. Neither the arrest nor the bounce runs on a captured
contact: the pair becomes one body instead, and stays one for the
rest of the episode. Mass is summed, and the two inertia tensors are
summed about the joint centre of mass, so the mated pair turns as the
pair and not as either craft. Both craft keep their own thrusters and
their own wheels, and either one accelerates the pair, which is what
a task that continues past docking needs. A program that ends its
episode on the capture channel never sees any of this; one that does
not, does.

The condition is judged from both ports and a capture needs both to
accept. Either port can be read as the arriving one and the two
readings differ slightly, so requiring both is the conservative
reading and the one that leaves the two ports' capture channels
agreeing about a fact of the pair.

#### Sensors, and the truth beside the measurement

```
sensor rangefinder
    noise normal 0.0 0.05
    bias_walk 0.02 600.0 0.001
    latency 2
    quantise 0.01
    dropout 0.005
end
```

A `sensor` block declares a chain of imperfection models. One term per
line, and the order is load-bearing: the chain applies its terms in the
order written, so a quantiser after a noise term quantises the noisy
value and one before it does not.

| Term | Operands | What it does |
|---|---|---|
| `noise normal` | mean (must be zero), standard deviation | Adds a normal draw of that standard deviation. One draw per step. |
| `scale` | relative standard deviation | Multiplies by `1 + s * n`, so a true value of zero stays zero. One draw per step. |
| `bias_walk` | turn-on standard deviation, correlation time in seconds, in-run standard deviation | A bias drawn once per episode, evolving as a first-order Gauss-Markov process. One draw per step, one per episode. |
| `latency` | whole control periods | Delays by exactly that many steps. No draws. |
| `quantise` | step, and optionally both ends of a range | Rounds to a multiple of the step, ties away from zero, after clamping to the range. No draws. |

A step and a range together decide how many grid positions the
quantiser needs, and a signed 64-bit integer holds about 9.2e18 of
them. A declared range that needs more is refused, naming both
operands; without a declared range the implied one is the narrower of a
working ceiling and what the step itself can reach, so a step of 1e-14
implies about 9.2e4 and not 1e15. A value beyond the grid saturates on
it with its sign intact rather than converting, which is what makes the
model total.
| `dropout` | probability from 0 up to but not including 1 | Holds the previously delivered value instead of the current one. One draw per step. |

The third operand of `bias_walk` is the process's steady-state
standard deviation, not the size of the per-step kick: the driving term
is scaled so the steady state is what was declared whatever the control
period is. The first operand is the turn-on bias, drawn once per
episode, and is usually the larger of the two.

One `bias_walk` and one `latency` per sensor, because one carries one
state of each. A sensor that declares no terms at all is refused, since
it would leave every channel it touches unchanged.

```
observe target from chaser mode=geometric through rangefinder with truth as look
```

`through <sensor>` routes each numeric component of the observe through
that chain; each component gets its own draws, so the components of one
observe are not corrupted identically. `with truth` additionally
publishes the uncorrupted components as a paired set, named with
`_truth` before the component: `look_truth_range` beside `look_range`.
Both clauses come before `as`, which stays the last clause on the line.
`with truth` without a `through` is refused, since the two halves would
be the same numbers.

Every channel carries a spec tag saying whether it is a measured or a
ground-truth channel and which channel it is paired with, so a consumer
can build a policy's observation space from the measured channels while
a privileged critic reads everything, without parsing names. A program
that declares no sensor publishes every channel as measured and
unpaired.

The first observation of an episode is the uncorrupted value. That
follows from how draws are addressed rather than being a separate
choice: a per-step model's draw index is the transition index within
the episode, and at an episode boundary no transition has been taken.
What the boundary does do is take the per-episode draws, fill any delay
with the true value, and prime the dropout hold.

Channel names must be unique within the world, or, in a world that
declares `agent` blocks, within each block, and at most 53 bytes
long. The compiled artifact's spec carries each derived component name
in a 96-byte entry, which holds a 53-byte name plus the longest suffix
any form derives: `_truth_range_rate`, seventeen bytes, being the
longest component suffix with the `_truth` a paired channel inserts.
The bound stays at 53 whatever the entry grows to, so a program that
compiles today keeps compiling. Every name readable in the objective
and termination expressions lives in one scope, so a derived component
may not collide with an action, a top-level world binding, or a form
argument, and an action may not collide with any of those either; the
compiler rejects the program rather than letting one silently shadow
another.

#### Defense payloads

```
astro_payload <name> body=<body> kind=<kind> [key=value ...]
```

A world-prefix statement declaring one payload carried by one craft.
`body=` names an `astro_body` of the same world that binds an
`assembly=`, since a payload attaches to a vehicle and a body without an
assembly carries none; a body without one is refused naming both
statements. Payloads are top level, not inside an `if`, `while`, or
`for_each`, because the payload set is part of the compiled program's
identity.

Each payload is constructed once per environment when the world is
built, and destroyed with the environment. Nothing is constructed while
stepping: the per-step path calls the libraries' evaluators, which
return a value and allocate nothing.

One call on that path is not an evaluator. The information state's
history push allocates a target's ring the first time it sees that
target, and the per-step path calls it once per sub-advance. It costs
nothing there only because the binding has already pushed every
observer-target pair once when the world was built and once at every
reset, so no push the step makes is ever a first one. That is the same
act as the seeding described under *Information state* below, and it
is what the allocation figures in this compiler's own gates measure.

The admissible keys are a function of `kind=`. A key that belongs to
another kind is refused naming that kind rather than reported as
unknown; a missing required key is refused naming it. Every key is a
scalar expression evaluated when the world is built and may take the
`uniform` and `normal` distribution forms, exactly as an `astro_body`
attribute may, except where a key is fixed when the payload is
constructed and a per-episode draw could not reach it.

| `kind=` | Required keys |
|---|---|
| `detect_ir` | `aperture_m`, `integration_s`, `passband_lo_um`, `passband_hi_um`, `throughput`, `snr_threshold`, `target_temp_k`, `target_emissivity`; optional `t_optics_k` and `optics_emissivity`, both 0 by default |
| `detect_radar` | `p_tx_w`, `g_tx_db`, `g_rx_db`, `freq_hz`, `loss_sys_db`, `bandwidth_hz`, `t_sys_k`, `noise_figure`, `snr_threshold` |
| `detect_lidar` | `pulse_energy_j`, `wavelength_nm`, `aperture_rx_m`, `atmospheric_tx`, `detector_efficiency`, `snr_threshold`, `target_albedo` |
| `infostate` | `history`, optional, 1024 by default; fixed when the payload is constructed, so it takes no distribution, and refused below 2, which is the fewest samples an interpolation needs |
| `impactor` | See *Effectors* below |
| `laser` | See *Effectors* below |

Every key naming the instrument is a parameter of the library function
that consumes it, spelled the same way: the constructor's for the six
`detect_ir` instrument keys, the nine `detect_radar` keys and the six
`detect_lidar` instrument keys, and the evaluator's for `t_optics_k`
and `optics_emissivity`.

The three named `target_` describe what is being looked at rather than
the instrument, which is why no constructor carries them.
`target_albedo` is the lidar evaluator's parameter of that name;
`target_temp_k` and `target_emissivity` are this grammar's names for
the target temperature and emissivity the infrared path needs, which
the library takes as `emitter_T_K` and as the emissivity argument of
its in-band radiated-power routine. They are declared on the payload
because they are the reference-target properties a detection threshold
is specified against. **One detection payload therefore models one
target class**, and a program observing two dissimilar targets declares
one payload per class. Radar needs none of them, its cross-section
being geometric.

`t_optics_k` and `optics_emissivity` describe the observing telescope's
own thermal emission, which for a warm instrument looking in its own
emission band sets the noise floor. Both default to 0, which is the
library's documented cosmic-background-only behaviour; a program
modelling a real instrument declares them.

**A body carries any number of detection payloads and at most one
information state.** A detection sensor binds into the vehicle's
payload list; an information state binds into its singleton slot, so a
second would evict the first and leave the evicted one's channels
reporting nothing for the rest of the run. A second is refused naming
both statements.

#### Detection

```
observe detect <payload> of <target> as <name>
```

`<payload>` names an `astro_payload` of a detection kind and `<target>`
an `astro_body` of the same world that binds an `assembly=` **declaring
at least one `collider`**, since the signature is computed from the
collision primitives the assembly declares and a body with none would
present no area at any aspect. A body with no assembly, and an assembly
with no collider, are each refused naming the body and what is missing.
A payload does not observe the craft that carries it. Seven components:

| Component | Value |
|---|---|
| `<name>_detected` | 1.0 when the computed signal-to-noise ratio meets the payload's declared threshold, 0.0 otherwise. |
| `<name>_snr` | The computed signal-to-noise ratio, dimensionless. |
| `<name>_range` | Observer-to-target distance, in metres. |
| `<name>_dir_x`, `<name>_dir_y`, `<name>_dir_z` | Unit direction from the observer to the target, in the world frame. |
| `<name>_aspect` | Cosine of the angle between the line of sight and the target's first body axis, in [-1, 1]. |

Both the hard decision and the continuous quantity are published, so a
program may shape a reward over one and terminate on the other without
deriving either from the other.

These channels are geometric and carry no light-time correction; the
corrected view is the track form's below. `_aspect` is published because
the signature models are aspect dependent: the target's silhouette is
the projected area of its assembly's collision primitives along the line
of sight, taken in the target's own frame, so a craft that turns changes
what its observer sees, and without the aspect channel an agent sees
detections come and go with nothing that explains them. The channel is
therefore taken against the same frame the silhouette is: the first body
axis, which the assembly format runs along the craft. For a craft flying
nose forward that is the angle between the line of sight and the
direction of flight. Overlapping primitives are summed rather than
unioned, which overstates the area of a craft whose colliders
interpenetrate.

The infrared model takes the in-band radiated power of that area at the
declared temperature and emissivity; the radar model takes the target as
a flat plate of that area facing the observer, which is the
geometric-optics form; the lidar model takes that area as the projected
area.

Three lidar quantities the statement does not declare are pinned here
rather than left to be discovered:

- the **transmit gain** is the diffraction-limited figure for the
  declared `aperture_rx_m` at the declared wavelength, so one aperture
  stands for both ends of the path;
- the **view-angle cosine** is 1.0, because the area already carries
  the projection and applying it twice would square it;
- the **beam-quality factor** is 1.0, the diffraction-limited value, so
  the returned photon count takes no degradation the statement has no
  way to declare.

**These channels carry no imperfection of their own.** Every model here
computes a noise-free quantity, and the libraries' own optional noise
generators are not used. A program that wants a noisy detector binds a
declared `sensor` to the channel exactly as it would to any other
observation, and may ask for the uncorrupted values beside the
corrupted ones:

```
observe detect eye of target through rangefinder with truth as ir
observe track picture of target through rangefinder as trk
```

That route puts the draws on this capability's own generator at the
coordinates a replay reproduces, which the libraries' own generators
would not.

#### Information state

```
observe track <payload> of <target> [modality=<modality>] as <name>
```

`<payload>` names an `astro_payload` of kind `infostate`; `<target>` is
an `astro_body` binding an `assembly=`. `modality=` takes `none`, `ir`,
`radar`, `lidar`, or `ephem` and defaults to `none`; it is recorded into
the observation unchanged, the library applying no per-modality
processing. Nine components:

| Component | Value |
|---|---|
| `<name>_valid` | 1.0 when an observation was produced, 0.0 otherwise. |
| `<name>_pos_x`, `<name>_pos_y`, `<name>_pos_z` | Target position at the retarded time, world frame, in metres. |
| `<name>_vel_x`, `<name>_vel_y`, `<name>_vel_z` | Target velocity at the retarded time, in metres per second. |
| `<name>_range` | Observer-to-target distance at that solution, in metres. |
| `<name>_age` | The observer's clock time less the retarded time, in seconds. |

When `_valid` reads 0.0 the other eight read 0.0. The solver's iteration
count is not published: it is a convergence diagnostic rather than a
state of the world.

The target's true state is pushed into the payload's history once per
sub-advance, so the history is finer than the light-time lag rather than
coarser. At the start of each episode, before any stepping, one sample
is pushed at the episode epoch. **The first observation of an episode is
therefore unavailable**: the only sample is the epoch itself and the
retarded time is strictly earlier, so the observer would be receiving
light emitted before the episode began, and no such state is invented to
supply it. The same holds on any later step whose elapsed time is
shorter than the light time to the target, which at long ranges is more
than one step.

At most 64 targets may be tracked against one information state, which
is the library's own per-observer limit; a program that names more is
refused naming the payload, the count, and the limit. And at most one
information state is carried by one body, for the reason given with the
statement above.

#### Effectors

```
engage <payload> at <target>
observe effect <payload> as <name>
```

`engage` fires an effector payload at a body. It is an act rather than
a declaration, so it is admissible **inside an `on_step` block and
nowhere else**, and it is refused elsewhere naming the block it belongs
in. `<payload>` names an `astro_payload` of an effector kind, and
`<target>` an `astro_body` of the same world that binds an `assembly=`
declaring at least one `collider`, since the effector acts on the area
the target presents and a body with none would present none. An
effector does not engage the craft that carries it.

`on_step` runs before the world advances, so an engagement lands on the
state the immediately following advance integrates, and the
observation, reward and termination the step reports are computed after
that advance. That is the same relation a body state write has.

**A payload is engaged at most once per step.** Two statements naming
one payload are refused where they are written, including when the
second is inside a conditional. One statement reached twice, which
today means a statement inside a loop, faults the environment instead:
the published result would otherwise depend on which call ran last,
with no channel saying so, and both engagements would already have
reached the world. A `fn` the step body calls cannot hold the
statement at all, since `engage` is a statement of this block and an
ordinary identifier everywhere else.

`observe effect` publishes what the last engagement of that payload
did. Two components are common to every effector kind: `<name>_engaged`
is 1.0 on a step where the payload was engaged and 0.0 otherwise, and
`<name>_effect` is the scalar magnitude that kind reports as its
principal result. On a step with no engagement `_engaged` reads 0.0 and
every other component reads 0.0. The rest of the set is the kind's own,
because an ablation event and an impact event share no fields.

| `kind=` | Required keys |
|---|---|
| `impactor` | `pattern`, one of `single` or `swarm`; `projectile_mass_kg`, `projectile_density_kg_per_m3`, `projectile_diameter_m`; `swarm_count` and `swarm_half_angle_rad`, required for `swarm` and refused for `single`; optional target-structure keys, below |
| `laser` | `primary_diam_m`, `wavelength_nm`, `p_output_w`, `m_squared`, `pointing_jitter_rad`, `rms_wavefront_m`, `plasma_attn_k`, `target_material`, `target_reflectivity` |

`pattern` and `target_material` take a word rather than a number,
because each names a library constant and a program that wrote the
number would depend on an internal numbering nothing promises it.
`target_material` takes `aluminum`, `steel`, `titanium`, `copper`,
`composite` or `fused_silica`, spelled as the library's own table
spells them. The library also offers a seventh value whose documented
meaning is a conservative generic-metal anchor rather than an absence
of material; this grammar does not admit it, because a program writing
it would reasonably expect no ablation and would get steel's numbers.
Neither keyword key takes a distribution form.

As with the detection kinds, the `target_` keys describe what is being
fired at rather than the payload, so **one effector payload models one
target class**. Every other key is the library constructor's own
parameter of the same name.

##### The kinetic impactor

Twelve components:

| Component | Value |
|---|---|
| `<name>_engaged` | 1.0 on a step where the payload was engaged. |
| `<name>_effect` | The velocity increment imparted to the target on a hit, in metres per second; 0.0 on a miss. |
| `<name>_hit` | 1.0 when the predicted intercept lands on the target. |
| `<name>_closing_speed` | The magnitude of the relative velocity, in metres per second. |
| `<name>_t_close` | Predicted time to closest approach, in seconds; negative when the target is already receding. |
| `<name>_miss` | Predicted closest-approach distance, in metres. |
| `<name>_fraction` | The fraction of the released projectile mass that lands on the target: 1.0 for `single`, and the target's silhouette over the cone's footprint for `swarm`. |
| `<name>_cos_angle` | Cosine of the impact angle between the closing direction and the target's first body axis, in [0, 1]; 0.0 when that axis points away from the projectile, since a surface is not struck from behind. |
| `<name>_penetrates` | 1.0 when the projectile diameter exceeds the Whipple critical diameter. |
| `<name>_critical_diameter` | That critical diameter, in metres. |
| `<name>_penetration` | Monolithic penetration depth, in metres. |
| `<name>_energy` | Energy delivered to the target's interior, in joules. |

Seven of the twelve are published whenever the payload is engaged,
because they describe the intercept the engagement set up: `_engaged`,
`_hit`, `_closing_speed`, `_t_close`, `_miss`, `_fraction` and
`_cos_angle`. The other five describe an impact and read 0.0 on a
miss: `_effect`, `_penetrates`, `_critical_diameter`, `_penetration`
and `_energy`.

Because the impact cosine is clamped at zero, a target presenting its
back to the projectile reports zero for it, and the two components the
Whipple analysis derives from it go to zero with it. That is the
library's own convention and is stated here because a reader who does
not know it would read a zero critical diameter as a defect.

**The hit test.** The projectile is released carrying its launcher's
own state and flies ballistically, so the engagement is resolved from
the relative state of the two craft rather than by adding a body to the
world. The intercept lands when the predicted closest approach falls
within the target's effective silhouette radius, which is the radius of
a disc of the projected area the target presents along the closing
direction, **and** the time to closest approach is positive. A negative
time to closest approach is a target already past its nearest point,
which is a miss however small the predicted separation.

**The effect.** Momentum transfer, and nothing beyond it. The
projectile arrives carrying its mass times the closing speed in the
target's frame, and the target takes that momentum along the closing
direction, scaled by the fraction that landed. The momentum
enhancement factor is therefore exactly 1. The published deflection
literature reports it above 1 for a cratering impact into a rubble
body, because the ejecta thrown back off the surface carries momentum
of its own; this model does not carry the ejecta mass and speed that
figure comes from, and does not claim it.

**The target-structure keys are optional**, and each is the parameter
of the same name that the library's impact analysis or its
delivered-energy routine takes:

| Key | What it feeds |
|---|---|
| `target_wall_thickness_m`, `target_bumper_density_kg_per_m3`, `target_bumper_spacing_m`, `target_wall_yield_stress_ksi` | The Whipple analysis, which runs when the wall thickness is positive and is skipped otherwise. |
| `target_brinell_hardness`, `target_density_kg_per_m3`, `target_speed_of_sound_m_per_s` | The monolithic-plate analysis, which runs when all three are positive and is skipped otherwise. |
| `target_inner_thickness_m` | The inner structural wall behind a Whipple stand-off, which lowers the coupled energy from full penetration to partial. |
| `target_monolithic_thickness_m` | The thickness the monolithic penetration depth is judged against. |

A skipped branch leaves its components at 0.0, which is the library's
own documented behaviour and not a failure. `_energy` is the exception
and is published whatever is declared: with no target geometry at all
the library returns its documented worst case, the full-penetration
fraction of the impact energy.

##### The directed-energy laser

Ten components:

| Component | Value |
|---|---|
| `<name>_engaged` | 1.0 on a step where the payload was engaged. |
| `<name>_effect` | The impulse delivered to the target, in newton seconds. |
| `<name>_dv` | The velocity increment that impulse imparted, in metres per second. |
| `<name>_mass_loss` | The target mass ablated and removed, in kilograms. |
| `<name>_range` | Emitter-to-target distance, in metres. |
| `<name>_spot` | Spot diameter at the target, in metres. |
| `<name>_encircled` | Fraction of the beam's energy falling inside the target's projected area. |
| `<name>_fluence` | Fluence at the target, in joules per square metre. |
| `<name>_transmissivity` | Plasma-plug transmissivity, in [0, 1]. |
| `<name>_ignited` | 1.0 when the fluence reached the material's plasma-ignition threshold. |

**The effect.** The ablation plume leaves along the beam, so the recoil
pushes the target away from the emitter: the impulse acts along the
unit vector from emitter to target, and the velocity increment is taken
against the mass the target had when the light arrived. The ablated
mass is then removed from the target. `_mass_loss` reports the mass
actually removed: a step that would ablate the whole body is outside
this model's range and removes nothing, since a massless body in the
integrator is not a state this layer will produce.

The engagement's **dwell is the step's own control period**, and its
**range** is the distance between the payload's body and the target,
both derived rather than declared. The **area** is the target's
silhouette along the line of sight, which is the same projected area
the detection channels take, in the target's own frame: a craft that
turns changes both what a detector sees and what a beam lands on, and
there is one description of that geometry rather than two.

Three quantities the library's ablation event carries are deliberately
not published. The threshold fluence is a constant of the declared
material; the effective dwell is the control period; and the on-target
intensity is the published fluence divided by that period. None of the
three can move within an episode, and a component nothing can move is a
component nothing can be shaped against.

##### Both kinds

Each payload is constructed once per environment when the world is
built, exactly as a detection payload is, and an engagement calls the
library's evaluator only: both return a value struct and allocate
nothing.

**These components carry no imperfection of their own**, on the same
rule as the detection channels: the models compute a noise-free
quantity, the libraries' own optional generators are not used, and the
swarm's direction sampler, which takes one, is not called. A program
that wants a noisy effector channel binds a declared `sensor` to it,
which puts the draws on this capability's own generator at the
coordinates a replay reproduces.

### The `objective` block

```
objective
    reward <expr>                      # required
    terminal <expr>
end
```

At most one `objective` block per world; `reward` is required and
`terminal` optional, each at most once. The reward is evaluated once per
step, after the world advances; a step whose reward is not finite ends
the episode as faulted. `terminal` is a terminal adjustment: when the
`terminated when` condition fires, its value is added to that final
step's reward. A truncated episode carries no terminal adjustment. A
world that declares `agent` blocks puts one `objective` in each block
instead; see *The `agent` block*.

### The `agent` block

```
agent <name>
    action <name> box <low> <high> [default <expr>]
    action <name> discrete <count> [default <expr>]
    observe <target> from <observer> [key=value ...] as <name>
    objective
        reward <expr>
        [terminal <expr>]
    end
end
```

An `agent` block is a scope, not a new declaration language: the three
constructs inside it are the `action`, `observe ... as`, and `objective`
above, unchanged in syntax and meaning. What the block changes is
ownership. Blocks are statements of the `fn world` body and must be top
level, not inside an `if`, `while`, or `for_each`; the agent index is
source order, starting at 0; the block's name is the agent's published
name.

**A world with no `agent` block is agent count 1.** Its declarations
belong to an implicit agent 0, its slices are the whole vectors, and its
published spec is the same bytes it was before this construct existed.

**World level and block level do not mix.** Once any `agent` block is
present, an `action`, an `objective`, or an `as`-bound `observe` at
world level is an error naming both sites. The alternative readings are
that a world-level channel belongs to every agent or to agent 0, and
both would be inventions. An `observe` without `as` is unaffected: it
prints, declares no channel, and stays at world level.

A block holds those three constructs and nothing else. It may declare at
most one `objective`; without one it publishes an all-zero reward stream,
which is the rule a program without an `objective` already follows. A
block with no `action` declares an observation-only agent, which is legal
for the same reason an action total of 0 is. A block with neither an
action nor an objective is an error: it publishes no action channel and
no reward and has no effect on the program.

**Above one agent, every block declares at least one `observe ... as`.**
An agent's name reaches a consumer only through the qualifier on its own
observation channel names, so a block that declares none publishes a
name that does not survive compilation: it is absent from the spec, from
the artifact, and therefore from the recorded episode file, and a
per-agent interface has nothing to key it by. At one agent the names
publish unqualified and nothing depends on the agent's own name, so the
rule applies above one agent only.

Agent names are unique within a world, are at most 31 bytes, and may not
equal an `astro_body` name in the same world or the name `episode`, both
of which are already read by dotted name in a program's expression scope.

**Channel names.** With one agent, channel names publish exactly as they
are declared. With more than one, two agents may reasonably both declare
a channel called `rel`, so names publish qualified, as `<agent>.<channel>`,
in the spec's channel-name entries. That entry holds 96 bytes: a
qualified name is the agent name, a dot, the declared channel name, and
the derived component suffix, and a combination that does not fit is an
error naming each part rather than only the total.

**Termination is per environment.** The episode ends for every agent at
once, on `terminated when`, on the horizon, or on a fault. There is no
per-agent termination.

**Batch mode** is unchanged in kind: every action channel of every agent
holds its `default` for the whole run, so a multi-agent program is batch
runnable and its episode file carries one reward stream per agent.

An example of the whole surface is
`integration_tests/rl_multi_agent.kfl`.

### Expression scope

`terminated when`, `reward`, and `terminal` are evaluated by the episode
machinery, outside the world prefix, and read a fixed scope:

- the declared action names;
- the four components of every `as`-bound observation channel;
- `episode.steps`, the count of control steps taken in the current
  episode, the step being evaluated included;
- the world body's top-level scalar `let` / `const` bindings (`double`,
  `int`, `bool`), captured once per environment when the world is built;
  a binding declared inside a nested block, or one of vector, matrix, or
  string type, is not readable here, and the compiler says so;
- `arg` globals, and the literals `true` / `false`.

Any other name is an error.

**With more than one agent** the first two entries need a rule about
whose channels are meant, and it is different in each of the three
positions a name can be written:

- Inside `on_step`, every agent's action channels are in scope. An
  unqualified name resolves when exactly one block declares it; when two
  blocks declare it, the unqualified use is an error naming both
  declaration sites and the qualified form, and `<agent>.<action>`
  resolves. Observation channels are not readable in `on_step` at all,
  qualified or not.
- Inside an `agent` block's `objective`, that agent's own action and
  observation channels resolve unqualified, and any agent's channels
  resolve qualified, including another agent's. That last part is
  deliberate: it lets a zero-sum reward be written once as the negation
  of the other agent's expression rather than twice. It does not widen
  what an agent observes; an agent's observation slice is exactly its own
  declarations, and a reward is computed by the environment rather than
  observed by an agent.
- Inside `episode`, `terminated when` reads any agent's channels and
  names the agent it reads, since there is one `episode` per program and
  it sits in no block.

### Distribution expressions

```
uniform(<low>, <high>)
normal(<mean>, <stddev>)
```

A distribution expression is valid in exactly two positions: as the whole
value of an `astro_body` attribute, and as the value of an episode
`reset` line. Anywhere else it is an error, and an `astro_body` value
that buries a distribution inside a larger expression is an error too: a
drawn parameter is the whole value or nothing.

Distributions are not random calls at run time. Every drawn parameter is
a deterministic function of the run's seed and the draw's coordinates
(the environment index, the episode index, and the parameter's position
in the source), so the same seed reproduces the same values, environment
by environment and episode by episode. A distribution-valued `astro_body`
attribute is redrawn at the start of every episode, exactly as `reset`
lines are; when both target the same field, the `reset` line wins.

A program that declares its own `fn uniform` or `fn normal` turns that
name back into an ordinary function call everywhere except `reset` lines,
which accept only distribution expressions. A Grammar 3.1 program with
its own `uniform` therefore keeps its meaning, distribution-free.

### Body state keys

Six scalar keys name an `astro_body`'s state components directly:
`pos_x`, `pos_y`, `pos_z` (position) and `vel_x`, `vel_y`, `vel_z`
(velocity). They are accepted in two places: as `astro_body` attribute
initialisers, with a constant value or a distribution, and as the targets
of episode `reset` lines. Constant-valued state keys are plain Grammar
3.1; using them does not make a program a reinforcement learning program.

### The compiled artifact

`kflc program.kfl -o program` on a reinforcement learning program
produces two artifacts from one emitted translation unit:

- `program`, the batch executable;
- `program.rlenv.so`, a shared object exporting the k26rl environment
  ABI.

Both are compiled from the same generated source, so batch runs and
library consumers step the same environment code by construction.

**The batch executable.** The generated `main()` is the episode loop; it
does not call `run()`. It steps `--envs` environments in lockstep,
driving every action at its declared default, until each environment has
completed `--episodes` episodes. An episode completes by termination,
truncation, or fault; the environment then resets automatically and keeps
stepping until every environment has reached the count.

| Flag              | Meaning                                                       |
|-------------------|---------------------------------------------------------------|
| `--envs <N>`      | Environments stepped side by side (default 1).                |
| `--episodes <K>`  | Episodes to complete per environment (default 1).             |
| `--seed <S>`      | The run's seed (default 0).                                   |
| `--out <path>`    | Record an episode file at `<path>`; the path must not exist.  |

Declared `arg` globals keep their `--<name>` flags (see *Command-line
arguments*).

**The environment library.** `program.rlenv.so` exports the k26rl
environment ABI and nothing else: the create, reset, step, observation,
reward, flags, fault, spec, and destroy calls plus the version and status
helpers declared in `k26rl_env.h` (shipped with `libk26rl`). A consumer
dlopens the shared object, checks `k26rl_abi_version()`, reads the
environment's geometry (channel counts and names, action bounds and
kinds, `control_dt`, `horizon`) from the `k26rl_env_spec` blob, and steps
every environment in the handle through flat arrays of doubles. The
header carries the full contract.

The same blob carries the agent count and, per agent in index order, the
offset and length of that agent's observation slice and action slice.
The slices are contiguous and their concatenation is the whole vector, so
a consumer splits one flat array rather than reading a second one. The
reward buffer is one double per agent per environment, environment major,
so agent `a` of environment `e` reads index `e * agent_count + a`.

**Determinism.** Given the same compiled artifact, the same seed, and the
same action stream, the observation, reward, and flag streams are
bit-identical across runs, and two identical batch runs write
byte-identical episode files. Each environment's episodes are a pure
function of the seed, the environment index, and the episode index; an
environment's streams do not change when neighbours run beside it in the
same handle. A recorded episode file re-simulates: creating a fresh
environment with the recorded seed, resetting to the recorded episode,
and replaying the recorded actions reproduces that episode's observation,
reward, and flag streams bitwise.

**Grammar 3.1 compatibility.** The surface is additive. A program that
uses none of the constructs in this section compiles exactly as before,
byte for byte, with no trace of the environment machinery.

### Worked example

An environment that randomises a craft's initial state each episode,
exposes a continuous and a discrete action, observes the craft from the
body it orbits, and rewards closing the range. This is the shape of
`integration_tests/rl_pointing.kfl`:

```
form RL_POINTING

    fn world pointing_world
        astro_body earth gm=3.986004418e14 mass=5.972e24
        astro_body craft gm=1.0 parent=earth pos_x=uniform(-7.0e6,7.0e6) pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=normal(0.0,10.0) vel_z=0.0

        episode
            control_dt 0.1
            horizon 1000
            terminated when episode.steps > 900
            reset craft.pos_x uniform(-7.0e6, 7.0e6)
            reset craft.vel_y normal(0.0, 10.0)
        end

        action thrust box -1.0 1.0 default 0.0
        action gear discrete 3 default 0

        on_step
            let damping: double = 0.99
            craft.vel_z = craft.vel_z * damping + gear
        end

        observe craft from earth mode=astrometric as track

        objective
            reward 0.0 - track_range + thrust
            terminal track_range < 0.5
        end
    end

end
```

A reinforcement learning program links the astronomy libraries plus
`libk26rl` and `libk26rng`:

```
KFLC_LDLIBS="-lk26rl -lk26rng -lk26astro_rt -lk26astro_vehicle \
    -lk26astro_atmos -lk26astro_grav -lk26astro_conics -lk26astro_body \
    -lk26astro_ephem -lk26astro_core -lk26compute -lk26tick -lk26m3d \
    -lgfortran -lm" \
    kflc rl_pointing.kfl -o pointing
./pointing --envs 8 --episodes 4 --seed 42 --out pointing.k26epi
```

The first command writes `pointing` and `pointing.rlenv.so`; the second
records eight environments for four episodes each into
`pointing.k26epi`.

Both of its action channels reach something a driver reads back, which
is worth copying rather than a detail of this fixture: `thrust` enters
the reward, and `gear` enters the out-of-plane velocity as a discrete
stage under first-order damping, so its effect settles at
`gear / (1 - damping)` instead of growing with the horizon. An action
channel no expression consumes is a channel a test cannot tell was
delivered.

Two worked control problems sit beside it,
`integration_tests/orbit_transfer.kfl` and
`integration_tests/stationkeeping.kfl`: each drives the craft's velocity
from a continuous action alone, reads the range and range-rate channels
for its endings, and draws its initial state per episode from the run's
seed.

---

## Command-line arguments

`arg <name> [default <value>] [mutable | readonly]` declares a global that
the program reads by name and the user sets on the command line. The type
is inferred from the default: a boolean default exposes `--<name>` /
`--no-<name>`, while a numeric or string default is set with
`--<name> <value>`. A `readonly` arg keeps its default and exposes no flag.

---

## Compilation and the generated program

`kflc program.kfl -o program` emits a self-contained C++ translation unit
and compiles it to a native binary. Useful modes:

| Mode          | Effect                                              |
|---------------|-----------------------------------------------------|
| `-o <file>`   | Emit, compile, and link a binary at `<file>`.       |
| `-c`          | Also keep the emitted `.kflc.cc` beside the binary. |
| `--emit`      | Write the generated C++ to stdout; do not compile.  |
| `--dump`      | Dump the parsed AST to stdout.                       |
| `--check`     | Parse and emit silently; exit non-zero on error.    |

The emitted program is an ordinary `main()`. It:

1. parses declared `arg`s from `argv`;
2. initialises any form-level arenas;
3. runs each `fn world` against a fresh world (in a portable,
   bit-reproducible mode), whose `observe` statements print to stdout;
4. renders each `plot` to `<name>.png` and `<name>.svg`;
5. calls `run()` if the program defines one;
6. returns 0.

Compilation is controlled by three environment variables: `CXX` (the C++
compiler, default `c++`), `KFLC_CFLAGS` (compile flags), and `KFLC_LDLIBS`
(link libraries, default `-lm`). A program that uses a numerical library
must extend `KFLC_LDLIBS` with the matching `-l` flags.

---

## Worked example — a compute program

A headless program that prints Keplerian quantities. It uses only scalar
maths, so the default `KFLC_LDLIBS=-lm` suffices. This is the shape of
`examples/compute_kepler.kfl`:

```
form COMPUTEKEPLER

    # Orbital period in years for a circular orbit of radius a (AU).
    fn double period_years(double a_au)
        return sqrt(a_au * a_au * a_au)
    end

    # Circular orbital speed (AU/yr) at radius a (AU).
    fn double circular_speed(double a_au)
        let mu: double = 4 * 3.141592653589793 * 3.141592653589793
        return sqrt(mu / a_au)
    end

    fn void run()
        print "planet      a[AU]   T[yr]     v[AU/yr]"
        print "Mercury  ", 0.3871, "   ", period_years(0.3871), "  ", circular_speed(0.3871)
        print "Earth    ", 1.0000, "   ", period_years(1.0000), "  ", circular_speed(1.0000)
        print "Mars     ", 1.5237, "   ", period_years(1.5237), "  ", circular_speed(1.5237)
    end

end
```

```
kflc compute_kepler.kfl -o kepler
./kepler
```

---

## Worked example — a plotting program

A program that samples a curve, marks a few points, and writes a figure.
It follows the shape of `examples/orbital_mechanics.kfl` (which combines
several such producers and plots with a `run()`):

```
form ORBITAL

    fn double circular_speed(double a_au)
        let mu: double = 4 * 3.141592653589793 * 3.141592653589793
        return sqrt(mu / a_au)
    end

    # Producer: a continuous curve plus discrete planet markers.
    fn data speed_curve
        let rs: vector = linspace(0.3, 35, 200)
        let vs: vector = zeros(200)
        let mu: double = 4 * 3.141592653589793 * 3.141592653589793
        let i: int = 0
        while i < 200
            vs[i] = sqrt(mu / rs[i])
            i = i + 1
        end
        series_line "v_circ" rs vs

        let pr: vector = zeros(3)
        let pv: vector = zeros(3)
        pr[0] = 0.3871
        pr[1] = 1.0000
        pr[2] = 1.5237
        let j: int = 0
        while j < 3
            pv[j] = sqrt(mu / pr[j])
            j = j + 1
        end
        series_scatter "planets" pr pv
    end

    plot p_speed data speed_curve
        title    "Circular orbital speed vs heliocentric distance"
        x_label  "r (AU)"
        y_label  "v (AU/yr)"

    fn void run()
        print "Earth circular speed = ", circular_speed(1.0), " AU/yr"
    end

end
```

```
KFLC_LDLIBS="-lk26plot -lk26compute -lk26m3d -lcairo -lm" \
    kflc orbital.kfl -o orbital
./orbital           # writes p_speed.png and p_speed.svg
```
