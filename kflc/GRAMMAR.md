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
  expressions and simulation statements.

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

Five constructs carry the surface, all of them statements inside a
`fn world` body:

| Construct                     | Purpose                                                    |
|-------------------------------|-------------------------------------------------------------|
| `episode` ... `end`           | The episode frame: control period, length bound, termination, per-episode state draws. |
| `action <name> ...`           | A scalar input channel, set from outside before each step.  |
| `on_step` ... `end`           | A statement block run once per control step.                |
| `observe ... as <name>`       | A named observation channel, recomputed each step.          |
| `objective` ... `end`         | The reward, and an optional terminal adjustment.            |

These words bind as keywords only at statement position inside a
`fn world` body. Everywhere else they stay ordinary identifiers, so
existing programs that use them as names keep compiling. (`episode`,
`action`, and `objective` sit on the compiler's reserved-name list, so a
`let`, `const`, or `arg` that binds one of them draws a warning.)

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
world. The `default` value, 0 when omitted, is what the batch executable
drives at every step; the declared bounds and kinds are published in the
compiled artifact's spec.

### The `on_step` block

The `on_step` body runs once per control step, before the world advances,
identically in batch mode and through the environment library. The
declared action names are in scope as read-only scalars. The body admits
ordinary statements only: world construction (`astro_body`), stepping
(`step`, `propagate`), `observe`, nested reinforcement learning
constructs, and `print` are all rejected; the stepping path performs no
I/O. A `print` inside a `fn` the body calls is rejected too, for the
same reason.

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

The assigned expression must be side-effect free: it may call the scalar
maths built-ins and the string helpers that only read their arguments
(`strlen`, `streq`, `starts_with`, `ends_with`), while `concat`, which
allocates, and every library built-in registered by a manifest are
rejected, whether called directly or through a `fn`. Outside `on_step` the dotted form is not a name at
all, and an objective or termination expression that names body state is
told to read it through an `observe ... as` channel instead.

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

Channel names must be unique within the world and at most 53 bytes
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
step's reward. A truncated episode carries no terminal adjustment.

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

Two worked control problems sit beside it,
`integration_tests/orbit_transfer.kfl` and
`integration_tests/stationkeeping.kfl`: each drives the craft's velocity
from its action in `on_step`, reads the range and range-rate channels
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
