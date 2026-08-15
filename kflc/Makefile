CC      ?= cc
PREFIX  ?= /usr
DESTDIR ?=

CFLAGS  ?= -O2 -g
override CFLAGS += -std=c11 -Wall -Wextra -Wpedantic \
                   -Iinclude -Isrc -I../libk26rl/include -I../libk26sense/include -I../libk26rng/include
# The float-control flags every library in this tree carries, on the
# compiler itself. They are appended with `override` rather than left
# to the caller because they are not a tuning choice: the compiler now
# does floating-point arithmetic of its own, deriving a vehicle's mass
# properties from its assembly and writing the results into the
# emitted source as constants. A contracted multiply-add here would
# make those constants a function of which compiler built this one,
# and the asset digest that pins the inputs would be pinning bytes
# whose derived numbers still moved. tests/test_assembly.c asserts the
# outcome, comparing constants from two optimisation levels, rather
# than asserting the presence of a flag.
override CFLAGS += -ffp-contract=off -fexcess-precision=standard

RL_LIB   = ../libk26rl/libk26rl.a
# The compiler calls the imperfection layer's own precondition check on
# every chain it assembles, so it links that library as it links the
# episode one: the refusal is the library's, and a copy of it here
# could drift from it.
SENSE_LIB = ../libk26sense/libk26sense.a ../libk26rng/libk26rng.a

BIN      = bin/kflc
LIB      = libkflc.a
# Library objects: parser + lexer + dump + arena/diag util. Linked by
# both the kflc binary and any embedding tool (k26form) that needs to
# load .kfl files. Since the assembly reader lives here, a consumer of
# this archive also links $(RL_LIB) for the asset digest.
# Every consumer of $(LIB) links the mathematical library: the
# assembly reader normalises the directions an asset declares, so the
# archive carries a square root whoever links it.
LIB_OBJS = src/util.o src/lexer.o src/parser.o src/dump.o src/serialize.o \
           src/expr.o src/stmt.o src/check.o src/opaque.o \
           src/assembly.o src/capture.o src/body_state.o \
           src/builtin_register_astro.o \
           src/builtin_loader.o src/runtime.o
# Driver objects: emitters + CLI driver. Only the kflc binary needs these.
DRV_OBJS = src/emit.o src/emit_rl.o src/main.o

.PHONY: all install clean test
all: $(BIN) $(LIB)

# Round-trip property test. Builds tests/round_trip against libkflc.a
# and runs it over every example .kfl, asserting parse / serialize /
# parse yields a structurally identical AST (the §3.3 semantic-
# equivalence promise). Standalone shell wrapper at tests/run.sh.
TEST_BIN = tests/round_trip
TEST_EXAMPLES = $(sort $(wildcard examples/*.kfl))

$(TEST_BIN): tests/round_trip.c $(LIB) $(RL_LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(RL_LIB) $(LDFLAGS) -lm

MANIFEST_TEST = tests/test_manifest_loader
$(MANIFEST_TEST): tests/test_manifest_loader.c $(LIB) $(RL_LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(RL_LIB) $(LDFLAGS) -lm

# Emit-side test for parse-time body bindings. Compiles a fixture
# via `kflc --emit` and greps the output for `_kfl_body_<NAME>_idx`
# declarations + idx-deref uses, and the absence of runtime
# `find_body` calls for known names.
BODY_BINDING_TEST = tests/test_body_binding_emit
$(BODY_BINDING_TEST): tests/test_body_binding_emit.c $(LIB) $(RL_LIB) $(BIN)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(RL_LIB) $(LDFLAGS) -lm

# Emit-side test for the KFL memory model. Verifies form-level
# arena declarations + init factory + fn-body allocator binding +
# conditional kflc_runtime.h include + slim include surface for
# forms that do not use the memory keywords.
W8_ARENA_TEST = tests/test_w8_arena_emit
$(W8_ARENA_TEST): tests/test_w8_arena_emit.c $(LIB) $(RL_LIB) $(BIN)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(RL_LIB) $(LDFLAGS) -lm

# Gate for compile-time enforcement of own / borrow / ptr / arena
# qualifiers. Writes small .kfl fixtures to /tmp, runs the built
# kflc binary against them, asserts that positive cases emit the
# expected C++ shape and negative cases trigger the right diag error.
MEMORY_QUAL_TEST = tests/test_memory_qualifiers
$(MEMORY_QUAL_TEST): tests/test_memory_qualifiers.c $(LIB) $(RL_LIB) $(BIN)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(RL_LIB) $(LDFLAGS) -lm

# Gate for the reserved-keyword warning list: identifiers that will
# become keywords in a future grammar version warn today, so authors
# rename before the keyword lands and binds.
RESERVED_KW_TEST = tests/test_reserved_keywords
$(RESERVED_KW_TEST): tests/test_reserved_keywords.c $(LIB) $(RL_LIB) $(BIN)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(RL_LIB) $(LDFLAGS) -lm

# Vehicle assemblies: the format, the mass-property derivation, the
# identity digest, and the provenance rules. Links libkflc.a for the
# reader and libk26rl.a for the digest.
ASSEMBLY_TEST = tests/test_assembly
$(ASSEMBLY_TEST): tests/test_assembly.c $(LIB) $(RL_LIB) $(BIN)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(RL_LIB) -lm

# Gate for the Grammar 3.2 reinforcement learning front end: episode /
# action / on_step / objective / observe-as parse + check, with the
# negative diagnostics pinned. Runs the built kflc binary over /tmp
# fixtures (test_reserved_keywords pattern).
# $(BIN) is a prerequisite because the actuator cases below drive the
# emitter, so a compiler left behind by an edit would have this gate
# measuring the previous one.
# Every gate below drives the built compiler: what each of them
# measures is an emitted artifact, so a compiler left behind by an edit
# would have the gate measuring the previous one and reporting green
# for code that is no longer there. `make test` is safe on its own
# because its target names $(BIN), but a gate built or run alone -
# which is what a review or a bisect does - was not. tests/rl_gate_util.h
# is what pulls the compiler into the gates that include it, so the two
# are named together and the gates that reach it directly name $(BIN)
# beside their own sources.
# A drive gate links only the episode library; the sibling stack
# archives are not its prerequisites and must not be added as such. A
# gate compiles its artifact when it runs, by shelling out to the built
# kflc with KFLC_LDLIBS naming those archives, so the artifact always
# links them as they stand on disk at that moment and no build ordering
# here affects it. Prerequisites were tried and are inert: a library
# can be changed and rebuilt without the gate binary changing at all,
# and the next run still catches the change because it recompiles the
# artifact from scratch.
RL_GATE_DEPS = tests/rl_gate_util.h $(BIN)

# The docking benchmark: the capture envelope's figures as the
# compiler carries them, each of its conditions at its limit, the port
# form's nine channels against an independent computation, the task's
# four outcomes, and two processes byte for byte. Links the collision
# library for the envelope test, which is where the conditions live.
RL_DOCKING_TEST = tests/test_rl_docking
$(RL_DOCKING_TEST): tests/test_rl_docking.c $(RL_GATE_DEPS) \
                    ../libk26rl/include/k26rl_env.h \
                    ../libk26rl/libk26rl.a \
                    ../libk26astro_coll/libk26astro_coll.a
	$(CC) $(CFLAGS) -I../libk26rl/include -I../libk26astro_coll/include \
	    -I../libk26m3d/include -o $@ $< \
	    ../libk26rl/libk26rl.a ../libk26astro_coll/libk26astro_coll.a \
	    ../libk26m3d/libk26m3d.a -ldl -lm

RL_GRAMMAR_TEST = tests/test_rl_grammar
$(RL_GRAMMAR_TEST): tests/test_rl_grammar.c $(LIB) $(RL_LIB) $(BIN)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(RL_LIB) $(LDFLAGS) -lm

# What a stepping path may reach: the registry's purity property, one
# program per expression position inside the per-step block, and the
# statements the block may not reach through a call. Links libkflc.a
# for the registry half and drives the built kflc for the rest. The
# refusal arms need no sibling archives; the positive arms use them to
# build a real artifact and say so when they are absent, so the gate
# runs everywhere and reports what it could not check.
RL_PURITY_TEST = tests/test_rl_purity
$(RL_PURITY_TEST): tests/test_rl_purity.c tests/rl_gate_util.h $(LIB) $(RL_LIB) $(BIN)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(RL_LIB) $(LDFLAGS) -lm

# Gate for the Grammar 3.2 dual-mode artifact: kflc -o on an RL
# program produces the batch executable plus the companion shared
# object with exactly the k26rl_env.h export set; a small batch run
# writes an episode file the libk26rl reader verifies; the shared
# object serves create/step/reset through dlopen with spec round trip,
# a fault path, and a batch/serve bit-identity check. Skips (77) when
# the sibling stack libraries are not built.
RL_EMIT_TEST = tests/test_rl_emit
$(RL_EMIT_TEST): tests/test_rl_emit.c ../libk26rl/include/k26rl_env.h \
                 ../libk26rl/include/k26rl_episode.h \
                 ../libk26rl/libk26rl.a $(BIN)
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# Determinism and equivalence gates for the dual-mode artifact:
# two-process batch determinism, two-session serve determinism,
# vector independence (batch and serve), spec round trip with
# unknown-tag injection, and batch/serve whole-file equivalence.
# Skips (77) when the sibling stack archives are not built.
RL_DET_TEST = tests/test_rl_determinism
$(RL_DET_TEST): tests/test_rl_determinism.c $(RL_GATE_DEPS) \
                ../libk26rl/include/k26rl_env.h \
                ../libk26rl/include/k26rl_episode.h \
                ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# Fault and auto-reset gates: a deterministically induced fault in
# one environment of a stepping set, the recorded fault record's
# fixed column families, boundary-reset semantics for all three
# episode endings, and episode index monotonicity. Skips (77) when
# the sibling stack archives are not built.
RL_FAULT_TEST = tests/test_rl_fault
$(RL_FAULT_TEST): tests/test_rl_fault.c $(RL_GATE_DEPS) \
                  ../libk26rl/include/k26rl_env.h \
                  ../libk26rl/include/k26rl_episode.h \
                  ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# Re-simulation gate: an episode rebuilt from the file alone by its
# identity triple (rekey ordinal, env index, episode index) with the
# recorded seed and action stream reproduces the recorded streams
# bitwise, terminated, truncated, faulted, and post-rekey episodes
# included. Skips (77) when the sibling stack archives are not built.
RL_RESIM_TEST = tests/test_rl_resim
$(RL_RESIM_TEST): tests/test_rl_resim.c $(RL_GATE_DEPS) \
                  ../libk26rl/include/k26rl_env.h \
                  ../libk26rl/include/k26rl_episode.h \
                  ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# Golden draw coordinates: every recorded domain-randomisation pair
# in a batch file is recomputed straight through libk26rng at the
# documented (class, channel, environment, episode, draw 0)
# coordinates and compared bitwise, tags included. Skips (77) when
# the sibling stack archives are not built.
RL_DRAWS_TEST = tests/test_rl_draws
$(RL_DRAWS_TEST): tests/test_rl_draws.c $(RL_GATE_DEPS) \
                  ../libk26rl/include/k26rl_env.h \
                  ../libk26rl/include/k26rl_episode.h \
                  ../libk26rng/include/k26rng.h \
                  ../libk26rl/libk26rl.a ../libk26rng/libk26rng.a
	$(CC) $(CFLAGS) -I../libk26rl/include -I../libk26rng/include \
	    -o $@ $< ../libk26rl/libk26rl.a ../libk26rng/libk26rng.a \
	    -ldl -lm

# Body state in on_step, and the range-rate observation component:
# write-before-advance ordering, the analytic single-step clause,
# metres-from-origin for position writes across a sector boundary,
# statement order and the boundary-reset step, the range rate against
# a finite difference of the range on two observation modes, and the
# two validation environments (same-seed byte identity, distinct-seed
# divergence, and the endings their sources claim). Skips (77) when
# the sibling stack archives are not built.
RL_STATE_TEST = tests/test_rl_state
$(RL_STATE_TEST): tests/test_rl_state.c $(RL_GATE_DEPS) \
                  ../libk26rl/include/k26rl_env.h \
                  ../libk26rl/include/k26rl_episode.h \
                  ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# Stepping hot-loop pin: the serve surface's create/reset/step
# path performs no allocation and no I/O (k26rl_env.h). The gate
# re-executes itself under an LD_PRELOAD interposer it compiles at
# run time, counts malloc/calloc/realloc/free/write/fwrite armed
# across the stepping window only, and proves the counters live
# with a positive control before asserting the zeros. Skips (77)
# when the sibling stack archives are not built.
# Telemetry tap gates on the serve surface: cross-transport equality
# between the ring and the episode file, bit identity of a run with
# the tap enabled and disabled and with consumers absent, attached,
# stalled, flapping, and killed, the read-only mapping proven by the
# fault a store raises, the enable surface with every refusal, and a
# late attach recovering the spec from the ring's preamble. Skips
# (77) when the sibling stack archives are not built.
# Viewer gates: every panel of the episode viewer driven through its
# headless dump and held against the file read directly through the
# format library's reader, plus the scrubbing cost counted, a file cut
# mid-write, channel names outside the drawable convention, and the
# re-simulation comparison's verdict. Skips (77) when the sibling
# stack archives or the viewer binary are not built.
# Body-state getter gates: sizing and refusals, determinism and
# vector independence, the bitwise pin between the getter's vector and
# the recorded geometric observation channels, and the two spec tags.
# Skips (77) when the sibling stack archives are not built.
RL_BODIES_TEST = tests/test_rl_bodies
$(RL_BODIES_TEST): tests/test_rl_bodies.c $(RL_GATE_DEPS) \
                   ../libk26rl/include/k26rl_env.h \
                   ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# Attitude through the stepping surface: the getter's geometry, the
# advance, the published subdivision, determinism, and the
# faulted-step contract.
# Actuators through the stepping surface: commands reaching the
# physics, actuator state resetting with the episode, and the split
# that keeps the field model's Fortran runtime out of a program that
# declares no magnetorquer.
RL_ACTUATOR_TEST = tests/test_rl_actuators
# $(BIN) is a prerequisite because this gate drives the compiler: what
# it measures is the emitted artifact, so a compiler left behind by an
# edit would have it measuring the previous emitter and reporting
# green for code that is no longer there.
$(RL_ACTUATOR_TEST): tests/test_rl_actuators.c $(RL_GATE_DEPS) \
                     ../libk26rl/include/k26rl_env.h \
                     ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# The collision pass through the stepping surface: contact between the
# sub-advances, termination through the declared predicate, the three
# contact channels, the applied duration, and determinism over a
# fixture whose episodes end in contact.
RL_COLL_TEST = tests/test_rl_collision
$(RL_COLL_TEST): tests/test_rl_collision.c $(RL_GATE_DEPS) \
                 ../libk26rl/include/k26rl_env.h \
                 ../libk26rl/include/k26rl_episode.h \
                 ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

RL_ATT_TEST = tests/test_rl_attitude
$(RL_ATT_TEST): tests/test_rl_attitude.c $(RL_GATE_DEPS) \
                ../libk26rl/include/k26rl_env.h \
                ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# The viewer's own gate drives the viewer binary, so the binary is
# built here before the gate runs rather than left to whatever a
# previous build put there: a gate run against a stale viewer measures
# the viewer that was, which is the shape of a green run that checked
# nothing. The build is conditional on the interface development
# packages exactly as the root target's is, and the gate skips on its
# own when the binary is absent, so a tree without them still tests
# green.
RL_VIEW_TEST = tests/test_rl_view
.PHONY: viewer
viewer:
	@if pkg-config --exists imgui glfw3 2>/dev/null; then \
	    $(MAKE) -s -C ../tools/k26rl_view; \
	fi

$(RL_VIEW_TEST): tests/test_rl_view.c $(RL_GATE_DEPS) \
                 ../libk26rl/include/k26rl_env.h \
                 ../libk26rl/include/k26rl_episode.h \
                 ../libk26rl/libk26rl.a viewer
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

RL_TAP_TEST = tests/test_rl_tap
$(RL_TAP_TEST): tests/test_rl_tap.c $(RL_GATE_DEPS) \
                ../libk26rl/include/k26rl_env.h \
                ../libk26rl/include/k26rl_episode.h \
                ../libk26rl/include/k26rl_tap.h \
                ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

RL_HOTPATH_TEST = tests/test_rl_hotpath
$(RL_HOTPATH_TEST): tests/test_rl_hotpath.c $(RL_GATE_DEPS) \
                    ../libk26rl/include/k26rl_env.h \
                    ../libk26rl/include/k26rl_tap.h \
                    ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# The imperfection layer through the stepping surface: the chain on the
# step path, the coordinates its draws take, the paired truth channels
# and the tag that resolves them.
RL_SENSOR_TEST = tests/test_rl_sensor
$(RL_SENSOR_TEST): tests/test_rl_sensor.c $(RL_GATE_DEPS) \
                   ../libk26rl/include/k26rl_env.h \
                   ../libk26sense/include/k26sense.h \
                   ../libk26sense/libk26sense.a \
                   ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -I../libk26sense/include \
	    -I../libk26rng/include -I../libk26compute/include -o $@ $< \
	    ../libk26rl/libk26rl.a ../libk26sense/libk26sense.a \
	    ../libk26rng/libk26rng.a ../libk26compute/libk26compute.a -ldl -lm

# The relative observe form through the stepping surface: the six
# channels, the axes they are resolved onto, and the rotating-frame
# velocity that an inertial difference would get wrong.
RL_REL_TEST = tests/test_rl_relative
$(RL_REL_TEST): tests/test_rl_relative.c $(RL_GATE_DEPS) \
                ../libk26rl/include/k26rl_env.h \
                ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# Handles and environments over the life of one process, over a
# program that binds more than one vehicle, and the buffer spans the
# two unsized getters write.
RL_LIFECYCLE_TEST = tests/test_rl_lifecycle
$(RL_LIFECYCLE_TEST): tests/test_rl_lifecycle.c $(RL_GATE_DEPS) \
                      ../libk26rl/include/k26rl_env.h \
                      ../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

# Grammar 3.1 pins: the 3.1 `step` statement still lowers to the
# clamped stepping entry (the exact-dt entry belongs to the episode
# machinery alone), and a reinforcement learning program stripped of
# its constructs emits as a plain 3.1 program. Needs only the built
# kflc; always runs.
RL_31_PIN_TEST = tests/test_rl_31_pin
$(RL_31_PIN_TEST): tests/test_rl_31_pin.c $(BIN)
	$(CC) $(CFLAGS) -o $@ $<

# Grammar 3.1 baseline byte-identity against the pre-RL base commit.
# Its own binary so that "history unavailable" is the harness's skip
# code (77), never a silent pass beside the always-runnable pins.
RL_31_BASE_TEST = tests/test_rl_31_baseline
$(RL_31_BASE_TEST): tests/test_rl_31_baseline.c $(BIN)
	$(CC) $(CFLAGS) -o $@ $<

# The drive gates link the sibling stack archives at build time, so
# their build (not just their run) must be conditional: with the
# archives absent, `make test` still completes, reporting each drive
# gate as skipped instead of dying at link.
RL_STACK_ARCHIVE = ../libk26rl/libk26rl.a
ifneq ($(wildcard $(RL_STACK_ARCHIVE)),)
RL_DRIVE_GATES = $(RL_EMIT_TEST) $(RL_DET_TEST) $(RL_FAULT_TEST) $(RL_RESIM_TEST) $(RL_DRAWS_TEST) $(RL_HOTPATH_TEST) $(RL_STATE_TEST) $(RL_TAP_TEST) $(RL_VIEW_TEST) $(RL_BODIES_TEST) $(RL_ATT_TEST) $(RL_ACTUATOR_TEST) $(RL_COLL_TEST) $(RL_LIFECYCLE_TEST) $(RL_REL_TEST) $(RL_SENSOR_TEST) $(RL_DOCKING_TEST)
else
RL_DRIVE_GATES =
endif

test: $(TEST_BIN) $(ASSEMBLY_TEST) $(MANIFEST_TEST) $(BODY_BINDING_TEST) $(W8_ARENA_TEST) $(MEMORY_QUAL_TEST) $(RESERVED_KW_TEST) $(RL_GRAMMAR_TEST) $(RL_PURITY_TEST) $(RL_DRIVE_GATES) $(RL_31_PIN_TEST) $(RL_31_BASE_TEST) $(BIN)
	./$(TEST_BIN) $(TEST_EXAMPLES)
	./$(TEST_BIN) integration_tests/astro_w8.kfl
	./$(TEST_BIN) integration_tests/rl_pointing.kfl
	./$(MANIFEST_TEST)
	./$(BODY_BINDING_TEST)
	./$(W8_ARENA_TEST)
	./$(MEMORY_QUAL_TEST)
	./$(RESERVED_KW_TEST)
	./$(ASSEMBLY_TEST)
	./$(RL_GRAMMAR_TEST)
	./$(RL_PURITY_TEST)
	@for t in $(RL_EMIT_TEST) $(RL_DET_TEST) $(RL_FAULT_TEST) \
	          $(RL_RESIM_TEST) $(RL_DRAWS_TEST) $(RL_HOTPATH_TEST) \
	          $(RL_STATE_TEST) $(RL_TAP_TEST) $(RL_VIEW_TEST) \
	          $(RL_BODIES_TEST) $(RL_ATT_TEST) \
	          $(RL_ACTUATOR_TEST) $(RL_COLL_TEST) \
	          $(RL_LIFECYCLE_TEST) $(RL_REL_TEST) \
	          $(RL_SENSOR_TEST) $(RL_DOCKING_TEST); do \
	     if [ ! -f $(RL_STACK_ARCHIVE) ] || [ ! -x $$t ]; then \
	         echo "$$(basename $$t): SKIP (stack libraries not built)"; \
	         continue; \
	     fi; \
	     ./$$t; rc=$$?; \
	     if [ $$rc -eq 77 ]; then \
	         echo "$$(basename $$t): SKIP (stack libraries not built)"; \
	     elif [ $$rc -ne 0 ]; then \
	         exit $$rc; \
	     fi; \
	 done
	./$(RL_31_PIN_TEST)
	@./$(RL_31_BASE_TEST); rc=$$?; \
	 if [ $$rc -eq 77 ]; then \
	     echo 'test_rl_31_baseline: SKIP (base commit not in history)'; \
	 elif [ $$rc -ne 0 ]; then \
	     exit $$rc; \
	 fi
	@printf 'form PROBE\n    fn double f()\n        return 1.0\n    end\nend\n' \
	     > /tmp/kflc_flags_probe.kfl; \
	 env -u KFLC_CFLAGS ./$(BIN) /tmp/kflc_flags_probe.kfl \
	     -o /tmp/kflc_flags_probe > /tmp/kflc_flags_probe.log 2>&1; \
	 if grep -q -- '-ffp-contract=off -fexcess-precision=standard' \
	     /tmp/kflc_flags_probe.log; then \
	     echo 'default cflags carry the FP determinism flags: OK'; \
	 else \
	     echo 'default cflags missing the FP determinism flags: FAIL'; \
	     cat /tmp/kflc_flags_probe.log; exit 1; \
	 fi

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

# The asset identity digest lives beside the record format's checksum
# in libk26rl, so the compiler that stamps a digest into a program and
# any consumer that verifies one share a single implementation. Static
# linkage pulls in that object and nothing else.
$(RL_LIB):
	$(MAKE) -C ../libk26rl

# -lm because the compiler now derives geometry of its own: a
# capsule's declared endpoints become a centre, a direction, and a
# half length once, here, rather than on every step of every run.
$(BIN): $(DRV_OBJS) $(LIB) $(RL_LIB) $(SENSE_LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(DRV_OBJS) $(LIB) $(RL_LIB) $(SENSE_LIB) $(LDFLAGS) -lm

src/%.o: src/%.c include/kflc.h src/internal.h
	$(CC) $(CFLAGS) -c -o $@ $<

# dump.c and serialize.c only need the public header.
src/dump.o: src/dump.c include/kflc.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/serialize.o: src/serialize.c include/kflc.h
	$(CC) $(CFLAGS) -c -o $@ $<

AR ?= ar

install: $(BIN) $(LIB)
	install -Dm755 $(BIN)                $(DESTDIR)$(PREFIX)/bin/kflc
	install -Dm644 $(LIB)                $(DESTDIR)$(PREFIX)/lib/$(LIB)
	install -Dm644 include/kflc.h        $(DESTDIR)$(PREFIX)/include/kflc.h
	install -Dm644 include/kflc_runtime.h \
	    $(DESTDIR)$(PREFIX)/include/kflc_runtime.h
	install -Dm644 GRAMMAR.md            $(DESTDIR)$(PREFIX)/share/doc/kflc/GRAMMAR.md
	install -Dm644 share/templates/template.kfl \
	    $(DESTDIR)$(PREFIX)/share/doc/kflc/templates/template.kfl
	@for f in $(wildcard examples/*.kfl); do \
	    install -Dm644 $$f \
	        $(DESTDIR)$(PREFIX)/share/doc/kflc/examples/$$(basename $$f); \
	done

clean:
	rm -f src/*.o $(BIN) $(LIB) $(TEST_BIN) $(MANIFEST_TEST) \
	      $(BODY_BINDING_TEST) $(W8_ARENA_TEST) $(MEMORY_QUAL_TEST) \
	      $(RESERVED_KW_TEST) $(RL_GRAMMAR_TEST) $(RL_PURITY_TEST) \
	      $(RL_EMIT_TEST) \
	      $(ASSEMBLY_TEST) \
	      $(RL_DET_TEST) $(RL_FAULT_TEST) $(RL_RESIM_TEST) \
	      $(RL_31_PIN_TEST) $(RL_31_BASE_TEST) $(RL_DRAWS_TEST) \
	      $(RL_HOTPATH_TEST) $(RL_STATE_TEST) $(RL_TAP_TEST) \
	      $(RL_VIEW_TEST) $(RL_BODIES_TEST) $(RL_ATT_TEST) \
	      $(RL_ACTUATOR_TEST) $(RL_COLL_TEST) $(RL_LIFECYCLE_TEST) $(RL_REL_TEST) \
	      $(RL_SENSOR_TEST) $(RL_DOCKING_TEST) \
	      tests/bench_parse
	rm -rf bin
