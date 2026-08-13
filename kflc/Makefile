CC      ?= cc
PREFIX  ?= /usr
DESTDIR ?=

CFLAGS  ?= -O2 -g
override CFLAGS += -std=c11 -Wall -Wextra -Wpedantic \
                   -Iinclude -Isrc

BIN      = bin/kflc
LIB      = libkflc.a
# Library objects: parser + lexer + dump + arena/diag util. Linked by
# both the kflc binary and any embedding tool (k26form) that needs to
# load .kfl files.
LIB_OBJS = src/util.o src/lexer.o src/parser.o src/dump.o src/serialize.o \
           src/expr.o src/stmt.o src/check.o src/opaque.o \
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

$(TEST_BIN): tests/round_trip.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

MANIFEST_TEST = tests/test_manifest_loader
$(MANIFEST_TEST): tests/test_manifest_loader.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

# Emit-side test for parse-time body bindings. Compiles a fixture
# via `kflc --emit` and greps the output for `_kfl_body_<NAME>_idx`
# declarations + idx-deref uses, and the absence of runtime
# `find_body` calls for known names.
BODY_BINDING_TEST = tests/test_body_binding_emit
$(BODY_BINDING_TEST): tests/test_body_binding_emit.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

# Emit-side test for the KFL memory model. Verifies form-level
# arena declarations + init factory + fn-body allocator binding +
# conditional kflc_runtime.h include + slim include surface for
# forms that do not use the memory keywords.
W8_ARENA_TEST = tests/test_w8_arena_emit
$(W8_ARENA_TEST): tests/test_w8_arena_emit.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

# Gate for compile-time enforcement of own / borrow / ptr / arena
# qualifiers. Writes small .kfl fixtures to /tmp, runs the built
# kflc binary against them, asserts that positive cases emit the
# expected C++ shape and negative cases trigger the right diag error.
MEMORY_QUAL_TEST = tests/test_memory_qualifiers
$(MEMORY_QUAL_TEST): tests/test_memory_qualifiers.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

# Gate for the reserved-keyword warning list: identifiers that will
# become keywords in a future grammar version warn today, so authors
# rename before the keyword lands and binds.
RESERVED_KW_TEST = tests/test_reserved_keywords
$(RESERVED_KW_TEST): tests/test_reserved_keywords.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

# Gate for the Grammar 3.2 reinforcement learning front end: episode /
# action / on_step / objective / observe-as parse + check, with the
# negative diagnostics pinned. Runs the built kflc binary over /tmp
# fixtures (test_reserved_keywords pattern).
RL_GRAMMAR_TEST = tests/test_rl_grammar
$(RL_GRAMMAR_TEST): tests/test_rl_grammar.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

# Gate for the Grammar 3.2 dual-mode artifact: kflc -o on an RL
# program produces the batch executable plus the companion shared
# object with exactly the k26rl_env.h export set; a small batch run
# writes an episode file the libk26rl reader verifies; the shared
# object serves create/step/reset through dlopen with spec round trip,
# a fault path, and a batch/serve bit-identity check. Skips (77) when
# the sibling stack libraries are not built.
RL_EMIT_TEST = tests/test_rl_emit
$(RL_EMIT_TEST): tests/test_rl_emit.c ../libk26rl/include/k26rl_env.h \
                 ../libk26rl/include/k26rl_episode.h
	$(CC) $(CFLAGS) -I../libk26rl/include -o $@ $< \
	    ../libk26rl/libk26rl.a -ldl -lm

test: $(TEST_BIN) $(MANIFEST_TEST) $(BODY_BINDING_TEST) $(W8_ARENA_TEST) $(MEMORY_QUAL_TEST) $(RESERVED_KW_TEST) $(RL_GRAMMAR_TEST) $(RL_EMIT_TEST) $(BIN)
	./$(TEST_BIN) $(TEST_EXAMPLES)
	./$(TEST_BIN) integration_tests/astro_w8.kfl
	./$(TEST_BIN) integration_tests/rl_pointing.kfl
	./$(MANIFEST_TEST)
	./$(BODY_BINDING_TEST)
	./$(W8_ARENA_TEST)
	./$(MEMORY_QUAL_TEST)
	./$(RESERVED_KW_TEST)
	./$(RL_GRAMMAR_TEST)
	@./$(RL_EMIT_TEST); rc=$$?; \
	 if [ $$rc -eq 77 ]; then \
	     echo 'test_rl_emit: SKIP (stack libraries not built)'; \
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

$(BIN): $(DRV_OBJS) $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(DRV_OBJS) $(LIB) $(LDFLAGS)

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
	      $(RESERVED_KW_TEST) $(RL_GRAMMAR_TEST) $(RL_EMIT_TEST) \
	      tests/bench_parse
	rm -rf bin
