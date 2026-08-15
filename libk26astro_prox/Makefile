CC      ?= gcc
AR      ?= ar

# Relative motion between two craft in proximity: the chief's
# local-vertical local-horizontal frame, the target's state resolved
# onto it, and the linearised propagator the gates and the
# initial-condition construction use.
#
# The frame and the relative state are measurement and are exact for
# any orbit. The linearised model is a tool and is not integrated by
# anything in this tree.
#
# Depends on: libk26astro_core (the sector-grid position type) and
# libk26m3d. It takes no view of what a body is beyond a position and
# a velocity.

CFLAGS  ?= -std=c11 -O2 -g -Wall -Wextra -Wpedantic -D_GNU_SOURCE
CFLAGS  += -Iinclude -Isrc
CFLAGS  += -I../libk26astro_core/include
CFLAGS  += -I../libk26m3d/include
CFLAGS  += -ffp-contract=off -fexcess-precision=standard
CFLAGS  += -fPIC

SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
LIB = libk26astro_prox.a

PUBLIC_HDRS  = $(wildcard include/k26astro_prox/*.h)
PRIVATE_HDRS = $(wildcard src/*.h)

PREFIX  ?= /usr
LIBDIR  ?= $(PREFIX)/lib
DESTDIR ?=

SONAME_MAJOR = 0
LIB_VER      = 0.1.0
LIB_BASE     = libk26astro_prox
SHLIB_FULL   = $(LIB_BASE).so.$(LIB_VER)
SHLIB_SONAME = $(LIB_BASE).so.$(SONAME_MAJOR)
SHLIB_LINK   = $(LIB_BASE).so
SHLIB_LDLIBS = -L../libk26astro_core -lk26astro_core -lm

.PHONY: all install clean test
all: $(LIB) $(SHLIB_FULL)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

$(SHLIB_FULL): $(OBJ)
	$(CC) -shared -Wl,-soname,$(SHLIB_SONAME) -o $@ $^ $(SHLIB_LDLIBS)

src/%.o: src/%.c $(PUBLIC_HDRS) $(PRIVATE_HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

install: $(LIB) $(SHLIB_FULL)
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(PREFIX)/include/k26astro_prox
	install -m755 $(SHLIB_FULL) $(DESTDIR)$(LIBDIR)/$(SHLIB_FULL)
	ln -sf $(SHLIB_FULL)   $(DESTDIR)$(LIBDIR)/$(SHLIB_SONAME)
	ln -sf $(SHLIB_SONAME) $(DESTDIR)$(LIBDIR)/$(SHLIB_LINK)
	install -m644 $(LIB) $(DESTDIR)$(LIBDIR)/$(LIB)
	for h in $(PUBLIC_HDRS); do \
	    install -m644 $$h $(DESTDIR)$(PREFIX)/include/k26astro_prox/; \
	done

# The archives a gate links, listed once and used both as the link
# line and as prerequisites, so a change in a sibling library cannot
# leave a stale gate binary in place reporting green for code that is
# no longer there.
TEST_ARCHIVES = $(LIB) \
                ../libk26astro_core/libk26astro_core.a \
                ../libk26m3d/libk26m3d.a
TEST_LINK = $(TEST_ARCHIVES) -lm

TEST_SRC = $(wildcard tests/test_*.c)
TEST_BIN = $(TEST_SRC:.c=)

test: $(LIB) $(TEST_BIN)
	@for t in $(TEST_BIN); do ./$$t || exit 1; done

tests/test_%: tests/test_%.c $(TEST_ARCHIVES)
	$(CC) $(CFLAGS) -o $@ $< $(TEST_LINK)

clean:
	rm -f src/*.o $(LIB) $(SHLIB_FULL) $(TEST_BIN)
