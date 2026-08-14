CC      ?= gcc
AR      ?= ar

# Spacecraft attitude on the exact stepping path: the advance that
# integrates a vehicle's orientation over a caller-named interval, and
# the actuators a vehicle commands. The rigid-body equations live in
# libk26astro_body and are consumed, not reimplemented.
#
# Depends on: libk26astro_vehicle, libk26astro_body, libk26astro_core,
# libk26m3d.

CFLAGS  ?= -std=c11 -O2 -g -Wall -Wextra -Wpedantic -D_GNU_SOURCE
CFLAGS  += -Iinclude -Isrc
CFLAGS  += -I../libk26astro_vehicle/include
CFLAGS  += -I../libk26astro_body/include
CFLAGS  += -I../libk26astro_core/include
CFLAGS  += -I../libk26m3d/include
CFLAGS  += -ffp-contract=off -fexcess-precision=standard
CFLAGS  += -fPIC

SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
LIB = libk26astro_att.a

PUBLIC_HDRS  = $(wildcard include/k26astro_att/*.h)
PRIVATE_HDRS = $(wildcard src/*.h)

PREFIX  ?= /usr
LIBDIR  ?= $(PREFIX)/lib
DESTDIR ?=

SONAME_MAJOR = 0
LIB_VER      = 0.1.0
LIB_BASE     = libk26astro_att
SHLIB_FULL   = $(LIB_BASE).so.$(LIB_VER)
SHLIB_SONAME = $(LIB_BASE).so.$(SONAME_MAJOR)
SHLIB_LINK   = $(LIB_BASE).so
SHLIB_LDLIBS = -lm

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
	install -d $(DESTDIR)$(PREFIX)/include/k26astro_att
	install -m755 $(SHLIB_FULL) $(DESTDIR)$(LIBDIR)/$(SHLIB_FULL)
	ln -sf $(SHLIB_FULL)   $(DESTDIR)$(LIBDIR)/$(SHLIB_SONAME)
	ln -sf $(SHLIB_SONAME) $(DESTDIR)$(LIBDIR)/$(SHLIB_LINK)
	install -m644 $(LIB) $(DESTDIR)$(LIBDIR)/$(LIB)
	for h in $(PUBLIC_HDRS); do \
	    install -m644 $$h $(DESTDIR)$(PREFIX)/include/k26astro_att/; \
	done

# The archives a gate links, listed once and used both as the link
# line and as prerequisites. Naming them as prerequisites is not
# tidiness: without it a change in a sibling library leaves a stale
# gate binary in place, so a defect introduced there passes its own
# gate. Found while mutation-testing this library's gates, where a
# removed term in the body library's Euler step appeared to survive
# and had in fact never been linked.
TEST_ARCHIVES = $(LIB) \
                ../libk26astro_vehicle/libk26astro_vehicle.a \
                ../libk26astro_body/libk26astro_body.a \
                ../libk26astro_ephem/libk26astro_ephem.a \
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
