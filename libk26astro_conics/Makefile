CC      ?= gcc
AR      ?= ar

# Analytic two-body machinery. Universal-variable Kepler propagator,
# single + multi-rev Lambert, SoI hierarchy. Pure functional.
#
# Depends on: libk26astro_core, libk26astro_body, libk26m3d.

CFLAGS  ?= -std=c11 -O2 -g -Wall -Wextra -Wpedantic -D_GNU_SOURCE
CFLAGS  += -Iinclude -Isrc
CFLAGS  += -I../libk26astro_core/include
CFLAGS  += -I../libk26astro_body/include
CFLAGS  += -I../libk26astro_ephem/include
CFLAGS  += -I../libk26m3d/include
CFLAGS  += -ffp-contract=off -fexcess-precision=standard -frounding-math
CFLAGS  += -fPIC

SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
LIB = libk26astro_conics.a

PUBLIC_HDRS  = $(wildcard include/k26astro_conics/*.h)
PRIVATE_HDRS = $(wildcard src/*.h)

PREFIX  ?= /usr
LIBDIR  ?= $(PREFIX)/lib
DESTDIR ?=

SONAME_MAJOR = 0
LIB_VER      = 0.4.1
LIB_BASE     = libk26astro_conics
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
	install -d $(DESTDIR)$(PREFIX)/include/k26astro_conics
	install -m755 $(SHLIB_FULL) $(DESTDIR)$(LIBDIR)/$(SHLIB_FULL)
	ln -sf $(SHLIB_FULL)   $(DESTDIR)$(LIBDIR)/$(SHLIB_SONAME)
	ln -sf $(SHLIB_SONAME) $(DESTDIR)$(LIBDIR)/$(SHLIB_LINK)
	install -m644 $(LIB) $(DESTDIR)$(LIBDIR)/$(LIB)
	for h in $(PUBLIC_HDRS); do \
	    install -m644 $$h $(DESTDIR)$(PREFIX)/include/k26astro_conics/; \
	done
	install -d $(DESTDIR)$(PREFIX)/share/kflc/builtins
	install -m644 kfl/k26astro_conics.kflbi \
	    $(DESTDIR)$(PREFIX)/share/kflc/builtins/k26astro_conics.kflbi

# The archives a gate links, listed once and used both as the link
# line and as prerequisites. Naming them as prerequisites is not
# tidiness: without it a change in a library this one links leaves a
# stale gate binary in place, so a defect introduced there passes this
# library's gates. The pattern rule below therefore depends on the
# archives, not only on the test's own source.
TEST_ARCHIVES = $(LIB) \
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
