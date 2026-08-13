CC      ?= gcc
AR      ?= ar

# Deterministic counter-based sampling (Philox4x32-10). Pure C, no
# external deps beyond libm's sqrt. Sits below the RL tier so sensor
# models and any future library needing reproducible noise can reach
# it without pulling RL machinery in.

CFLAGS  ?= -std=c11 -O2 -g -Wall -Wextra -Wpedantic -D_GNU_SOURCE
CFLAGS  += -Iinclude -Isrc
CFLAGS  += -ffp-contract=off -fexcess-precision=standard
CFLAGS  += -fPIC

SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
LIB = libk26rng.a

PUBLIC_HDRS  = $(wildcard include/*.h)
PRIVATE_HDRS = $(wildcard src/*.h)

PREFIX  ?= /usr
LIBDIR  ?= $(PREFIX)/lib
DESTDIR ?=

SONAME_MAJOR = 0
LIB_VER      = 0.1.0
LIB_BASE     = libk26rng
SHLIB_FULL   = $(LIB_BASE).so.$(LIB_VER)
SHLIB_SONAME = $(LIB_BASE).so.$(SONAME_MAJOR)
SHLIB_LINK   = $(LIB_BASE).so
SHLIB_LDLIBS = -lm

TEST_SRC = $(wildcard tests/test_*.c)
TEST_BIN = $(TEST_SRC:.c=)

.PHONY: all test install clean
all: $(LIB) $(SHLIB_FULL)

test: $(LIB) $(TEST_BIN)
	@for t in $(TEST_BIN); do ./$$t || exit 1; done

tests/test_%: tests/test_%.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) -lm

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

$(SHLIB_FULL): $(OBJ)
	$(CC) -shared -Wl,-soname,$(SHLIB_SONAME) -o $@ $^ $(SHLIB_LDLIBS)

%.o: %.c $(PUBLIC_HDRS) $(PRIVATE_HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

install: all
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(LIBDIR)
	install -m 644 $(PUBLIC_HDRS) $(DESTDIR)$(PREFIX)/include
	install -m 644 $(LIB) $(DESTDIR)$(LIBDIR)
	install -m 755 $(SHLIB_FULL) $(DESTDIR)$(LIBDIR)
	ln -sf $(SHLIB_FULL) $(DESTDIR)$(LIBDIR)/$(SHLIB_SONAME)
	ln -sf $(SHLIB_SONAME) $(DESTDIR)$(LIBDIR)/$(SHLIB_LINK)

clean:
	rm -f $(OBJ) $(LIB) $(SHLIB_FULL) $(TEST_BIN)
