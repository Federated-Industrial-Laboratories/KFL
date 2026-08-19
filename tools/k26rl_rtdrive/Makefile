# k26rl_rtdrive - the wall-clock driver as a program.
#
# Runs a trained policy over a compiled world against a monotonic
# clock at a declared rate and reports periods run and missed, the
# decision-latency distribution, and the wall clock against the
# simulated duration. The loop is libk26rl's; this builds the part
# that opens an artifact, loads a policy and prints the result.
#
# The world itself is opened at run time with dlopen, so nothing of
# the simulation stack is linked here.

CC      ?= gcc
CFLAGS  ?= -std=c11 -O2 -g -Wall -Wextra -Wpedantic -D_GNU_SOURCE

INC = -I../../libk26rl/include

LIBS = ../../libk26rl/libk26rl.a -ldl -lm

.PHONY: all clean
all: k26rl_rtdrive

k26rl_rtdrive: main.c ../../libk26rl/libk26rl.a
	$(CC) $(CFLAGS) $(INC) -o $@ $< $(LIBS)

../../libk26rl/libk26rl.a:
	$(MAKE) -C ../../libk26rl

clean:
	rm -f k26rl_rtdrive
