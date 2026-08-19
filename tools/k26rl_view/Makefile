# k26rl_view: episode file viewer for the reinforcement learning
# environment layer (host-only).
#
# Replay mode over the binary episode format: timeline scrubbing,
# reward, observation, action, trajectory and metadata panels, a
# three-dimensional scene view, and a re-simulation comparison when an
# artifact is supplied. The headless --dump mode needs no display and
# is what the viewer gates drive.
#
# The interface is C++ because Dear ImGui is; nothing in the library
# tiers links any of this, and the compiled environments link none of
# it either. ImPlot is vendored under upstream/ (MIT; see the tree's
# NOTICE); Dear ImGui and GLFW come from the system.

CXX     ?= g++
CC      ?= gcc
CXXFLAGS ?= -std=c++11 -O2 -g -Wall -Wextra
CFLAGS  ?= -std=c99 -O2 -g -Wall -Wextra

PKGS = imgui glfw3
PKG_CFLAGS := $(shell pkg-config --cflags $(PKGS) 2>/dev/null)
PKG_LIBS   := $(shell pkg-config --libs $(PKGS) 2>/dev/null)

INC = -I../../libk26rl/include \
      -I../../libk26m3d/include \
      -I../../kflc/src \
      -Iupstream/implot \
      $(PKG_CFLAGS)

LIBS = ../../libk26rl/libk26rl.a \
       ../../libk26m3d/libk26m3d.a \
       $(PKG_LIBS) -lGL -ldl -lm

# The vendored plotting library is compiled here rather than kept as a
# prebuilt object, so what ships is source and its provenance.
UPSTREAM_SRC = upstream/implot/implot.cpp upstream/implot/implot_items.cpp
OWN_SRC = main.cpp model.cpp asset.cpp scene.cpp dump.cpp resim.cpp gui.cpp

# The compiler's capture envelope table, compiled in rather than
# transcribed. A docking port's mating plane is the diameter its named
# envelope publishes, and the scene draws that plane; a second copy of
# the figure here would be a picture that stops agreeing with the
# program the first time either moves.
SHARED_SRC = ../../kflc/src/capture.c
SHARED_OBJ = capture.o

OBJ = $(OWN_SRC:.cpp=.o) $(SHARED_OBJ) $(UPSTREAM_SRC:.cpp=.o)

.PHONY: all clean
all: k26rl_view

k26rl_view: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(LIBS)

# The vendored sources are compiled as they arrived: no warning flags
# of ours are imposed on code we did not write and do not modify.
upstream/implot/%.o: upstream/implot/%.cpp
	$(CXX) -std=c++11 -O2 -g $(PKG_CFLAGS) -Iupstream/implot -c -o $@ $<

capture.o: $(SHARED_SRC) ../../kflc/src/capture.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<

main.o:  main.cpp dump.h gui.h model.h scene.h
model.o: model.cpp model.h
asset.o: asset.cpp asset.h
scene.o: scene.cpp scene.h model.h asset.h resim.h
dump.o:  dump.cpp dump.h model.h resim.h asset.h scene.h
resim.o: resim.cpp resim.h model.h
gui.o:   gui.cpp gui.h dump.h model.h resim.h asset.h scene.h

clean:
	rm -f k26rl_view $(OBJ)
