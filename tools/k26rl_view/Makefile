# k26rl_view: episode file viewer for the reinforcement learning
# environment layer (host-only).
#
# Replay mode over the binary episode format: timeline scrubbing,
# reward, observation, action, trajectory and metadata panels, and a
# re-simulation comparison when an artifact is supplied. The headless
# --dump mode needs no display and is what the viewer gate drives.
#
# The interface is C++ because Dear ImGui is; nothing in the library
# tiers links any of this, and the compiled environments link none of
# it either. ImPlot is vendored under upstream/ (MIT; see the tree's
# NOTICE); Dear ImGui and GLFW come from the system.

CXX     ?= g++
CXXFLAGS ?= -std=c++11 -O2 -g -Wall -Wextra

PKGS = imgui glfw3
PKG_CFLAGS := $(shell pkg-config --cflags $(PKGS) 2>/dev/null)
PKG_LIBS   := $(shell pkg-config --libs $(PKGS) 2>/dev/null)

INC = -I../../libk26rl/include \
      -I../../libk26m3d/include \
      -Iupstream/implot \
      $(PKG_CFLAGS)

LIBS = ../../libk26rl/libk26rl.a \
       ../../libk26m3d/libk26m3d.a \
       $(PKG_LIBS) -lGL -ldl -lm

# The vendored plotting library is compiled here rather than kept as a
# prebuilt object, so what ships is source and its provenance.
UPSTREAM_SRC = upstream/implot/implot.cpp upstream/implot/implot_items.cpp
OWN_SRC = main.cpp model.cpp dump.cpp resim.cpp gui.cpp
OBJ = $(OWN_SRC:.cpp=.o) $(UPSTREAM_SRC:.cpp=.o)

.PHONY: all clean
all: k26rl_view

k26rl_view: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(LIBS)

# The vendored sources are compiled as they arrived: no warning flags
# of ours are imposed on code we did not write and do not modify.
upstream/implot/%.o: upstream/implot/%.cpp
	$(CXX) -std=c++11 -O2 -g $(PKG_CFLAGS) -Iupstream/implot -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<

main.o:  main.cpp dump.h gui.h model.h
model.o: model.cpp model.h
dump.o:  dump.cpp dump.h model.h resim.h
resim.o: resim.cpp resim.h model.h
gui.o:   gui.cpp gui.h dump.h model.h resim.h

clean:
	rm -f k26rl_view $(OBJ)
