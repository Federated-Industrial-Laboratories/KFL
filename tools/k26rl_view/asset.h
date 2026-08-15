/* asset.h - the vehicle assembly a recording names, read for drawing.
 *
 * An assembly is the text description of a vehicle that the compiler
 * derived a body's mass properties from, and the compiled artifact
 * carries the digest of its bytes. This reads the same asset from
 * disk, recomputes that digest through the format library's own
 * function, and builds the wireframe its meshes describe.
 *
 * The digest is the whole point of reading it here. A wireframe drawn
 * beside a recording is a claim that this is the craft that flew, and
 * the only thing that makes the claim true is that the bytes on disk
 * are the bytes the artifact was built from. When they are not, this
 * viewer says so and draws nothing, which is the design's own rule:
 * report the mismatch rather than render a craft that is not the
 * craft that flew.
 *
 * What is read here is a subset of what the compiler reads, and
 * deliberately so: the viewer needs the assembly's name, the meshes
 * its components reference in the order the compiler contributes
 * them, and the geometry those meshes hold. It does not need
 * colliders, actuators, ports, or mass properties, and does not
 * derive any. A subset reader can disagree with the compiler about
 * what an assembly references, and the consequence of disagreeing is
 * bounded by construction: the digest then fails to match and the
 * viewer refuses to draw. It cannot draw the wrong craft quietly.
 */
#ifndef K26RL_VIEW_ASSET_H
#define K26RL_VIEW_ASSET_H

#include <stdint.h>

#include <string>
#include <vector>

namespace k26rl_view {

/* One line of the wireframe, as two vertex indices. */
struct Edge {
    uint32_t a;
    uint32_t b;
};

/* An assembly read from disk: what it is called, what its bytes
 * digest to, and the wireframe its meshes make.
 *
 * Vertices are in the assembly's body frame, the component placement
 * applied, because that is the frame the craft's own geometry is
 * expressed in and the frame a viewer draws it in. */
struct Asset {
    bool loaded = false;
    std::string error;                  /* why not, when !loaded */
    std::string path;
    std::string name;                   /* the `assembly <name>` line */
    uint8_t digest[32];
    std::vector<std::string> meshes;    /* referenced, in contribution order */
    uint32_t mesh_vertices = 0;         /* as the mesh files declare them */
    uint32_t mesh_triangles = 0;
    std::vector<double> vertices;       /* 3 per vertex, body frame */
    std::vector<Edge> edges;            /* unique, undirected */
};

/* Read an assembly and every mesh it references, recompute the
 * identity digest over their bytes, and build the wireframe. A
 * missing file, a malformed number, or a mesh directive outside the
 * accepted subset leaves `loaded` false with `error` saying which. */
Asset asset_load(const std::string &path);

/* Lowercase hexadecimal, for a panel and for a diagnostic. */
std::string digest_hex(const uint8_t d[32]);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_ASSET_H */
