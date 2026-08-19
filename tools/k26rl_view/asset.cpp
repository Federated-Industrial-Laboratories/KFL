/* asset.cpp: the assembly reader behind the wireframe and the scene.
 *
 * The digest is recomputed through the format library's own function,
 * not through a second implementation of it, so the framing this
 * viewer applies is the framing the compiler applied: the assembly's
 * own bytes first, then each component's mesh in the order the
 * components are declared, each contribution framed by its length.
 * Two readers of one format is already one more than the tree wants;
 * two implementations of the identity function would be worse, since
 * a difference there would be invisible until it wrongly cleared or
 * wrongly refused a craft.
 */

#include "asset.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>

extern "C" {
#include "k26rl_digest.h"
#include "capture.h"
}

namespace k26rl_view {

namespace {

bool read_file_(const std::string &path, std::vector<uint8_t> *out)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
        return false;
    out->clear();
    for (;;) {
        uint8_t buf[8192];
        size_t n = fread(buf, 1, sizeof buf, f);
        if (n)
            out->insert(out->end(), buf, buf + n);
        if (n < sizeof buf)
            break;
    }
    bool ok = ferror(f) == 0;
    fclose(f);
    return ok;
}

/* The directory a path sits in, so a mesh resolves against the
 * assembly that named it exactly as it does for the compiler. */
std::string dirname_(const std::string &path)
{
    size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? std::string(".")
                                      : path.substr(0, slash);
}

std::string resolve_(const std::string &rel, const std::string &base)
{
    if (!rel.empty() && rel[0] == '/')
        return rel;
    return dirname_(base) + "/" + rel;
}

/* One line's whitespace-separated tokens, with `#` to end of line
 * dropped and a quoted token taken whole with its quotes removed.
 * The accepted subset of both formats is line-oriented, so a line is
 * the whole of the lexical state.
 *
 * The quotes matter: the compiler's own tokenizer strips them, so a
 * quoted mesh path is an ordinary path to it. A reader that kept them
 * would refuse a file the compiler accepted, which fails safe and
 * blames the file for its own reading. */
void tokens_(const std::string &line, std::vector<std::string> *out)
{
    out->clear();
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t' ||
                                   line[i] == '\r'))
            i++;
        if (i >= line.size() || line[i] == '#')
            break;
        if (line[i] == '"') {
            size_t j = ++i;
            while (j < line.size() && line[j] != '"')
                j++;
            out->push_back(line.substr(i, j - i));
            i = (j < line.size()) ? j + 1 : j;
            continue;
        }
        size_t j = i;
        while (j < line.size() && line[j] != ' ' && line[j] != '\t' &&
               line[j] != '\r' && line[j] != '#')
            j++;
        out->push_back(line.substr(i, j - i));
        i = j;
    }
}

void lines_(const std::vector<uint8_t> &bytes,
            std::vector<std::string> *out)
{
    out->clear();
    std::string cur;
    for (size_t i = 0; i < bytes.size(); i++) {
        char c = (char)bytes[i];
        if (c == '\n') {
            out->push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty())
        out->push_back(cur);
}

bool number_(const std::string &s, double *out)
{
    char *end = 0;
    *out = strtod(s.c_str(), &end);
    return end && *end == '\0' && end != s.c_str();
}

/* A component's placement: a translation, and an optional unit
 * quaternion written scalar first. */
struct Placement {
    double at[3];
    double q[4];
};

void rotate_(const Placement &p, const double v[3], double out[3])
{
    double w = p.q[0], x = p.q[1], y = p.q[2], z = p.q[3];
    double t[3];
    t[0] = 2.0 * (y * v[2] - z * v[1]);
    t[1] = 2.0 * (z * v[0] - x * v[2]);
    t[2] = 2.0 * (x * v[1] - y * v[0]);
    out[0] = v[0] + w * t[0] + (y * t[2] - z * t[1]);
    out[1] = v[1] + w * t[1] + (z * t[0] - x * t[2]);
    out[2] = v[2] + w * t[2] + (x * t[1] - y * t[0]);
}

/* The component's orientation as a matrix, built by rotating the
 * basis vectors with the same quaternion the vertices use, so a
 * collider's axes and its mesh's vertices cannot be placed by two
 * different rotations. */
void placement_matrix_(const Placement &p, double out[9])
{
    for (int c = 0; c < 3; c++) {
        double e[3] = { 0.0, 0.0, 0.0 };
        double w[3];
        e[c] = 1.0;
        rotate_(p, e, w);
        for (int r = 0; r < 3; r++)
            out[r * 3 + c] = w[r];
    }
}

void add_edge_(std::map<std::pair<uint32_t, uint32_t>, int> *seen,
               std::vector<Edge> *edges, uint32_t a, uint32_t b)
{
    std::pair<uint32_t, uint32_t> key(a < b ? a : b, a < b ? b : a);
    if (seen->find(key) != seen->end())
        return;
    (*seen)[key] = 1;
    Edge e;
    e.a = key.first;
    e.b = key.second;
    edges->push_back(e);
}

}  /* namespace */

std::string digest_hex(const uint8_t d[32])
{
    static const char *const hexd = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 32; i++) {
        out.push_back(hexd[(d[i] >> 4) & 0xF]);
        out.push_back(hexd[d[i] & 0xF]);
    }
    return out;
}

AssetVerdict asset_verdict(const Spec &sp, const Asset &a,
                           const AssemblyRef **out)
{
    const AssemblyRef *match = 0;
    for (size_t i = 0; i < sp.assemblies.size(); i++) {
        if (sp.assemblies[i].name == a.name)
            match = &sp.assemblies[i];
    }
    if (out)
        *out = match;
    if (!match)
        return ASSET_NO_BODY;
    if (!match->has_digest)
        return ASSET_NO_DIGEST;
    if (memcmp(match->digest, a.digest, K26RL_SHA256_BYTES) != 0)
        return ASSET_MISMATCH;
    return ASSET_DRAWABLE;
}

Asset asset_load(const std::string &path)
{
    Asset a;
    a.path = path;
    memset(a.digest, 0, sizeof a.digest);

    std::vector<uint8_t> bytes;
    if (!read_file_(path, &bytes)) {
        a.error = "cannot open assembly `" + path + "`";
        return a;
    }

    K26RlSha256 dig;
    k26rl_digest_begin(&dig);
    k26rl_digest_add(&dig, bytes.empty() ? 0 : &bytes[0],
                     (uint64_t)bytes.size());

    std::vector<std::string> src;
    lines_(bytes, &src);

    /* Walk the assembly for what the scene draws: its name, each
     * component's placement, mesh and colliders, and the ports and
     * thrusters its features declare. Everything else is skipped by
     * name rather than refused, because the compiler has already
     * accepted this file and this reader is not a second gate on it.
     *
     * A component's placement is settled at the block's `end` and not
     * at whatever line the mesh happened to follow. The compiler
     * parses the whole block and derives afterwards, so key order
     * inside a block is free for it; a reader that snapshotted the
     * placement at the `mesh` line would make that order load-bearing
     * for itself alone, and would draw a craft at its raw mesh
     * coordinates while the digest, which is over bytes and says
     * nothing about how they are read, still matched. */
    Placement cur;
    /* Which block the walk is inside. The assembly's own top level is
     * BLK_NONE; a block this reader draws nothing from is BLK_OTHER,
     * so its keys cannot be read as the enclosing block's. */
    enum Block { BLK_NONE, BLK_COMPONENT, BLK_PORT, BLK_THRUSTER, BLK_OTHER };
    Block blk = BLK_NONE;
    bool have_mesh = false;
    std::string pending_mesh;
    std::vector<Placement> placements;
    std::vector<std::string> mesh_paths;
    std::vector<std::string> tok;
    /* Colliders wait for their component's `end` for the same reason
     * the mesh placement does: the placement that bakes them into the
     * body frame is not settled until then. */
    std::vector<Collider> pending;
    Port port = Port();
    Thruster thruster = Thruster();

    for (size_t i = 0; i < src.size(); i++) {
        tokens_(src[i], &tok);
        if (tok.empty())
            continue;
        if (tok[0] == "assembly" && tok.size() >= 2 && blk == BLK_NONE) {
            a.name = tok[1];
            continue;
        }
        if (blk == BLK_NONE &&
            (tok[0] == "component" || tok[0] == "port" ||
             tok[0] == "thruster" || tok[0] == "wheel" ||
             tok[0] == "magnetorquer")) {
            memset(&cur, 0, sizeof cur);
            cur.q[0] = 1.0;
            have_mesh = false;
            pending_mesh.clear();
            pending.clear();
            if (tok[0] == "component") {
                blk = BLK_COMPONENT;
            } else if (tok[0] == "port") {
                blk = BLK_PORT;
                port = Port();
                port.name = tok.size() >= 2 ? tok[1] : "";
            } else if (tok[0] == "thruster") {
                blk = BLK_THRUSTER;
                thruster = Thruster();
                thruster.name = tok.size() >= 2 ? tok[1] : "";
            } else {
                blk = BLK_OTHER;
            }
            continue;
        }
        if (tok[0] == "end") {
            if (blk == BLK_COMPONENT) {
                double rot[9];
                if (have_mesh) {
                    mesh_paths.push_back(pending_mesh);
                    placements.push_back(cur);
                }
                placement_matrix_(cur, rot);
                for (size_t k = 0; k < pending.size(); k++) {
                    Collider c = pending[k];
                    double w[3];
                    memcpy(c.rot, rot, sizeof c.rot);
                    if (c.kind == COLLIDER_BOX) {
                        /* A box's centre is its component's origin and
                         * its half extents lie along the component's
                         * own axes, which `rot` carries. */
                        for (int q = 0; q < 3; q++)
                            c.centre[q] = cur.at[q];
                    } else if (c.kind == COLLIDER_SPHERE) {
                        rotate_(cur, c.a, w);
                        for (int q = 0; q < 3; q++)
                            c.centre[q] = cur.at[q] + w[q];
                    } else {
                        double wb[3];
                        rotate_(cur, c.a, w);
                        rotate_(cur, c.b, wb);
                        for (int q = 0; q < 3; q++) {
                            c.a[q] = cur.at[q] + w[q];
                            c.b[q] = cur.at[q] + wb[q];
                            c.centre[q] = 0.5 * (c.a[q] + c.b[q]);
                        }
                    }
                    a.colliders.push_back(c);
                }
            } else if (blk == BLK_PORT) {
                /* The envelope's own figures, from the compiler's
                 * table: an unknown name is left unresolved rather
                 * than filled with a guess, and the port then draws
                 * its axis and its roll reference alone. */
                if (!port.envelope.empty()) {
                    const KflcCaptureEnvelope *env =
                        kflc_capture_envelope(port.envelope.c_str());
                    if (env) {
                        port.has_envelope = true;
                        port.mating_diameter =
                            kflc_capture_mm_to_m(env->mating_diameter_mm);
                        port.lateral_limit = env->lateral;
                        port.pitchyaw_limit_deg = env->pitchyaw_deg;
                    }
                }
                a.ports.push_back(port);
            } else if (blk == BLK_THRUSTER) {
                a.thrusters.push_back(thruster);
            }
            blk = BLK_NONE;
            have_mesh = false;
            pending.clear();
            continue;
        }
        if (blk == BLK_COMPONENT) {
            if (tok[0] == "at" && tok.size() >= 4) {
                for (int c = 0; c < 3; c++) {
                    if (!number_(tok[1 + c], &cur.at[c])) {
                        a.error = "malformed `at` in `" + path + "`";
                        return a;
                    }
                }
            } else if (tok[0] == "rotate" && tok.size() >= 5) {
                for (int c = 0; c < 4; c++) {
                    if (!number_(tok[1 + c], &cur.q[c])) {
                        a.error = "malformed `rotate` in `" + path + "`";
                        return a;
                    }
                }
            } else if (tok[0] == "mesh" && tok.size() >= 2) {
                /* A second mesh line replaces the first, which is what
                 * the compiler's own single-valued field does. */
                pending_mesh = tok[1];
                have_mesh = true;
            } else if (tok[0] == "collider" && tok.size() >= 2) {
                Collider c;
                size_t need;
                memset(&c, 0, sizeof c);
                if (tok[1] == "sphere") {
                    c.kind = COLLIDER_SPHERE;
                    need = 6;
                } else if (tok[1] == "capsule") {
                    c.kind = COLLIDER_CAPSULE;
                    need = 9;
                } else if (tok[1] == "box") {
                    c.kind = COLLIDER_BOX;
                    need = 5;
                } else {
                    a.error = "`" + tok[1] + "` is not a collider shape in `" +
                              path + "`";
                    return a;
                }
                if (tok.size() < need) {
                    a.error = "malformed `collider` in `" + path + "`";
                    return a;
                }
                bool ok = true;
                for (int q = 0; q < 3; q++)
                    ok = ok && number_(tok[2 + q], &c.a[q]);
                if (c.kind == COLLIDER_CAPSULE) {
                    for (int q = 0; q < 3; q++)
                        ok = ok && number_(tok[5 + q], &c.b[q]);
                    ok = ok && number_(tok[8], &c.radius);
                } else if (c.kind == COLLIDER_SPHERE) {
                    ok = ok && number_(tok[5], &c.radius);
                }
                if (!ok) {
                    a.error = "malformed `collider` in `" + path + "`";
                    return a;
                }
                pending.push_back(c);
            }
        } else if (blk == BLK_PORT) {
            double *vec = tok[0] == "at" ? port.at
                        : tok[0] == "axis" ? port.axis
                        : tok[0] == "roll_ref" ? port.roll_ref : 0;
            if (vec && tok.size() >= 4) {
                for (int c = 0; c < 3; c++) {
                    if (!number_(tok[1 + c], &vec[c])) {
                        a.error = "malformed `" + tok[0] + "` in `" + path +
                                  "`";
                        return a;
                    }
                }
            } else if (tok[0] == "capture" && tok.size() >= 2) {
                port.envelope = tok[1];
            }
        } else if (blk == BLK_THRUSTER) {
            double *vec = tok[0] == "at" ? thruster.at
                        : tok[0] == "dir" ? thruster.dir : 0;
            if (vec && tok.size() >= 4) {
                for (int c = 0; c < 3; c++) {
                    if (!number_(tok[1 + c], &vec[c])) {
                        a.error = "malformed `" + tok[0] + "` in `" + path +
                                  "`";
                        return a;
                    }
                }
            } else if (tok[0] == "thrust" && tok.size() >= 2) {
                if (!number_(tok[1], &thruster.thrust)) {
                    a.error = "malformed `thrust` in `" + path + "`";
                    return a;
                }
            }
        }
    }

    /* Each mesh contributes its bytes to the digest in the order its
     * component was declared, which is the order the compiler
     * contributes them, and its geometry to the wireframe. */
    std::map<std::pair<uint32_t, uint32_t>, int> seen;
    for (size_t m = 0; m < mesh_paths.size(); m++) {
        std::string mpath = resolve_(mesh_paths[m], path);
        std::vector<uint8_t> mbytes;
        if (!read_file_(mpath, &mbytes)) {
            a.error = "cannot open mesh `" + mpath + "`";
            return a;
        }
        k26rl_digest_add(&dig, mbytes.empty() ? 0 : &mbytes[0],
                         (uint64_t)mbytes.size());
        a.meshes.push_back(mpath);

        /* Two passes over the mesh, because the compiler makes two:
         * it counts the vertices over the whole file first and bounds
         * every face index by that total, so a mesh whose faces are
         * written before its vertices is a mesh it accepts. A reader
         * that bounded by the vertices it had seen so far would
         * refuse a file the compiler compiled. */
        std::vector<std::string> mlines;
        lines_(mbytes, &mlines);
        uint32_t base = (uint32_t)(a.vertices.size() / 3);
        uint32_t local = 0;
        for (size_t i = 0; i < mlines.size(); i++) {
            tokens_(mlines[i], &tok);
            if (tok.empty())
                continue;
            if (tok[0] == "v" && tok.size() >= 4) {
                double v[3], w[3];
                for (int c = 0; c < 3; c++) {
                    if (!number_(tok[1 + c], &v[c])) {
                        a.error = "malformed vertex in `" + mpath + "`";
                        return a;
                    }
                }
                rotate_(placements[m], v, w);
                for (int c = 0; c < 3; c++)
                    a.vertices.push_back(w[c] + placements[m].at[c]);
                local++;
                a.mesh_vertices++;
            } else if (tok[0] != "f") {
                a.error = "`" + tok[0] + "` is not a mesh directive in `" +
                          mpath + "`";
                return a;
            }
        }
        for (size_t i = 0; i < mlines.size(); i++) {
            tokens_(mlines[i], &tok);
            if (tok.empty() || tok[0] != "f")
                continue;
            if (tok.size() < 4) {
                a.error = "malformed face in `" + mpath + "`";
                return a;
            }
            double idx[3];
            for (int c = 0; c < 3; c++) {
                if (!number_(tok[1 + c], &idx[c]) || idx[c] < 1.0 ||
                    idx[c] > (double)local) {
                    a.error = "malformed face in `" + mpath + "`";
                    return a;
                }
            }
            uint32_t f[3];
            Face face;
            for (int c = 0; c < 3; c++)
                f[c] = base + (uint32_t)idx[c] - 1u;
            add_edge_(&seen, &a.edges, f[0], f[1]);
            add_edge_(&seen, &a.edges, f[1], f[2]);
            add_edge_(&seen, &a.edges, f[2], f[0]);
            face.a = f[0];
            face.b = f[1];
            face.c = f[2];
            a.faces.push_back(face);
            a.mesh_triangles++;
        }
    }

    k26rl_digest_final(&dig, a.digest);
    a.loaded = true;
    return a;
}

}  /* namespace k26rl_view */
