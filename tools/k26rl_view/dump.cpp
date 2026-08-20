/* dump.cpp - the headless presenter.
 *
 * Writes the values behind a named panel, for a named episode and
 * step range, without opening a window. It exists so the panels are
 * testable: a panel whose numbers nothing checks is an assertion, and
 * the interface and this dump read the same model, so what is checked
 * here is what the interface draws.
 *
 * The scene panel is here on the same terms and for a stronger
 * reason: everything between a body's state and a pixel lives in the
 * model, and this is what reaches it without a display.
 *
 * Every double is written as the sixteen hex digits of its IEEE-754
 * binary64 bit pattern. The format's own rule is that values travel
 * as bit patterns and never through decimal text, for the last-ULP
 * reason recorded when the episode format was decided; a dump whose
 * whole purpose is bitwise comparison keeps that rule rather than
 * inviting an argument about decimal digits.
 */
#include "dump.h"

#include "asset.h"
#include "scene.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

namespace k26rl_view {

static void hx(FILE *f, double v)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);
    fprintf(f, "%016" PRIx64, bits);
}

/* The trajectory panel's standing label, shortened to the limit it
 * carries. Each channel's own mode is a reading beside it, so the
 * label no longer repeats that the modes are published; what it keeps
 * is the condition under which the reconstruction is exact, which is
 * a fact no reading on the panel states. */
const char *const TRAJECTORY_LABEL =
    "observer-relative; exact under a geometric observe, otherwise an "
    "apparent direction at a geometric range";

/* A trajectory's mode is its channels' mode, which the five channels
 * of one observe share. A file written before the tag existed carries
 * none, and this says so rather than guessing. */
static const char *trajectory_mode_name(Model &m, const Trajectory &t)
{
    const std::vector<Channel> &ch = m.spec().channels;

    for (size_t i = 0; i < ch.size(); i++) {
        if (ch[i].index == t.dir_x)
            return ch[i].has_mode ? observer_mode_name(ch[i].mode)
                                  : "unpublished";
    }
    return "unpublished";
}

static void range_for(const Episode &ep, const DumpOptions &o,
                      uint32_t *lo, uint32_t *hi)
{
    *lo = o.step_lo;
    *hi = (o.step_hi == UINT32_MAX) ? ep.step_count : o.step_hi;
    if (*hi > ep.step_count)
        *hi = ep.step_count;
    if (*lo > *hi)
        *lo = *hi;
}

static void dump_meta(FILE *f, Model &m)
{
    const FileInfo &i = m.info();
    const Spec &s = m.spec();

    /* Which source produced this, said once and first. A ring has no
     * path and no length, so it says what it is instead of reporting
     * a file's fields with nothing behind them. */
    if (i.live) {
        fprintf(f, "source tap %s\n", i.tap.c_str());
        fprintf(f, "ring_slot_size %u\n", m.ring_slot_size());
        fprintf(f, "ring_slot_count %u\n", m.ring_slot_count());
        fprintf(f, "frames_accepted %" PRIu64 "\n", m.frames_accepted());
        fprintf(f, "frames_lost %" PRIu64 "\n", m.frames_lost());
        fprintf(f, "producer_closed %d\n", m.producer_closed() ? 1 : 0);
    } else {
        fprintf(f, "file %s\n", i.path.c_str());
    }
    fprintf(f, "file_bytes %" PRIu64 "\n", i.file_bytes);
    fprintf(f, "clean_close %d\n", i.clean_close ? 1 : 0);
    fprintf(f, "readable_bytes %" PRIu64 "\n", i.readable_bytes);
    if (!i.clean_close) {
        /* A file cut mid-write reads to its last complete frame; the
         * truncation point is where the valid prefix ended. */
        fprintf(f, "truncation_point %" PRIu64 "\n", i.readable_bytes);
    }
    fprintf(f, "unindexed_starts %u\n", i.unindexed_episode_starts);
    fprintf(f, "format_version %u\n", i.format_version);
    fprintf(f, "governing_seed %016" PRIx64 "\n", i.governing_seed);
    fprintf(f, "rekey_ordinal %u\n", i.rekey_ordinal);
    fprintf(f, "n_envs %u\n", i.n_envs);
    fprintf(f, "steps_per_chunk %u\n", i.steps_per_chunk);
    fprintf(f, "episode_count %u\n", i.episode_count);

    fprintf(f, "spec abi_version %08x\n", s.abi_version);
    fprintf(f, "spec endian_probe %08x\n", s.endian_probe);
    fprintf(f, "spec agent_count %u\n", s.agent_count);
    fprintf(f, "spec n_envs %u\n", s.n_envs);
    fprintf(f, "spec horizon %u\n", s.horizon);
    fprintf(f, "spec obs_total %u\n", s.obs_total);
    fprintf(f, "spec act_total %u\n", s.act_total);
    fprintf(f, "spec episode_flags %08x\n", s.episode_flags);
    fprintf(f, "spec control_dt ");
    hx(f, s.control_dt);
    fprintf(f, "\n");
    fprintf(f, "spec raw_len %u\n", (unsigned)s.raw.size());
    for (size_t k = 0; k < s.channels.size(); k++) {
        fprintf(f, "channel %u %u %s\n", s.channels[k].index,
                (unsigned)s.channels[k].kind, s.channels[k].name.c_str());
    }
    for (size_t k = 0; k < s.obs_slices.size(); k++) {
        fprintf(f, "obs_slice %u %u %u\n", s.obs_slices[k].agent,
                s.obs_slices[k].offset, s.obs_slices[k].count);
    }
    for (size_t k = 0; k < s.act_slices.size(); k++) {
        fprintf(f, "act_slice %u %u %u\n", s.act_slices[k].agent,
                s.act_slices[k].offset, s.act_slices[k].count);
    }
    for (size_t k = 0; k < s.actions.size(); k++) {
        const ActionDecl &a = s.actions[k];
        fprintf(f, "action %u %u %u ", a.offset, (unsigned)a.kind, a.arity);
        hx(f, a.lo);
        fprintf(f, " ");
        hx(f, a.hi);
        fprintf(f, "\n");
    }
    for (size_t k = 0; k < s.unknown_tags.size(); k++) {
        fprintf(f, "unknown_tag %u %u\n", (unsigned)s.unknown_tags[k].first,
                s.unknown_tags[k].second);
    }
    fprintf(f, "trajectory_count %u\n", (unsigned)m.trajectories().size());
    for (size_t k = 0; k < m.trajectories().size(); k++) {
        const Trajectory &t = m.trajectories()[k];
        fprintf(f, "trajectory %s %u %u %u %u\n", t.base.c_str(), t.dir_x,
                t.dir_y, t.dir_z, t.range);
        fprintf(f, "trajectory_mode %s %s\n", t.base.c_str(),
                trajectory_mode_name(m, t));
    }
    for (size_t k = 0; k < m.spec().body_names.size(); k++)
        fprintf(f, "body %u %s\n", (unsigned)k,
                m.spec().body_names[k].c_str());
    fprintf(f, "trajectory_label %s\n", TRAJECTORY_LABEL);
}

/* Every episode panel begins with this, so what an episode is missing
 * travels with the episode rather than sitting in one panel a reader
 * may not have asked for.
 *
 * A recording read from a file is missing nothing and these lines are
 * absent from it, which is what makes a live dump comparable with the
 * file the same run wrote: identical when the viewer kept up, and
 * different by exactly these records when it did not. A viewer that
 * drew through a hole without one of these would be drawing a path
 * the craft never flew. */
static void dump_episode_header(FILE *f, uint32_t k, const Episode &e)
{
    fprintf(f, "episode %u %u %u %u %u %u %s %u %016" PRIx64 "\n", k,
            e.ordinal, e.env, e.episode, e.step_count, e.transitions(),
            e.complete ? end_reason_name(e.end_reason) : "open",
            (unsigned)e.fault_code, e.seed);
    if (!e.start_seen) {
        /* The opening frame never arrived, so this episode has no
         * initial observation and no randomisation record to report
         * and does not present its earliest held values as them. */
        fprintf(f, "episode_start_missing %u\n", k);
    }
    for (size_t g = 0; g < e.gaps.size(); g++) {
        fprintf(f, "episode_gap %u %u %u\n", k, e.gaps[g].first,
                e.gaps[g].count);
    }
}

static void dump_timeline(FILE *f, Model &m, const DumpOptions &o)
{
    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;
        uint32_t lo, hi;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e) {
            fprintf(f, "error %s\n", err.c_str());
            continue;
        }
        dump_episode_header(f, k, *e);
        range_for(*e, o, &lo, &hi);
        for (uint32_t i = lo; i < hi; i++) {
            fprintf(f, "flag %u %u %08x\n", k, e->step_at(i),
                    i < e->flags.size() ? e->flags[i] : 0u);
            fprintf(f, "dt %u %u ", k, e->step_at(i));
            hx(f, i < e->applied_dt.size() ? e->applied_dt[i] : 0.0);
            fprintf(f, "\n");
        }
    }
}

static void dump_reward(FILE *f, Model &m, const DumpOptions &o)
{
    const uint32_t agents = m.spec().agent_count ? m.spec().agent_count : 1;

    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;
        uint32_t lo, hi;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e)
            continue;
        dump_episode_header(f, k, *e);
        std::vector<double> ret = Model::returns(*e, agents);
        range_for(*e, o, &lo, &hi);
        for (uint32_t i = lo; i < hi; i++) {
            for (uint32_t a = 0; a < agents; a++) {
                size_t idx = (size_t)i * agents + a;
                fprintf(f, "reward %u %u %u ", k, e->step_at(i), a);
                hx(f, idx < e->rewards.size() ? e->rewards[idx] : 0.0);
                fprintf(f, "\n");
                fprintf(f, "return %u %u %u ", k, e->step_at(i), a);
                hx(f, idx < ret.size() ? ret[idx] : 0.0);
                fprintf(f, "\n");
            }
        }
        /* The terminal adjustment arrives with the closing record, so
         * an episode still running has none and is not given one. */
        if (e->complete) {
            for (uint32_t a = 0; a < agents; a++) {
                fprintf(f, "terminal_adj %u %u ", k, a);
                hx(f, a < e->terminal_adjustments.size()
                       ? e->terminal_adjustments[a] : 0.0);
                fprintf(f, "\n");
            }
        }
    }
}

static void dump_obs(FILE *f, Model &m, const DumpOptions &o)
{
    const uint32_t obs_total = m.spec().obs_total;

    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;
        uint32_t lo, hi;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e)
            continue;
        dump_episode_header(f, k, *e);
        /* An episode whose opening record never arrived has no
         * initial observation and no randomisation draws to report.
         * Printing zeros for them would be presenting a value the
         * episode never had, which the header has already said is
         * unknown. */
        if (e->start_seen) {
            for (uint32_t j = 0; j < obs_total; j++) {
                fprintf(f, "initial_obs %u %u ", k, j);
                hx(f, j < e->initial_obs.size() ? e->initial_obs[j] : 0.0);
                fprintf(f, "\n");
            }
            for (uint32_t j = 0; j < e->dr_tags.size(); j++) {
                fprintf(f, "dr %u %u %u ", k, j, e->dr_tags[j]);
                hx(f, e->dr_values[j]);
                fprintf(f, "\n");
            }
        }
        range_for(*e, o, &lo, &hi);
        for (uint32_t i = lo; i < hi; i++) {
            for (uint32_t j = 0; j < obs_total; j++) {
                size_t idx = (size_t)i * obs_total + j;
                fprintf(f, "obs %u %u %u ", k, e->step_at(i), j);
                hx(f, idx < e->obs.size() ? e->obs[idx] : 0.0);
                fprintf(f, "\n");
            }
        }
    }
}

static void dump_action(FILE *f, Model &m, const DumpOptions &o)
{
    const uint32_t act_total = m.spec().act_total;
    const Spec &s = m.spec();

    for (size_t k = 0; k < s.actions.size(); k++) {
        const ActionDecl &a = s.actions[k];
        fprintf(f, "action_bound %u %u %u ", a.offset, (unsigned)a.kind,
                a.arity);
        hx(f, a.lo);
        fprintf(f, " ");
        hx(f, a.hi);
        fprintf(f, "\n");
    }
    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;
        uint32_t lo, hi;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e)
            continue;
        dump_episode_header(f, k, *e);
        range_for(*e, o, &lo, &hi);
        for (uint32_t i = lo; i < hi; i++) {
            for (uint32_t j = 0; j < act_total; j++) {
                size_t idx = (size_t)i * act_total + j;
                fprintf(f, "act %u %u %u ", k, e->step_at(i), j);
                hx(f, idx < e->act.size() ? e->act[idx] : 0.0);
                fprintf(f, "\n");
            }
        }
    }
}

static void dump_traj(FILE *f, Model &m, const DumpOptions &o)
{
    fprintf(f, "trajectory_count %u\n", (unsigned)m.trajectories().size());
    fprintf(f, "trajectory_label %s\n", TRAJECTORY_LABEL);
    for (size_t t = 0; t < m.trajectories().size(); t++) {
        const Trajectory &tr = m.trajectories()[t];
        fprintf(f, "trajectory %s %u %u %u %u\n", tr.base.c_str(), tr.dir_x,
                tr.dir_y, tr.dir_z, tr.range);
        fprintf(f, "trajectory_mode %s %s\n", tr.base.c_str(),
                trajectory_mode_name(m, tr));
    }
    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;
        uint32_t lo, hi;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e)
            continue;
        dump_episode_header(f, k, *e);
        range_for(*e, o, &lo, &hi);
        for (size_t t = 0; t < m.trajectories().size(); t++) {
            const Trajectory &tr = m.trajectories()[t];
            for (uint32_t i = lo; i < hi; i++) {
                double xyz[3];
                Model::point(*e, m.spec(), tr, i, xyz);
                fprintf(f, "traj %s %u %u ", tr.base.c_str(), k,
                        e->step_at(i));
                hx(f, xyz[0]);
                fprintf(f, " ");
                hx(f, xyz[1]);
                fprintf(f, " ");
                hx(f, xyz[2]);
                fprintf(f, "\n");
            }
        }
    }
}

/* The scrubbing claim, made countable. A seek to a step of the
 * episode already loaded costs no reader traffic at all; a seek that
 * changes episode costs exactly one index lookup and one bounded
 * read. The counters are printed beside every sample so both halves
 * are visible in the output rather than inferred from it. */
static void dump_scrub(FILE *f, Model &m, const DumpOptions &o)
{
    (void)o;
    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e = m.load(k, &err);
        uint32_t marks[3];
        if (!e)
            continue;
        marks[0] = 0;
        marks[1] = e->step_count ? e->step_count / 2 : 0;
        marks[2] = e->step_count ? e->step_count - 1 : 0;
        for (int s = 0; s < 3; s++) {
            const Episode *again = m.load(k, &err);
            size_t idx = (size_t)marks[s] * m.spec().obs_total;
            fprintf(f, "scrub_sample %u %u %" PRIu64 " %" PRIu64 " ", k,
                    e->step_at(marks[s]), m.index_lookups(),
                    m.episode_reads());
            hx(f, (again && idx < again->obs.size()) ? again->obs[idx] : 0.0);
            fprintf(f, "\n");
        }
    }
    fprintf(f, "scrub_index_lookups %" PRIu64 "\n", m.index_lookups());
    fprintf(f, "scrub_episode_reads %" PRIu64 "\n", m.episode_reads());
}

/* One rebuild per episode, shared by the panels that need one.
 *
 * The world frame, the attitude panel and the re-simulation verdict
 * are three views of one rebuild, and `--dump all` asked for it three
 * times over: the same artifact loaded, the same episode driven, the
 * same numbers produced, three times. The cache is one deep because
 * the panels walk the episodes in the same order, and it is keyed on
 * the episode so a second panel asking for a different one still gets
 * its own rebuild rather than the previous panel's. */
static ResimResult rebuild_(Model &m, const Episode &e, uint32_t k,
                            const DumpOptions &o, uint32_t reference)
{
    static ResimResult cached;
    static uint32_t cached_k = UINT32_MAX;
    static uint32_t cached_ref = UINT32_MAX;
    static std::string cached_artifact;
    static bool have = false;

    /* The reference is part of the key because it changes the answer:
     * two panels asking for two frames must not share one rebuild. */
    if (have && cached_k == k && cached_ref == reference &&
        cached_artifact == o.artifact)
        return cached;
    cached = resimulate(m, e, o.artifact, reference);
    cached_k = k;
    cached_ref = reference;
    cached_artifact = o.artifact;
    have = true;
    return cached;
}

/* The world frame, which only re-simulation can produce: the file
 * records observation channels, and the bodies come from the
 * artifact's body getter as the rebuild runs. */
static void dump_world(FILE *f, Model &m, const DumpOptions &o)
{
    if (o.artifact.empty()) {
        fprintf(f, "world unavailable no artifact supplied\n");
        return;
    }
    for (size_t k = 0; k < m.spec().body_names.size(); k++)
        fprintf(f, "body %u %s\n", (unsigned)k,
                m.spec().body_names[k].c_str());
    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e)
            continue;
        dump_episode_header(f, k, *e);
        ResimResult r = rebuild_(m, *e, k, o, K26RL_BODY_REF_ORIGIN);
        if (!r.ran || !r.has_bodies) {
            fprintf(f, "world_unavailable %u %s\n", k,
                    r.ran ? "artifact publishes no body getter"
                          : r.message.c_str());
            continue;
        }
        fprintf(f, "world_bodies %u %u\n", k, r.body_count);
        for (uint32_t i = 0; i < r.steps_compared; i++) {
            for (uint32_t b = 0; b < r.body_count; b++) {
                size_t base = ((size_t)i * r.body_count + b) * 6;
                if (base + 6 > r.bodies.size())
                    break;
                fprintf(f, "world %u %u %u", k, i, b);
                for (int c = 0; c < 6; c++) {
                    fprintf(f, " ");
                    hx(f, r.bodies[base + c]);
                }
                fprintf(f, "\n");
            }
        }
    }
}

/* The attitude panel. Like the world frame it exists only with an
 * artifact: an episode file records observation channels, and a
 * body's orientation is not one of them unless the programme happened
 * to declare an attitude observe. The values here are the artifact's
 * own getter, read as the rebuild runs. */
static void dump_attitude(FILE *f, Model &m, const DumpOptions &o)
{
    if (o.artifact.empty()) {
        fprintf(f, "attitude unavailable no artifact supplied\n");
        return;
    }
    for (size_t k = 0; k < m.spec().body_names.size(); k++)
        fprintf(f, "body %u %s\n", (unsigned)k,
                m.spec().body_names[k].c_str());
    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e)
            continue;
        dump_episode_header(f, k, *e);
        ResimResult r = rebuild_(m, *e, k, o, K26RL_BODY_REF_ORIGIN);
        if (!r.ran || !r.has_attitudes) {
            fprintf(f, "attitude_unavailable %u %s\n", k,
                    r.ran ? "artifact publishes no attitude getter"
                          : r.message.c_str());
            continue;
        }
        fprintf(f, "attitude_bodies %u %u\n", k, r.body_count);
        for (uint32_t i = 0; i < r.steps_compared; i++) {
            for (uint32_t b = 0; b < r.body_count; b++) {
                size_t base = ((size_t)i * r.body_count + b) * 7;
                if (base + 7 > r.attitudes.size())
                    break;
                fprintf(f, "attitude %u %u %u", k, i, b);
                for (int c = 0; c < 7; c++) {
                    fprintf(f, " ");
                    hx(f, r.attitudes[base + c]);
                }
                fprintf(f, "\n");
            }
        }
    }
}

/* The ground truth beside the measurement. The pairing is the file's
 * own statement, read from the channel-source tag: which channels a
 * sensor corrupted, and which channel holds what the corruption was
 * applied to. Nothing here parses a name to find a pair. */
static void dump_overlay(FILE *f, Model &m, const DumpOptions &o)
{
    const Spec &sp = m.spec();
    std::vector<std::pair<uint32_t, uint32_t> > pairs = m.overlay_pairs();

    if (pairs.empty()) {
        fprintf(f, "overlay_none no paired channels in this file\n");
        return;
    }
    for (size_t i = 0; i < pairs.size(); i++) {
        const char *mn = "?", *tn = "?";
        for (size_t c = 0; c < sp.channels.size(); c++) {
            if (sp.channels[c].index == pairs[i].first)
                mn = sp.channels[c].name.c_str();
            if (sp.channels[c].index == pairs[i].second)
                tn = sp.channels[c].name.c_str();
        }
        fprintf(f, "overlay_pair %u %u %s %s\n", pairs[i].first,
                pairs[i].second, mn, tn);
    }
    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;
        uint32_t lo, hi;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e)
            continue;
        dump_episode_header(f, k, *e);
        range_for(*e, o, &lo, &hi);
        for (uint32_t i = lo; i < hi; i++) {
            for (size_t p = 0; p < pairs.size(); p++) {
                size_t base = (size_t)i * sp.obs_total;
                if (base + sp.obs_total > e->obs.size())
                    break;
                fprintf(f, "overlay %u %u %u ", k, e->step_at(i),
                        (unsigned)p);
                hx(f, e->obs[base + pairs[p].first]);
                fprintf(f, " ");
                hx(f, e->obs[base + pairs[p].second]);
                fprintf(f, "\n");
            }
        }
    }
}

/* The craft's own wireframe, and the check that makes drawing it
 * honest. The artifact carries the digest of the asset bytes it was
 * built from; this recomputes that digest over the asset on disk and
 * refuses to draw a craft whose bytes differ, because a wireframe
 * beside a recording is a claim about what flew. */
static void dump_wireframe(FILE *f, Model &m, const DumpOptions &o)
{
    const Spec &sp = m.spec();

    for (size_t i = 0; i < sp.assemblies.size(); i++) {
        const AssemblyRef &a = sp.assemblies[i];
        fprintf(f, "assembly %u %s %s\n", a.body,
                a.name.empty() ? "?" : a.name.c_str(),
                a.has_digest ? digest_hex(a.digest).c_str() : "nodigest");
    }
    if (sp.assemblies.empty())
        fprintf(f, "assembly_none no body in this file binds an assembly\n");
    if (o.assets.empty()) {
        fprintf(f, "wireframe unavailable no asset supplied\n");
        return;
    }

    /* One record set per supplied assembly, keyed by the request's
     * position, because a recording of two craft supplies two and a
     * reader has to be able to tell them apart. Which body each
     * belongs to, and whether its bytes are the recorded bytes, are
     * one question asked in one place, so the window and this dump
     * cannot answer it differently. */
    std::vector<AssetBound> bound = asset_bind(sp, o.assets);
    for (size_t k = 0; k < bound.size(); k++) {
        const Asset &as = bound[k].asset;
        const AssemblyRef *match = 0;
        unsigned n = (unsigned)k;

        fprintf(f, "wireframe_request %u %s %s\n", n,
                bound[k].request.name.empty() ? "-"
                                              : bound[k].request.name.c_str(),
                bound[k].request.path.c_str());
        if (!as.loaded) {
            fprintf(f, "wireframe_error %u %s\n", n, as.error.c_str());
            continue;
        }
        fprintf(f, "wireframe_asset %u %s %s\n", n, as.name.c_str(),
                digest_hex(as.digest).c_str());
        fprintf(f, "wireframe_meshes %u %u\n", n, (unsigned)as.meshes.size());
        if (bound[k].request.body == ASSET_UNBOUND)
            asset_verdict(sp, as, &match);
        else
            asset_verdict_at(sp, as, bound[k].request.body, &match);
        if (bound[k].verdict == ASSET_NO_BODY) {
            fprintf(f, "wireframe_digest %u unmatched %s\n", n,
                    as.name.c_str());
            continue;
        }
        fprintf(f, "wireframe_body %u %u %s\n", n, match->body,
                match->body < sp.body_names.size()
                    ? sp.body_names[match->body].c_str() : "?");
        if (bound[k].verdict == ASSET_NO_DIGEST) {
            fprintf(f, "wireframe_digest %u absent\n", n);
            continue;
        }
        if (bound[k].verdict == ASSET_MISMATCH) {
            /* Reported, and not drawn. The bytes on disk are not the
             * bytes that flew, and a picture of them would be a
             * picture of a different craft. */
            fprintf(f, "wireframe_digest %u mismatch %s\n", n,
                    digest_hex(match->digest).c_str());
            continue;
        }
        fprintf(f, "wireframe_digest %u match %s\n", n,
                digest_hex(match->digest).c_str());
        fprintf(f, "wireframe_counts %u %u %u %u\n", n, as.mesh_vertices,
                (unsigned)as.edges.size(), as.mesh_triangles);
        for (size_t i = 0; i * 3 + 2 < as.vertices.size(); i++) {
            fprintf(f, "wireframe_vertex %u %u", n, (unsigned)i);
            for (int c = 0; c < 3; c++) {
                fprintf(f, " ");
                hx(f, as.vertices[i * 3 + c]);
            }
            fprintf(f, "\n");
        }
        for (size_t i = 0; i < as.edges.size(); i++)
            fprintf(f, "wireframe_edge %u %u %u %u\n", n, (unsigned)i,
                    as.edges[i].a, as.edges[i].b);
    }
}

/* The per-agent panel groups, and the one thing about them a gate has
 * to be able to fail on.
 *
 * Which channels and which action offsets belong to an agent comes
 * from the spec's own slice tags. It never comes from the channel
 * names, even though the names carry an agent prefix and reading the
 * prefix would be easier: two agents may declare a channel of one
 * name, and a viewer that grouped by prefix would then disagree with
 * the ABI about which agent owns which number. This writes both, the
 * slice the tags gave and the name each channel publishes, so a
 * comparison can show that the first decided and the second did not.
 */
static void dump_agents(FILE *f, Model &m, const DumpOptions &o)
{
    const Spec &sp = m.spec();
    std::vector<AgentGroup> g = m.agent_groups();

    fprintf(f, "agents %u\n", (unsigned)g.size());
    for (size_t a = 0; a < g.size(); a++) {
        fprintf(f, "agent %u %s\n", g[a].index,
                g[a].name.empty() ? "-" : g[a].name.c_str());
        fprintf(f, "agent_obs_slice %u %u %u %d\n", g[a].index,
                g[a].obs_offset, g[a].obs_count, g[a].has_obs_slice ? 1 : 0);
        fprintf(f, "agent_act_slice %u %u %u %d\n", g[a].index,
                g[a].act_offset, g[a].act_count, g[a].has_act_slice ? 1 : 0);
        for (size_t c = 0; c < g[a].channels.size(); c++) {
            std::string nm;
            for (size_t q = 0; q < sp.channels.size(); q++) {
                if (sp.channels[q].index == g[a].channels[c])
                    nm = sp.channels[q].name;
            }
            fprintf(f, "agent_channel %u %u %s\n", g[a].index,
                    g[a].channels[c], nm.empty() ? "-" : nm.c_str());
        }
        for (uint32_t j = 0; j < g[a].act_count; j++) {
            const ActionDecl *d = 0;
            uint32_t off = g[a].act_offset + j;
            for (size_t q = 0; q < sp.actions.size(); q++) {
                if (sp.actions[q].offset == off)
                    d = &sp.actions[q];
            }
            fprintf(f, "agent_action %u %u %u %u ", g[a].index, off,
                    d && d->has_kind ? (unsigned)d->kind : 0u,
                    d ? d->arity : 0u);
            hx(f, d && d->has_bounds ? d->lo : 0.0);
            fprintf(f, " ");
            hx(f, d && d->has_bounds ? d->hi : 0.0);
            fprintf(f, "\n");
        }
    }

    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;
        uint32_t lo, hi;
        uint32_t agents = sp.agent_count ? sp.agent_count : 1;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e)
            continue;
        dump_episode_header(f, k, *e);
        range_for(*e, o, &lo, &hi);
        for (size_t a = 0; a < g.size(); a++) {
            std::vector<double> ret = Model::returns(*e, agents);
            for (uint32_t i = lo; i < hi; i++) {
                size_t j = (size_t)i * agents + g[a].index;
                fprintf(f, "agent_reward %u %u %u ", k, g[a].index,
                        e->step_at(i));
                hx(f, j < e->rewards.size() ? e->rewards[j] : 0.0);
                fprintf(f, " ");
                hx(f, j < ret.size() ? ret[j] : 0.0);
                fprintf(f, "\n");
            }
            fprintf(f, "agent_terminal %u %u ", k, g[a].index);
            hx(f, g[a].index < e->terminal_adjustments.size()
                   ? e->terminal_adjustments[g[a].index] : 0.0);
            fprintf(f, "\n");
        }
    }
}

static void dump_scene_header_(FILE *f, Model &m, const DumpOptions &o)
{
    const Spec &sp = m.spec();
    const SceneOptions &s = o.scene;
    const char *mode = s.camera.mode == CAMERA_ORBIT ? "orbit"
                     : s.camera.mode == CAMERA_CHASE ? "chase" : "free";

    fprintf(f, "scene_frame %s\n", scene_body_name(sp, s.frame).c_str());
    fprintf(f, "scene_camera %s %s\n", mode,
            scene_body_name(sp, s.camera.target).c_str());
    if (s.camera.mode == CAMERA_ORBIT) {
        fprintf(f, "scene_camera_orbit ");
        hx(f, s.camera.azimuth_deg);
        fprintf(f, " ");
        hx(f, s.camera.elevation_deg);
        fprintf(f, " ");
        hx(f, s.camera.radius);
        fprintf(f, "\n");
    } else if (s.camera.mode == CAMERA_CHASE) {
        fprintf(f, "scene_camera_chase");
        for (int c = 0; c < 3; c++) {
            fprintf(f, " ");
            hx(f, s.camera.chase[c]);
        }
        fprintf(f, "\n");
    } else {
        fprintf(f, "scene_camera_free");
        for (int c = 0; c < 3; c++) {
            fprintf(f, " ");
            hx(f, s.camera.eye[c]);
        }
        for (int c = 0; c < 3; c++) {
            fprintf(f, " ");
            hx(f, s.camera.look[c]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "scene_camera_up");
    for (int c = 0; c < 3; c++) {
        fprintf(f, " ");
        hx(f, s.camera.up[c]);
    }
    fprintf(f, "\n");
    fprintf(f, "scene_projection %s ",
            s.camera.projection == PROJECTION_PERSPECTIVE ? "perspective"
                                                          : "orthographic");
    hx(f, s.camera.projection == PROJECTION_PERSPECTIVE
           ? s.camera.fov_y_deg : s.camera.ortho_height);
    fprintf(f, " ");
    hx(f, s.camera.near_plane);
    fprintf(f, " ");
    hx(f, s.camera.far_plane);
    fprintf(f, "\n");
    fprintf(f, "scene_viewport %u %u\n", s.viewport.width, s.viewport.height);
    for (int i = 0; i < ELEM_KIND_COUNT; i++)
        fprintf(f, "scene_toggle %s %d\n", element_name((ElementKind)i),
                s.enabled[i] ? 1 : 0);
    fprintf(f, "scene_shading %d", s.shading ? 1 : 0);
    for (int c = 0; c < 3; c++) {
        fprintf(f, " ");
        hx(f, s.light[c]);
    }
    fprintf(f, "\n");
    fprintf(f, "scene_shading_label %s\n", SHADING_LABEL);
    fprintf(f, "scene_trajectory_label %s\n", SCENE_TRAJECTORY_LABEL);
    fprintf(f, "scene_detection_label %s\n", SCENE_DETECTION_LABEL);
    /* The detection channel sets the line of sight is built from, so
     * a reader can hold a drawn line against the flag that governs
     * it without inferring which channels are which. */
    {
        const std::vector<Detection> &dt = m.detections();
        for (size_t d = 0; d < dt.size(); d++) {
            fprintf(f, "scene_detection %u %s %u %u %u %u %u\n", (unsigned)d,
                    dt[d].base.c_str(), dt[d].detected, dt[d].dir_x,
                    dt[d].dir_y, dt[d].dir_z, dt[d].range);
        }
    }
    fprintf(f, "scene_scale ");
    hx(f, s.velocity_seconds);
    fprintf(f, " ");
    hx(f, s.axis_length);
    fprintf(f, " ");
    hx(f, s.thruster_scale);
    fprintf(f, " ");
    hx(f, s.spin_scale);
    /* The datalink element's figure sits with the scales because that
     * is where a reader looks for what governs an element's drawing,
     * though it is a duration and not a scale. */
    fprintf(f, " ");
    hx(f, s.link_fade_seconds);
    fprintf(f, "\n");
    fprintf(f, "scene_datalink_label %s\n", SCENE_DATALINK_LABEL);
}

static void dump_scene_element_(FILE *f, const Spec &sp, uint32_t k,
                                uint32_t step, size_t i,
                                const SceneElement &e)
{
    size_t nv = e.local.size() / 3;

    fprintf(f, "scene_element %u %u %u %s %s %u %u %u %s\n", k, step,
            (unsigned)i, element_name(e.kind),
            e.body == SCENE_NO_BODY ? "-"
                                    : scene_body_name(sp, e.body).c_str(),
            (unsigned)nv, (unsigned)e.segments.size(),
            (unsigned)e.faces.size(), e.name.c_str());
    fprintf(f, "scene_note %u %u %u %s\n", k, step, (unsigned)i,
            e.note.c_str());
    /* What a datalink line was drawn from: its receiver, and the three
     * figures the getter reported, so a reader holds the line against
     * the budget and the age rather than against its brightness. */
    if (e.kind == ELEM_DATALINK) {
        fprintf(f, "scene_link %u %u %u %s %s ", k, step, (unsigned)i,
                scene_body_name(sp, e.body).c_str(),
                scene_body_name(sp, e.link.receiver).c_str());
        hx(f, e.link.margin_db);
        fprintf(f, " ");
        hx(f, e.link.age_s);
        fprintf(f, " ");
        hx(f, e.link.fade);
        fprintf(f, "\n");
    }
    fprintf(f, "scene_mvp %u %u %u", k, step, (unsigned)i);
    for (int c = 0; c < 16; c++) {
        fprintf(f, " ");
        hx(f, (double)e.mvp[c]);
    }
    fprintf(f, "\n");
    for (size_t v = 0; v < nv; v++) {
        fprintf(f, "scene_vertex %u %u %u %u", k, step, (unsigned)i,
                (unsigned)v);
        for (int c = 0; c < 3; c++) {
            fprintf(f, " ");
            hx(f, (double)e.ndc[v * 3 + c]);
        }
        fprintf(f, " %u\n", (unsigned)e.behind[v]);
    }
    for (size_t sgi = 0; sgi < e.segments.size(); sgi++) {
        fprintf(f, "scene_segment %u %u %u %u %u %u %d\n", k, step,
                (unsigned)i, (unsigned)sgi, e.segments[sgi].a,
                e.segments[sgi].b, e.segments[sgi].drawn ? 1 : 0);
    }
    for (size_t fi = 0; fi < e.faces.size(); fi++) {
        fprintf(f, "scene_face %u %u %u %u %u %u %u ", k, step, (unsigned)i,
                (unsigned)fi, e.faces[fi].a, e.faces[fi].b, e.faces[fi].c);
        hx(f, (double)e.faces[fi].intensity);
        fprintf(f, " %d\n", e.faces[fi].drawn ? 1 : 0);
    }
}

/* The projection dump: the scene view made checkable without a
 * display.
 *
 * For a named camera pose, a named projection and a named viewport it
 * writes every drawn element's vertices in normalised device
 * coordinates, with the element's identity, its source body, and a
 * per-vertex behind-the-eye flag. There is no window, no context and
 * no display anywhere in the path.
 *
 * That is the answer to a question a scene view otherwise cannot
 * answer: whether the arithmetic between a body's state and a pixel
 * is right. What it does not cover is stated rather than hidden. The
 * draw calls themselves, and therefore the shaders, need a display
 * and a framebuffer read-back to check; what this establishes is that
 * the geometry handed to those calls is right.
 *
 * Coordinates are single precision, because single precision is what
 * the pipeline these numbers are going to runs in, and they are
 * written as the binary64 bit pattern of that value so the format's
 * standing rule about decimal text still holds.
 */
static void dump_scene(FILE *f, Model &m, const DumpOptions &o)
{
    const Spec &sp = m.spec();
    SceneInput in;
    std::vector<AssetBound> bound;

    dump_scene_header_(f, m, o);
    for (size_t k = 0; k < sp.body_names.size(); k++)
        fprintf(f, "body %u %s\n", (unsigned)k, sp.body_names[k].c_str());

    /* The assemblies, and the one check that lets each be drawn. A
     * craft whose bytes are not the recorded bytes is never drawn, in
     * this panel exactly as in the wireframe panel and through the
     * same binding function, so the two cannot disagree. */
    bound = asset_bind(sp, o.assets);
    if (bound.empty())
        fprintf(f, "scene_asset none no assembly supplied\n");
    for (size_t k = 0; k < bound.size(); k++) {
        const Asset &as = bound[k].asset;
        unsigned n = (unsigned)k;

        if (!as.loaded) {
            fprintf(f, "scene_asset_error %u %s\n", n, as.error.c_str());
            continue;
        }
        fprintf(f, "scene_asset %u %s %s\n", n, as.name.c_str(),
                digest_hex(as.digest).c_str());
        fprintf(f, "scene_asset_verdict %u %s %s %s\n", n,
                asset_verdict_name(bound[k].verdict),
                bound[k].request.name.empty() ? "-"
                                              : bound[k].request.name.c_str(),
                bound[k].verdict == ASSET_DRAWABLE
                    ? scene_body_name(sp, bound[k].body).c_str() : "-");
        fprintf(f, "scene_asset_parts %u %u colliders %u ports %u "
                   "thrusters %u faces\n", n,
                (unsigned)as.colliders.size(), (unsigned)as.ports.size(),
                (unsigned)as.thrusters.size(), (unsigned)as.faces.size());
        if (bound[k].verdict == ASSET_DRAWABLE) {
            AssetBinding b;
            b.asset = &bound[k].asset;
            b.body = bound[k].body;
            in.assets.push_back(b);
        }
    }

    in.model = &m;
    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;
        uint32_t lo, hi;
        ResimResult r;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e)
            continue;
        dump_episode_header(f, k, *e);
        if (!o.artifact.empty())
            r = rebuild_(m, *e, k, o, o.scene.frame);
        in.episode = e;
        in.resim = o.artifact.empty() ? 0 : &r;
        range_for(*e, o, &lo, &hi);
        for (uint32_t step = lo; step < hi; step++) {
            Scene sc = scene_build(in, o.scene, step);
            fprintf(f, "scene_pose %u %u %s %s\n", k, step,
                    sc.pose_from_artifact ? "artifact" : "synthetic",
                    sc.message.c_str());
            fprintf(f, "scene_eye %u %u", k, step);
            for (int c = 0; c < 3; c++) {
                fprintf(f, " ");
                hx(f, sc.eye[c]);
            }
            fprintf(f, "\n");
            fprintf(f, "scene_elements %u %u %u\n", k, step,
                    (unsigned)sc.elements.size());
            for (size_t i = 0; i < sc.elements.size(); i++)
                dump_scene_element_(f, sp, k, step, i, sc.elements[i]);
        }
    }
}

static void dump_resim(FILE *f, Model &m, const DumpOptions &o)
{
    const uint32_t obs_total = m.spec().obs_total;
    const uint32_t agents = m.spec().agent_count ? m.spec().agent_count : 1;

    if (o.artifact.empty()) {
        fprintf(f, "resim unavailable no artifact supplied\n");
        return;
    }
    fprintf(f, "resim artifact %s\n", o.artifact.c_str());
    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e;

        if (o.episode != UINT32_MAX && k != o.episode)
            continue;
        e = m.load(k, &err);
        if (!e)
            continue;
        dump_episode_header(f, k, *e);
        ResimResult r = rebuild_(m, *e, k, o, K26RL_BODY_REF_ORIGIN);
        fprintf(f, "resim_abi %u %08x\n", k, r.abi_version);
        fprintf(f, "resim_ran %u %d\n", k, r.ran ? 1 : 0);
        fprintf(f, "resim_verdict %u %s\n", k,
                !r.ran ? "failed" : (r.equal ? "equal-bitwise" : "diverged"));
        fprintf(f, "resim_steps %u %u\n", k, r.steps_compared);
        if (r.ran && !r.equal) {
            fprintf(f, "resim_divergence %u %u %s\n", k, r.first_divergence,
                    r.divergence_kind.c_str());
        }
        fprintf(f, "resim_message %u %s\n", k, r.message.c_str());
        if (!r.ran)
            continue;
        for (uint32_t j = 0; j < obs_total && j < r.initial_obs.size(); j++) {
            fprintf(f, "resim_initial_obs %u %u ", k, j);
            hx(f, r.initial_obs[j]);
            fprintf(f, "\n");
        }
        for (uint32_t i = 0; i < r.steps_compared; i++) {
            for (uint32_t j = 0; j < obs_total; j++) {
                size_t idx = (size_t)i * obs_total + j;
                fprintf(f, "resim_obs %u %u %u ", k, i, j);
                hx(f, idx < r.obs.size() ? r.obs[idx] : 0.0);
                fprintf(f, "\n");
            }
            for (uint32_t a = 0; a < agents; a++) {
                size_t idx = (size_t)i * agents + a;
                fprintf(f, "resim_reward %u %u %u ", k, i, a);
                hx(f, idx < r.rewards.size() ? r.rewards[idx] : 0.0);
                fprintf(f, "\n");
            }
            fprintf(f, "resim_flags %u %u %08x\n", k, i,
                    i < r.flags.size() ? r.flags[i] : 0u);
        }
    }
}

/* The total number of step records held, over every episode. It is
 * the one number that rises with every step frame accepted and never
 * falls, which is what makes the advance a countable fact: an episode
 * ending resets a per-episode count, and a viewer that had stopped
 * receiving would look the same as one that had moved on. */
static uint64_t steps_held_(Model &m)
{
    uint64_t total = 0;

    for (uint32_t k = 0; k < m.info().episode_count; k++) {
        std::string err;
        const Episode *e = m.load(k, &err);
        if (e)
            total += e->step_count;
    }
    return total;
}

const char *live_watch(FILE *f, Model &m, const LiveOptions &o, bool report)
{
    uint64_t round = 0, idle = 0;
    const char *reason = "closed";

    if (report) {
        fprintf(f, "live_tap %s\n", o.tap.c_str());
        fprintf(f, "live_from_start %d\n", o.from_start ? 1 : 0);
        fprintf(f, "live_slot_size %u\n", m.ring_slot_size());
        fprintf(f, "live_slot_count %u\n", m.ring_slot_count());
    }
    for (;;) {
        struct timespec ts;
        uint64_t before_lost = m.frames_lost();
        uint32_t frames = m.poll();
        uint64_t lost = m.frames_lost() - before_lost;

        round++;
        if (frames || lost) {
            idle = 0;
            if (report) {
                fprintf(f, "live_poll %" PRIu64 " %u %" PRIu64 " %" PRIu64
                        " %" PRIu64 " %u %" PRIu64 "\n", round, frames, lost,
                        m.frames_accepted(), m.frames_lost(),
                        m.info().episode_count, steps_held_(m));
            }
        } else {
            idle++;
        }
        /* The producer marks the ring closed before it unlinks it, so
         * a drained ring that carries the mark is a finished run.
         * One more round after the mark, because the mark and the
         * last frames are separate stores and the frames may not
         * have been taken yet. */
        if (m.producer_closed() && !frames) {
            reason = "closed";
            break;
        }
        if (o.polls && round >= o.polls) {
            reason = "polls";
            break;
        }
        /* A producer that died without marking the ring leaves no
         * signal at all: the cursor simply stops. Waiting forever on
         * one is the same as hanging, so a watcher that has seen
         * nothing for long enough says so and stops. */
        if (o.idle_polls && idle >= o.idle_polls) {
            reason = "idle";
            break;
        }
        if (o.poll_ms) {
            ts.tv_sec = (time_t)(o.poll_ms / 1000u);
            ts.tv_nsec = (long)(o.poll_ms % 1000u) * 1000000L;
            nanosleep(&ts, 0);
        }
    }
    if (report) {
        fprintf(f, "live_end %s %" PRIu64 " %" PRIu64 " %" PRIu64 " %u %"
                PRIu64 "\n", reason, round, m.frames_accepted(),
                m.frames_lost(), m.info().episode_count, steps_held_(m));
    }
    return reason;
}

int dump(FILE *f, Model &m, const DumpOptions &o)
{
    const std::string &p = o.panel;

    fprintf(f, "k26rl_view dump 1\n");
    fprintf(f, "panel %s\n", p.c_str());
    /* A run in progress is watched before it is reported: the panels
     * below draw what a source holds, and a live source holds what
     * has arrived by the time they are asked. */
    if (m.live())
        live_watch(f, m, o.live, p == "live" || p == "all");
    if (p == "meta" || p == "all")
        dump_meta(f, m);
    if (p == "timeline" || p == "all")
        dump_timeline(f, m, o);
    if (p == "reward" || p == "all")
        dump_reward(f, m, o);
    if (p == "obs" || p == "all")
        dump_obs(f, m, o);
    if (p == "action" || p == "all")
        dump_action(f, m, o);
    if (p == "agents" || p == "all")
        dump_agents(f, m, o);
    if (p == "traj" || p == "all")
        dump_traj(f, m, o);
    if (p == "scrub" || p == "all")
        dump_scrub(f, m, o);
    if (p == "world" || p == "all")
        dump_world(f, m, o);
    if (p == "attitude" || p == "all")
        dump_attitude(f, m, o);
    if (p == "overlay" || p == "all")
        dump_overlay(f, m, o);
    if (p == "wireframe" || p == "all")
        dump_wireframe(f, m, o);
    if (p == "scene" || p == "all")
        dump_scene(f, m, o);
    if (p == "resim" || p == "all")
        dump_resim(f, m, o);
    if (p != "meta" && p != "timeline" && p != "reward" && p != "obs" &&
        p != "action" && p != "agents" && p != "traj" && p != "scrub" &&
        p != "resim" &&
        p != "world" && p != "attitude" && p != "overlay" &&
        p != "wireframe" && p != "scene" && p != "live" && p != "all") {
        fprintf(stderr, "k26rl_view: unknown panel `%s`\n", p.c_str());
        return 2;
    }
    fprintf(f, "end\n");
    return 0;
}

}  /* namespace k26rl_view */
