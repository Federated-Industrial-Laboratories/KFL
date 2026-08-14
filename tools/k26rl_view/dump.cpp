/* dump.cpp - the headless presenter.
 *
 * Writes the values behind a named panel, for a named episode and
 * step range, without opening a window. It exists so the panels are
 * testable: a panel whose numbers nothing checks is an assertion, and
 * the interface and this dump read the same model, so what is checked
 * here is what the interface draws.
 *
 * Every double is written as the sixteen hex digits of its IEEE-754
 * binary64 bit pattern. The format's own rule is that values travel
 * as bit patterns and never through decimal text, for the last-ULP
 * reason recorded when the episode format was decided; a dump whose
 * whole purpose is bitwise comparison keeps that rule rather than
 * inviting an argument about decimal digits.
 */
#include "dump.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

namespace k26rl_view {

static void hx(FILE *f, double v)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);
    fprintf(f, "%016" PRIx64, bits);
}

/* The trajectory panel's standing label. The spec publishes no
 * observer mode, so this viewer cannot say which mode produced a
 * channel and does not claim to: it states the construction and the
 * one condition under which the reconstruction is exact, which is
 * what keeps the panel honest without inventing a fact. */
const char *const TRAJECTORY_LABEL =
    "observer-relative: each channel's own mode is published by the "
    "spec and stated per trajectory below; the reconstruction is the "
    "target's position exactly under a geometric observe, and an "
    "apparent direction taken at a geometric range otherwise";

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

    fprintf(f, "file %s\n", i.path.c_str());
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

static void dump_episode_header(FILE *f, uint32_t k, const Episode &e)
{
    fprintf(f, "episode %u %u %u %u %u %u %s %u %016" PRIx64 "\n", k,
            e.ordinal, e.env, e.episode, e.step_count, e.transitions(),
            end_reason_name(e.end_reason), (unsigned)e.fault_code, e.seed);
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
            fprintf(f, "flag %u %u %08x\n", k, i,
                    i < e->flags.size() ? e->flags[i] : 0u);
            fprintf(f, "dt %u %u ", k, i);
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
                fprintf(f, "reward %u %u %u ", k, i, a);
                hx(f, idx < e->rewards.size() ? e->rewards[idx] : 0.0);
                fprintf(f, "\n");
                fprintf(f, "return %u %u %u ", k, i, a);
                hx(f, idx < ret.size() ? ret[idx] : 0.0);
                fprintf(f, "\n");
            }
        }
        for (uint32_t a = 0; a < agents; a++) {
            fprintf(f, "terminal_adj %u %u ", k, a);
            hx(f, a < e->terminal_adjustments.size()
                   ? e->terminal_adjustments[a] : 0.0);
            fprintf(f, "\n");
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
        range_for(*e, o, &lo, &hi);
        for (uint32_t i = lo; i < hi; i++) {
            for (uint32_t j = 0; j < obs_total; j++) {
                size_t idx = (size_t)i * obs_total + j;
                fprintf(f, "obs %u %u %u ", k, i, j);
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
                fprintf(f, "act %u %u %u ", k, i, j);
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
                fprintf(f, "traj %s %u %u ", tr.base.c_str(), k, i);
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
                    marks[s], m.index_lookups(), m.episode_reads());
            hx(f, (again && idx < again->obs.size()) ? again->obs[idx] : 0.0);
            fprintf(f, "\n");
        }
    }
    fprintf(f, "scrub_index_lookups %" PRIu64 "\n", m.index_lookups());
    fprintf(f, "scrub_episode_reads %" PRIu64 "\n", m.episode_reads());
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
        ResimResult r = resimulate(m, *e, o.artifact);
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
        ResimResult r = resimulate(m, *e, o.artifact);
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

int dump(FILE *f, Model &m, const DumpOptions &o)
{
    const std::string &p = o.panel;

    fprintf(f, "k26rl_view dump 1\n");
    fprintf(f, "panel %s\n", p.c_str());
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
    if (p == "traj" || p == "all")
        dump_traj(f, m, o);
    if (p == "scrub" || p == "all")
        dump_scrub(f, m, o);
    if (p == "world" || p == "all")
        dump_world(f, m, o);
    if (p == "resim" || p == "all")
        dump_resim(f, m, o);
    if (p != "meta" && p != "timeline" && p != "reward" && p != "obs" &&
        p != "action" && p != "traj" && p != "scrub" && p != "resim" &&
        p != "world" && p != "all") {
        fprintf(stderr, "k26rl_view: unknown panel `%s`\n", p.c_str());
        return 2;
    }
    fprintf(f, "end\n");
    return 0;
}

}  /* namespace k26rl_view */
