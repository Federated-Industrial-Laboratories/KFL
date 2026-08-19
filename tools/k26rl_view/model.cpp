/* model.cpp - the viewer's data layer.
 *
 * Reads an episode file through the format library's own reader and
 * nothing else: no second parser exists here, so a file this viewer
 * shows is a file the library agrees with. The spec blob is walked
 * once at open for the channel names, action bounds, and agent
 * slices the panels label themselves with; unknown tags are skipped
 * by length and remembered, because a file carrying a tag this build
 * does not know is a fact worth reporting rather than hiding.
 */
#include "model.h"

#include <string.h>

#include <sys/stat.h>

#include "live.h"

namespace k26rl_view {

const char *end_reason_name(uint16_t reason)
{
    switch (reason) {
    case K26RL_END_TERMINATED: return "terminated";
    case K26RL_END_TRUNCATED:  return "truncated";
    case K26RL_END_FAULT:      return "fault";
    default:                   return "unknown";
    }
}

const char *observer_mode_name(uint16_t mode)
{
    switch (mode) {
    case K26RL_OBS_MODE_GEOMETRIC:   return "geometric";
    case K26RL_OBS_MODE_ASTROMETRIC: return "astrometric";
    case K26RL_OBS_MODE_APPARENT:    return "apparent";
    case K26RL_OBS_MODE_TOPOCENTRIC: return "topocentric";
    default:                         return "unknown";
    }
}

Model::Model()
    : reader_(0), feed_(0), current_k_(0), have_(false), index_lookups_(0),
      episode_reads_(0)
{
}

Model::~Model()
{
    close();
}

void Model::close()
{
    if (reader_) {
        k26rl_episode_reader_close(reader_);
        reader_ = 0;
    }
    if (feed_) {
        delete feed_;
        feed_ = 0;
    }
    have_ = false;
    index_lookups_ = 0;
    episode_reads_ = 0;
    traj_.clear();
    det_.clear();
    spec_ = Spec();
    info_ = FileInfo();
}

/* The declaration for an action offset, created on first mention:
 * bounds and kind arrive as separate tags and either may come
 * first. */
static ActionDecl *find_action_(std::vector<ActionDecl> &v,
                                uint32_t offset)
{
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i].offset == offset)
            return &v[i];
    }
    v.push_back(ActionDecl());
    v.back().offset = offset;
    v.back().lo = 0.0;
    v.back().hi = 0.0;
    v.back().kind = 0;
    v.back().arity = 0;
    v.back().has_bounds = false;
    v.back().has_kind = false;
    return &v.back();
}

/* The assembly record for a body, created on first mention: the two
 * tags arrive in either order and each carries only its own half. */
static AssemblyRef *find_assembly_(std::vector<AssemblyRef> &v, uint32_t body)
{
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i].body == body)
            return &v[i];
    }
    AssemblyRef r;
    r.body = body;
    r.has_digest = false;
    memset(r.digest, 0, sizeof r.digest);
    v.push_back(r);
    return &v.back();
}

void Model::parse_spec_(const uint8_t *blob, uint32_t len)
{
    uint32_t off = 0;

    spec_.raw.assign(blob, blob + len);
    while (off + 6 <= len) {
        uint16_t tag = get_u16_(blob + off);
        uint32_t l = get_u32_(blob + off + 2);
        const uint8_t *v = blob + off + 6;

        if ((uint64_t)off + 6 + l > len)
            break;   /* a malformed tail ends the walk, as a reader must */
        switch (tag) {
        case K26RL_TAG_ABI_VERSION:  spec_.abi_version = get_u32_(v); break;
        case K26RL_TAG_ENDIAN_PROBE: spec_.endian_probe = get_u32_(v); break;
        case K26RL_TAG_AGENT_COUNT:  spec_.agent_count = get_u32_(v); break;
        case K26RL_TAG_N_ENVS:       spec_.n_envs = get_u32_(v); break;
        case K26RL_TAG_CONTROL_DT:   spec_.control_dt = get_f64_(v); break;
        case K26RL_TAG_HORIZON:      spec_.horizon = get_u32_(v); break;
        case K26RL_TAG_OBS_TOTAL:    spec_.obs_total = get_u32_(v); break;
        case K26RL_TAG_ACT_TOTAL:    spec_.act_total = get_u32_(v); break;
        case K26RL_TAG_EPISODE_FLAGS: spec_.episode_flags = get_u32_(v); break;
        case K26RL_TAG_AGENT_OBS_SLICE:
            if (l >= 12) {
                Slice s;
                s.agent = get_u32_(v);
                s.offset = get_u32_(v + 4);
                s.count = get_u32_(v + 8);
                spec_.obs_slices.push_back(s);
            }
            break;
        case K26RL_TAG_AGENT_ACT_SLICE:
            if (l >= 12) {
                Slice s;
                s.agent = get_u32_(v);
                s.offset = get_u32_(v + 4);
                s.count = get_u32_(v + 8);
                spec_.act_slices.push_back(s);
            }
            break;
        case K26RL_TAG_ACT_BOUNDS:
            if (l >= 20) {
                ActionDecl *a = find_action_(spec_.actions, get_u32_(v));
                a->lo = get_f64_(v + 4);
                a->hi = get_f64_(v + 12);
                a->has_bounds = true;
            }
            break;
        case K26RL_TAG_ACT_KIND:
            if (l >= 6) {
                ActionDecl *a = find_action_(spec_.actions, get_u32_(v));
                a->kind = get_u16_(v + 4);
                a->arity = (l >= 10) ? get_u32_(v + 6) : 0;
                a->has_kind = true;
            }
            break;
        case K26RL_TAG_OBS_CHANNEL_NAME:
            if (l >= 4) {
                Channel c;
                c.index = get_u32_(v);
                c.kind = K26RL_OBS_KIND_VECTOR;
                c.mode = 0;
                c.has_mode = false;
                c.source = K26RL_OBS_SOURCE_MEASURED;
                c.pair = K26RL_OBS_PAIR_NONE;
                c.has_source = false;
                c.name.assign((const char *)v + 4, l - 4);
                spec_.channels.push_back(c);
            }
            break;
        case K26RL_TAG_OBS_CHANNEL_MODE:
            if (l >= 6) {
                uint32_t idx = get_u32_(v);
                uint16_t md = get_u16_(v + 4);
                for (size_t i = 0; i < spec_.channels.size(); i++) {
                    if (spec_.channels[i].index == idx) {
                        spec_.channels[i].mode = md;
                        spec_.channels[i].has_mode = true;
                    }
                }
            }
            break;
        case K26RL_TAG_OBS_CHANNEL_SOURCE:
            if (l >= 10) {
                uint32_t idx = get_u32_(v);
                uint16_t src = get_u16_(v + 4);
                uint32_t pr = get_u32_(v + 6);
                for (size_t i = 0; i < spec_.channels.size(); i++) {
                    if (spec_.channels[i].index == idx) {
                        spec_.channels[i].source = src;
                        spec_.channels[i].pair = pr;
                        spec_.channels[i].has_source = true;
                    }
                }
            }
            break;
        case K26RL_TAG_ASSEMBLY_NAME:
            if (l >= 4) {
                AssemblyRef *r = find_assembly_(spec_.assemblies, get_u32_(v));
                r->name.assign((const char *)v + 4, l - 4);
            }
            break;
        case K26RL_TAG_ASSEMBLY_DIGEST:
            if (l >= 4 + K26RL_SHA256_BYTES) {
                AssemblyRef *r = find_assembly_(spec_.assemblies, get_u32_(v));
                memcpy(r->digest, v + 4, K26RL_SHA256_BYTES);
                r->has_digest = true;
            }
            break;
        case K26RL_TAG_BODY_NAME:
            if (l >= 4) {
                uint32_t bi = get_u32_(v);
                if (spec_.body_names.size() <= bi)
                    spec_.body_names.resize(bi + 1);
                spec_.body_names[bi].assign((const char *)v + 4, l - 4);
            }
            break;
        case K26RL_TAG_OBS_CHANNEL_KIND:
            if (l >= 6) {
                uint32_t idx = get_u32_(v);
                uint16_t kind = get_u16_(v + 4);
                for (size_t i = 0; i < spec_.channels.size(); i++) {
                    if (spec_.channels[i].index == idx)
                        spec_.channels[i].kind = kind;
                }
            }
            break;
        default:
            spec_.unknown_tags.push_back(std::make_pair(tag, l));
            break;
        }
        off += 6 + l;
    }
}

/* Group channels into drawable trajectories by the published naming
 * convention: an `as`-bound observe publishes <base>_dir_x, _dir_y,
 * _dir_z, _range and _range_rate, so a base with the first four
 * present is drawable. A channel set that does not match the
 * convention is not drawn here; it stays in the observation panel,
 * where every channel appears whatever it is named. */
static bool suffixed_(const std::string &name, const char *suffix,
                      std::string *base)
{
    size_t n = strlen(suffix);
    if (name.size() <= n)
        return false;
    if (name.compare(name.size() - n, n, suffix) != 0)
        return false;
    base->assign(name, 0, name.size() - n);
    return true;
}

void Model::find_trajectories_()
{
    traj_.clear();
    for (size_t i = 0; i < spec_.channels.size(); i++) {
        std::string base;
        if (!suffixed_(spec_.channels[i].name, "_dir_x", &base))
            continue;
        Trajectory t;
        t.base = base;
        t.dir_x = spec_.channels[i].index;
        bool y = false, z = false, r = false;
        for (size_t j = 0; j < spec_.channels.size(); j++) {
            const std::string &n = spec_.channels[j].name;
            if (n == base + "_dir_y") { t.dir_y = spec_.channels[j].index; y = true; }
            if (n == base + "_dir_z") { t.dir_z = spec_.channels[j].index; z = true; }
            if (n == base + "_range") { t.range = spec_.channels[j].index; r = true; }
        }
        if (y && z && r)
            traj_.push_back(t);
    }
}

/* A detection observe publishes seven channels of one base, and the
 * four this needs are the flag, the direction and the range: the line
 * of sight is that direction taken out to that range, and the flag
 * says whether the payload saw anything at this step. The other two
 * are carried when the file has them, for the panel to read.
 *
 * The convention is the file's own naming and not a guess: the
 * channels of one observe share a base and differ only in the
 * suffix. */
void Model::find_detections_()
{
    det_.clear();
    for (size_t i = 0; i < spec_.channels.size(); i++) {
        std::string base;
        Detection d;
        bool x = false, y = false, z = false, r = false;

        if (!suffixed_(spec_.channels[i].name, "_detected", &base))
            continue;
        d.base = base;
        d.detected = spec_.channels[i].index;
        d.snr = 0;
        d.aspect = 0;
        d.has_snr = false;
        d.has_aspect = false;
        d.dir_x = d.dir_y = d.dir_z = d.range = 0;
        for (size_t j = 0; j < spec_.channels.size(); j++) {
            const std::string &n = spec_.channels[j].name;
            uint32_t k = spec_.channels[j].index;
            if (n == base + "_dir_x") { d.dir_x = k; x = true; }
            if (n == base + "_dir_y") { d.dir_y = k; y = true; }
            if (n == base + "_dir_z") { d.dir_z = k; z = true; }
            if (n == base + "_range") { d.range = k; r = true; }
            if (n == base + "_snr") { d.snr = k; d.has_snr = true; }
            if (n == base + "_aspect") { d.aspect = k; d.has_aspect = true; }
        }
        if (x && y && z && r)
            det_.push_back(d);
    }
}

/* The agent groups, read from the slice tags.
 *
 * A file written before the slice tags existed carries none, and a
 * one-agent file's agent owns the whole vector by the ABI's own
 * definition, so that case falls back to the totals. With more than
 * one agent there is no such definition and a missing tag leaves the
 * group empty, which the group reports rather than filling in.
 */
std::vector<AgentGroup> Model::agent_groups() const
{
    const uint32_t n = spec_.agent_count ? spec_.agent_count : 1;
    std::vector<AgentGroup> out;

    for (uint32_t a = 0; a < n; a++) {
        AgentGroup g;
        g.index = a;
        g.obs_offset = 0;
        g.obs_count = 0;
        g.act_offset = 0;
        g.act_count = 0;
        g.has_obs_slice = false;
        g.has_act_slice = false;
        for (size_t k = 0; k < spec_.obs_slices.size(); k++) {
            if (spec_.obs_slices[k].agent != a)
                continue;
            g.obs_offset = spec_.obs_slices[k].offset;
            g.obs_count = spec_.obs_slices[k].count;
            g.has_obs_slice = true;
        }
        for (size_t k = 0; k < spec_.act_slices.size(); k++) {
            if (spec_.act_slices[k].agent != a)
                continue;
            g.act_offset = spec_.act_slices[k].offset;
            g.act_count = spec_.act_slices[k].count;
            g.has_act_slice = true;
        }
        if (!g.has_obs_slice && n == 1)
            g.obs_count = spec_.obs_total;
        if (!g.has_act_slice && n == 1)
            g.act_count = spec_.act_total;
        for (size_t c = 0; c < spec_.channels.size(); c++) {
            uint32_t idx = spec_.channels[c].index;
            if (idx >= g.obs_offset && idx - g.obs_offset < g.obs_count)
                g.channels.push_back(idx);
        }
        /* The agent's name, taken from the prefix its channels
         * publish. It is a label and never a membership test: the
         * channels above are already chosen, and a group whose
         * channels disagree about the prefix keeps no name at all
         * rather than adopting the first one it read. */
        if (n > 1) {
            bool agreed = !g.channels.empty();
            std::string prefix;
            for (size_t c = 0; c < g.channels.size() && agreed; c++) {
                std::string nm;
                size_t dot;
                for (size_t q = 0; q < spec_.channels.size(); q++) {
                    if (spec_.channels[q].index == g.channels[c])
                        nm = spec_.channels[q].name;
                }
                dot = nm.find('.');
                if (dot == std::string::npos || dot == 0) {
                    agreed = false;
                    break;
                }
                if (c == 0)
                    prefix = nm.substr(0, dot);
                else if (nm.compare(0, dot, prefix) != 0 ||
                         dot != prefix.size())
                    agreed = false;
            }
            if (agreed)
                g.name = prefix;
        }
        out.push_back(g);
    }
    return out;
}

bool Model::open(const std::string &path, std::string *err)
{
    K26RlEpisodeInfo ei;
    const uint8_t *blob = 0;
    uint32_t blob_len = 0;
    struct stat sb;
    K26RlStatus st;

    close();
    st = k26rl_episode_reader_open(path.c_str(), &reader_);
    if (st != K26RL_OK) {
        reader_ = 0;
        if (err)
            *err = std::string("cannot open ") + path + ": " +
                   k26rl_status_str(st);
        return false;
    }
    st = k26rl_episode_reader_info(reader_, &ei);
    if (st != K26RL_OK) {
        if (err)
            *err = std::string("cannot read file info: ") +
                   k26rl_status_str(st);
        close();
        return false;
    }
    info_.path = path;
    info_.format_version = ei.format_version;
    info_.governing_seed = ei.governing_seed;
    info_.rekey_ordinal = ei.rekey_ordinal;
    info_.n_envs = ei.n_envs;
    info_.steps_per_chunk = ei.steps_per_chunk;
    info_.episode_count = ei.episode_count;
    info_.unindexed_episode_starts = ei.unindexed_episode_starts;
    info_.readable_bytes = ei.readable_bytes;
    info_.clean_close = ei.clean_close != 0;
    info_.file_bytes = (stat(path.c_str(), &sb) == 0) ? (uint64_t)sb.st_size : 0;

    if (k26rl_episode_reader_spec(reader_, &blob, &blob_len) == K26RL_OK &&
        blob && blob_len)
        parse_spec_(blob, blob_len);
    find_trajectories_();
    find_detections_();
    return true;
}

/* Attaching to a run in progress differs from opening a recording in
 * one respect only: what is held grows. The spec comes from the ring
 * preamble and is walked by the same parser that walks a file's, so
 * the channel names, action bounds and agent slices the panels label
 * themselves with are the same facts read from the same bytes. */
bool Model::attach(const std::string &tap_name, bool from_start,
                   std::string *err)
{
    const uint8_t *blob = 0;
    uint32_t blob_len = 0;

    close();
    feed_ = new LiveFeed();
    if (!feed_->attach(tap_name, from_start, err)) {
        delete feed_;
        feed_ = 0;
        return false;
    }
    info_ = feed_->info();
    blob = feed_->spec(&blob_len);
    if (blob && blob_len)
        parse_spec_(blob, blob_len);
    find_trajectories_();
    find_detections_();
    /* The decoder needs the widths before it reads a step frame, and
     * they are the spec's, walked once above rather than a second
     * time inside the feed. */
    feed_->geometry(spec_.obs_total, spec_.act_total, spec_.agent_count);
    return true;
}

uint32_t Model::poll()
{
    LivePoll p;

    if (!feed_)
        return 0;
    p = feed_->poll();
    info_ = feed_->info();
    return p.frames;
}

bool Model::producer_closed() const
{
    return feed_ && feed_->closed();
}

uint64_t Model::frames_accepted() const
{
    return feed_ ? feed_->accepted() : 0;
}

uint64_t Model::frames_lost() const
{
    return feed_ ? feed_->lost() : 0;
}

uint32_t Model::ring_slot_size() const
{
    return feed_ ? feed_->slot_size() : 0;
}

uint32_t Model::ring_slot_count() const
{
    return feed_ ? feed_->slot_count() : 0;
}

/* The pairing predicate: a channel is overlaid when the file says it
 * carries a measurement and names the channel holding the truth that
 * measurement was applied to. Only the measured side answers, so a
 * pair asked from both ends is drawn once. */
uint32_t Model::truth_pair_of(uint32_t channel) const
{
    for (size_t i = 0; i < spec_.channels.size(); i++) {
        const Channel &c = spec_.channels[i];
        if (c.index != channel)
            continue;
        if (!c.has_source || c.source != K26RL_OBS_SOURCE_MEASURED)
            return K26RL_OBS_PAIR_NONE;
        return c.pair;
    }
    return K26RL_OBS_PAIR_NONE;
}

/* The measured half of every pair the file declares, in channel
 * order, built from that one predicate. */
std::vector<std::pair<uint32_t, uint32_t> > Model::overlay_pairs() const
{
    std::vector<std::pair<uint32_t, uint32_t> > out;
    for (size_t i = 0; i < spec_.channels.size(); i++) {
        uint32_t t = truth_pair_of(spec_.channels[i].index);
        if (t == K26RL_OBS_PAIR_NONE)
            continue;
        out.push_back(std::make_pair(spec_.channels[i].index, t));
    }
    return out;
}

bool Model::identity(uint32_t k, uint32_t *ordinal, uint32_t *env,
                     uint32_t *episode) const
{
    if (feed_) {
        const std::vector<Episode *> &eps = feed_->episodes();
        if (k >= eps.size())
            return false;
        if (ordinal) *ordinal = eps[k]->ordinal;
        if (env)     *env     = eps[k]->env;
        if (episode) *episode = eps[k]->episode;
        return true;
    }
    if (!reader_ || k >= info_.episode_count)
        return false;
    return k26rl_episode_reader_at(reader_, k, ordinal, env, episode) ==
           K26RL_OK;
}

const Episode *Model::load(uint32_t k, std::string *err)
{
    uint32_t ordinal = 0, env = 0, episode = 0;
    K26RlEpisodeData d;
    K26RlStatus st;

    /* A live source holds every episode it has decoded, so there is
     * nothing to seek and nothing to cache: the episode is already
     * where it will stay until detach, and the pointer stays good
     * across the polls that lengthen it. */
    if (feed_) {
        const std::vector<Episode *> &eps = feed_->episodes();
        if (k >= eps.size()) {
            if (err)
                *err = "episode index out of range";
            return 0;
        }
        current_k_ = k;
        return eps[k];
    }
    if (!reader_ || k >= info_.episode_count) {
        if (err)
            *err = "episode index out of range";
        return 0;
    }
    /* Scrubbing inside the episode already loaded is free: the seek
     * the timeline performs is an index into memory, and no reader
     * call happens at all. */
    if (have_ && current_k_ == k)
        return &current_;

    index_lookups_++;
    st = k26rl_episode_reader_at(reader_, k, &ordinal, &env, &episode);
    if (st != K26RL_OK) {
        if (err)
            *err = std::string("cannot address episode: ") +
                   k26rl_status_str(st);
        return 0;
    }
    episode_reads_++;
    st = k26rl_episode_read(reader_, ordinal, env, episode, &d);
    if (st != K26RL_OK) {
        if (err)
            *err = std::string("cannot read episode: ") +
                   k26rl_status_str(st);
        return 0;
    }

    Episode e;
    e.ordinal = d.rekey_ordinal;
    e.env = d.env;
    e.episode = d.episode;
    e.step_count = d.step_count;
    e.end_reason = d.end_reason;
    e.fault_code = d.fault_code;
    e.seed = info_.governing_seed;
    (void)k26rl_episode_reader_seed(reader_, d.rekey_ordinal, &e.seed);

    const uint32_t obs_total = spec_.obs_total;
    const uint32_t act_total = spec_.act_total;
    const uint32_t agents = spec_.agent_count ? spec_.agent_count : 1;

    if (d.initial_obs && obs_total)
        e.initial_obs.assign(d.initial_obs, d.initial_obs + obs_total);
    if (d.dr_count) {
        e.dr_tags.assign(d.dr_tags, d.dr_tags + d.dr_count);
        e.dr_values.assign(d.dr_values, d.dr_values + d.dr_count);
    }
    if (d.obs && obs_total && d.step_count)
        e.obs.assign(d.obs, d.obs + (size_t)d.step_count * obs_total);
    if (d.act && act_total && d.step_count)
        e.act.assign(d.act, d.act + (size_t)d.step_count * act_total);
    if (d.rewards && d.step_count)
        e.rewards.assign(d.rewards, d.rewards + (size_t)d.step_count * agents);
    if (d.flags && d.step_count)
        e.flags.assign(d.flags, d.flags + d.step_count);
    if (d.applied_dt && d.step_count)
        e.applied_dt.assign(d.applied_dt, d.applied_dt + d.step_count);
    if (d.terminal_adjustments)
        e.terminal_adjustments.assign(d.terminal_adjustments,
                                      d.terminal_adjustments + agents);
    k26rl_episode_free(&d);

    current_ = e;
    current_k_ = k;
    have_ = true;
    return &current_;
}

std::vector<double> Model::returns(const Episode &ep, uint32_t agent_count)
{
    uint32_t agents = agent_count ? agent_count : 1;
    std::vector<double> out;

    out.assign((size_t)ep.step_count * agents, 0.0);
    for (uint32_t a = 0; a < agents; a++) {
        double acc = 0.0;
        for (uint32_t i = 0; i < ep.step_count; i++) {
            size_t k = (size_t)i * agents + a;
            if (k < ep.rewards.size())
                acc += ep.rewards[k];
            out[k] = acc;
        }
        /* The terminal adjustment is the episode's, not a step's, and
         * belongs to the return the last step reports. */
        if (ep.step_count && a < ep.terminal_adjustments.size()) {
            size_t last = (size_t)(ep.step_count - 1) * agents + a;
            out[last] += ep.terminal_adjustments[a];
        }
    }
    return out;
}

void Model::point(const Episode &ep, const Spec &sp, const Trajectory &t,
                  uint32_t step, double *xyz)
{
    size_t base = (size_t)step * sp.obs_total;

    xyz[0] = xyz[1] = xyz[2] = 0.0;
    if (base + sp.obs_total > ep.obs.size())
        return;
    {
        double r = ep.obs[base + t.range];
        xyz[0] = ep.obs[base + t.dir_x] * r;
        xyz[1] = ep.obs[base + t.dir_y] * r;
        xyz[2] = ep.obs[base + t.dir_z] * r;
    }
}

}  /* namespace k26rl_view */
