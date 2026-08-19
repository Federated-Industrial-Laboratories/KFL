/* live.cpp - decoding a running simulation's telemetry ring.
 *
 * The ring reader beneath this hands out whole frames and counts the
 * ones the producer overwrote first. Everything here turns those
 * frames into the episodes the panels draw, and turns that count into
 * the thing a person needs, which is where in the step stream the
 * missing records were.
 *
 * The frame payloads are the record format's own, field for field
 * with the file transport's, which is why the two sources produce
 * episodes that compare bitwise: one schema, two transports. A step
 * record travels the ring as a step chunk of one step, and at a count
 * of one the format's column-major payload is the same bytes as row
 * order, so this reads the file's layout and not a second encoding.
 */
#include "live.h"

#include <stdio.h>
#include <string.h>

namespace k26rl_view {

LiveFeed::LiveFeed()
    : reader_(0), slot_size_(0), slot_count_(0), spec_(0), spec_len_(0),
      obs_total_(0), act_total_(0), agent_count_(0), accepted_(0), lost_(0)
{
}

LiveFeed::~LiveFeed()
{
    detach();
}

void LiveFeed::detach()
{
    if (reader_) {
        k26rl_tap_detach(reader_);
        reader_ = 0;
    }
    for (size_t i = 0; i < episodes_.size(); i++)
        delete episodes_[i];
    episodes_.clear();
    open_.clear();
    seeds_.clear();
    buf_.clear();
    name_.clear();
    slot_size_ = 0;
    slot_count_ = 0;
    spec_ = 0;
    spec_len_ = 0;
    obs_total_ = 0;
    act_total_ = 0;
    agent_count_ = 0;
    accepted_ = 0;
    lost_ = 0;
    info_ = FileInfo();
}

const uint8_t *LiveFeed::spec(uint32_t *len) const
{
    if (len)
        *len = spec_len_;
    return spec_;
}

bool LiveFeed::closed() const
{
    return reader_ && k26rl_tap_reader_closed(reader_) != 0;
}

/* The preamble's file-header frame carries the format version, the
 * seed and ordinal in force when the tap was enabled, the geometry
 * and the spec blob, in the episode file's own file-header layout.
 * It is written once and no overwrite reaches it, so this succeeds
 * however late the attach was. */
bool LiveFeed::read_header_(std::string *err)
{
    const uint8_t *frame = 0, *spec = 0;
    uint32_t len = 0, slen = 0, plen;
    const uint8_t *p;

    if (k26rl_tap_reader_header(reader_, &frame, &len) != K26RL_OK ||
        len < K26RL_EPISODE_FRAME_HEADER_SIZE) {
        if (err)
            *err = "the ring's header frame will not read";
        return false;
    }
    plen = get_u32_(frame + 4);
    if (plen < 24 || K26RL_EPISODE_FRAME_HEADER_SIZE + plen > len) {
        if (err)
            *err = "the ring's header frame is malformed";
        return false;
    }
    p = frame + K26RL_EPISODE_FRAME_HEADER_SIZE;
    info_.format_version = get_u32_(p);
    info_.governing_seed = get_u64_(p + 4);
    info_.rekey_ordinal = get_u32_(p + 12);
    info_.n_envs = get_u32_(p + 16);
    info_.steps_per_chunk = get_u32_(p + 20);

    if (k26rl_tap_reader_spec(reader_, &spec, &slen) != K26RL_OK) {
        if (err)
            *err = "the ring's header frame carries no readable spec";
        return false;
    }
    spec_ = spec;
    spec_len_ = slen;

    /* The seed table starts at the ordinal in force at enable; a
     * rekey frame extends it. A viewer that joined after a rekey
     * knows the current key and not the ones before it, which is
     * exactly what the ring can tell it. */
    seeds_.assign((size_t)info_.rekey_ordinal + 1, info_.governing_seed);
    open_.assign(info_.n_envs ? info_.n_envs : 1, (size_t)-1);
    return true;
}

bool LiveFeed::attach(const std::string &name, bool from_start,
                      std::string *err)
{
    K26RlStatus st;

    detach();
    st = k26rl_tap_attach(name.c_str(), from_start ? 1 : 0, &reader_);
    if (st != K26RL_OK) {
        reader_ = 0;
        if (err) {
            *err = "cannot attach to ring ";
            *err += name;
            *err += ": ";
            *err += k26rl_status_str(st);
        }
        return false;
    }
    if (k26rl_tap_reader_info(reader_, &slot_size_, &slot_count_) != K26RL_OK) {
        if (err)
            *err = "the ring will not report its geometry";
        detach();
        return false;
    }
    if (!read_header_(err)) {
        detach();
        return false;
    }
    name_ = name;
    buf_.assign(slot_size_, 0);
    info_.tap = name;
    info_.live = true;
    info_.episode_count = 0;
    return true;
}

LivePoll LiveFeed::poll()
{
    LivePoll r;

    if (!reader_)
        return r;
    for (;;) {
        uint32_t len = 0;
        uint64_t lost = 0;

        if (k26rl_tap_read(reader_, &buf_[0], slot_size_, &len, &lost) !=
            K26RL_OK)
            break;
        lost_ += lost;
        r.lost += lost;
        if (!len)
            break;                  /* nothing further published yet */
        accepted_++;
        r.frames++;
        frame_(&buf_[0], len);
    }
    info_.episode_count = (uint32_t)episodes_.size();
    r.closed = closed();
    info_.clean_close = r.closed;
    return r;
}

/* The episode an environment has open, or a fresh one when this
 * viewer never saw the opening frame: mid-run attach and a lost
 * opening frame are the same situation and get the same answer, which
 * is an episode that says what it does not know. */
Episode *LiveFeed::episode_for_(uint32_t env, uint32_t ordinal)
{
    Episode *e;

    if (env >= open_.size())
        open_.resize((size_t)env + 1, (size_t)-1);
    if (open_[env] != (size_t)-1)
        return episodes_[open_[env]];

    e = new Episode();
    e->ordinal = ordinal;
    e->env = env;
    e->episode = UINT32_MAX;     /* only the closing frame carries it */
    e->seed = ordinal < seeds_.size() ? seeds_[ordinal] : info_.governing_seed;
    e->start_seen = false;
    e->complete = false;
    e->initial_obs.clear();
    open_[env] = episodes_.size();
    episodes_.push_back(e);
    return e;
}

void LiveFeed::start_(const uint8_t *p, uint32_t plen)
{
    uint32_t ordinal, env, episode, dr_count, j;
    uint64_t need;
    Episode *e;

    if (plen < 16)
        return;
    ordinal = get_u32_(p);
    env = get_u32_(p + 4);
    episode = get_u32_(p + 8);
    need = (uint64_t)16 + (uint64_t)obs_total_ * 8;
    if (plen < need)
        return;
    dr_count = get_u32_(p + 12 + (size_t)obs_total_ * 8);
    if (need + (uint64_t)dr_count * 12 > plen)
        return;

    if (env < open_.size() && open_[env] != (size_t)-1) {
        /* An environment whose previous episode's closing frame was
         * overwritten: that episode ends here, unfinished and saying
         * so, rather than absorbing the next one's steps. */
        open_[env] = (size_t)-1;
    }
    e = new Episode();
    e->ordinal = ordinal;
    e->env = env;
    e->episode = episode;
    e->seed = ordinal < seeds_.size() ? seeds_[ordinal] : info_.governing_seed;
    e->start_seen = true;
    e->complete = false;
    e->initial_obs.resize(obs_total_);
    for (j = 0; j < obs_total_; j++)
        e->initial_obs[j] = get_f64_(p + 12 + (size_t)j * 8);
    for (j = 0; j < dr_count; j++) {
        const uint8_t *d = p + 16 + (size_t)obs_total_ * 8 + (size_t)j * 12;
        e->dr_tags.push_back(get_u32_(d));
        e->dr_values.push_back(get_f64_(d + 4));
    }
    if (env >= open_.size())
        open_.resize((size_t)env + 1, (size_t)-1);
    open_[env] = episodes_.size();
    episodes_.push_back(e);
}

/* One step record. Its own step number is in the payload, so a record
 * that does not follow the previous one is a gap of exactly the
 * difference: the ring's frame count says how much was lost, and this
 * says where. */
void LiveFeed::step_(const uint8_t *p, uint32_t plen)
{
    uint32_t ordinal, env, first, count, j;
    uint64_t ncols8 = (uint64_t)obs_total_ + act_total_ + agent_count_;
    const uint8_t *cols;
    Episode *e;

    if (plen < 16)
        return;
    ordinal = get_u32_(p);
    env = get_u32_(p + 4);
    first = get_u32_(p + 8);
    count = get_u32_(p + 12);
    if (count != 1 || plen != 16 + ncols8 * 8 + 12)
        return;                  /* the ring publishes one step a frame */
    cols = p + 16;

    e = episode_for_(env, ordinal);
    if (!e->step_no.empty()) {
        uint32_t last = e->step_no.back();
        if (first <= last)
            return;              /* not a successor: nothing to record */
        if (first > last + 1) {
            StepGap g;
            g.first = last + 1;
            g.count = first - last - 1;
            e->gaps.push_back(g);
        }
    } else if (first > 0) {
        /* The head of the stream is missing: either this viewer
         * joined after the episode began or the opening records were
         * overwritten. Either way the steps before this one are not
         * held and the episode says so. */
        StepGap g;
        g.first = 0;
        g.count = first;
        e->gaps.push_back(g);
    }
    e->step_no.push_back(first);
    for (j = 0; j < obs_total_; j++)
        e->obs.push_back(get_f64_(cols + (size_t)j * 8));
    for (j = 0; j < act_total_; j++)
        e->act.push_back(get_f64_(cols + (size_t)(obs_total_ + j) * 8));
    for (j = 0; j < agent_count_; j++)
        e->rewards.push_back(
            get_f64_(cols + (size_t)(obs_total_ + act_total_ + j) * 8));
    e->flags.push_back(get_u32_(cols + (size_t)ncols8 * 8));
    e->applied_dt.push_back(get_f64_(cols + (size_t)ncols8 * 8 + 4));
    e->step_count = (uint32_t)e->step_no.size();
}

void LiveFeed::end_(const uint8_t *p, uint32_t plen)
{
    uint32_t ordinal, env, episode, steps, j;
    Episode *e;

    if (plen < 20 + (uint64_t)agent_count_ * 8)
        return;
    ordinal = get_u32_(p);
    env = get_u32_(p + 4);
    episode = get_u32_(p + 8);
    steps = get_u32_(p + 12);
    e = episode_for_(env, ordinal);
    e->episode = episode;
    e->end_reason = get_u16_(p + 16);
    e->fault_code = get_u16_(p + 18);
    e->complete = true;
    e->terminal_adjustments.resize(agent_count_);
    for (j = 0; j < agent_count_; j++)
        e->terminal_adjustments[j] = get_f64_(p + 20 + (size_t)j * 8);
    /* The producer's own step count, which is what the episode had.
     * It is not the count this viewer holds when frames were lost,
     * and the difference is the gaps already recorded. */
    if (!e->step_no.empty() && steps > e->step_no.back() + 1) {
        StepGap g;
        g.first = e->step_no.back() + 1;
        g.count = steps - e->step_no.back() - 1;
        e->gaps.push_back(g);
    }
    if (env < open_.size())
        open_[env] = (size_t)-1;
}

void LiveFeed::rekey_(const uint8_t *p, uint32_t plen)
{
    uint64_t seed;
    uint32_t ordinal;

    if (plen < 12)
        return;
    seed = get_u64_(p);
    ordinal = get_u32_(p + 8);
    if (seeds_.size() <= ordinal)
        seeds_.resize((size_t)ordinal + 1, seed);
    seeds_[ordinal] = seed;
}

void LiveFeed::frame_(const uint8_t *frame, uint32_t total)
{
    uint16_t kind;
    uint32_t plen;
    const uint8_t *p;

    if (total < K26RL_EPISODE_FRAME_HEADER_SIZE)
        return;
    kind = get_u16_(frame);
    plen = get_u32_(frame + 4);
    if ((uint64_t)K26RL_EPISODE_FRAME_HEADER_SIZE + plen != total)
        return;
    p = frame + K26RL_EPISODE_FRAME_HEADER_SIZE;
    switch (kind) {
    case K26RL_FRAME_EPISODE_START: start_(p, plen); break;
    case K26RL_FRAME_STEP_CHUNK:    step_(p, plen);  break;
    case K26RL_FRAME_EPISODE_END:   end_(p, plen);   break;
    case K26RL_FRAME_REKEY:         rekey_(p, plen); break;
    default: break;              /* a kind this build does not know */
    }
}

}  /* namespace k26rl_view */
