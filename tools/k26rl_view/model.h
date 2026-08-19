/* model.h - the viewer's data layer.
 *
 * Everything the panels draw is computed here, from the episode file
 * alone, with no window, no graphics library, and no interface
 * dependency of any kind. The two presenters sit on top: the headless
 * dump and the graphical interface. They share this one model rather
 * than each computing its own, because a dump that exercised
 * different code from the interface would test nothing about the
 * interface.
 *
 * Reader traffic is counted here as well, so the scrubbing claim (one
 * index lookup and one bounded chunk read per seek, and nothing at
 * all for a step inside an episode already loaded) is a countable
 * fact rather than a timing measurement.
 *
 * There are two sources and one model. A finished recording is read
 * from a file; a run still in progress is read from the telemetry
 * ring it publishes into, decoded beside this in live.h. Both fill
 * the same episodes, so no panel above knows or needs to know which
 * one it is drawing, and the one difference a live source can carry
 * is stated in the episode rather than hidden in it: a ring
 * overwrites, so a viewer the producer outran holds a step stream
 * with holes in it, and the holes are recorded.
 */
#ifndef K26RL_VIEW_MODEL_H
#define K26RL_VIEW_MODEL_H

#include <stdint.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

extern "C" {
#include "k26rl_digest.h"
#include "k26rl_env.h"
#include "k26rl_episode.h"
}

namespace k26rl_view {

/* Little-endian field reads, matching the format's own discipline:
 * every integer is assembled byte by byte, never read as a struct.
 * They live in the header because both sources decode the same
 * fields and one of them would otherwise write these again. */
inline uint16_t get_u16_(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

inline uint32_t get_u32_(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline uint64_t get_u64_(const uint8_t *p)
{
    return (uint64_t)get_u32_(p) | ((uint64_t)get_u32_(p + 4) << 32);
}

inline double get_f64_(const uint8_t *p)
{
    uint64_t bits = get_u64_(p);
    double v;
    memcpy(&v, &bits, sizeof v);
    return v;
}

/* One declared observation channel, as the spec publishes it. */
struct Channel {
    uint32_t index;
    uint16_t kind;
    uint16_t mode;      /* K26RL_TAG_OBS_CHANNEL_MODE, added at ABI 1.2 */
    bool has_mode;      /* false for a file written before the tag existed */
    /* K26RL_TAG_OBS_CHANNEL_SOURCE, added at ABI 1.5: whether this
     * channel carries what a sensor measured or the truth beside it,
     * and which channel is its pair. A viewer draws the two as one
     * overlaid pair, which is the whole reason the tag exists: the
     * pairing is a fact the file states, not a naming convention the
     * viewer guesses at. */
    uint16_t source;
    uint32_t pair;
    bool has_source;
    std::string name;
};

/* One assembly a body binds, as the spec publishes it: the name the
 * asset declared and the digest of the bytes the artifact was built
 * from. A wireframe drawn from an asset on disk is only the craft
 * that flew if the asset's own digest equals this one. */
struct AssemblyRef {
    uint32_t body;
    std::string name;
    uint8_t digest[32];
    bool has_digest;
};

/* One action channel's declared bounds and kind. */
struct ActionDecl {
    uint32_t offset;
    double lo;
    double hi;
    uint16_t kind;      /* K26RL_ACT_KIND_BOX or _DISCRETE */
    uint32_t arity;     /* discrete only */
    bool has_bounds;
    bool has_kind;
};

/* An agent's slice of the observation or action vector. */
struct Slice {
    uint32_t agent;
    uint32_t offset;
    uint32_t count;
};

struct Spec {
    uint32_t abi_version = 0;
    uint32_t endian_probe = 0;
    uint32_t agent_count = 0;
    uint32_t n_envs = 0;
    uint32_t horizon = 0;
    uint32_t obs_total = 0;
    uint32_t act_total = 0;
    uint32_t episode_flags = 0;
    double control_dt = 0.0;
    std::vector<Channel> channels;
    std::vector<ActionDecl> actions;
    std::vector<Slice> obs_slices;
    std::vector<Slice> act_slices;
    std::vector<std::string> body_names;   /* declaration order */
    std::vector<AssemblyRef> assemblies;   /* bodies binding one */
    std::vector<uint8_t> raw;
    /* Tags this build does not know, kept so the metadata panel can
     * say the file carried them rather than pretending it did not. */
    std::vector<std::pair<uint16_t, uint32_t> > unknown_tags;
};

/* A drawable trajectory: four channels of one `as`-bound observe,
 * found by the published naming convention.
 *
 * The reconstruction is observer-relative by construction. The
 * direction components are apparent and the range is the geometric
 * magnitude, and the spec publishes no observer mode, so the viewer
 * cannot say which mode produced a given channel and does not
 * pretend to: the label states the construction and the one
 * condition under which it is exact. */
struct Trajectory {
    std::string base;
    uint32_t dir_x, dir_y, dir_z, range;
};

/* One detection observe, found by the published naming convention: a
 * `_detected` channel with the direction and range channels of the
 * same base beside it.
 *
 * The signal-to-noise ratio and the aspect cosine are optional here
 * because a viewer that refused a set missing one of them would draw
 * nothing for a file that carries everything the line of sight needs.
 * The four that are not optional are the four the line is built
 * from. */
struct Detection {
    std::string base;
    uint32_t detected;
    uint32_t dir_x, dir_y, dir_z, range;
    uint32_t snr;
    uint32_t aspect;
    bool has_snr;
    bool has_aspect;
};

/* One agent's share of the vectors, as the spec's slice tags publish
 * it, with the channels that fall inside the observation slice.
 *
 * Membership comes from the slice tags and from nothing else. The
 * names carry an agent prefix as well, and reading membership off the
 * prefix would be a viewer that disagrees with the ABI the first time
 * two agents declare a channel of one name.
 *
 * `name` is the prefix those qualified names share, which is the only
 * published source for an agent's name, and it is empty when the
 * channels in the slice do not agree on one. */
struct AgentGroup {
    uint32_t index;
    std::string name;
    uint32_t obs_offset;
    uint32_t obs_count;
    uint32_t act_offset;
    uint32_t act_count;
    bool has_obs_slice;
    bool has_act_slice;
    std::vector<uint32_t> channels;   /* channel indices, in slice order */
};

/* A run of step records the ring overwrote before this viewer read
 * them: the first step number missing and how many. A file source
 * never produces one, because a file records every step. */
struct StepGap {
    uint32_t first = 0;
    uint32_t count = 0;
};

struct Episode {
    uint32_t ordinal = 0;
    uint32_t env = 0;
    uint32_t episode = 0;
    uint32_t step_count = 0;
    uint16_t end_reason = 0;
    uint16_t fault_code = 0;
    uint64_t seed = 0;              /* the key its ordinal resolves to */
    std::vector<double> initial_obs;
    std::vector<uint32_t> dr_tags;
    std::vector<double> dr_values;
    std::vector<double> obs;        /* step_count * obs_total */
    std::vector<double> act;        /* step_count * act_total */
    std::vector<double> rewards;    /* step_count * agent_count */
    std::vector<uint32_t> flags;    /* step_count */
    std::vector<double> applied_dt; /* step_count */
    std::vector<double> terminal_adjustments;  /* agent_count */

    /* Live sources only, and each of them a statement about what did
     * not arrive rather than about what did.
     *
     * step_no carries each stored record's own step number, because
     * a stream with a hole in it no longer has position equal to
     * number and a panel labelling a point with its position would
     * be labelling it wrongly. It is empty for a file source, where
     * the two are the same by construction.
     *
     * gaps are the holes themselves. start_seen is false when the
     * episode's opening frame never arrived, in which case the
     * initial observation and the randomisation record are absent
     * rather than guessed. complete is false while the episode is
     * still running, when its ending is not yet a fact. */
    std::vector<uint32_t> step_no;
    std::vector<StepGap> gaps;
    bool start_seen = true;
    bool complete = true;

    /* The step number of the record stored at position i. */
    uint32_t step_at(uint32_t i) const
    {
        return i < step_no.size() ? step_no[i] : i;
    }

    /* True when the last step record is a fault record, which is the
     * format's one step record that is not a transition. */
    bool ends_by_fault() const { return end_reason == K26RL_END_FAULT; }
    /* Transitions, which is the step count less the fault record. */
    uint32_t transitions() const
    {
        return ends_by_fault() && step_count ? step_count - 1 : step_count;
    }
};

/* What the metadata panel reports about the recording as a whole.
 * A live source fills the same fields from the ring preamble's
 * file-header frame, which carries them field for field, and leaves
 * the byte counts at zero because a ring has no length. */
struct FileInfo {
    std::string path;
    /* The ring's name when the source is a running simulation, and
     * empty when it is a file. The two are never both set. */
    std::string tap;
    bool live = false;
    uint32_t format_version = 0;
    uint64_t governing_seed = 0;
    uint32_t rekey_ordinal = 0;
    uint32_t n_envs = 0;
    uint32_t steps_per_chunk = 0;
    uint32_t episode_count = 0;
    uint32_t unindexed_episode_starts = 0;
    uint64_t readable_bytes = 0;
    uint64_t file_bytes = 0;
    bool clean_close = false;
};

class LiveFeed;

class Model {
public:
    Model();
    ~Model();

    /* Open an episode file. A file cut mid-write opens to its
     * readable prefix; info().clean_close is false and
     * info().readable_bytes marks where the valid prefix ended. */
    bool open(const std::string &path, std::string *err);

    /* Attach to a running simulation's telemetry ring instead. The
     * spec, the geometry and the seed come from the ring preamble,
     * which is written once and never overwritten, so a viewer
     * joining an hour into a run gets them whole. from_start joins
     * at the ring's oldest surviving frame rather than at the
     * producer's current position.
     *
     * Nothing about this attaches to the simulation: the mapping is
     * read only, the ring holds no field a consumer may write, and
     * the producer never learns that anybody looked. */
    bool attach(const std::string &tap_name, bool from_start,
                std::string *err);
    bool live() const { return feed_ != 0; }
    void close();

    /* Take whatever the producer has published since the last call,
     * appending to the episodes and to the loss account. Returns the
     * number of frames accepted, zero when nothing new has arrived,
     * and zero always for a file source. Never blocks. */
    uint32_t poll();
    /* The producer has marked the ring closed: the run is over and
     * what is held is all there will be. False for a file source and
     * false for a producer that died without marking it, which is
     * why a caller distinguishes stalled from finished by whether
     * frames are still arriving. */
    bool producer_closed() const;
    uint64_t frames_accepted() const;
    uint64_t frames_lost() const;
    uint32_t ring_slot_size() const;
    uint32_t ring_slot_count() const;

    const FileInfo &info() const { return info_; }
    const Spec &spec() const { return spec_; }
    const std::vector<Trajectory> &trajectories() const { return traj_; }
    const std::vector<Detection> &detections() const { return det_; }

    /* One group per agent, from the slice tags. Both presenters ask
     * this one function, so the groups the headless dump reports are
     * the groups the window draws. */
    std::vector<AgentGroup> agent_groups() const;

    /* Identity of the k-th indexed episode. */
    bool identity(uint32_t k, uint32_t *ordinal, uint32_t *env,
                  uint32_t *episode) const;

    /* The k-th indexed episode, decoded. Loading is cached: asking
     * again for the episode already loaded costs no reader traffic,
     * which is what makes scrubbing inside an episode free. */
    const Episode *load(uint32_t k, std::string *err);
    const Episode *loaded() const { return have_ ? &current_ : 0; }
    uint32_t loaded_index() const { return have_ ? current_k_ : 0xFFFFFFFFu; }

    /* Reader traffic since open, the scrubbing claim's evidence. */
    uint64_t index_lookups() const { return index_lookups_; }
    uint64_t episode_reads() const { return episode_reads_; }

    /* Per-agent cumulative return over an episode: the sum of the
     * agent's per-step rewards, the terminal adjustment included,
     * which is what the episode-return curve draws. */
    static std::vector<double> returns(const Episode &ep,
                                       uint32_t agent_count);
    /* The reconstructed observer-relative point of trajectory t at
     * step i, in metres. */
    static void point(const Episode &ep, const Spec &sp, const Trajectory &t,
                      uint32_t step, double *xyz);
    /* The ground-truth channel paired with a measured one, or
     * K26RL_OBS_PAIR_NONE when it has none.
     *
     * This is the pairing predicate, and it is one function because
     * both presenters ask it: the headless dump through the list
     * below, which is built from it, and the window channel by
     * channel. A predicate written twice is a window overlaying a
     * different pair from the one the gate checks. */
    uint32_t truth_pair_of(uint32_t channel) const;

    /* The measured channels that carry a ground-truth pair, in
     * channel order. A file whose program declared no sensor has
     * none, which is not a failure: it is a file with nothing to
     * overlay. */
    std::vector<std::pair<uint32_t, uint32_t> > overlay_pairs() const;

private:
    K26RlEpisodeReader *reader_;
    LiveFeed *feed_;
    FileInfo info_;
    Spec spec_;
    std::vector<Trajectory> traj_;
    std::vector<Detection> det_;
    Episode current_;
    uint32_t current_k_;
    bool have_;
    uint64_t index_lookups_;
    uint64_t episode_reads_;

    void parse_spec_(const uint8_t *blob, uint32_t len);
    void find_trajectories_();
    void find_detections_();
};

/* The ending's name, for a panel that distinguishes the three the
 * format distinguishes. */
const char *end_reason_name(uint16_t reason);

/* The observer mode's name, as the spec publishes it. A file written
 * before the mode tag existed carries none, and the viewer says so
 * rather than guessing. */
const char *observer_mode_name(uint16_t mode);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_MODEL_H */
