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
 */
#ifndef K26RL_VIEW_MODEL_H
#define K26RL_VIEW_MODEL_H

#include <stdint.h>

#include <string>
#include <vector>

extern "C" {
#include "k26rl_env.h"
#include "k26rl_episode.h"
}

namespace k26rl_view {

/* One declared observation channel, as the spec publishes it. */
struct Channel {
    uint32_t index;
    uint16_t kind;
    std::string name;
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

    /* True when the last step record is a fault record, which is the
     * format's one step record that is not a transition. */
    bool ends_by_fault() const { return end_reason == K26RL_END_FAULT; }
    /* Transitions, which is the step count less the fault record. */
    uint32_t transitions() const
    {
        return ends_by_fault() && step_count ? step_count - 1 : step_count;
    }
};

/* What the metadata panel reports about the file as a whole. */
struct FileInfo {
    std::string path;
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

class Model {
public:
    Model();
    ~Model();

    /* Open an episode file. A file cut mid-write opens to its
     * readable prefix; info().clean_close is false and
     * info().readable_bytes marks where the valid prefix ended. */
    bool open(const std::string &path, std::string *err);
    void close();

    const FileInfo &info() const { return info_; }
    const Spec &spec() const { return spec_; }
    const std::vector<Trajectory> &trajectories() const { return traj_; }

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

private:
    K26RlEpisodeReader *reader_;
    FileInfo info_;
    Spec spec_;
    std::vector<Trajectory> traj_;
    Episode current_;
    uint32_t current_k_;
    bool have_;
    uint64_t index_lookups_;
    uint64_t episode_reads_;

    void parse_spec_(const uint8_t *blob, uint32_t len);
    void find_trajectories_();
};

/* The ending's name, for a panel that distinguishes the three the
 * format distinguishes. */
const char *end_reason_name(uint16_t reason);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_MODEL_H */
