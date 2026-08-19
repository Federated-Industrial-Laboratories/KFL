/* live.h - the viewer's second source: a simulation that is still
 * running.
 *
 * A serving simulation publishes the record format's frames into a
 * shared memory ring as it steps. This attaches to that ring and
 * decodes what arrives into the same episodes the file reader
 * produces, so every panel above draws a run in progress with no
 * knowledge that its numbers came from anywhere new.
 *
 * Three properties are the whole reason this file exists separately
 * from the ring reader beneath it.
 *
 * It reads and nothing else. The mapping is read only, the ring holds
 * no field a consumer may write, and no call here reaches a running
 * simulation: watching a run cannot change it, and a viewer that is
 * attached, stalled, or killed leaves the run's output byte for byte
 * what it would have been with nobody watching.
 *
 * It reports what it missed. The ring overwrites, so a viewer the
 * producer outran loses whole frames. The reader beneath counts them
 * from the sequence numbers; this turns that count into the thing a
 * person needs, which is where in the step stream the hole is. A step
 * record carries its own step number, so a record arriving at a
 * number the previous one does not lead to is a gap of exactly the
 * difference, recorded against the episode it belongs to. A panel
 * that drew through such a hole without saying so would draw a path
 * the craft never flew.
 *
 * It does not invent the beginning of an episode it joined late. A
 * viewer attaching mid-run, or one that lost an episode's opening
 * frame, has no initial observation and no randomisation record for
 * that episode, and says so rather than presenting the first values
 * it happens to hold as the ones the episode started from.
 */
#ifndef K26RL_VIEW_LIVE_H
#define K26RL_VIEW_LIVE_H

#include <stdint.h>

#include <string>
#include <vector>

#include "model.h"

extern "C" {
#include "k26rl_tap.h"
}

namespace k26rl_view {

/* What one drain of the ring did, so a caller can say whether the
 * picture moved and why it stopped moving. */
struct LivePoll {
    uint32_t frames;    /* frames accepted this round */
    uint64_t lost;      /* frames overwritten before them, this round */
    bool closed;        /* the producer has marked the ring closed */

    LivePoll() : frames(0), lost(0), closed(false) {}
};

class LiveFeed {
public:
    LiveFeed();
    ~LiveFeed();

    /* Attach to the named ring. from_start joins at the ring's oldest
     * surviving frame, which is what makes a viewer report the frames
     * a producer overwrote before it ever looked; otherwise it joins
     * at the producer's current position, which is the ordinary live
     * case and loses nothing because nothing was owed. */
    bool attach(const std::string &name, bool from_start, std::string *err);
    void detach();
    bool attached() const { return reader_ != 0; }

    /* Take everything published since the last call. Never blocks and
     * never waits on the producer: an empty result means nothing new
     * has been published yet, which closed() tells apart from the end
     * of the run. */
    LivePoll poll();

    bool closed() const;
    uint64_t accepted() const { return accepted_; }
    uint64_t lost() const { return lost_; }
    const std::string &name() const { return name_; }
    uint32_t slot_size() const { return slot_size_; }
    uint32_t slot_count() const { return slot_count_; }

    /* Decoded from the preamble's file-header frame, which is written
     * once when the tap is enabled and which no overwrite reaches, so
     * these are known from the moment of attach however late that
     * was. */
    const FileInfo &info() const { return info_; }
    const uint8_t *spec(uint32_t *len) const;

    /* The vector widths a step frame is laid out by. They are the
     * spec's, and the spec is walked by the model above rather than
     * a second time here, so one parser serves both sources. Set
     * after attach and before the first poll. */
    void geometry(uint32_t obs_total, uint32_t act_total,
                  uint32_t agent_count)
    {
        obs_total_ = obs_total;
        act_total_ = act_total;
        agent_count_ = agent_count;
    }

    /* The episodes seen so far, in the order they began. Pointers
     * stay valid until detach, so a caller may hold one across a
     * poll that appends another. */
    const std::vector<Episode *> &episodes() const { return episodes_; }

private:
    K26RlTapReader *reader_;
    std::string name_;
    std::vector<uint8_t> buf_;
    uint32_t slot_size_;
    uint32_t slot_count_;
    FileInfo info_;
    const uint8_t *spec_;
    uint32_t spec_len_;
    uint32_t obs_total_;
    uint32_t act_total_;
    uint32_t agent_count_;
    uint64_t accepted_;
    uint64_t lost_;
    std::vector<Episode *> episodes_;
    /* Per environment, the episode of it that is open, or none. */
    std::vector<size_t> open_;
    /* The seed each rekey ordinal resolves to, from the header frame
     * and the rekey frames, so a live episode reports the seed a
     * file's episode of the same ordinal reports. */
    std::vector<uint64_t> seeds_;

    bool read_header_(std::string *err);
    void frame_(const uint8_t *frame, uint32_t total);
    void start_(const uint8_t *p, uint32_t plen);
    void step_(const uint8_t *p, uint32_t plen);
    void end_(const uint8_t *p, uint32_t plen);
    void rekey_(const uint8_t *p, uint32_t plen);
    Episode *episode_for_(uint32_t env, uint32_t ordinal);

    LiveFeed(const LiveFeed &);
    LiveFeed &operator=(const LiveFeed &);
};

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_LIVE_H */
