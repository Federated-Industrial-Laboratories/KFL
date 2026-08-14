/* test_k26rl_ring.c - the telemetry ring's protocol and geometry.
 *
 * The ring is lossy by design, so the property that matters is not
 * that a consumer sees everything: it is that a consumer's account of
 * what it saw and what it lost adds up to what the producer published,
 * and that nothing half-written is ever delivered as a frame.
 *
 * Gates:
 *   1. Geometry. The documented worked example is pinned exactly: a
 *      twenty-channel, four-action, one-agent, eight-draw environment
 *      sizes a 320-byte slot and a 32768-slot ring. The slot floor is
 *      pinned on a geometry whose budget arithmetic would fall below
 *      it, and the size ceiling refuses a geometry no budget can hold.
 *   2. Naming and exclusivity: empty, over-long, and separator-bearing
 *      names refused; a second producer on one name refused.
 *   3. Round trip: every frame kind published and read back, its
 *      decoded values bitwise equal to what was fed in, and a step
 *      frame's payload equal byte for byte to the layout the file
 *      transport writes for the same values.
 *   4. Overwrite accounting: a producer driven far past the ring's
 *      depth while the consumer reads slowly. Accepted sequences
 *      strictly increase, and accepted plus lost equals the producer's
 *      own published total, which is the producer's count and not the
 *      consumer's arithmetic checking itself.
 *   5. Tear rejection, deterministically induced: a raw read is
 *      interrupted between the sequence load and the copy by enough
 *      publication to overwrite the slot under it. The re-read must
 *      reject it, and the copied bytes must also fail the checksum, so
 *      both guards are shown to fire on the same frame.
 *   6. Late attach: a consumer attaching after the run began recovers
 *      the spec blob from the preamble and joins at the producer's
 *      current position.
 *   7. Close: the closed mark is visible to an attached consumer, the
 *      name is released, and a fresh producer may take it.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#include "k26rl_tap.h"
#include "k26rl_internal.h"

/* The documented worked example's geometry. */
#define OBS 20u
#define ACT 4u
#define AGENTS 1u
#define DRMAX 8u

static char name_buf_[4][64];

static const char *tap_name_(int which, const char *tag)
{
    snprintf(name_buf_[which], sizeof name_buf_[which], "test.%ld.%s",
             (long)getpid(), tag);
    return name_buf_[which];
}

static K26RlEpisodeGeom geom_(uint32_t obs, uint32_t act, uint32_t agents,
                              uint32_t dr_max)
{
    K26RlEpisodeGeom g;

    g.n_envs = 2;
    g.agent_count = agents;
    g.obs_total = obs;
    g.act_total = act;
    g.steps_per_chunk = 1024;
    g.dr_max = dr_max;
    return g;
}

static const uint8_t SPEC_[] = { 0x01, 0x00, 0x04, 0x00, 0x00, 0x00,
                                 0xAA, 0xBB, 0xCC, 0xDD };

static K26RlStatus open_(const char *name, K26RlEpisodeGeom g, K26RlTap **out)
{
    return k26rl_tap_open(name, &g, 0x0123456789ABCDEFull, 3, "3.2", "0.4.1",
                          SPEC_, (uint32_t)sizeof SPEC_, out);
}

/* ---- Gate 1: geometry ---------------------------------------------- */

static void gate_geometry_(void)
{
    K26RlTap *t = NULL;
    K26RlTapReader *r = NULL;
    uint32_t slot = 0, count = 0;
    const char *n = tap_name_(0, "geom");

    shm_unlink("/k26rl.unused");
    ASSERT(open_(n, geom_(OBS, ACT, AGENTS, DRMAX), &t) == K26RL_OK);
    ASSERT(k26rl_tap_attach(n, 1, &r) == K26RL_OK);
    ASSERT(k26rl_tap_reader_info(r, &slot, &count) == K26RL_OK);
    /* The largest frame this geometry admits is the episode-start
     * carrying its eight draws: 16 + 20*8 + 8*12 = 272 payload bytes,
     * 296 with the frame header, 320 rounded to the line. The budget
     * then takes the largest power of two that fits: 32768 slots in
     * 10 MiB. */
    ASSERT(slot == 320);
    ASSERT(count == 32768);
    k26rl_tap_detach(r);
    k26rl_tap_close(t);

    /* A geometry whose budget arithmetic lands below the floor gets
     * the floor: 2100 channels size a 16896-byte slot, of which the
     * budget holds 512, so the floor raises the ring to 1024. */
    n = tap_name_(0, "floor");
    ASSERT(open_(n, geom_(2100, 0, 1, 0), &t) == K26RL_OK);
    ASSERT(k26rl_tap_attach(n, 1, &r) == K26RL_OK);
    ASSERT(k26rl_tap_reader_info(r, &slot, &count) == K26RL_OK);
    ASSERT(slot == 16896);
    ASSERT(count == K26RL_TAP_SLOTS_MIN);
    k26rl_tap_detach(r);
    k26rl_tap_close(t);

    /* And a geometry the ceiling cannot hold even at the floor is
     * refused before anything is created or mapped. */
    n = tap_name_(0, "huge");
    ASSERT(open_(n, geom_(40000, 0, 1, 0), &t) == K26RL_E_GEOMETRY);
    ASSERT(t == NULL);
    ASSERT(k26rl_tap_attach(n, 1, &r) == K26RL_E_TAP_UNAVAILABLE);
}

/* ---- Gate 2: naming and exclusivity -------------------------------- */

static void gate_naming_(void)
{
    K26RlTap *t = NULL, *t2 = NULL;
    char over[K26RL_TAP_NAME_MAX + 3];
    const char *n = tap_name_(1, "name");

    ASSERT(open_("", geom_(OBS, ACT, AGENTS, DRMAX), &t) == K26RL_E_TAP_NAME);
    memset(over, 'a', sizeof over - 1);
    over[sizeof over - 1] = '\0';
    ASSERT(open_(over, geom_(OBS, ACT, AGENTS, DRMAX), &t) ==
           K26RL_E_TAP_NAME);
    ASSERT(open_("has/separator", geom_(OBS, ACT, AGENTS, DRMAX), &t) ==
           K26RL_E_TAP_NAME);
    ASSERT(open_("has space", geom_(OBS, ACT, AGENTS, DRMAX), &t) ==
           K26RL_E_TAP_NAME);
    ASSERT(t == NULL);

    /* Exactly at the bound is accepted; one over is not. */
    memset(over, 'a', K26RL_TAP_NAME_MAX);
    over[K26RL_TAP_NAME_MAX] = '\0';
    ASSERT(open_(over, geom_(OBS, ACT, AGENTS, DRMAX), &t) == K26RL_OK);
    k26rl_tap_close(t);

    ASSERT(open_(n, geom_(OBS, ACT, AGENTS, DRMAX), &t) == K26RL_OK);
    ASSERT(open_(n, geom_(OBS, ACT, AGENTS, DRMAX), &t2) ==
           K26RL_E_TAP_EXISTS);
    ASSERT(t2 == NULL);
    k26rl_tap_close(t);
    /* The name is free again once the producer released it. */
    ASSERT(open_(n, geom_(OBS, ACT, AGENTS, DRMAX), &t) == K26RL_OK);
    k26rl_tap_close(t);
}

/* ---- Gate 3: round trip -------------------------------------------- */

static void gate_roundtrip_(void)
{
    K26RlTap *t = NULL;
    K26RlTapReader *r = NULL;
    const char *n = tap_name_(2, "trip");
    double obs[OBS], act[ACT], rew[AGENTS], adj[AGENTS];
    double dr_vals[DRMAX];
    uint32_t dr_tags[DRMAX];
    uint8_t buf[4096];
    uint32_t len = 0, i, slot = 0, count = 0;
    uint64_t lost = 0;
    const uint8_t *p;

    for (i = 0; i < OBS; i++)
        obs[i] = 1.5 + (double)i;
    for (i = 0; i < ACT; i++)
        act[i] = -3.25 * (double)(i + 1);
    rew[0] = 7.125;
    adj[0] = -0.5;
    for (i = 0; i < DRMAX; i++) {
        dr_tags[i] = 0x100u + i;
        dr_vals[i] = 100.0 + (double)i / 8.0;
    }

    ASSERT(open_(n, geom_(OBS, ACT, AGENTS, DRMAX), &t) == K26RL_OK);
    ASSERT(k26rl_tap_attach(n, 1, &r) == K26RL_OK);
    ASSERT(k26rl_tap_reader_info(r, &slot, &count) == K26RL_OK);

    k26rl_tap_start(t, 1, 4, obs, dr_tags, dr_vals, DRMAX);
    k26rl_tap_step(t, 1, obs, act, rew, K26RL_FLAG_TERMINATED, 0.25);
    k26rl_tap_end(t, 1, K26RL_END_TERMINATED, 0, adj);
    k26rl_tap_rekey(t, 0xFEEDFACEu);
    ASSERT(k26rl_tap_published(t) == 4);

    /* The episode-start frame. */
    ASSERT(k26rl_tap_read(r, buf, slot, &len, &lost) == K26RL_OK);
    ASSERT(len > 0 && lost == 0);
    ASSERT(k26rl_get_u16_(buf + K26RL_FH_OFF_KIND) == K26RL_FRAME_EPISODE_START);
    ASSERT(k26rl_get_u64_(buf + K26RL_FH_OFF_SEQUENCE) == 1);
    p = buf + K26RL_EPISODE_FRAME_HEADER_SIZE;
    ASSERT(k26rl_get_u32_(p) == 3);        /* the rekey ordinal at open */
    ASSERT(k26rl_get_u32_(p + 4) == 1);    /* environment */
    ASSERT(k26rl_get_u32_(p + 8) == 4);    /* episode */
    for (i = 0; i < OBS; i++)
        ASSERT(k26rl_get_f64_(p + 12 + i * 8) == obs[i]);
    ASSERT(k26rl_get_u32_(p + 12 + OBS * 8) == DRMAX);
    for (i = 0; i < DRMAX; i++) {
        const uint8_t *q = p + 16 + OBS * 8 + i * 12;
        ASSERT(k26rl_get_u32_(q) == dr_tags[i]);
        ASSERT(k26rl_get_f64_(q + 4) == dr_vals[i]);
    }

    /* The step frame, published as the step-chunk kind with a count of
     * one. At K equal to one the format's column-major payload is the
     * same bytes as row order, which is what is checked here field by
     * field in the file transport's order. */
    ASSERT(k26rl_tap_read(r, buf, slot, &len, &lost) == K26RL_OK);
    ASSERT(len == K26RL_EPISODE_FRAME_HEADER_SIZE + 28 + 8 * (OBS + ACT + AGENTS));
    ASSERT(k26rl_get_u16_(buf + K26RL_FH_OFF_KIND) == K26RL_FRAME_STEP_CHUNK);
    p = buf + K26RL_EPISODE_FRAME_HEADER_SIZE;
    ASSERT(k26rl_get_u32_(p) == 3);
    ASSERT(k26rl_get_u32_(p + 4) == 1);
    ASSERT(k26rl_get_u32_(p + 8) == 0);    /* first step index */
    ASSERT(k26rl_get_u32_(p + 12) == 1);   /* K */
    p += 16;
    for (i = 0; i < OBS; i++, p += 8)
        ASSERT(k26rl_get_f64_(p) == obs[i]);
    for (i = 0; i < ACT; i++, p += 8)
        ASSERT(k26rl_get_f64_(p) == act[i]);
    for (i = 0; i < AGENTS; i++, p += 8)
        ASSERT(k26rl_get_f64_(p) == rew[i]);
    ASSERT(k26rl_get_u32_(p) == K26RL_FLAG_TERMINATED);
    ASSERT(k26rl_get_f64_(p + 4) == 0.25);

    /* The episode-end frame, its step count the transition this
     * episode recorded. */
    ASSERT(k26rl_tap_read(r, buf, slot, &len, &lost) == K26RL_OK);
    ASSERT(k26rl_get_u16_(buf + K26RL_FH_OFF_KIND) == K26RL_FRAME_EPISODE_END);
    p = buf + K26RL_EPISODE_FRAME_HEADER_SIZE;
    ASSERT(k26rl_get_u32_(p) == 3);
    ASSERT(k26rl_get_u32_(p + 4) == 1);
    ASSERT(k26rl_get_u32_(p + 8) == 4);
    ASSERT(k26rl_get_u32_(p + 12) == 1);
    ASSERT(k26rl_get_u16_(p + 16) == K26RL_END_TERMINATED);
    ASSERT(k26rl_get_u16_(p + 18) == 0);
    ASSERT(k26rl_get_f64_(p + 20) == adj[0]);

    /* The rekey frame carries the next ordinal. */
    ASSERT(k26rl_tap_read(r, buf, slot, &len, &lost) == K26RL_OK);
    ASSERT(k26rl_get_u16_(buf + K26RL_FH_OFF_KIND) == K26RL_FRAME_REKEY);
    p = buf + K26RL_EPISODE_FRAME_HEADER_SIZE;
    ASSERT(k26rl_get_u64_(p) == 0xFEEDFACEu);
    ASSERT(k26rl_get_u32_(p + 8) == 4);

    /* Nothing more, and nothing lost along the way. */
    ASSERT(k26rl_tap_read(r, buf, slot, &len, &lost) == K26RL_OK);
    ASSERT(len == 0);
    ASSERT(k26rl_tap_reader_accepted(r) == 4);
    ASSERT(k26rl_tap_reader_lost(r) == 0);

    k26rl_tap_detach(r);
    k26rl_tap_close(t);
}

/* ---- Gate 4: overwrite accounting ---------------------------------- */

static void gate_overwrite_(void)
{
    K26RlTap *t = NULL;
    K26RlTapReader *r = NULL;
    const char *n = tap_name_(3, "over");
    double obs[OBS], act[ACT], rew[AGENTS];
    uint8_t *buf;
    uint32_t slot = 0, count = 0, len = 0, i;
    uint64_t lost = 0, published, last_seq = 0, reads = 0;
    /* Well past the ring's depth, so overwrite is certain rather than
     * likely: the consumer takes one frame for every ten published. */
    const uint32_t TOTAL = 40000;

    for (i = 0; i < OBS; i++)
        obs[i] = (double)i;
    for (i = 0; i < ACT; i++)
        act[i] = (double)i;
    rew[0] = 1.0;

    ASSERT(open_(n, geom_(OBS, ACT, AGENTS, DRMAX), &t) == K26RL_OK);
    ASSERT(k26rl_tap_attach(n, 1, &r) == K26RL_OK);
    ASSERT(k26rl_tap_reader_info(r, &slot, &count) == K26RL_OK);
    ASSERT(TOTAL > count);
    buf = malloc(slot);
    ASSERT(buf != NULL);

    k26rl_tap_start(t, 0, 0, obs, NULL, NULL, 0);
    for (i = 0; i < TOTAL; i++) {
        k26rl_tap_step(t, 0, obs, act, rew, 0, 0.25);
        if (i % 10 == 0) {
            ASSERT(k26rl_tap_read(r, buf, slot, &len, &lost) == K26RL_OK);
            if (len) {
                uint64_t seq = k26rl_get_u64_(buf + K26RL_FH_OFF_SEQUENCE);
                ASSERT(seq > last_seq);   /* strictly increasing */
                last_seq = seq;
                reads++;
            }
        }
    }
    /* Drain what is still standing in the ring. */
    for (;;) {
        ASSERT(k26rl_tap_read(r, buf, slot, &len, &lost) == K26RL_OK);
        if (!len)
            break;
        {
            uint64_t seq = k26rl_get_u64_(buf + K26RL_FH_OFF_SEQUENCE);
            ASSERT(seq > last_seq);
            last_seq = seq;
            reads++;
        }
    }

    published = k26rl_tap_published(t);
    ASSERT(published == (uint64_t)TOTAL + 1);   /* the start frame too */
    ASSERT(k26rl_tap_reader_accepted(r) == reads);
    /* The accounting check that matters: what the consumer accepted
     * plus what it reports lost is what the producer says it
     * published. A consumer that miscounts a gap fails here rather
     * than agreeing with its own arithmetic. */
    ASSERT(k26rl_tap_reader_accepted(r) + k26rl_tap_reader_lost(r) ==
           published);
    ASSERT(k26rl_tap_reader_lost(r) > 0);       /* overwrite did happen */

    free(buf);
    k26rl_tap_detach(r);
    k26rl_tap_close(t);
}

/* ---- Gate 5: tear rejection ---------------------------------------- */

/* The raw protocol, opened up so the copy can be interrupted. A
 * consumer that sleeps here and hopes for a race proves nothing on a
 * quiet machine; this induces the overwrite deterministically, in the
 * exact window the protocol guards. */
static void gate_tear_(void)
{
    K26RlTap *t = NULL;
    K26RlTapReader *r = NULL;
    const char *n = tap_name_(0, "tear");
    double obs[OBS], act[ACT], rew[AGENTS];
    uint8_t buf[4096];
    uint32_t slot = 0, count = 0, len = 0, i;
    uint64_t lost = 0, target;
    int fd;
    const uint8_t *base, *sl;
    uint64_t seq0, seq1;

    for (i = 0; i < OBS; i++)
        obs[i] = (double)i;
    for (i = 0; i < ACT; i++)
        act[i] = (double)i;
    rew[0] = 1.0;

    ASSERT(open_(n, geom_(OBS, ACT, AGENTS, DRMAX), &t) == K26RL_OK);
    ASSERT(k26rl_tap_attach(n, 1, &r) == K26RL_OK);
    ASSERT(k26rl_tap_reader_info(r, &slot, &count) == K26RL_OK);
    ASSERT(slot <= sizeof buf);

    k26rl_tap_start(t, 0, 0, obs, NULL, NULL, 0);
    for (i = 0; i < 4; i++)
        k26rl_tap_step(t, 0, obs, act, rew, 0, 0.25);

    /* Map the object independently and walk the protocol by hand. */
    {
        char object[K26RL_TAP_NAME_MAX + 16];
        struct stat sb;
        snprintf(object, sizeof object, "%s%s", K26RL_TAP_OBJECT_PREFIX, n);
        fd = shm_open(object, O_RDONLY, 0);
        ASSERT(fd >= 0);
        ASSERT(fstat(fd, &sb) == 0);
        base = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        ASSERT(base != MAP_FAILED);
        target = 2;   /* the first step frame */
        sl = base + k26rl_get_u32_(base + K26RL_TAP_OFF_PREAMBLE) +
             (size_t)(target & (count - 1)) * slot;

        /* Step one of the protocol: the sequence reads as expected. */
        seq0 = k26rl_pub_load_u64_(sl + K26RL_FH_OFF_SEQUENCE);
        ASSERT(seq0 == target);

        /* The interruption: publish a full lap of the ring, so the
         * slot under this reader now holds a later frame. */
        for (i = 0; i < count; i++)
            k26rl_tap_step(t, 0, obs, act, rew, 0, 0.25);

        /* Step two: the copy, then the re-read that must reject it. */
        memcpy(buf, sl, slot);
        seq1 = k26rl_pub_load_u64_(sl + K26RL_FH_OFF_SEQUENCE);
        ASSERT(seq1 != seq0);
        ASSERT(seq1 == target + count);

        /* And the independent guard: read as the frame it was meant
         * to be, the copied bytes fail their own checksum, because
         * the sequence the checksum covers is no longer the one in
         * the image. */
        {
            uint8_t hdr[K26RL_EPISODE_FRAME_HEADER_SIZE];
            uint32_t want, got, plen;
            plen = k26rl_get_u32_(buf + K26RL_FH_OFF_LENGTH);
            ASSERT(plen + K26RL_EPISODE_FRAME_HEADER_SIZE <= slot);
            want = k26rl_get_u32_(buf + K26RL_FH_OFF_CRC);
            memcpy(hdr, buf, sizeof hdr);
            k26rl_put_u64_(hdr + K26RL_FH_OFF_SEQUENCE, target);
            k26rl_put_u32_(hdr + K26RL_FH_OFF_CRC, 0);
            got = k26rl_crc32c(K26RL_CRC32C_INIT, hdr, sizeof hdr);
            got = k26rl_crc32c(got, buf + K26RL_EPISODE_FRAME_HEADER_SIZE,
                               plen);
            ASSERT(got != want);
        }
        munmap((void *)(uintptr_t)base, (size_t)sb.st_size);
    }

    /* The library reader, over the same overwritten ring, reports the
     * loss and never delivers the frame that went. */
    ASSERT(k26rl_tap_read(r, buf, slot, &len, &lost) == K26RL_OK);
    ASSERT(lost > 0);

    k26rl_tap_detach(r);
    k26rl_tap_close(t);
}

/* ---- Gates 6 and 7: late attach, and close ------------------------- */

static void gate_late_and_close_(void)
{
    K26RlTap *t = NULL;
    K26RlTapReader *early = NULL, *late = NULL;
    const char *n = tap_name_(1, "late");
    double obs[OBS], act[ACT], rew[AGENTS];
    const uint8_t *spec = NULL;
    uint8_t buf[4096];
    uint32_t spec_len = 0, slot = 0, len = 0, i;
    uint64_t lost = 0;

    for (i = 0; i < OBS; i++)
        obs[i] = (double)i;
    for (i = 0; i < ACT; i++)
        act[i] = (double)i;
    rew[0] = 1.0;

    ASSERT(open_(n, geom_(OBS, ACT, AGENTS, DRMAX), &t) == K26RL_OK);
    ASSERT(k26rl_tap_attach(n, 1, &early) == K26RL_OK);
    k26rl_tap_start(t, 0, 0, obs, NULL, NULL, 0);
    for (i = 0; i < 50; i++)
        k26rl_tap_step(t, 0, obs, act, rew, 0, 0.25);

    /* Attaching now: the preamble is still whole, so the spec blob
     * comes back verbatim however long the run has been going. */
    ASSERT(k26rl_tap_attach(n, 0, &late) == K26RL_OK);
    ASSERT(k26rl_tap_reader_spec(late, &spec, &spec_len) == K26RL_OK);
    ASSERT(spec_len == sizeof SPEC_);
    ASSERT(memcmp(spec, SPEC_, sizeof SPEC_) == 0);
    ASSERT(k26rl_tap_reader_info(late, &slot, NULL) == K26RL_OK);

    /* A late reader joins where the producer is: nothing is waiting
     * for it, and it reports no loss for a run it was not watching. */
    ASSERT(k26rl_tap_read(late, buf, slot, &len, &lost) == K26RL_OK);
    ASSERT(len == 0 && lost == 0);
    k26rl_tap_step(t, 0, obs, act, rew, 0, 0.25);
    ASSERT(k26rl_tap_read(late, buf, slot, &len, &lost) == K26RL_OK);
    ASSERT(len > 0 && lost == 0);
    ASSERT(k26rl_tap_reader_lost(late) == 0);

    /* Close: the mark is visible to both readers, the early one can
     * still drain, and the name is free for a fresh producer. */
    ASSERT(!k26rl_tap_reader_closed(late));
    k26rl_tap_close(t);
    ASSERT(k26rl_tap_reader_closed(late));
    ASSERT(k26rl_tap_reader_closed(early));
    ASSERT(k26rl_tap_read(early, buf, slot, &len, &lost) == K26RL_OK);
    ASSERT(len > 0);   /* the mapping outlived the name */

    k26rl_tap_detach(early);
    k26rl_tap_detach(late);

    ASSERT(open_(n, geom_(OBS, ACT, AGENTS, DRMAX), &t) == K26RL_OK);
    k26rl_tap_close(t);
}

int main(void)
{
    gate_geometry_();
    gate_naming_();
    gate_roundtrip_();
    gate_overwrite_();
    gate_tear_();
    gate_late_and_close_();
    printf("test_k26rl_ring: all gates pass\n");
    return 0;
}
