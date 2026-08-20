/* rl_tap_drain.c: a telemetry-ring reader for the package's tap gate.
 *
 * The gate arms the ring from Python and needs a consumer that is not
 * the thing under test. This is that consumer: it attaches to a named
 * ring with the library's own reader, drains it to the producer's
 * close, and checks the arithmetic a lossless drain must satisfy.
 *
 * The check is the sequence arithmetic, not a frame count: the first
 * slot frame is sequence 1, every accepted frame's sequence is one
 * past the last, and the reader reports no loss. A drain that skipped
 * a frame, double-counted one, or silently restarted its numbering
 * fails here rather than being reported as a clean run.
 *
 * Usage:  rl_tap_drain <ring-name> <timeout-seconds>
 *
 * It prints one line, "attached", as soon as it holds the ring, so
 * the caller can start stepping without racing the attach, and then
 * one report line per finding. Frames are counted by kind so the
 * caller can assert that the run's shape reached the ring.
 *
 * Exit codes: 0 a clean drain, 1 a defect in what was drained
 * (a sequence gap, reported loss, no frames at all, or the producer
 * never closing inside the timeout), 2 a usage or attach failure.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k26rl_episode.h"
#include "k26rl_tap.h"

/* Frame header field offsets, as k26rl_episode.h states them. */
#define FH_OFF_KIND     0
#define FH_OFF_LENGTH   4
#define FH_OFF_SEQUENCE 8

static uint16_t get_u16_(const uint8_t *p)
{
    return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8));
}

static uint64_t get_u64_(const uint8_t *p)
{
    uint64_t v = 0;
    int i;
    for (i = 7; i >= 0; i--)
        v = (v << 8) | (uint64_t)p[i];
    return v;
}

static void nap_(long micros)
{
    struct timespec ts;
    ts.tv_sec = micros / 1000000L;
    ts.tv_nsec = (micros % 1000000L) * 1000L;
    nanosleep(&ts, NULL);
}

static double now_(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    K26RlTapReader *r = NULL;
    K26RlStatus st;
    uint32_t slot_size = 0, slot_count = 0;
    uint8_t *buf;
    uint64_t expect = 1, first_seq = 0, last_seq = 0;
    uint64_t accepted = 0, reported_lost = 0;
    uint64_t starts = 0, chunks = 0, ends = 0, others = 0;
    double deadline, limit;
    int gap = 0, closed = 0;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <ring-name> <timeout-seconds>\n",
                argv[0]);
        return 2;
    }
    limit = atof(argv[2]);
    if (limit <= 0.0) {
        fprintf(stderr, "timeout must be positive\n");
        return 2;
    }

    /* The producer arms the ring before spawning this reader, so one
     * attach attempt would still race the spawn on a loaded machine;
     * retry to the deadline and fail only if the ring never appears. */
    deadline = now_() + limit;
    for (;;) {
        st = k26rl_tap_attach(argv[1], 1, &r);
        if (st == K26RL_OK)
            break;
        if (now_() >= deadline) {
            fprintf(stderr, "cannot attach to ring %s: %s\n", argv[1],
                    k26rl_status_str(st));
            return 2;
        }
        nap_(1000);
    }
    if (k26rl_tap_reader_info(r, &slot_size, &slot_count) != K26RL_OK) {
        fprintf(stderr, "the ring will not report its geometry\n");
        k26rl_tap_detach(r);
        return 2;
    }
    buf = (uint8_t *)malloc(slot_size);
    if (!buf) {
        fprintf(stderr, "cannot size a read buffer of %u bytes\n",
                slot_size);
        k26rl_tap_detach(r);
        return 2;
    }
    printf("attached\n");
    fflush(stdout);

    deadline = now_() + limit;
    for (;;) {
        uint32_t len = 0;
        uint64_t lost = 0;

        if (k26rl_tap_read(r, buf, slot_size, &len, &lost) != K26RL_OK)
            break;
        reported_lost += lost;
        if (!len) {
            /* Nothing further published yet. The producer's close is
             * the end of the run, and the frames it published before
             * closing are still readable after it, so the drain ends
             * only when a closed ring has nothing left. */
            if (k26rl_tap_reader_closed(r)) {
                closed = 1;
                break;
            }
            if (now_() >= deadline)
                break;
            nap_(500);
            continue;
        }
        {
            uint64_t seq = get_u64_(buf + FH_OFF_SEQUENCE);
            uint16_t kind = get_u16_(buf + FH_OFF_KIND);

            if (seq != expect) {
                printf("sequence gap: expected %llu, read %llu\n",
                       (unsigned long long)expect,
                       (unsigned long long)seq);
                gap = 1;
            }
            expect = seq + 1;
            if (!accepted)
                first_seq = seq;
            last_seq = seq;
            accepted++;
            if (kind == K26RL_FRAME_EPISODE_START)
                starts++;
            else if (kind == K26RL_FRAME_STEP_CHUNK)
                chunks++;
            else if (kind == K26RL_FRAME_EPISODE_END)
                ends++;
            else
                others++;
        }
    }

    printf("slot_size %u slot_count %u\n", slot_size, slot_count);
    printf("accepted %llu lost %llu first_seq %llu last_seq %llu\n",
           (unsigned long long)accepted,
           (unsigned long long)reported_lost,
           (unsigned long long)first_seq,
           (unsigned long long)last_seq);
    printf("kinds starts %llu chunks %llu ends %llu others %llu\n",
           (unsigned long long)starts, (unsigned long long)chunks,
           (unsigned long long)ends, (unsigned long long)others);
    printf("closed %d\n", closed);
    fflush(stdout);

    free(buf);
    k26rl_tap_detach(r);

    if (gap || reported_lost || !accepted || !closed)
        return 1;
    /* The arithmetic a clean drain satisfies: contiguous from the
     * first slot frame, and as many frames accepted as the span
     * covers. */
    if (first_seq != 1 || last_seq - first_seq + 1 != accepted)
        return 1;
    return 0;
}
