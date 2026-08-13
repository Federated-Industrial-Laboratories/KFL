/* test_rl_draws.c: golden draw coordinates.
 *
 * Channel allocation is part of the compiled program's identity: a
 * reset line's value is drawn at stream class 0x0001 and a
 * distribution-valued astro_body attribute at class 0x0002, each on
 * its source-order channel, at draw index 0, for its (environment,
 * episode). This gate pins that end to end: a fixture with two
 * channels in each class records episodes, and every recorded
 * domain-randomisation pair is checked bitwise against the value
 * recomputed HERE, straight through libk26rng at the documented
 * coordinates. A swapped channel, a shifted draw index, a wrong
 * class, or a broken tag composition all go red on the value or the
 * tag, not on a downstream coincidence.
 *
 * Wire: see kflc/Makefile RL_DRAWS_TEST + test target. Skips 77 when
 * the sibling archives are absent (rl_gate_util.h).
 */
#define _GNU_SOURCE
#include "rl_gate_util.h"

#include "k26rl_episode.h"
#include "k26rng.h"

#define WORK_DIR "/tmp/kflc_rl_draws_test"

/* Two class-0x0002 channels (attribute source order: craft pos_x
 * then vel_y) and two class-0x0001 channels (reset source order:
 * pos_y then vel_x). */
static const char *const DRAWS_KFL =
    "form RL_DRAWS\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=uniform(6.6e6, 7.0e6) pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=normal(7350.0, 10.0) vel_z=0.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 2\n"
    "        reset craft.pos_y uniform(-1000.0, 1000.0)\n"
    "        reset craft.vel_x normal(0.0, 5.0)\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.0\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 1.0\n"
    "    end\n"
    "end\n"
    "end\n";

static double golden_uniform_(uint64_t seed, uint16_t cls, uint16_t ch,
                              uint32_t envi, uint32_t ep,
                              double a, double b)
{
    K26RngCoords c;
    c.stream = cls; c.channel = ch;
    c.environment = envi; c.episode = ep; c.draw = 0;
    return k26rng_uniform(k26rng_key(seed), c, a, b);
}

static double golden_normal_(uint64_t seed, uint16_t cls, uint16_t ch,
                             uint32_t envi, uint32_t ep,
                             double mu, double sigma)
{
    K26RngCoords c;
    c.stream = cls; c.channel = ch;
    c.environment = envi; c.episode = ep; c.draw = 0;
    return mu + sigma * k26rng_normal(k26rng_key(seed), c);
}

static void expect_bits_(double got, double want, const char *what,
                         uint32_t envi, uint32_t ep)
{
    if (memcmp(&got, &want, sizeof got) != 0) {
        fprintf(stderr, "draw mismatch: %s env %u episode %u: "
                "recorded %.17g, recomputed %.17g\n",
                what, envi, ep, got, want);
        ASSERT(0);
    }
}

int main(void)
{
    if (!rl_libs_present_("test_rl_draws")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/draws.kfl", DRAWS_KFL);
    rl_compile_(WORK_DIR "/draws.kfl", WORK_DIR "/draws", WORK_DIR);
    rl_run_or_die_(WORK_DIR "/draws --envs 2 --episodes 3 --seed 99"
                   " --out " WORK_DIR "/draws.k26epi > /dev/null");

    const uint64_t seed = 99;
    K26RlEpisodeReader *rd = NULL;
    ASSERT(k26rl_episode_reader_open(WORK_DIR "/draws.k26epi", &rd)
           == K26RL_OK);
    K26RlEpisodeInfo info;
    ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
    ASSERT(info.episode_count >= 6);

    int checked = 0;
    for (uint32_t k = 0; k < info.episode_count; k++) {
        uint32_t ord = 0, envi = 0, ep = 0;
        ASSERT(k26rl_episode_reader_at(rd, k, &ord, &envi, &ep)
               == K26RL_OK);
        K26RlEpisodeData d;
        ASSERT(k26rl_episode_read(rd, ord, envi, ep, &d) == K26RL_OK);

        /* Four channels, source order within each class; the tag is
         * (class << 16) | channel; values recomputed at the
         * documented coordinates must match bitwise. */
        ASSERT(d.dr_count == 4);
        static const uint32_t want_tags[4] = {
            0x00010000u, 0x00010001u, 0x00020000u, 0x00020001u
        };
        for (uint32_t i = 0; i < 4; i++) {
            ASSERT(d.dr_tags[i] == want_tags[i]);
        }
        expect_bits_(d.dr_values[0],
                     golden_uniform_(seed, 0x0001u, 0, envi, ep,
                                     -1000.0, 1000.0),
                     "reset craft.pos_y (class 0x0001 channel 0)",
                     envi, ep);
        expect_bits_(d.dr_values[1],
                     golden_normal_(seed, 0x0001u, 1, envi, ep,
                                    0.0, 5.0),
                     "reset craft.vel_x (class 0x0001 channel 1)",
                     envi, ep);
        expect_bits_(d.dr_values[2],
                     golden_uniform_(seed, 0x0002u, 0, envi, ep,
                                     6.6e6, 7.0e6),
                     "attribute craft.pos_x (class 0x0002 channel 0)",
                     envi, ep);
        expect_bits_(d.dr_values[3],
                     golden_normal_(seed, 0x0002u, 1, envi, ep,
                                    7350.0, 10.0),
                     "attribute craft.vel_y (class 0x0002 channel 1)",
                     envi, ep);
        checked++;

        k26rl_episode_free(&d);
    }
    k26rl_episode_reader_close(rd);

    printf("test_rl_draws: %d episode(s), every recorded draw"
           " recomputed bitwise at its documented coordinates\n",
           checked);
    return 0;
}
