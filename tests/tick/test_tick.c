/* test_tick.c — libk26tick accumulator + clamp + channel dispatch. */
#include "k26tick.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int   g_n_phys = 0;
static int   g_n_ai   = 0;
static int   g_n_frame = 0;
static double g_last_phys_dt = 0.0;
static double g_last_ai_dt = 0.0;

static void cb_phys(double dt, void *u) { (void)u; g_n_phys++; g_last_phys_dt = dt; }
static void cb_ai  (double dt, void *u) { (void)u; g_n_ai++;   g_last_ai_dt = dt; }
static void cb_frame(double dt, void *u) { (void)u; g_n_frame++; (void)dt; }

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

int main(void)
{
    /* ---- World setup ---------------------------------------------- */
    K26TickWorld *w = k26tick_open();
    ASSERT(w);

    K26TickChannel cp = k26tick_add_channel(w, "phys", 120.0, cb_phys, NULL);
    K26TickChannel ca = k26tick_add_channel(w, "ai",    30.0, cb_ai,   NULL);
    K26TickChannel cf = k26tick_add_channel(w, "frame",  0.0, cb_frame, NULL);
    (void)cp; (void)ca; (void)cf;
    ASSERT(k26tick_channel_count(w) == 3);

    /* ---- Single advance: 1/60s → exactly 2 phys steps, 0 ai steps,
     * 1 frame step. ----------------------------------------------- */
    g_n_phys = g_n_ai = g_n_frame = 0;
    k26tick_advance(w, 1.0 / 60.0);
    ASSERT(g_n_phys == 2);    /* (1/60) / (1/120) = 2 */
    ASSERT(g_n_ai   == 0);    /* (1/60) / (1/30) = 0.5 — partial */
    ASSERT(g_n_frame == 1);   /* render-rate */

    /* ---- After enough sub-frame accumulation, ai eventually fires. */
    for (int i = 0; i < 6; i++) k26tick_advance(w, 1.0 / 60.0);
    ASSERT(g_n_ai >= 3);      /* 7 frames × 1/60 = 7/60 ≥ 3 × 1/30 */

    /* ---- Spiral-of-death clamp: a 5s wall hitch must NOT queue 600
     * phys steps. The clamp caps at 5 × longest_step (= 5/30 = ~166ms),
     * so we expect at most 5/30 × 120 = ~20 phys steps. ---------- */
    g_n_phys = 0;
    k26tick_advance(w, 5.0);
    ASSERT(g_n_phys <= 22);

    /* ---- step_dt is consistent (channel's own, not wallclock). --- */
    ASSERT(g_last_phys_dt > 0.0083 && g_last_phys_dt < 0.0084);  /* 1/120 */

    /* ---- Disable + re-enable resets accumulator: no burst on enable. */
    k26tick_set_enabled(w, cp, 0);
    g_n_phys = 0;
    for (int i = 0; i < 10; i++) k26tick_advance(w, 1.0 / 60.0);
    ASSERT(g_n_phys == 0);
    k26tick_set_enabled(w, cp, 1);
    g_n_phys = 0;
    k26tick_advance(w, 1.0 / 60.0);
    /* Re-enable cleared accumulator — so a single 1/60 frame should
     * fire 2 steps, not a backlog. */
    ASSERT(g_n_phys == 2);

    /* ---- now_s is monotonic. ------------------------------------ */
    double t0 = k26tick_now_s();
    double t1 = k26tick_now_s();
    ASSERT(t1 >= t0);

    /* ---- Exact advance: simulated dt, no clamps. ----------------- */
    K26TickWorld *we = k26tick_open();
    ASSERT(we);
    K26TickChannel co = k26tick_add_channel(we, "orbit", 1.0, cb_phys, NULL);
    K26TickChannel cr = k26tick_add_channel(we, "render", 0.0, cb_frame, NULL);
    (void)co; (void)cr;

    /* The clamped entry drops a 3600 s request to the 0.5 s hard cap:
     * zero 1 s steps fire. That is the wallclock guard doing its job,
     * and exactly what a simulated-time caller must not get. */
    g_n_phys = 0; g_n_frame = 0;
    k26tick_advance(we, 3600.0);
    ASSERT(g_n_phys == 0);

    /* The exact entry drops nothing: the 0.5 s residue plus 3599.5 s
     * more is 3600 whole 1 s steps, and the render-rate channel sees
     * the full simulated dt once. */
    g_n_phys = 0; g_n_frame = 0;
    k26tick_advance_exact(we, 3599.5);
    ASSERT(g_n_phys == 3600);
    ASSERT(g_n_frame == 1);
    ASSERT(g_last_phys_dt == 1.0);

    /* Remainder carries across exact advances. */
    g_n_phys = 0;
    k26tick_advance_exact(we, 0.75);
    ASSERT(g_n_phys == 0);
    k26tick_advance_exact(we, 0.25);
    ASSERT(g_n_phys == 1);

    k26tick_close(we);

    k26tick_close(w);
    printf("test_tick: all assertions passed\n");
    return 0;
}
