/* k26rl_realtime.h - advancing a simulation against the wall clock,
 * where an action carries a deadline.
 *
 * A second driver beside the stepping one. The stepping driver
 * advances a world as fast as the machine allows and the policy is
 * given all the time it wants; this one runs a fixed control period
 * against a monotonic clock, and time passes whether or not the model
 * has answered. Same world, same actuators, same sensing; what
 * differs is when the calls are made and what happens when an answer
 * is late.
 *
 * It exists to answer a question the stepping driver cannot: would
 * this controller run on the vehicle's computer, at rate, and hold.
 * It is not a training path. Training belongs on the stepping driver,
 * where a run is reproducible and faster than real time; this driver
 * validates what that produced.
 *
 * The loop, one period at a time. The schedule is absolute: period p
 * is due to begin at start + p * period, and a period that runs long
 * does not push the ones after it back. Each period the driver asks
 * the caller's decide hook for an action, and the action is applied
 * if it arrived before the period's own wall-clock end.
 *
 * What a miss is. If the action has not arrived by then the driver
 * holds the action already in force and counts a miss. Holding is
 * what a command register does: it keeps its contents until something
 * writes to it, and a late guidance update leaves the previous
 * command standing rather than commanding nothing. Substituting a
 * declared default would make a late decision a discontinuity no real
 * system has, and ending the run on a miss would make the deadline a
 * cliff and hide how often it is approached, which is the quantity
 * worth knowing. A miss is an outcome and not an error: the run
 * continues, the miss is counted, and its overrun is reported.
 *
 * Because the schedule is absolute, a driver that has fallen behind
 * carries its lateness into the next period's budget, which is what a
 * fixed-tick controller does: a loop that misses its tick is late for
 * the following one whatever the model then costs. The report
 * therefore carries the worst overrun and the worst start lateness
 * separately, so a reader can tell a slow decision from an accumulated
 * schedule slip.
 *
 * Determinism, and what stands in its place. A run against a wall
 * clock cannot be bit-identical to another: which periods miss depends
 * on what else the machine was doing. What holds instead is a replay
 * property. The driver records the action stream it actually applied,
 * held actions on missed periods included, and replaying that stream
 * through the stepping driver reproduces the run's physics exactly,
 * bit for bit. That says the clock changed only which actions were
 * chosen, never how the world answered them, and it is what makes a
 * result taken here admissible as evidence: a run nobody can reproduce
 * is an anecdote, while a run whose recorded actions replay exactly is
 * a measurement of timing on top of physics anyone can check.
 *
 * What the driver asks the operating system for, which is nothing. It
 * pins no core, raises no priority, locks no pages and needs no
 * particular kernel. It reads CLOCK_MONOTONIC and sleeps to an
 * absolute time on that clock, and that is the whole of it. A result
 * that silently depended on a real-time kernel would not transfer to
 * the machine it is a claim about, so the driver asks for nothing and
 * a reader may take a run's numbers as what an ordinary scheduler
 * gave it.
 *
 * Allocation. The loop allocates nothing, opens nothing and takes no
 * lock: every buffer it touches is the caller's and is sized before
 * the first period. The summary that follows the loop sorts a
 * caller-provided scratch buffer through the C library's sort, so it
 * is the one part of a run that may reach an allocator, and it runs
 * after the last period rather than inside one.
 *
 * Threading. One run occupies one thread and the decide hook is
 * called on that thread. A policy evaluator's own rule applies
 * unchanged: one loaded policy is evaluated by one thread at a time,
 * so two concurrent runs load two policies.
 */
#ifndef K26RL_REALTIME_H
#define K26RL_REALTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status registry for this header. Append-only: values are never
 * renumbered. It is a registry of its own rather than an extension of
 * k26rl_env.h's, because that surface is frozen and adding to its
 * enum would be a change to it. */
typedef enum {
    K26RL_RT_OK          = 0,
    K26RL_RT_E_NULL      = 1,   /* null pointer argument */
    K26RL_RT_E_CONFIG    = 2,   /* a configured value is unusable */
    K26RL_RT_E_CLOCK     = 3,   /* the monotonic clock refused */
    K26RL_RT_E_DECIDE    = 4,   /* the decide hook reported failure */
    K26RL_RT_E_ADVANCE   = 5,   /* the advance hook reported failure */
    K26RL_RT_E_SAMPLES   = 6    /* no samples to summarise */
} K26RlRtStatus;

/* The two calls a run is made of. Both are the caller's: this header
 * knows nothing about what produces an action or what advances a
 * world, so the same loop drives a compiled artifact, a stub, or a
 * deliberately slowed stand-in that exists to make a gate fail. */
typedef struct {
    /* Produce this period's action from the current observation. The
     * deadline is on this call. Returns 0 on success; any other value
     * ends the run with K26RL_RT_E_DECIDE. */
    int (*decide)(void *context, const double *obs, double *action);

    /* Apply the action the period settled on and advance the world by
     * one control period of simulated time, writing the observation
     * that results. Returns 0 on success; any other value ends the run
     * with K26RL_RT_E_ADVANCE. */
    int (*advance)(void *context, const double *action, double *obs);

    /* Passed to both hooks and otherwise untouched. */
    void *context;
} K26RlRtHooks;

/* What a run is asked to do. */
typedef struct {
    /* The control rate in hertz. The period is this rounded to the
     * nanosecond, and the report carries the rounded value, so what a
     * run is measured against is the schedule it actually ran. */
    double   rate_hz;

    /* Simulated seconds one advance covers, which the artifact
     * declares. A run is real time in the ordinary sense when this
     * equals one period; when it does not, the report's real-time
     * factor says by how much the run outran or lagged the world. */
    double   control_dt;

    /* Periods to run. Each is one advance, always: a driver that
     * skipped an advance to catch up would be reporting a simulated
     * duration it never computed. */
    uint64_t periods;

    /* Buffer widths, in doubles. */
    uint32_t obs_width;
    uint32_t act_width;

    /* The fraction of the nominal duration the wall clock may exceed
     * while the run is still said to have held rate. Zero demands the
     * schedule exactly, which no ordinary machine gives. */
    double   rate_tolerance;
} K26RlRtConfig;

/* Every buffer a run touches. All are the caller's and none is freed
 * or retained here. The per-period arrays hold `periods` entries, and
 * `action_log` holds `periods * act_width` doubles.
 *
 * Two are inputs as well as outputs. `obs` must carry the world's
 * initial observation before the first period, since the driver's
 * first decision is made from it. `applied` must carry the command
 * register's initial contents, which is what a miss in the very first
 * period holds; the artifact's declared default action is the natural
 * choice, and a caller that leaves it zeroed is declaring that. */
typedef struct {
    double  *obs;         /* obs_width */
    double  *candidate;   /* act_width, scratch the decide hook writes */
    double  *applied;     /* act_width, the command register */
    double  *action_log;  /* periods * act_width, the applied stream */
    double  *latency_us;  /* periods, decision latency in microseconds */
    double  *lateness_us; /* periods, how late the period began against
                           * the schedule, in microseconds */
    double  *scratch;     /* periods, sorted in place by the summary */
    uint8_t *missed;      /* periods, 1 where the deadline was missed */
} K26RlRtBuffers;

/* A latency distribution. The mean is carried beside the quantiles
 * rather than instead of them: a controller that is usually fast and
 * occasionally very slow is the dangerous shape and an average hides
 * it, so the mean is here to be compared against the tail and not to
 * stand for it.
 *
 * Quantiles are by nearest rank over the samples taken, so a quantile
 * deeper than one part in `count` resolves to the largest sample and
 * is not evidence of anything; `count` is reported so a reader can see
 * which quantiles the run could resolve. */
typedef struct {
    uint64_t count;
    double   min_us;
    double   p50_us;
    double   p90_us;
    double   p99_us;
    double   p999_us;
    double   max_us;
    double   mean_us;
} K26RlRtLatency;

/* What a run reports.
 *
 * `periods_run` counts periods whose advance returned, so the
 * simulated duration below is a duration the world computed. A run cut
 * short by a failing hook reports the periods that completed; a run
 * stopped by a failing advance may carry a miss count that includes the
 * period that failed, whose action was applied and whose world did not
 * move. */
typedef struct {
    uint64_t periods_run;
    uint64_t periods_missed;

    /* The period actually scheduled, rate_hz rounded to the
     * nanosecond. */
    uint64_t period_ns;

    /* The largest amount by which an action arrived after its
     * period's end. Zero when nothing missed. */
    double   worst_overrun_us;

    /* The largest amount by which a period began after the time the
     * schedule set for it. Reported apart from the overrun because
     * the two have different causes: one is a slow decision, the other
     * is a schedule the run has not caught up with. */
    double   worst_start_lateness_us;

    double   wall_clock_s;   /* measured, first period start to last end */
    double   nominal_s;      /* periods * period_ns, what was asked for */
    double   simulated_s;    /* periods * control_dt, what was computed */
    double   real_time_factor;   /* simulated_s / wall_clock_s */

    /* 1 when the wall clock stayed inside the nominal duration and its
     * declared tolerance. A run that fell behind says so here, and the
     * two durations above are reported beside it so the verdict can be
     * checked rather than believed. */
    int      rate_held;

    /* The decision latency distribution, and the distribution of how
     * late each period began against the schedule. The second is here
     * because the worst overrun alone cannot separate a decision that
     * ran long from a wakeup that arrived late, and on an ordinary
     * kernel the wakeup's own jitter is a measurable share of a
     * millisecond-scale period. A caller sizing a control period needs
     * both numbers and neither stands for the other. */
    K26RlRtLatency decision;
    K26RlRtLatency start_lateness;
} K26RlRtReport;

/**
 * @brief Decode a real-time status code.
 * @param status The code.
 * @return A stable string; unknown values decode to one string rather
 *         than failing, as the registry is append-only.
 */
const char *k26rl_rt_status_str(K26RlRtStatus status);

/**
 * @brief Run one policy against the wall clock at a declared rate.
 * @param config  The rate, the period count and the buffer widths.
 * @param hooks   The decide and advance calls, and their context.
 * @param buffers Every buffer the run writes; see K26RlRtBuffers for
 *                the two that must be filled before the call.
 * @param out     Receives the run's report, and is cleared before any
 *                refusal can return, so a refused run reads as a run of
 *                no periods rather than as whatever the caller's memory
 *                held.
 * @return K26RL_RT_OK, or the refusal. A hook that fails ends the run
 *         and the report describes the periods completed before it.
 * @note  The loop allocates nothing, performs no I/O of its own and
 *        asks the operating system for no scheduling favour. It reads
 *        CLOCK_MONOTONIC and sleeps to absolute times on it. Any I/O a
 *        run performs is the advance hook's, which is where episode
 *        recording sits.
 */
K26RlRtStatus k26rl_rt_run(const K26RlRtConfig *config,
                           const K26RlRtHooks *hooks,
                           const K26RlRtBuffers *buffers,
                           K26RlRtReport *out);

/**
 * @brief Summarise timing samples as a distribution.
 * @param samples The samples, in the order they were taken.
 * @param count   How many, at least one.
 * @param scratch `count` doubles the sort may use; the samples
 *                themselves are left in the order they were measured.
 * @param out     Receives the summary.
 * @return K26RL_RT_OK, K26RL_RT_E_NULL, or K26RL_RT_E_SAMPLES.
 * @note  Published beside the run because a caller that post-processes
 *        its own samples should reach the same numbers a report
 *        carries, and because a summary nothing outside the run can
 *        call is a summary no gate can hold against a known
 *        distribution.
 */
K26RlRtStatus k26rl_rt_summarise(const double *samples, uint64_t count,
                                 double *scratch, K26RlRtLatency *out);

#ifdef __cplusplus
}
#endif

#endif /* K26RL_REALTIME_H */
