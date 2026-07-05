// bngsim/include/bngsim/wallclock.hpp — Wall-clock timeout/cancellation support
//
// `WallClockBudget` snapshots a monotonic start time when a simulator's run()
// is entered, and exposes a cheap `check()` callable from inside the inner
// integration/iteration loop. When the elapsed time exceeds the configured
// limit, `check()` throws `TimeoutError` — a distinct exception type so the
// Python wrapper can surface a `SimulationTimeout` separate from generic
// solver errors.
//
// All operations are GIL-free (std::chrono::steady_clock) and safe to call
// from inside a `py::gil_scoped_release` block.

#pragma once

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bngsim {

// Exception raised when a simulator's wall-clock budget is exceeded.
//
// Distinct from std::runtime_error so the pybind11 layer can map it to
// `bngsim.SimulationTimeout` rather than `bngsim.SimulationError`. Carries
// both the configured limit and the actual elapsed wall-clock time so the
// Python exception can expose them as attributes.
class TimeoutError : public std::runtime_error {
  public:
    TimeoutError(double timeout_seconds, double elapsed_seconds)
        : std::runtime_error(format_message(timeout_seconds, elapsed_seconds)),
          timeout_seconds_(timeout_seconds), elapsed_seconds_(elapsed_seconds) {}

    double timeout_seconds() const noexcept { return timeout_seconds_; }
    double elapsed_seconds() const noexcept { return elapsed_seconds_; }

  private:
    static std::string format_message(double limit, double elapsed) {
        std::ostringstream os;
        os << "Simulation exceeded wall-clock timeout: elapsed " << elapsed << "s"
           << " > limit " << limit << "s";
        return os.str();
    }

    double timeout_seconds_;
    double elapsed_seconds_;
};

// Wall-clock budget tracker.
//
// `limit_seconds <= 0` disables the budget (check() is a no-op). The hot-path
// cost is one steady_clock::now() call plus a double comparison, ~30 ns on
// modern x86. Callers that check inside tight loops should consider amortizing
// (e.g. one check per N iterations); CVODE/ODE checks once per integration
// step which is naturally rate-limited, while SSA checks every reaction event
// because per-step cost is small.
class WallClockBudget {
  public:
    explicit WallClockBudget(double limit_seconds = 0.0)
        : limit_(limit_seconds), start_(std::chrono::steady_clock::now()) {}

    // True iff a positive limit was supplied at construction.
    bool active() const noexcept { return limit_ > 0.0; }

    // Elapsed wall-clock seconds since construction.
    double elapsed() const noexcept {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

    // Throw TimeoutError if elapsed time exceeds the limit. No-op when inactive.
    void check() const {
        if (!active())
            return;
        double e = elapsed();
        if (e > limit_) {
            throw TimeoutError(limit_, e);
        }
    }

    double limit_seconds() const noexcept { return limit_; }

  private:
    double limit_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace bngsim
