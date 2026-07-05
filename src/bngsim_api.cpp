// bngsim/src/bngsim_api.cpp — C API implementation
//
// Stable C ABI for Python ctypes, Julia, R, and similar consumers.
// All functions return 0 on success, negative on error. Never exit().
// Includes null safety on all inputs and uses unique_ptr for
// exception-safe handle allocation.

#include "bngsim/bngsim_api.h"
#include "bngsim/bngsim.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Thread-local error message buffer
static thread_local std::string last_error;

static void set_error(const std::string &msg) { last_error = msg; }

// ─── Opaque handle types ─────────────────────────────────────────────────────

struct BNGModel {
    std::unique_ptr<bngsim::NetworkModel> model;
};

struct BNGResult {
    std::unique_ptr<bngsim::Result> result;
};

// ─── Model lifecycle ─────────────────────────────────────────────────────────

extern "C" {

BNGModel *bng_model_create(const char *net_file_path) {
    if (!net_file_path) {
        set_error("Null file path");
        return nullptr;
    }
    try {
        auto handle = std::make_unique<BNGModel>();
        handle->model =
            std::make_unique<bngsim::NetworkModel>(bngsim::NetworkModel::from_net(net_file_path));
        return handle.release(); // caller owns
    } catch (const std::exception &e) {
        set_error(e.what());
        return nullptr;
    }
}

BNGModel *bng_model_clone(const BNGModel *model) {
    if (!model || !model->model) {
        set_error("Null model");
        return nullptr;
    }
    try {
        auto handle = std::make_unique<BNGModel>();
        handle->model = std::make_unique<bngsim::NetworkModel>(model->model->clone());
        return handle.release();
    } catch (const std::exception &e) {
        set_error(e.what());
        return nullptr;
    }
}

void bng_model_free(BNGModel *model) { delete model; }

// ─── Parameter access ────────────────────────────────────────────────────────

int bng_set_param(BNGModel *model, const char *name, double value) {
    if (!model || !model->model) {
        set_error("Null model");
        return -1;
    }
    if (!name) {
        set_error("Null parameter name");
        return -1;
    }
    try {
        model->model->set_param(name, value);
        return 0;
    } catch (const std::exception &e) {
        set_error(e.what());
        return -1;
    }
}

int bng_get_param(const BNGModel *model, const char *name, double *value_out) {
    if (!model || !model->model) {
        set_error("Null model");
        return -1;
    }
    if (!name) {
        set_error("Null parameter name");
        return -1;
    }
    if (!value_out) {
        set_error("Null output pointer");
        return -1;
    }
    try {
        *value_out = model->model->get_param(name);
        return 0;
    } catch (const std::exception &e) {
        set_error(e.what());
        return -1;
    }
}

void bng_reset(BNGModel *model) {
    if (model && model->model) {
        model->model->reset();
    }
}

// ─── Simulation ──────────────────────────────────────────────────────────────

BNGResult *bng_simulate_ode(BNGModel *model, double t_start, double t_end, int n_points,
                            double rtol, double atol) {
    if (!model || !model->model) {
        set_error("Null model");
        return nullptr;
    }
    try {
        bngsim::CvodeSimulator sim(*model->model);
        bngsim::TimeSpec times{t_start, t_end, n_points};
        bngsim::SolverOptions opts;
        opts.rtol = rtol;
        opts.atol = atol;

        auto handle = std::make_unique<BNGResult>();
        handle->result = std::make_unique<bngsim::Result>(sim.run(times, opts));
        return handle.release();
    } catch (const std::exception &e) {
        set_error(e.what());
        return nullptr;
    }
}

BNGResult *bng_simulate_ssa(BNGModel *model, double t_start, double t_end, int n_points,
                            unsigned long long seed) {
    if (!model || !model->model) {
        set_error("Null model");
        return nullptr;
    }
    try {
        bngsim::SsaSimulator sim(*model->model);
        bngsim::TimeSpec times{t_start, t_end, n_points};

        auto handle = std::make_unique<BNGResult>();
        handle->result = std::make_unique<bngsim::Result>(sim.run(times, seed));
        return handle.release();
    } catch (const std::exception &e) {
        set_error(e.what());
        return nullptr;
    }
}

// ─── Result access ───────────────────────────────────────────────────────────

int bng_result_n_times(const BNGResult *r) { return r ? r->result->n_times() : 0; }
int bng_result_n_species(const BNGResult *r) { return r ? r->result->n_species() : 0; }
int bng_result_n_observables(const BNGResult *r) { return r ? r->result->n_observables() : 0; }

const double *bng_result_time(const BNGResult *r) { return r ? r->result->time().data() : nullptr; }
const double *bng_result_species(const BNGResult *r) {
    return r ? r->result->species_data().data() : nullptr;
}
const double *bng_result_observables(const BNGResult *r) {
    return r ? r->result->observable_data().data() : nullptr;
}

void bng_result_free(BNGResult *r) { delete r; }

// ─── Model introspection ─────────────────────────────────────────────────────

int bng_n_species(const BNGModel *m) { return m ? m->model->n_species() : 0; }
int bng_n_reactions(const BNGModel *m) { return m ? m->model->n_reactions() : 0; }
int bng_n_observables(const BNGModel *m) { return m ? m->model->n_observables() : 0; }
int bng_n_parameters(const BNGModel *m) { return m ? m->model->n_parameters() : 0; }

// ─── Error handling ──────────────────────────────────────────────────────────

const char *bng_last_error(void) { return last_error.c_str(); }

// ─── Reserved names ──────────────────────────────────────────────────────────

static const char *reserved_constants_[] = {"_pi", "_e", "_kB", "_NA", "_R", "_h", "_F", nullptr};
static const char *reserved_functions_[] = {
    "time", "t",     "sin",   "cos",   "tan",  "asin",  "acos",   "atan", "sinh",  "cosh",
    "tanh", "asinh", "acosh", "atanh", "exp",  "log",   "ln",     "log2", "log10", "sqrt",
    "abs",  "floor", "ceil",  "round", "rint", "trunc", "min",    "max",  "clamp", "avg",
    "sum",  "erf",   "erfc",  "sign",  "sgn",  "if",    "mratio", "tgamma", nullptr};

const char **bng_reserved_constants(int *count) {
    if (count)
        *count = 7;
    return reserved_constants_;
}

const char **bng_reserved_functions(int *count) {
    if (count)
        *count = 38;
    return reserved_functions_;
}

} // extern "C"
