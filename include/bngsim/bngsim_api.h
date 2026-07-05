/* bngsim/include/bngsim/bngsim_api.h — C API for language-agnostic access
 *
 * Stable C ABI for other languages (Python ctypes, Julia, R, etc.)
 * All functions return 0 on success, negative on error.
 * Thread-safe: each BNGModel* is independent.
 */

#ifndef BNGSIM_API_H
#define BNGSIM_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a loaded model */
typedef struct BNGModel BNGModel;

/* Opaque handle to simulation results */
typedef struct BNGResult BNGResult;

/* ─── Model lifecycle ─────────────────────────────────────────────────────── */

/* Load a .net file and create a model instance.
 * Returns NULL on error. Error message available via bng_last_error(). */
BNGModel *bng_model_create(const char *net_file_path);

/* Deep-copy a model (for parallel workers). */
BNGModel *bng_model_clone(const BNGModel *model);

/* Free a model instance. */
void bng_model_free(BNGModel *model);

/* ─── Parameter access ────────────────────────────────────────────────────── */

/* Set a parameter value by name. Returns 0 on success, -1 if not found. */
int bng_set_param(BNGModel *model, const char *name, double value);

/* Get a parameter value by name. Returns 0 on success, -1 if not found. */
int bng_get_param(const BNGModel *model, const char *name, double *value_out);

/* Reset species to initial concentrations. */
void bng_reset(BNGModel *model);

/* ─── Simulation ──────────────────────────────────────────────────────────── */

/* Run ODE (CVODE) simulation.
 * Returns NULL on error. */
BNGResult *bng_simulate_ode(BNGModel *model, double t_start, double t_end, int n_points,
                            double rtol, double atol);

/* Run SSA (Gillespie) simulation.
 * Returns NULL on error. */
BNGResult *bng_simulate_ssa(BNGModel *model, double t_start, double t_end, int n_points,
                            unsigned long long seed);

/* ─── Result access ───────────────────────────────────────────────────────── */

/* Get dimensions */
int bng_result_n_times(const BNGResult *result);
int bng_result_n_species(const BNGResult *result);
int bng_result_n_observables(const BNGResult *result);

/* Get data pointers (valid until bng_result_free).
 * Data is row-major: array[time_index * n_cols + col_index]. */
const double *bng_result_time(const BNGResult *result);
const double *bng_result_species(const BNGResult *result);
const double *bng_result_observables(const BNGResult *result);

/* Free result. */
void bng_result_free(BNGResult *result);

/* ─── Model introspection ─────────────────────────────────────────────────── */

int bng_n_species(const BNGModel *model);
int bng_n_reactions(const BNGModel *model);
int bng_n_observables(const BNGModel *model);
int bng_n_parameters(const BNGModel *model);

/* ─── Error handling ──────────────────────────────────────────────────────── */

/* Get the last error message (thread-local). */
const char *bng_last_error(void);

/* ─── Reserved names ─────────────────────────────────────────────────────── */

/* Returns a NULL-terminated array of reserved constant names. Do not free. */
const char **bng_reserved_constants(int *count);

/* Returns a NULL-terminated array of reserved function names. Do not free. */
const char **bng_reserved_functions(int *count);

#ifdef __cplusplus
}
#endif

#endif /* BNGSIM_API_H */
