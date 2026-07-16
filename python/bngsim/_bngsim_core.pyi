"""
bngsim C++ engine bindings (low-level)
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['CvodeSimulator', 'HAS_KLU', 'HAS_LAPACK_DENSE', 'HAS_MIR', 'HAS_NFSIM', 'HAS_RULEMONKEY', 'ModelBuilder', 'NetworkModel', 'NfsimSimulator', 'ResultCore', 'RuleMonkeySimulator', 'SolverOptions', 'SolverStats', 'SsaDiagnostics', 'SsaSimulator', 'SteadyStateOptions', 'SteadyStateResultCore', 'TimeSpec', 'bench_ssa_propensity_jit', 'emit_ssa_propensity_source_structure', 'find_steady_state', 'reserved_names']
class CvodeSimulator:
    def __init__(self, model: NetworkModel) -> None:
        """
        Create an ODE (CVODE) simulator for the given model
        """
    def run(self, times: TimeSpec, opts: SolverOptions = ...) -> ResultCore:
        """
        Run ODE simulation (releases GIL)
        """
    def set_max_steps(self, max_steps: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_tolerances(self, rtol: typing.SupportsFloat | typing.SupportsIndex, atol: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class ModelBuilder:
    def __init__(self) -> None:
        ...
    def add_discontinuity_trigger(self, condition_expr: str) -> None:
        """
        Register a time-dependent inequality condition (e.g. 'time()<=0.125') as a discontinuity trigger (GH #72). It is compiled and registered as a CVODE root so the ODE integrator stops at the time threshold and cannot step over a narrow forcing pulse encoded by a piecewise assignment rule. No state assignment; duplicate strings are ignored.
        """
    def add_event(self, id: str, trigger_expr: str, assignments: collections.abc.Sequence[tuple[typing.SupportsInt | typing.SupportsIndex, str]], delay: typing.SupportsFloat | typing.SupportsIndex = 0.0, priority: typing.SupportsInt | typing.SupportsIndex = 0, persistent: bool = True, initial_value: bool = True, use_values_from_trigger_time: bool = True, delay_expr: str = '', priority_expr: str = '', assignment_ode_only: collections.abc.Sequence[bool] = []) -> None:
        """
        Add a discrete event. assignments = [(sp_idx_0based, value_expr_str), ...]. delay_expr/priority_expr (when non-empty) are evaluated at trigger time and override the constant delay/priority. assignment_ode_only (GH #81) is a parallel bool list; true entries apply under ODE only and are skipped under SSA (the compartment-resize concentration rescale that must not perturb counts).
        """
    def add_function(self, name: str, expression: str) -> int:
        """
        Add a function (named expression). Returns 0-based index.
        """
    def add_observable(self, name: str, entries: collections.abc.Sequence[tuple[typing.SupportsInt | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex]]) -> int:
        """
        Add an observable. entries = [(sp_idx_0based, factor), ...]. Returns 0-based index.
        """
    def add_parameter(self, name: str, value: typing.SupportsFloat | typing.SupportsIndex, expression: str = '', is_expression: bool = False) -> int:
        """
        Add a parameter. Returns 0-based index.
        """
    def add_reaction(self, reactants: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], products: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], type: str, rate_law: str, stat_factor: typing.SupportsFloat | typing.SupportsIndex = 1.0, apply_species_factor: bool = True, ssa_volume_factor: typing.SupportsFloat | typing.SupportsIndex = 1.0, per_species_volume_scaling: bool = False, is_rate_rule_ode: bool = False, ssa_live_volume_idx0: typing.SupportsInt | typing.SupportsIndex = -1, ssa_live_volume_exp: typing.SupportsFloat | typing.SupportsIndex = 0.0, ode_only: bool = False) -> int:
        """
        Add a reaction. type = 'elementary', 'functional', or 'mm'. apply_species_factor=False marks an SBML-style functional rate where the kinetic-law expression already includes the reactant population factor. ssa_volume_factor is an SSA-only multiplier converting an ODE-units rate (storage/time) to amount/time propensity; SBML loaders pass V_c (the reaction's compartment volume). Default 1.0 leaves .net and V=1 SBML behavior unchanged. per_species_volume_scaling=True is the cross-compartment ODE accumulator: compute_derivs divides the rate by each affected species's volume_factor instead of using a scalar rate. Default false preserves .net and uniform-V_s SBML behavior. is_rate_rule_ode=True (GH #81) marks a Functional `[] -> [X]` reaction compiled from an SBML rate rule; the SSA/PSA loop integrates X deterministically instead of firing it as a stochastic channel (ODE path unaffected). ssa_live_volume_idx0/ssa_live_volume_exp (GH #81) apply the SSA live-volume correction for a variable-volume compartment: the propensity is multiplied by (ssa_volume_factor/conc[idx])^exp where idx is the promoted compartment species and exp = n_h-1. Default idx0=-1 leaves .net / static-V byte-identical. Returns 0-based index.
        """
    def add_reaction_live_volume_term(self, rxn_idx0: typing.SupportsInt | typing.SupportsIndex, live_idx0: typing.SupportsInt | typing.SupportsIndex, v_static: typing.SupportsFloat | typing.SupportsIndex, exp: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        GH #144 (case 4): append a cross-compartment SSA live-volume term to a reaction. compute_rxn_rate multiplies the SSA propensity by (v_static / conc[live_idx0])^exp. One term per variable-volume compartment the reaction touches. No-op if rxn_idx0 is out of range.
        """
    def add_species(self, name: str, init_conc: typing.SupportsFloat | typing.SupportsIndex, fixed: bool = False, volume_factor: typing.SupportsFloat | typing.SupportsIndex = 1.0, amount_valued: bool = False, reported: bool = True) -> int:
        """
        Add a species. Returns 0-based index. volume_factor is the storage→amount conversion factor used by the SSA fire step (Δstorage = ±1/volume_factor per fire); SBML loaders pass the species's compartment volume V_c. Default 1.0 leaves .net and V=1 SBML behavior unchanged. amount_valued=True makes the species participate in reaction species factors by its amount (stored × volume_factor) rather than the stored concentration (GH #75, hasOnlySubstanceUnits=true semantics); default False is byte-identical. reported=False keeps the species as full integrator state but omits it from the trajectory output columns (GH #71 — event-mutated parameters/compartments promoted to species); default True is byte-identical.
        """
    def add_species_param_ref(self, species_idx0: typing.SupportsInt | typing.SupportsIndex, param_name: str) -> None:
        """
        Record that species[species_idx0]'s initial concentration is controlled by the parameter named param_name. Used by CVODES forward sensitivity analysis to seed dY_i(0)/dp_k = 1 when p_k is requested via sensitivity_params.
        """
    def build(self) -> NetworkModel:
        """
        Finalize and build the NetworkModel. Builder is consumed.
        """
    def enable_rateof(self) -> None:
        """
        Enable SBML rateOf csymbol support (GH #106). Call when the model references rateOf(species) in event triggers or rate-rule / assignment-rule functions (emitted as 'rate_of__<species>' tokens). build() then sizes the live derivative buffer and registers the accessor variables those expressions read.
        """
    def set_compute_conservation_laws(self, enabled: bool) -> None:
        """
        Enable/disable conservation-law detection in build() (GH #102). The detector is dense O(n_species^3) Gaussian elimination consumed only by the steady-state solver; disable it to keep setup O(reactions) for very large ODE-only networks (~100K species). Default True preserves existing behavior.
        """
    def set_reaction_live_volume(self, rxn_idx0: typing.SupportsInt | typing.SupportsIndex, ssa_live_volume_idx0: typing.SupportsInt | typing.SupportsIndex, ssa_live_volume_exp: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        GH #81: set the SSA live-volume correction on an already-added reaction (rxn_idx0 = 0-based index from add_reaction). Used by the SBML loader to tag a variable-volume reaction once its event-resized compartment has been promoted to a species (which happens after reaction emission). No-op if rxn_idx0 is out of range.
        """
    def set_species_ode_live_volume(self, species_idx0: typing.SupportsInt | typing.SupportsIndex, live_idx0: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        GH #144 (case 4): set the cross-compartment ODE live-volume divide on a species. compute_derivs divides this species's per-species accumulation by conc[live_idx0] (the promoted compartment species = V_live) instead of its static volume_factor. No-op if species_idx0 is out of range.
        """
    def set_species_rateof_amount(self, species_idx0: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        GH #231 (rateOf): mark a hasOnlySubstanceUnits=true species so its rateOf csymbol reports the amount-rate (volume_factor * stored-rate) instead of the stored d(conc)/dt. Correct for constant- and variable-volume compartments alike (the integrator stores amount/V_static). No-op if species_idx0 is out of range.
        """
class NetworkModel:
    @staticmethod
    def from_net(path: str) -> NetworkModel:
        """
        Load a model from a BNG .net file
        """
    def __repr__(self) -> str:
        ...
    def _dense_analytical_jacobian(self, t: typing.SupportsFloat | typing.SupportsIndex, conc: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> list[float]:
        """
        Assemble the dense analytical Jacobian at (t, conc), flat column-major. Test/diagnostic hook for the FD cross-check. GH #76.
        """
    def _eval_functions(self, t: typing.SupportsFloat | typing.SupportsIndex, conc: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> dict:
        """
        Evaluate functions at (t, conc) → {name: value}. Test/diagnostic hook.
        """
    def _eval_rhs(self, t: typing.SupportsFloat | typing.SupportsIndex, conc: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> list[float]:
        """
        Evaluate dy/dt at (t, conc). Test/diagnostic hook.
        """
    def add_table_function_arrays(self, name: str, xs: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], ys: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], index_name: str = 'time', method: str = 'linear') -> None:
        """
        Add a table function from arrays
        """
    def add_table_function_file(self, name: str, filepath: str, index_name: str = 'time', method: str = 'linear') -> None:
        """
        Add a table function from a .tfun file
        """
    def clone(self) -> NetworkModel:
        """
        Deep copy the model (for parallel workers)
        """
    def codegen_data(self) -> dict:
        """
        Return model data for code generation
        """
    def codegen_jacobian_plan(self) -> dict:
        """
        Dense analytical-Jacobian scatter plan (Elementary + MM, rows resolved) for the codegen .so. GH #76 Task 4.
        """
    def event_sensitivity_unsupported_reason(self, sens_param_names: collections.abc.Sequence[str]) -> str | None:
        """
        Return a reason string if any event blocks Phase-1 forward sensitivity for the given sensitivity-parameter names, else None (GH #212).
        """
    def functional_jacobian_context(self) -> dict:
        """
        Read-only context for the Python analytical-Jacobian differentiator (GH #76).
        """
    def get_concentration(self, name: str) -> float:
        """
        Get a single species concentration by name
        """
    def get_param(self, name: str) -> float:
        """
        Get a parameter value by name
        """
    def get_state(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Bulk-copy all species concentrations into a new float64 ndarray, ordered like species_names(). O(n_species), one Python call (GH #102).
        """
    def pending_sensitivity_seed(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        The pending carry-over forward-sensitivity seed dx/dθ as an (n_species, n_params) ndarray, or shape (0, 0) when none is pending. Columns are pending_sensitivity_seed_param_names() (GH #210).
        """
    def reported_volume_factors(self) -> list[float]:
        """
        Per-reported-species volume_factor (V_c), in reported-species order.
        """
    def reset(self) -> None:
        """
        Reset species to initial concentrations
        """
    def reset_rhs_counters(self) -> None:
        """
        Reset the RHS instrumentation counters to zero.
        """
    def save_concentrations(self) -> None:
        """
        Snapshot current concentrations as new initial state
        """
    def set_concentration(self, name: str, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set a single species concentration by name
        """
    def set_functional_jacobian(self, terms: list) -> bool:
        """
        Compile and attach symbolically-derived Functional Jacobian terms (GH #76). Returns True if the analytical Jacobian was populated.
        """
    def set_param(self, name: str, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set a parameter value by name
        """
    def set_params(self, params: dict) -> None:
        """
        Set multiple parameters from a dict
        """
    def set_state(self, state: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Bulk-assign all species concentrations from a 1-D float64 ndarray, ordered like species_names(). O(n_species), one Python call (GH #102).
        """
    @property
    def analytical_jacobian_complete(self) -> bool:
        """
        True iff the analytical Jacobian covers every reaction (Elementary + attached Functional terms). GH #76.
        """
    @property
    def conservation_laws(self) -> dict:
        """
        Conservation laws detected from stoichiometry matrix
        """
    @property
    def has_pending_sensitivity_seed(self) -> bool:
        """
        True iff a forward-sensitivity carry-over seed (dx/dθ) from a prior phase is pending, ready to seed a carry_sensitivities=True run (GH #210).
        """
    @property
    def ic_state_dirty(self) -> bool:
        """
        True iff the current species state is carried-over dynamics from a previous run() (not a fresh initial condition). Forward sensitivities requested on a dirty state require carry_sensitivities=True (else they raise). Cleared by reset()/save_concentrations() (GH #210).
        """
    @property
    def load_warnings(self) -> list[str]:
        ...
    @property
    def n_discontinuity_triggers(self) -> int:
        ...
    @property
    def n_events(self) -> int:
        ...
    @property
    def n_functions(self) -> int:
        ...
    @property
    def n_observables(self) -> int:
        ...
    @property
    def n_parameters(self) -> int:
        ...
    @property
    def n_reactions(self) -> int:
        ...
    @property
    def n_species(self) -> int:
        ...
    @property
    def n_table_functions(self) -> int:
        ...
    @property
    def observable_names(self) -> list[str]:
        ...
    @property
    def param_is_expression(self) -> list[bool]:
        """
        Per-parameter ``is_expression`` flag (True for derived ConstantExpression parameters such as BNG2.pl-emitted ``_rateLaw{N}``).
        """
    @property
    def param_names(self) -> list[str]:
        """
        List of all parameter names
        """
    @property
    def pending_sensitivity_seed_param_names(self) -> list[str]:
        """
        Parameter names labeling the columns of pending_sensitivity_seed() (GH #210).
        """
    @property
    def rhs_eval_count(self) -> int:
        """
        Number of RHS (compute_derivs_core) evaluations so far.
        """
    @property
    def rhs_evaluates_observables(self) -> bool:
        """
        Whether compute_derivs_core refreshes observables/functions on the RHS (true iff the model has functions; false for pure mass-action).
        """
    @property
    def rhs_observable_eval_count(self) -> int:
        """
        Number of RHS evaluations that ran update_observables + evaluate_functions.
        """
    @property
    def species_names(self) -> list[str]:
        ...
    @property
    def table_function_names(self) -> list[str]:
        ...
    @property
    def uses_rateof(self) -> bool:
        """
        Whether the model uses the SBML rateOf csymbol (GH #106).
        """
class NfsimSimulator:
    def __init__(self, xml_path: str) -> None:
        """
        Create an NFsim simulator from a BNG XML file path
        """
    def add_molecules(self, molecule_type_name: str, count: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Add count molecules of named type in default (unbound) state
        """
    def add_species(self, pattern: str, count: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Add count exact single-molecule BNGL species instances
        """
    def clear_param_overrides(self) -> None:
        """
        Clear all parameter overrides
        """
    def destroy_session(self) -> None:
        """
        Destroy the current session, freeing the NFsim System
        """
    def evaluate_expression(self, expr: str, extra: collections.abc.Mapping[str, typing.SupportsFloat | typing.SupportsIndex] = {}) -> float:
        """
        Evaluate a BNG expression with current overrides (and optional extra bindings)
        """
    def get_molecule_count(self, molecule_type_name: str) -> int:
        """
        Get current molecule count for named type
        """
    def get_observable_names(self) -> list[str]:
        """
        Get observable names (session must be active)
        """
    def get_observable_values(self) -> list[float]:
        """
        Get current observable values (session must be active)
        """
    def get_parameter(self, name: str) -> float:
        """
        Get a parameter value by name (session must be active)
        """
    def get_species_count(self, pattern: str) -> int:
        """
        Get current count for an exact single-molecule BNGL species pattern
        """
    def has_saved_concentrations(self, label: str = '') -> bool:
        """
        Whether a save_concentrations() snapshot is available to restore under 'label'
        """
    def has_session(self) -> bool:
        """
        Whether a session is currently active
        """
    def initialize(self, seed: typing.SupportsInt | typing.SupportsIndex = 42) -> None:
        """
        Initialize a simulation session (parse XML, seed RNG, prepare)
        """
    def remove_species(self, pattern: str, count: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Remove count exact single-molecule BNGL species instances
        """
    def restore_concentrations(self, label: str = '') -> None:
        """
        Restore the molecular state captured by save_concentrations() into slot 'label' (BNG resetConcentrations()). Raises if none saved under 'label'.
        """
    def run(self, times: TimeSpec, seed: typing.SupportsInt | typing.SupportsIndex = 42, timeout_seconds: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> ResultCore:
        """
        Run NFsim simulation with deterministic seed (releases GIL). timeout_seconds > 0 enables a wall-clock budget checked between output points; on overrun, raises bngsim.SimulationTimeout.
        """
    def save_concentrations(self, label: str = '') -> None:
        """
        Snapshot the live System's molecular state into the named slot 'label' for later in-process restore (BNG saveConcentrations()). Each label owns its own snapshot; '' is the default/unlabeled slot.
        """
    def save_species(self, path: str) -> None:
        """
        Write the live System's molecular species to a BNG-format .species file
        """
    def saved_concentration_labels(self) -> list[str]:
        """
        Sorted names of currently-held save_concentrations() slots ('' = default)
        """
    def set_block_same_complex_binding(self, enabled: bool) -> None:
        """
        Block same-complex binding (NFsim CLI -bscb). Default: false.
        """
    def set_connectivity(self, enabled: bool) -> None:
        """
        Enable or disable NFsim connectivity inference at XML initialization
        """
    def set_molecule_limit(self, limit: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set global molecule limit (default: INT_MAX, no artificial limit)
        """
    def set_nfsim_v1143_compat(self, enabled: bool) -> None:
        """
        Enable or disable NFsim v1.14.3 selector compatibility mode
        """
    def set_param(self, name: str, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set a parameter override (applied before each run)
        """
    def set_species_count(self, pattern: str, count: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set count for an exact single-molecule BNGL species pattern
        """
    def set_traversal_limit(self, limit: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the universal traversal limit (NFsim CLI -utl N). Negative = auto.
        """
    def simulate(self, t_start: typing.SupportsFloat | typing.SupportsIndex, t_end: typing.SupportsFloat | typing.SupportsIndex, n_points: typing.SupportsInt | typing.SupportsIndex, timeout_seconds: typing.SupportsFloat | typing.SupportsIndex = 0.0, relative_time: bool = False, sample_times: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] = []) -> ResultCore:
        """
        Simulate from t_start to t_end, capturing n_points snapshots. timeout_seconds > 0 enables a wall-clock budget checked between output points; on overrun, raises bngsim.SimulationTimeout and the session must be destroyed before reuse. relative_time=true offsets the returned time axis to start at 0 (matching BNG2.pl simulate_nf convention) without changing the internal NFsim clock. sample_times (non-empty) overrides the uniform grid: the result is labelled with exactly those sorted absolute times and the live clock advances by their span from the current session time.
        """
    def step_to(self, time: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Advance simulation to target time (no output, state preserved)
        """
    @property
    def xml_path(self) -> str:
        """
        Path to the XML file
        """
class ResultCore:
    def __init__(self) -> None:
        ...
    def to_cdat(self, path: str) -> None:
        """
        Export species in BNG .cdat format
        """
    def to_gdat(self, path: str, print_functions: bool = False, print_rate_laws: bool = False) -> None:
        """
        Export observables in BNG .gdat format. Function headers are bare (no '()') and identical across methods. With print_functions=True, append user-named function columns; with print_rate_laws=True, also append the auto-generated _rateLawN columns (still bare).
        """
    @property
    def expression_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def expression_names(self) -> list[str]:
        ...
    @property
    def expression_sensitivity_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def expression_sensitivity_ic_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def gdat_expression_names(self) -> list[str]:
        ...
    @property
    def n_expressions(self) -> int:
        ...
    @property
    def n_observables(self) -> int:
        ...
    @property
    def n_sens_ic_species(self) -> int:
        ...
    @property
    def n_sens_params(self) -> int:
        ...
    @property
    def n_species(self) -> int:
        ...
    @property
    def n_times(self) -> int:
        ...
    @property
    def observable_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def observable_names(self) -> list[str]:
        ...
    @property
    def observable_sensitivity_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def observable_sensitivity_ic_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def raw_expression_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def raw_expression_names(self) -> list[str]:
        ...
    @property
    def raw_n_expressions(self) -> int:
        ...
    @property
    def sens_ic_species_names(self) -> list[str]:
        ...
    @property
    def sens_param_names(self) -> list[str]:
        ...
    @property
    def sensitivity_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def sensitivity_ic_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def solver_stats(self) -> SolverStats:
        ...
    @property
    def species_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def species_names(self) -> list[str]:
        ...
    @property
    def ssa_diagnostics(self) -> SsaDiagnostics:
        ...
    @property
    def time(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
class RuleMonkeySimulator:
    def __init__(self, xml_path: str) -> None:
        """
        Create a RuleMonkey simulator from a BNG XML file path
        """
    def add_molecules(self, molecule_type_name: str, count: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Add count molecules of named type in default (unbound) state
        """
    def add_species(self, pattern: str, count: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Add count exact BNGL species instances
        """
    def clear_param_overrides(self) -> None:
        """
        Clear all parameter overrides
        """
    def current_time(self) -> float:
        """
        Get the active session current time
        """
    def destroy_session(self) -> None:
        """
        Destroy the current session
        """
    def evaluate_expression(self, expr: str, extra: collections.abc.Mapping[str, typing.SupportsFloat | typing.SupportsIndex] = {}) -> float:
        """
        Evaluate a BNG expression against the active session (optional extra bindings)
        """
    def get_molecule_count(self, molecule_type_name: str) -> int:
        """
        Get current molecule count for named type
        """
    def get_observable_names(self) -> list[str]:
        """
        Get observable names
        """
    def get_observable_values(self) -> list[float]:
        """
        Get current observable values
        """
    def get_parameter(self, name: str) -> float:
        """
        Get a parameter value by name
        """
    def get_species_count(self, pattern: str) -> int:
        """
        Get current count for an exact, fully-specified BNGL species pattern
        """
    def has_session(self) -> bool:
        """
        Whether a session is currently active
        """
    def initialize(self, seed: typing.SupportsInt | typing.SupportsIndex = 42) -> None:
        """
        Initialize a simulation session
        """
    def load_state(self, path: str) -> None:
        """
        Load a session state
        """
    def remove_species(self, pattern: str, count: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Remove count exact BNGL species instances
        """
    def run(self, times: TimeSpec, seed: typing.SupportsInt | typing.SupportsIndex = 42, timeout_seconds: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> ResultCore:
        """
        Run RuleMonkey simulation with deterministic seed (releases GIL). timeout_seconds > 0 enables a wall-clock budget polled by upstream every ~1024 SSA events; on overrun, raises bngsim.SimulationTimeout.
        """
    def save_species(self, path: str) -> None:
        """
        Write the live pool to a BNG-format .species file (readNFspecies-compatible)
        """
    def save_state(self, path: str) -> None:
        """
        Save the active session state
        """
    def set_block_same_complex_binding(self, enabled: bool) -> None:
        """
        Block same-complex binding. Default: true.
        """
    def set_molecule_limit(self, limit: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set global molecule limit
        """
    def set_param(self, name: str, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set a parameter override
        """
    def set_species_count(self, pattern: str, count: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set count for an exact BNGL species pattern (adds/removes the diff)
        """
    def simulate(self, t_start: typing.SupportsFloat | typing.SupportsIndex, t_end: typing.SupportsFloat | typing.SupportsIndex, n_points: typing.SupportsInt | typing.SupportsIndex, timeout_seconds: typing.SupportsFloat | typing.SupportsIndex = 0.0, sample_times: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] = []) -> ResultCore:
        """
        Simulate from t_start to t_end, capturing n_points snapshots. timeout_seconds > 0 enables a wall-clock budget polled by upstream every ~1024 SSA events; on overrun, raises bngsim.SimulationTimeout and the session must be destroyed before reuse. sample_times (non-empty) overrides the uniform grid: the result is labelled with exactly those sorted absolute times and the live session advances by their span from the current session time.
        """
    def step_to(self, time: typing.SupportsFloat | typing.SupportsIndex, timeout_seconds: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        """
        Advance simulation to target time without returning output. timeout_seconds > 0 enables a wall-clock budget; on overrun, raises bngsim.SimulationTimeout.
        """
    @property
    def xml_path(self) -> str:
        """
        Path to the XML file
        """
class SolverOptions:
    carry_sensitivities: bool
    codegen_c_source: str
    codegen_so_path: str
    force_dense_linear_solver: bool
    jacobian: str
    steady_state: bool
    def __init__(self) -> None:
        ...
    def set_jax_jac_fn(self, fn: typing.Any) -> None:
        """
        Set JAX AD Jacobian callback. fn(t, y_array) -> flat_jac_array (col-major)
        """
    def set_sensitivity_error_control(self, error_control: bool) -> None:
        """
        Include sensitivities in error test (default: true)
        """
    def set_sensitivity_ic(self, species: collections.abc.Sequence[str]) -> None:
        """
        Set species names for forward initial-condition sensitivity. Each named species's IC contributes a sensitivity column dY(t)/dY_k(0); the variational ODE has zero source term and is integrated by the codegen sens RHS, so codegen must be enabled for IC-sens workflows.
        """
    def set_sensitivity_method(self, method: str) -> None:
        """
        Set the CVODES corrector strategy for the coupled state + sensitivity ODE system. Both modes integrate state and all sensitivity equations as one extended ODE in a single CVODES pass; they differ only in how each step's nonlinear solve is structured.
        
          'staggered' (default, CV_STAGGERED): state first, then sensitivities as a separate solve. Two smaller nonlinear solves per step. Often more robust for stiff or large systems; CVODES' default.
          'simultaneous' (CV_SIMULTANEOUS): state and all sensitivities solved together as one coupled nonlinear system at every step. Often a touch faster per step on small / well-conditioned problems. AMICI's default.
        """
    def set_sensitivity_params(self, params: collections.abc.Sequence[str]) -> None:
        """
        Set parameter names for forward sensitivity analysis
        """
    @property
    def atol(self) -> float:
        ...
    @atol.setter
    def atol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def event_seed(self) -> int:
        ...
    @event_seed.setter
    def event_seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_step_size(self) -> float:
        ...
    @max_step_size.setter
    def max_step_size(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_steps(self) -> int:
        ...
    @max_steps.setter
    def max_steps(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def rtol(self) -> float:
        ...
    @rtol.setter
    def rtol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def steady_state_tol(self) -> float:
        ...
    @steady_state_tol.setter
    def steady_state_tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def timeout_seconds(self) -> float:
        ...
    @timeout_seconds.setter
    def timeout_seconds(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SolverStats:
    steady_state_reached: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def to_dict(self) -> dict:
        ...
    @property
    def linear_solver(self) -> int:
        ...
    @linear_solver.setter
    def linear_solver(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_dense_blas_factorizations(self) -> int:
        ...
    @n_dense_blas_factorizations.setter
    def n_dense_blas_factorizations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_err_test_fails(self) -> int:
        ...
    @n_err_test_fails.setter
    def n_err_test_fails(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_jac_evals(self) -> int:
        ...
    @n_jac_evals.setter
    def n_jac_evals(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_nonlin_conv_fails(self) -> int:
        ...
    @n_nonlin_conv_fails.setter
    def n_nonlin_conv_fails(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_nonlin_iters(self) -> int:
        ...
    @n_nonlin_iters.setter
    def n_nonlin_iters(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_rhs_evals(self) -> int:
        ...
    @n_rhs_evals.setter
    def n_rhs_evals(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_steps(self) -> int:
        ...
    @n_steps.setter
    def n_steps(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def steady_state_residual(self) -> float:
        ...
    @steady_state_residual.setter
    def steady_state_residual(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SsaDiagnostics:
    first_negative_species: str
    first_reverse_reaction: str
    propensity_backend: str
    psa_activation_crossed: bool
    psa_active: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def to_dict(self) -> dict:
        ...
    @property
    def n_negative_crossings(self) -> int:
        ...
    @n_negative_crossings.setter
    def n_negative_crossings(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_reverse_fires(self) -> int:
        ...
    @n_reverse_fires.setter
    def n_reverse_fires(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def psa_exact_event_integral(self) -> float:
        ...
    @psa_exact_event_integral.setter
    def psa_exact_event_integral(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def psa_mbar_integral(self) -> list[float]:
        ...
    @psa_mbar_integral.setter
    def psa_mbar_integral(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def psa_peak_population(self) -> float:
        ...
    @psa_peak_population.setter
    def psa_peak_population(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def psa_qexc_integral(self) -> list[float]:
        ...
    @psa_qexc_integral.setter
    def psa_qexc_integral(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def psa_reaction_index(self) -> list[int]:
        ...
    @psa_reaction_index.setter
    def psa_reaction_index(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @property
    def psa_scaled_event_integral(self) -> float:
        ...
    @psa_scaled_event_integral.setter
    def psa_scaled_event_integral(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def psa_time(self) -> float:
        ...
    @psa_time.setter
    def psa_time(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SsaSimulator:
    def __init__(self, model: NetworkModel) -> None:
        """
        Create a stochastic simulator (SSA/PSA) for the given model
        """
    def run(self, times: TimeSpec, seed: typing.SupportsInt | typing.SupportsIndex = 42, timeout_seconds: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> ResultCore:
        """
        Run exact SSA simulation with deterministic seed (releases GIL). timeout_seconds > 0 enables a wall-clock budget; on overrun, raises bngsim.SimulationTimeout.
        """
    def run_psa(self, times: TimeSpec, seed: typing.SupportsInt | typing.SupportsIndex = 42, poplevel: typing.SupportsFloat | typing.SupportsIndex = 100.0, timeout_seconds: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> ResultCore:
        """
        Run PSA (Partial Scaling Algorithm) simulation.
        Lin, Feng, Hlavacek, J. Chem. Phys. 150, 244101 (2019).
        poplevel = N_c (critical population size, must be > 1). Releases GIL. timeout_seconds > 0 enables a wall-clock budget; on overrun, raises bngsim.SimulationTimeout.
        """
    def set_propensity_library(self, so_path: str) -> None:
        """
        GH #190: supply a cc-compiled value-specialized propensity .so (symbol bngsim_ssa_propensities). When set and the model is recompute-all eligible (pure mass-action exact SSA, no events, small nr), the run takes the RR-style recompute-all + flat-scan loop by default. No-op for ineligible models; '' clears it.
        """
class SteadyStateOptions:
    codegen_so_path: str
    jacobian: str
    method: str
    def __init__(self) -> None:
        ...
    @property
    def atol(self) -> float:
        ...
    @atol.setter
    def atol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_steps(self) -> int:
        ...
    @max_steps.setter
    def max_steps(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_time(self) -> float:
        ...
    @max_time.setter
    def max_time(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rtol(self) -> float:
        ...
    @rtol.setter
    def rtol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sensitivity_params(self) -> list[str]:
        ...
    @sensitivity_params.setter
    def sensitivity_params(self, arg0: collections.abc.Sequence[str]) -> None:
        ...
    @property
    def tol(self) -> float:
        ...
    @tol.setter
    def tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SteadyStateResultCore:
    def __init__(self) -> None:
        ...
    @property
    def concentrations(self) -> list[float]:
        ...
    @property
    def converged(self) -> bool:
        ...
    @property
    def expression_names(self) -> list[str]:
        ...
    @property
    def expression_sensitivity_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def method_used(self) -> str:
        ...
    @property
    def n_rhs_evals(self) -> int:
        ...
    @property
    def n_sens_params(self) -> int:
        ...
    @property
    def n_steps(self) -> int:
        ...
    @property
    def observable_names(self) -> list[str]:
        ...
    @property
    def observable_sensitivity_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def raw_expression_names(self) -> list[str]:
        ...
    @property
    def raw_expression_sensitivity_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def residual(self) -> float:
        ...
    @property
    def sens_param_names(self) -> list[str]:
        ...
    @property
    def sensitivity_data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def species_names(self) -> list[str]:
        ...
class TimeSpec:
    def __init__(self) -> None:
        ...
    @property
    def n_points(self) -> int:
        ...
    @n_points.setter
    def n_points(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def sample_times(self) -> list[float]:
        ...
    @sample_times.setter
    def sample_times(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def t_end(self) -> float:
        ...
    @t_end.setter
    def t_end(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def t_start(self) -> float:
        ...
    @t_start.setter
    def t_start(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
def bench_ssa_propensity_jit(model: NetworkModel, n_iters: typing.SupportsInt | typing.SupportsIndex = 200000) -> dict:
    """
    GH #190: benchmark the structure-specialized SSA propensity kernel (cc -O3 and MIR) vs the per-reaction compute_propensity loop.
    """
def emit_ssa_propensity_source_structure(model: NetworkModel) -> tuple[str, int]:
    """
    Emit STRUCTURE-specialized C for the full SSA propensity vector — rate constants read from a runtime params[] argument, signature bngsim_ssa_propensities(const double* x, const double* p, double* a); the source/.so cache key depends only on model structure. Returns (source, n_unsupported).
    """
def find_steady_state(model: NetworkModel, opts: SteadyStateOptions = ...) -> SteadyStateResultCore:
    """
    Find steady state of the ODE system (releases GIL)
    """
def reserved_names() -> dict:
    """
    Return dict of reserved constant and function names
    """
HAS_KLU: bool = True
HAS_LAPACK_DENSE: bool = True
HAS_MIR: bool = False
HAS_NFSIM: bool = True
HAS_RULEMONKEY: bool = True
__build_commit__: str = '771a40931f4d+dirty'
__version__: str = '0.11.35'
