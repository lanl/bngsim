// Diagnostic: coloring stats for metapopulation SIR model
#include <bngsim/bngsim.hpp>
#include <iostream>
#include <chrono>

int main() {
    auto model = bngsim::NetworkModel::from_net(
        "../benchmarks/models/net/ode/metapop_sir_100.net");
    auto& sp = model.jacobian_sparsity();
    double comp = (sp.n_colors > 0) ? (double)sp.n / sp.n_colors : 0;
    std::cout << "metapop_sir_100: " << sp.n << " sp, "
              << model.n_reactions() << " rxn"
              << " nnz=" << sp.nnz
              << " density=" << sp.density
              << " n_colors=" << sp.n_colors
              << " compress=" << comp << "x"
              << std::endl;

    // Time ODE
    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec ts{0.0, 200.0, 201};
    bngsim::SolverOptions opts;
    opts.max_steps = 100000;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = sim.run(ts, opts);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "  ODE: " << elapsed << "s"
              << " steps=" << result.solver_stats().n_steps
              << " rhs=" << result.solver_stats().n_rhs_evals
              << std::endl;
    return 0;
}
