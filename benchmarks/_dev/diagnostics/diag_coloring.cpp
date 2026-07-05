// Diagnostic: check coloring stats for all large benchmark models
#include <bngsim/bngsim.hpp>
#include <iostream>
#include <chrono>

int main() {
    struct Info { const char* name; const char* path; };
    Info models[] = {
        {"egfr_net",       "../benchmarks/models/net/ssa/egfr_net.net"},
        {"multisite_phos", "../benchmarks/models/net/ssa/multisite_phos.net"},
        {"fceri_gamma",    "../benchmarks/models/net/ssa/fceri_gamma.net"},
    };

    for (auto& m : models) {
        auto model = bngsim::NetworkModel::from_net(m.path);
        int elem = 0, func = 0, mm = 0;
        for (auto& r : model.reactions()) {
            if (r.rate_law_type == bngsim::RateLawType::Elementary) elem++;
            else if (r.rate_law_type == bngsim::RateLawType::Functional) func++;
            else mm++;
        }
        auto& sp = model.jacobian_sparsity();
        double comp = (sp.n_colors > 0) ? (double)sp.n / sp.n_colors : 0;
        std::cout << m.name << ": " << sp.n << " sp, "
                  << model.n_reactions() << " rxn"
                  << " (E=" << elem << " F=" << func << " M=" << mm << ")"
                  << " nnz=" << sp.nnz
                  << " density=" << sp.density
                  << " n_colors=" << sp.n_colors
                  << " compress=" << comp << "x"
                  << std::endl;
    }

    // Time multisite_phos ODE (short horizon)
    std::cout << "\nTiming multisite_phos ODE (t=0..1, 3pts)...\n";
    auto model = bngsim::NetworkModel::from_net(
        "../benchmarks/models/net/ssa/multisite_phos.net");
    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec ts{0.0, 1.0, 3};
    bngsim::SolverOptions opts;
    opts.max_steps = 50000;
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
