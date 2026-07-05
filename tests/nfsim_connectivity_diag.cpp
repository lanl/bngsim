#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "NFcore/NFcore.hh"
#include "NFinput/NFinput.hh"
#include "NFutil/NFutil.hh"

using namespace NFcore;

namespace {

class CwdGuard {
public:
    explicit CwdGuard(const std::filesystem::path& new_cwd) {
        try {
            old_cwd_ = std::filesystem::current_path();
            if (!new_cwd.empty()) {
                std::filesystem::current_path(new_cwd);
                changed_ = true;
            }
        } catch (...) {
            changed_ = false;
        }
    }

    ~CwdGuard() {
        if (!changed_) return;
        try {
            std::filesystem::current_path(old_cwd_);
        } catch (...) {
        }
    }

private:
    std::filesystem::path old_cwd_;
    bool changed_ = false;
};

struct Options {
    std::string xml_path;
    std::string out_path;
    std::string connected_log_path;
    std::string connected_list_path;
    std::string reaction_log_path;
    unsigned long seed = 1;
    double t_end = 20.0;
    int n_steps = 100;
    int gml = 1000000;
    int log_buffer_size = 1;
    bool connectivity = false;
    bool output_event_counter = true;
    bool nfsim_v1143_compat = false;
    bool track_rxn_num = false;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --xml PATH --out PATH [options]\n"
        << "Options:\n"
        << "  --seed N                    RNG seed (default: 1)\n"
        << "  --t-end T                   Simulation end time (default: 20)\n"
        << "  --n-steps N                 Number of output steps (default: 100)\n"
        << "  --gml N                     Global molecule limit (default: 1000000)\n"
        << "  --connectivity on|off       Enable inferred connectivity updates\n"
        << "  --v1143-compat on|off       Enable NFsim v1.14.3 selector behavior\n"
        << "  --event-counter on|off      Include EventCounter column (default: on)\n"
        << "  --connected-log PATH        Write connected-rate updates to file\n"
        << "  --connected-list PATH       Write inferred connected reaction pairs\n"
        << "  --rxn-log PATH              Write reaction firing JSON log\n"
        << "  --track-rxn-num on|off      Log numeric reaction ids instead of names\n"
        << "  --log-buffer N              Reaction log flush interval (default: 1)\n";
    std::exit(2);
}

bool parse_on_off(const std::string& value, const char* flag_name) {
    if (value == "on" || value == "true" || value == "1") return true;
    if (value == "off" || value == "false" || value == "0") return false;
    throw std::runtime_error(std::string("Invalid value for ") + flag_name + ": " + value);
}

Options parse_args(int argc, char* argv[]) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* flag_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag_name);
            }
            return argv[++i];
        };

        if (arg == "--xml") {
            options.xml_path = need_value("--xml");
        } else if (arg == "--out") {
            options.out_path = need_value("--out");
        } else if (arg == "--seed") {
            options.seed = std::stoul(need_value("--seed"));
        } else if (arg == "--t-end") {
            options.t_end = std::stod(need_value("--t-end"));
        } else if (arg == "--n-steps") {
            options.n_steps = std::stoi(need_value("--n-steps"));
        } else if (arg == "--gml") {
            options.gml = std::stoi(need_value("--gml"));
        } else if (arg == "--connectivity") {
            options.connectivity = parse_on_off(need_value("--connectivity"), "--connectivity");
        } else if (arg == "--v1143-compat") {
            options.nfsim_v1143_compat = parse_on_off(
                need_value("--v1143-compat"), "--v1143-compat");
        } else if (arg == "--event-counter") {
            options.output_event_counter = parse_on_off(
                need_value("--event-counter"), "--event-counter");
        } else if (arg == "--connected-log") {
            options.connected_log_path = need_value("--connected-log");
        } else if (arg == "--connected-list") {
            options.connected_list_path = need_value("--connected-list");
        } else if (arg == "--rxn-log") {
            options.reaction_log_path = need_value("--rxn-log");
        } else if (arg == "--track-rxn-num") {
            options.track_rxn_num = parse_on_off(need_value("--track-rxn-num"), "--track-rxn-num");
        } else if (arg == "--log-buffer") {
            options.log_buffer_size = std::stoi(need_value("--log-buffer"));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.xml_path.empty() || options.out_path.empty()) {
        usage(argv[0]);
    }
    return options;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Options options = parse_args(argc, argv);
        std::filesystem::path xml_abs = std::filesystem::absolute(options.xml_path);
        CwdGuard cwd_guard(xml_abs.parent_path());

        int suggested_traversal_limit = -1;
        std::unique_ptr<System> system(NFinput::initializeFromXML(
            xml_abs.string(),
            /*blockSameComplexBinding=*/true,
            options.gml,
            /*verbose=*/false,
            suggested_traversal_limit,
            /*evaluateComplexScopedLocalFunctions=*/true,
            /*connectivityFlag=*/options.connectivity));
        if (!system) {
            throw std::runtime_error("initializeFromXML returned nullptr");
        }

        system->setUniversalTraversalLimit(suggested_traversal_limit);
        system->setNFsimV1143Compatibility(options.nfsim_v1143_compat);
        if (!options.connected_log_path.empty()) {
            system->setTrackConnected(true);
            system->registerConnectedRxnFileLocation(options.connected_log_path);
        }
        if (!options.connected_list_path.empty()) {
            system->setPrintConnected(true);
            system->registerListOfConnectedRxnFileLocation(options.connected_list_path);
        }
        if (!options.reaction_log_path.empty()) {
            system->setLogBufferSize(options.log_buffer_size);
            system->setRxnNumberTrack(options.track_rxn_num);
            system->registerReactionFileLocation(options.reaction_log_path);
        }
        if (options.output_event_counter) {
            system->turnOnOutputEventCounter();
        }

        system->seedRNG(options.seed);
        NFutil::SEED_RANDOM(options.seed);
        system->registerOutputFileLocation(options.out_path);
        system->outputAllObservableNames();
        system->prepareForSimulation();
        double final_time = system->sim(options.t_end, options.n_steps, /*verbose=*/false);

        std::cout
            << "xml=" << xml_abs.string() << "\n"
            << "out=" << std::filesystem::absolute(options.out_path).string() << "\n"
            << "connectivity=" << (options.connectivity ? "on" : "off") << "\n"
            << "v1143_compat=" << (options.nfsim_v1143_compat ? "on" : "off") << "\n"
            << "seed=" << options.seed << "\n"
            << "final_time=" << final_time << "\n"
            << "event_count=" << system->getGlobalEventCounter() << "\n";
        if (!options.connected_log_path.empty()) {
            std::cout << "connected_log="
                      << std::filesystem::absolute(options.connected_log_path).string() << "\n";
        }
        if (!options.connected_list_path.empty()) {
            std::cout << "connected_list="
                      << std::filesystem::absolute(options.connected_list_path).string() << "\n";
        }
        if (!options.reaction_log_path.empty()) {
            std::cout << "reaction_log="
                      << std::filesystem::absolute(options.reaction_log_path).string() << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "nfsim_connectivity_diag: " << e.what() << std::endl;
        return 1;
    }
}
