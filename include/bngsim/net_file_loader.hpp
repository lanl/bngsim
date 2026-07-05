// bngsim/include/bngsim/net_file_loader.hpp — .net file parser
//
// Defines the loader interface and the BioNetGen .net implementation.
// NetFileLoader parses .net files and routes construction through ModelBuilder.

#pragma once

#include "bngsim/model.hpp"

#include <string>

namespace bngsim {

// ─── Abstract loader interface ───────────────────────────────────────────────
class ModelLoader {
  public:
    virtual ~ModelLoader() = default;
    virtual NetworkModel load(const std::string &source) = 0;
};

// ─── .net file loader ────────────────────────────────────────────────────────
class NetFileLoader : public ModelLoader {
  public:
    NetworkModel load(const std::string &path) override;
};

} // namespace bngsim
