// bngsim/include/bngsim/net_file_loader.hpp — .net file parser
//
// Defines the loader interface and the BioNetGen .net implementation.
// NetFileLoader parses .net files and routes construction through ModelBuilder.

#pragma once

#include "bngsim/model.hpp"

#include <string>
#include <vector>

namespace bngsim {

// ─── .net table functions ────────────────────────────────────────────────────

/// One table function a `.net` functions line asks for, read out of the
/// `tfun(...)` call in its expression.
struct NetTableFunction {
    /// Runtime identifier — ExprTk calls the table `tfun_<name>()`. It is the
    /// BNG function's own name when the expression is nothing but a `tfun(...)`
    /// call, and a synthetic `<bng_func>__tfun<k>` when the call sits inside
    /// surrounding arithmetic.
    std::string name;
    /// `.tfun` column-2 header to accept; empty means `name`. Set to the BNG
    /// function name for a synthetic table, whose file still labels its value
    /// column by the name the modeller wrote.
    std::string header_name;
    /// File-based spec: a path relative to the `.net` file's own directory, or
    /// an absolute one. Empty when `is_inline`.
    std::string filepath;
    std::vector<double> xs; ///< Inline data — `tfun([xs],[ys],index)`.
    std::vector<double> ys;
    std::string index_name; ///< "time", or a parameter/observable name.
    std::string method;     ///< "linear" or "step".
    bool is_inline = false;
};

/// What a `.net` functions line becomes once its `tfun(...)` calls are lifted
/// out of the expression.
struct NetFunctionTables {
    /// The expression to hand `ModelBuilder::add_function`. Unchanged when the
    /// line names no table; the whole `tfun(...)` body when the line is one
    /// (`build()` rewrites that to `tfun_<name>()` once the table is loaded);
    /// otherwise the original with each call replaced by `tfun_<synthetic>()`,
    /// so the arithmetic around it survives.
    std::string expression;
    /// The tables to register with the builder before it compiles `expression`.
    std::vector<NetTableFunction> tables;
};

/// Read the `tfun(...)` calls out of one `.net` functions line.
///
/// This is the loader's own post-parse step, reachable on its own so a caller
/// that parsed the `.net` elsewhere — `bngsim._net_reader`, which parses in
/// Python and builds through the same `ModelBuilder` — registers the same
/// tables from the same spec parse rather than growing a second reading of
/// `tfun(...)` syntax to disagree with this one (issue #597).
///
/// @param func_name  The BNG function's name, which names the table it becomes
///                   and validates the `.tfun` header of the ones it contains.
/// @param expression The functions line's expression, as parsed.
NetFunctionTables net_function_tables(const std::string &func_name, const std::string &expression);

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
