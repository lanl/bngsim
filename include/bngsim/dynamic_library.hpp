// bngsim/include/bngsim/dynamic_library.hpp — Platform-abstract dynamic library loading
//
// Wraps dlopen/dlsym/dlclose on Unix and LoadLibrary/GetProcAddress/FreeLibrary
// on Windows. Provides RAII ownership semantics (non-copyable, movable).
//
// Used by code-generated RHS loading on Unix and Windows.

#pragma once

#include <stdexcept>
#include <string>
#include <utility> // std::exchange

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// NOMINMAX stops <windows.h> from defining min()/max() macros, which collide
// with std::min/std::max in translation units that include this header (e.g.
// cvode_simulator.cpp). GH #150.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace bngsim {

/// Platform-abstract dynamic library loader with RAII ownership.
///
/// Usage:
///   DynamicLibrary lib("path/to/library.so");  // throws on failure
///   auto fn = lib.symbol<FnType>("function_name");  // throws on failure
///   // lib automatically unloaded when destroyed
///
class DynamicLibrary {
  public:
    DynamicLibrary() = default;

    /// Load a shared library from the given path.
    /// Throws std::runtime_error if loading fails.
    explicit DynamicLibrary(const std::string &path) {
#ifdef _WIN32
        handle_ = LoadLibraryA(path.c_str());
        if (!handle_) {
            throw std::runtime_error("DynamicLibrary: LoadLibrary failed for '" + path +
                                     "' (error " + std::to_string(GetLastError()) + ")");
        }
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            throw std::runtime_error("DynamicLibrary: dlopen failed for '" + path +
                                     "': " + std::string(dlerror()));
        }
#endif
    }

    ~DynamicLibrary() { close(); }

    // Non-copyable
    DynamicLibrary(const DynamicLibrary &) = delete;
    DynamicLibrary &operator=(const DynamicLibrary &) = delete;

    // Movable
    DynamicLibrary(DynamicLibrary &&other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}

    DynamicLibrary &operator=(DynamicLibrary &&other) noexcept {
        if (this != &other) {
            close();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    /// Check if the library is loaded.
    explicit operator bool() const { return handle_ != nullptr; }

    /// Look up a symbol and cast to the requested function pointer type.
    /// Throws std::runtime_error if the symbol is not found.
    template <typename T> T symbol(const std::string &name) const {
        void *sym = raw_symbol(name);
        // reinterpret_cast from void* to function pointer — required by POSIX
        // (dlsym returns void*) and Windows (GetProcAddress returns FARPROC).
        return reinterpret_cast<T>(sym);
    }

    /// Look up a symbol, returning nullptr if not found (no-throw).
    template <typename T> T try_symbol(const std::string &name) const noexcept {
        void *sym = raw_symbol_nothrow(name);
        return reinterpret_cast<T>(sym);
    }

    /// Get the raw platform handle (for interop). May be nullptr.
    void *native_handle() const { return handle_; }

  private:
    void *handle_ = nullptr;

    void close() {
        if (handle_) {
#ifdef _WIN32
            FreeLibrary(static_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
            handle_ = nullptr;
        }
    }

    void *raw_symbol(const std::string &name) const {
#ifdef _WIN32
        auto sym =
            reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(handle_), name.c_str()));
        if (!sym) {
            throw std::runtime_error("DynamicLibrary: GetProcAddress failed for '" + name +
                                     "' (error " + std::to_string(GetLastError()) + ")");
        }
        return sym;
#else
        dlerror(); // clear previous error
        void *sym = dlsym(handle_, name.c_str());
        const char *err = dlerror();
        if (err) {
            throw std::runtime_error("DynamicLibrary: dlsym failed for '" + name +
                                     "': " + std::string(err));
        }
        return sym;
#endif
    }

    void *raw_symbol_nothrow(const std::string &name) const noexcept {
#ifdef _WIN32
        return reinterpret_cast<void *>(
            GetProcAddress(static_cast<HMODULE>(handle_), name.c_str()));
#else
        dlerror();
        void *sym = dlsym(handle_, name.c_str());
        dlerror(); // consume any error
        return sym;
#endif
    }
};

} // namespace bngsim
