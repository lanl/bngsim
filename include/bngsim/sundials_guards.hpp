// bngsim/include/bngsim/sundials_guards.hpp — RAII wrappers for SUNDIALS resources
//
// Eliminates manual cleanup blocks in cvode_simulator.cpp and steady_state.cpp.
// Each guard owns a single SUNDIALS resource and frees it in its destructor.
// Non-copyable, movable. All guards are safe to destroy when holding nullptr.
//
// Used by the CVODE and steady-state solvers to keep cleanup exception-safe.

#pragma once

#include <cvodes/cvodes.h>
#include <kinsol/kinsol.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>

#include <utility> // std::exchange

namespace bngsim {

// ─── SUNContext ──────────────────────────────────────────────────────────────

struct SunContextGuard {
    SUNContext ctx = nullptr;

    SunContextGuard() {
        if (SUNContext_Create(SUN_COMM_NULL, &ctx) != 0) {
            ctx = nullptr;
        }
    }
    ~SunContextGuard() {
        if (ctx)
            SUNContext_Free(&ctx);
    }

    explicit operator bool() const { return ctx != nullptr; }
    operator SUNContext() const { return ctx; }

    SunContextGuard(const SunContextGuard &) = delete;
    SunContextGuard &operator=(const SunContextGuard &) = delete;
    SunContextGuard(SunContextGuard &&o) noexcept : ctx(std::exchange(o.ctx, nullptr)) {}
    SunContextGuard &operator=(SunContextGuard &&o) noexcept {
        if (this != &o) {
            if (ctx)
                SUNContext_Free(&ctx);
            ctx = std::exchange(o.ctx, nullptr);
        }
        return *this;
    }
};

// ─── N_Vector ────────────────────────────────────────────────────────────────

struct NVectorGuard {
    N_Vector v = nullptr;

    NVectorGuard() = default;
    explicit NVectorGuard(N_Vector nv) : v(nv) {}
    ~NVectorGuard() {
        if (v)
            N_VDestroy(v);
    }

    explicit operator bool() const { return v != nullptr; }
    operator N_Vector() const { return v; }
    double *data() { return N_VGetArrayPointer(v); }
    const double *data() const { return N_VGetArrayPointer(v); }

    NVectorGuard(const NVectorGuard &) = delete;
    NVectorGuard &operator=(const NVectorGuard &) = delete;
    NVectorGuard(NVectorGuard &&o) noexcept : v(std::exchange(o.v, nullptr)) {}
    NVectorGuard &operator=(NVectorGuard &&o) noexcept {
        if (this != &o) {
            if (v)
                N_VDestroy(v);
            v = std::exchange(o.v, nullptr);
        }
        return *this;
    }
};

// ─── N_Vector array (from N_VCloneVectorArray) ──────────────────────────────

struct NVectorArrayGuard {
    N_Vector *arr = nullptr;
    int count = 0;

    NVectorArrayGuard() = default;
    NVectorArrayGuard(N_Vector *a, int n) : arr(a), count(n) {}
    ~NVectorArrayGuard() {
        if (arr)
            N_VDestroyVectorArray(arr, count);
    }

    explicit operator bool() const { return arr != nullptr; }
    N_Vector operator[](int i) const { return arr[i]; }

    NVectorArrayGuard(const NVectorArrayGuard &) = delete;
    NVectorArrayGuard &operator=(const NVectorArrayGuard &) = delete;
    NVectorArrayGuard(NVectorArrayGuard &&o) noexcept
        : arr(std::exchange(o.arr, nullptr)), count(std::exchange(o.count, 0)) {}
    NVectorArrayGuard &operator=(NVectorArrayGuard &&o) noexcept {
        if (this != &o) {
            if (arr)
                N_VDestroyVectorArray(arr, count);
            arr = std::exchange(o.arr, nullptr);
            count = std::exchange(o.count, 0);
        }
        return *this;
    }
};

// ─── CVODE memory ────────────────────────────────────────────────────────────

struct CvodeMemGuard {
    void *mem = nullptr;

    CvodeMemGuard() = default;
    explicit CvodeMemGuard(void *m) : mem(m) {}
    ~CvodeMemGuard() {
        if (mem)
            CVodeFree(&mem);
    }

    explicit operator bool() const { return mem != nullptr; }
    operator void *() const { return mem; }

    CvodeMemGuard(const CvodeMemGuard &) = delete;
    CvodeMemGuard &operator=(const CvodeMemGuard &) = delete;
    CvodeMemGuard(CvodeMemGuard &&o) noexcept : mem(std::exchange(o.mem, nullptr)) {}
    CvodeMemGuard &operator=(CvodeMemGuard &&o) noexcept {
        if (this != &o) {
            if (mem)
                CVodeFree(&mem);
            mem = std::exchange(o.mem, nullptr);
        }
        return *this;
    }
};

// ─── KINSOL memory ───────────────────────────────────────────────────────────

struct KinsolMemGuard {
    void *mem = nullptr;

    KinsolMemGuard() = default;
    explicit KinsolMemGuard(void *m) : mem(m) {}
    ~KinsolMemGuard() {
        if (mem)
            KINFree(&mem);
    }

    explicit operator bool() const { return mem != nullptr; }
    operator void *() const { return mem; }

    KinsolMemGuard(const KinsolMemGuard &) = delete;
    KinsolMemGuard &operator=(const KinsolMemGuard &) = delete;
    KinsolMemGuard(KinsolMemGuard &&o) noexcept : mem(std::exchange(o.mem, nullptr)) {}
    KinsolMemGuard &operator=(KinsolMemGuard &&o) noexcept {
        if (this != &o) {
            if (mem)
                KINFree(&mem);
            mem = std::exchange(o.mem, nullptr);
        }
        return *this;
    }
};

// ─── SUNMatrix ───────────────────────────────────────────────────────────────

struct SUNMatrixGuard {
    SUNMatrix mat = nullptr;

    SUNMatrixGuard() = default;
    explicit SUNMatrixGuard(SUNMatrix m) : mat(m) {}
    ~SUNMatrixGuard() {
        if (mat)
            SUNMatDestroy(mat);
    }

    explicit operator bool() const { return mat != nullptr; }
    operator SUNMatrix() const { return mat; }

    SUNMatrixGuard(const SUNMatrixGuard &) = delete;
    SUNMatrixGuard &operator=(const SUNMatrixGuard &) = delete;
    SUNMatrixGuard(SUNMatrixGuard &&o) noexcept : mat(std::exchange(o.mat, nullptr)) {}
    SUNMatrixGuard &operator=(SUNMatrixGuard &&o) noexcept {
        if (this != &o) {
            if (mat)
                SUNMatDestroy(mat);
            mat = std::exchange(o.mat, nullptr);
        }
        return *this;
    }
};

// ─── SUNLinearSolver ─────────────────────────────────────────────────────────

struct SUNLinSolGuard {
    SUNLinearSolver ls = nullptr;

    SUNLinSolGuard() = default;
    explicit SUNLinSolGuard(SUNLinearSolver s) : ls(s) {}
    ~SUNLinSolGuard() {
        if (ls)
            SUNLinSolFree(ls);
    }

    explicit operator bool() const { return ls != nullptr; }
    operator SUNLinearSolver() const { return ls; }

    SUNLinSolGuard(const SUNLinSolGuard &) = delete;
    SUNLinSolGuard &operator=(const SUNLinSolGuard &) = delete;
    SUNLinSolGuard(SUNLinSolGuard &&o) noexcept : ls(std::exchange(o.ls, nullptr)) {}
    SUNLinSolGuard &operator=(SUNLinSolGuard &&o) noexcept {
        if (this != &o) {
            if (ls)
                SUNLinSolFree(ls);
            ls = std::exchange(o.ls, nullptr);
        }
        return *this;
    }
};

} // namespace bngsim
