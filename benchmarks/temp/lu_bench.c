// Isolated LU microbenchmark for GH #84 feasibility.
// Compares SUNDIALS' built-in unblocked dense LU (denseGETRF/GETRS, replicated
// verbatim from sundials_dense.c v7.2.1) against Accelerate's blocked dgetrf_/
// dgetrs_ — the exact lever SUNLinSol_LapackDense would pull. Measures the
// crossover-N and speedup that gate the issue.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// f77 LAPACK (Accelerate provides these; column-major, 32-bit int)
extern void dgetrf_(const int *m, const int *n, double *a, const int *lda,
                    int *ipiv, int *info);
extern void dgetrs_(const char *trans, const int *n, const int *nrhs,
                    const double *a, const int *lda, const int *ipiv,
                    double *b, const int *ldb, int *info);

#define ONE 1.0
#define ZERO 0.0

// --- SUNDIALS built-in unblocked LU, replicated verbatim ---
// `a` is array of column pointers (column-major), m rows, n cols.
static long sun_GETRF(double **a, long m, long n, long *p) {
    long i, j, k, l;
    double *col_j, *col_k, temp, mult, a_kj;
    for (k = 0; k < n; k++) {
        col_k = a[k];
        l = k;
        for (i = k + 1; i < m; i++)
            if (fabs(col_k[i]) > fabs(col_k[l])) l = i;
        p[k] = l;
        if (col_k[l] == ZERO) return (k + 1);
        if (l != k)
            for (i = 0; i < n; i++) {
                temp = a[i][l]; a[i][l] = a[i][k]; a[i][k] = temp;
            }
        mult = ONE / col_k[k];
        for (i = k + 1; i < m; i++) col_k[i] *= mult;
        for (j = k + 1; j < n; j++) {
            col_j = a[j];
            a_kj = col_j[k];
            if (a_kj != ZERO)
                for (i = k + 1; i < m; i++) col_j[i] -= a_kj * col_k[i];
        }
    }
    return 0;
}

static void sun_GETRS(double **a, long n, long *p, double *b) {
    long i, k, pk;
    double *col_k, tmp;
    for (k = 0; k < n; k++) {
        pk = p[k];
        if (pk != k) { tmp = b[k]; b[k] = b[pk]; b[pk] = tmp; }
    }
    for (k = 0; k < n - 1; k++) {
        col_k = a[k];
        for (i = k + 1; i < n; i++) b[i] -= col_k[i] * b[k];
    }
    for (k = n - 1; k > 0; k--) {
        col_k = a[k];
        b[k] /= col_k[k];
        for (i = 0; i < k; i++) b[i] -= col_k[i] * b[k];
    }
    b[0] /= a[0][0];
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Build a well-conditioned (diagonally dominant) matrix into flat col-major buf.
static void fill_matrix(double *flat, int n, unsigned seed) {
    srand(seed);
    for (int col = 0; col < n; col++)
        for (int row = 0; row < n; row++) {
            double v = ((double)rand() / RAND_MAX) - 0.5;
            flat[(long)col * n + row] = v;
        }
    for (int d = 0; d < n; d++)
        flat[(long)d * n + d] += (double)n; // diagonally dominant => stable LU
}

int main(int argc, char **argv) {
    int sizes[] = {30, 50, 100, 150, 200, 300, 400, 600, 800};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    // target ~0.4s of work per method per size; reps scale ~1/N^3
    printf("%5s %10s %12s %12s %9s\n", "N", "reps", "builtin_us", "lapack_us", "speedup");
    for (int s = 0; s < nsizes; s++) {
        int n = sizes[s];
        long n2 = (long)n * n;
        // reps: aim for stable timing; fewer at large N
        long reps = (long)(4e9 / ((double)n * n * n));
        if (reps < 3) reps = 3;
        if (reps > 200000) reps = 200000;

        double *master = malloc(n2 * sizeof(double));
        fill_matrix(master, n, 12345u + s);

        // --- builtin: needs array-of-column-pointers over a working copy ---
        double *work = malloc(n2 * sizeof(double));
        double **cols = malloc(n * sizeof(double *));
        long *p = malloc(n * sizeof(long));
        double *b = malloc(n * sizeof(double));
        double checksum_b = 0.0;
        double t0 = now_s();
        for (long r = 0; r < reps; r++) {
            memcpy(work, master, n2 * sizeof(double));
            for (int c = 0; c < n; c++) cols[c] = work + (long)c * n;
            for (int i = 0; i < n; i++) b[i] = 1.0 + (i % 7);
            sun_GETRF(cols, n, n, p);
            sun_GETRS(cols, n, p, b);
            checksum_b += b[0] + b[n - 1];
        }
        double t_builtin = (now_s() - t0) / reps * 1e6; // us per solve

        // --- lapack (Accelerate) ---
        int *ipiv = malloc(n * sizeof(int));
        double *workl = malloc(n2 * sizeof(double));
        double *bl = malloc(n * sizeof(double));
        int N = n, NRHS = 1, LDA = n, LDB = n, INFO;
        double checksum_l = 0.0;
        t0 = now_s();
        for (long r = 0; r < reps; r++) {
            memcpy(workl, master, n2 * sizeof(double));
            for (int i = 0; i < n; i++) bl[i] = 1.0 + (i % 7);
            dgetrf_(&N, &N, workl, &LDA, ipiv, &INFO);
            dgetrs_("N", &N, &NRHS, workl, &LDA, ipiv, bl, &LDB, &INFO);
            checksum_l += bl[0] + bl[n - 1];
        }
        double t_lapack = (now_s() - t0) / reps * 1e6;

        // correctness: both solve the same system; checksums must match closely
        double rel = fabs(checksum_b - checksum_l) /
                     (fabs(checksum_b) + 1e-30);
        printf("%5d %10ld %12.3f %12.3f %8.2fx%s\n", n, reps, t_builtin,
               t_lapack, t_builtin / t_lapack,
               rel > 1e-6 ? "  <-- MISMATCH!" : "");

        free(master); free(work); free(cols); free(p); free(b);
        free(ipiv); free(workl); free(bl);
    }
    return 0;
}
