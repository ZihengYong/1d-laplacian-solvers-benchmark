"""
The 1D Laplacian: Direct vs. Iterative Structure-Exploiting Solvers Benchmark
=============================================================================
Solves A_n x = b where A_n = tridiag(-1, 2, -1) in R^{n x n}, b = A_n 1,
and exact solution x* = 1 across n in {20, 40, 80}.

Compares 6 structure-exploiting solvers:
  - Stationary iterative methods (x_0 = 0, stop when ||b - A_n x_k||_2 / ||b||_2 < 1e-8):
      1. Jacobi
      2. Gauss-Seidel (GS)
      3. Symmetric Gauss-Seidel (SGS, counting both forward and backward sweeps)
  - O(n) banded direct solvers (including factorization in total FLOPs & runtime):
      4. Thomas Algorithm
      5. Tridiagonal LU Factorization
      6. Tridiagonal Cholesky Factorization (A_n = L L^T)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt


def build_1d_laplacian(n: int):
    A = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    x_exact = np.ones(n)
    b = A @ x_exact
    return A, b, x_exact


def solve_jacobi(n: int, b: np.ndarray, tol: float = 1e-8, max_iter: int = 50000):
    """Matrix-free Jacobi iteration exploiting the 3-point stencil."""
    x = np.zeros(n)
    norm_b = np.linalg.norm(b)
    flops = 0
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        # Matrix-free residual r = b - A_n x (~6n flops incl. norm)
        Ax = 2.0 * x
        Ax[1:] -= x[:-1]
        Ax[:-1] -= x[1:]
        r = b - Ax
        flops += 6 * n
        if np.linalg.norm(r) / norm_b < tol:
            return x, it, flops, time.perf_counter() - t0
        # Jacobi update: x_new = 0.5 * (b + x_left + x_right) (~3n flops)
        x_new = b.copy()
        x_new[1:] += x[:-1]
        x_new[:-1] += x[1:]
        x = 0.5 * x_new
        flops += 3 * n
    return x, max_iter, flops, time.perf_counter() - t0


def solve_gs(n: int, b: np.ndarray, tol: float = 1e-8, max_iter: int = 50000):
    """In-place Gauss-Seidel exploiting tridiagonal nearest-neighbor coupling."""
    x = np.zeros(n)
    norm_b = np.linalg.norm(b)
    flops = 0
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        for i in range(n):
            left = x[i - 1] if i > 0 else 0.0
            right = x[i + 1] if i < n - 1 else 0.0
            x[i] = 0.5 * (b[i] + left + right)
        flops += 3 * n
        Ax = 2.0 * x
        Ax[1:] -= x[:-1]
        Ax[:-1] -= x[1:]
        flops += 5 * n
        if np.linalg.norm(b - Ax) / norm_b < tol:
            return x, it, flops, time.perf_counter() - t0
    return x, max_iter, flops, time.perf_counter() - t0


def solve_sgs(n: int, b: np.ndarray, tol: float = 1e-8, max_iter: int = 50000):
    """Symmetric Gauss-Seidel counting BOTH forward and backward sweeps."""
    x = np.zeros(n)
    norm_b = np.linalg.norm(b)
    flops = 0
    sweeps = 0
    t0 = time.perf_counter()
    for _ in range(1, max_iter + 1):
        # Forward sweep
        for i in range(n):
            left = x[i - 1] if i > 0 else 0.0
            right = x[i + 1] if i < n - 1 else 0.0
            x[i] = 0.5 * (b[i] + left + right)
        # Backward sweep
        for i in range(n - 1, -1, -1):
            left = x[i - 1] if i > 0 else 0.0
            right = x[i + 1] if i < n - 1 else 0.0
            x[i] = 0.5 * (b[i] + left + right)
        sweeps += 2
        flops += 6 * n
        Ax = 2.0 * x
        Ax[1:] -= x[:-1]
        Ax[:-1] -= x[1:]
        flops += 5 * n
        if np.linalg.norm(b - Ax) / norm_b < tol:
            return x, sweeps, flops, time.perf_counter() - t0
    return x, sweeps, flops, time.perf_counter() - t0


def solve_thomas(n: int, b: np.ndarray):
    """Thomas algorithm (specialized O(n) tridiagonal elimination): 8n - 7 flops."""
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)
    c_prime[0] = -0.5
    d_prime[0] = b[0] * 0.5
    for i in range(1, n - 1):
        denom = 2.0 + c_prime[i - 1]
        c_prime[i] = -1.0 / denom
        d_prime[i] = (b[i] + d_prime[i - 1]) / denom
    d_prime[n - 1] = (b[n - 1] + d_prime[n - 2]) / (2.0 + c_prime[n - 2])
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x, 1, 8 * n - 7


def solve_tridiag_lu(n: int, b: np.ndarray):
    """Bidiagonal LU factorization A_n = L U + triangular solves: 8n - 7 flops."""
    l_sub = np.zeros(n - 1)
    u_diag = np.zeros(n)
    u_diag[0] = 2.0
    for i in range(1, n):
        l_sub[i - 1] = -1.0 / u_diag[i - 1]
        u_diag[i] = 2.0 + l_sub[i - 1]
    y = np.zeros(n)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - l_sub[i - 1] * y[i - 1]
    x = np.zeros(n)
    x[n - 1] = y[n - 1] / u_diag[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] + x[i + 1]) / u_diag[i]
    return x, 1, 8 * n - 7


def solve_tridiag_cholesky(n: int, b: np.ndarray):
    """Tridiagonal Cholesky factorization A_n = L L^T + triangular solves: 7n - 5 flops."""
    l_diag = np.zeros(n)
    l_sub = np.zeros(n - 1)
    l_diag[0] = np.sqrt(2.0)
    for i in range(1, n):
        l_sub[i - 1] = -1.0 / l_diag[i - 1]
        l_diag[i] = np.sqrt(2.0 - l_sub[i - 1] ** 2)
    y = np.zeros(n)
    y[0] = b[0] / l_diag[0]
    for i in range(1, n):
        y[i] = (b[i] - l_sub[i - 1] * y[i - 1]) / l_diag[i]
    x = np.zeros(n)
    x[n - 1] = y[n - 1] / l_diag[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - l_sub[i] * x[i + 1]) / l_diag[i]
    return x, 1, 7 * n - 5


def benchmark_direct(fn, n: int, b: np.ndarray, repeats: int = 200):
    """Average direct solver runtime over multiple calls for accurate microsecond timing."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        x, cnt, flp = fn(n, b)
    elapsed = (time.perf_counter() - t0) / repeats
    return x, cnt, flp, elapsed


def main():
    ns_lap = [20, 40, 80]
    methods = ["Jacobi", "Gauss-Seidel", "SGS (sweeps)", "Thomas", "Tridiag LU", "Tridiag Cholesky"]
    lap_summary = {m: {"iters": [], "flops": [], "err": [], "time_ms": []} for m in methods}

    for n in ns_lap:
        A, b, x_ex = build_1d_laplacian(n)
        rho_j = np.cos(np.pi / (n + 1))
        rho_gs = rho_j ** 2
        D = 2.0 * np.eye(n)
        L_s = -np.eye(n, k=-1)
        U_s = -np.eye(n, k=1)
        G_sgs = np.linalg.inv(D + U_s) @ (-L_s) @ np.linalg.inv(D + L_s) @ (-U_s)
        rho_sgs = float(np.max(np.abs(np.linalg.eigvals(G_sgs))))

        print(
            f"\n=== n = {n} | kappa_2(A) = {np.linalg.cond(A):.1f} | "
            f"rho(J)={rho_j:.5f}, rho(GS)={rho_gs:.5f}, rho(SGS)={rho_sgs:.5f} ==="
        )
        solvers = [
            ("Jacobi", solve_jacobi(n, b)),
            ("Gauss-Seidel", solve_gs(n, b)),
            ("SGS (sweeps)", solve_sgs(n, b)),
            ("Thomas", benchmark_direct(solve_thomas, n, b)),
            ("Tridiag LU", benchmark_direct(solve_tridiag_lu, n, b)),
            ("Tridiag Cholesky", benchmark_direct(solve_tridiag_cholesky, n, b)),
        ]
        for name, (x_sol, it_cnt, flp, elp) in solvers:
            rel_err = float(np.linalg.norm(x_sol - x_ex) / np.linalg.norm(x_ex))
            t_ms = elp * 1e3
            lap_summary[name]["iters"].append(it_cnt)
            lap_summary[name]["flops"].append(flp)
            lap_summary[name]["err"].append(rel_err)
            lap_summary[name]["time_ms"].append(t_ms)
            print(
                f"  {name:<18} | Iters/Sweeps: {it_cnt:<6d} | FLOPs: {flp:<9d} | "
                f"Time: {t_ms:<8.3f} ms | RelErr: {rel_err:.3e}"
            )

    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=150)

    styles = {
        "Jacobi": ("o-", "#7f8c8d"),
        "Gauss-Seidel": ("s-", "#2980b9"),
        "SGS (sweeps)": ("^-", "#8e44ad"),
        "Thomas": ("D--", "#27ae60"),
        "Tridiag LU": ("v--", "#e67e22"),
        "Tridiag Cholesky": ("P--", "#c0392b"),
    }
    for name, (sty, col) in styles.items():
        axes[0].loglog(ns_lap, lap_summary[name]["flops"], sty, color=col, linewidth=2.0, markersize=7, label=name)
    axes[0].set_title(r"Total Work (FLOPs) Scaling: $O(n^3)$ Iterative vs. $O(n)$ Direct", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("System Dimension $n$ (log scale)")
    axes[0].set_ylabel("Total FLOPs (log scale)")
    axes[0].set_xticks(ns_lap)
    axes[0].set_xticklabels([str(v) for v in ns_lap])
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.2, loc="upper left")

    x_pos = np.arange(len(ns_lap))
    width = 0.25
    axes[1].bar(x_pos - width, lap_summary["Jacobi"]["iters"], width, color="#7f8c8d", label="Jacobi (Iters)")
    axes[1].bar(x_pos, lap_summary["Gauss-Seidel"]["iters"], width, color="#2980b9", label="Gauss-Seidel (Iters)")
    axes[1].bar(x_pos + width, lap_summary["SGS (sweeps)"]["iters"], width, color="#8e44ad", label="SGS (Total Sweeps)")
    axes[1].set_yscale("log")
    axes[1].set_title(r"Iterative Work Count ($\|r_k\|_2 / \|b\|_2 < 10^{-8}$)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("System Dimension $n$")
    axes[1].set_ylabel("Iterations / Sweeps (log scale)")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"$n={v}$" for v in ns_lap])
    axes[1].grid(True, axis="y", which="both", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8.5, loc="upper left")

    plt.tight_layout()
    plt.savefig("results/laplacian_1d_benchmark.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
