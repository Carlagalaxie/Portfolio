# ADRS adaptive mesh + interpolation for Aij and Bi
# Save as adrs_adapt_interpolation.py and run with Python 3 (requires numpy, matplotlib)

import numpy as np
import matplotlib.pyplot as plt

def solve_ADRS_on_mesh(x, xcontrol, Target, K=0.1, V=1.0, lamda=1.0,
                       NT=2000, eps=1e-6, dt_factor=0.4, max_iters=5000):
    """
    Solve ADRS on a given non-uniform mesh x (1D array of node coordinates).
    Returns final solution T on that mesh and the mesh itself (unchanged).
    Uses an explicit time-stepping with stable dt computed from local spacing.
    """
    NX = len(x)
    dxs = np.diff(x)  # length NX-1
    # Build source F on nodes using gaussian bumps centered at L/(ic+1) with same L as before (L=1)
    L = x[-1] - x[0]
    F = np.zeros(NX)
    for ic, a in enumerate(xcontrol):
        center = L / (ic+1)
        F += a * np.exp(-100.0 * (x - center)**2)
    # initial T
    T = np.zeros(NX)
    # compute stable dt from minimal spacing
    dx_min = np.min(dxs)
    dt = dt_factor * dx_min**2 / (abs(V)*dx_min + 2*K + abs(np.max(F))*dx_min**2 + 1e-12)

    # time integration explicit with non-uniform finite differences for second derivative
    def second_deriv_nonuniform(T, x):
        """Compute second derivative at interior nodes for non-uniform grid with 3-point formula."""
        N = len(x)
        Txx = np.zeros_like(T)
        for j in range(1, N-1):
            h1 = x[j] - x[j-1]
            h2 = x[j+1] - x[j]
            # second derivative approximation (3-point on non-uniform grid)
            # formula: 2 * ( (T[j+1]-T[j])/h2 - (T[j]-T[j-1])/h1 ) / (h1+h2)
            Txx[j] = 2.0 * (((T[j+1]-T[j]) / h2) - ((T[j]-T[j-1]) / h1)) / (h1 + h2)
        return Txx

    # time loop
    res = 1.0
    res0 = None
    n = 0
    while n < max_iters and (res0 is None or res > eps * res0):
        n += 1
        Txx = second_deriv_nonuniform(T, x)
        # approximate first derivative with simple central difference on non-uniform grid
        Tx = np.zeros_like(T)
        for j in range(1, NX-1):
            h1 = x[j] - x[j-1]
            h2 = x[j+1] - x[j]
            Tx[j] = ( (T[j+1]-T[j])/(h2) * (h1/(h1+h2)) + (T[j]-T[j-1])/(h1) * (h2/(h1+h2)) )
        RHS = np.zeros_like(T)
        for j in range(1, NX-1):
            RHS[j] = dt * (-V * Tx[j] + K * Txx[j] - lamda * T[j] + F[j])
        res = np.sum(np.abs(RHS))
        if res0 is None:
            res0 = res + 1e-16
        T[1:-1] += RHS[1:-1]
    # compute cost integral on non-uniform mesh via trapezoidal rule
    integrand = (T - Target)**2
    cost = np.trapz(integrand, x)
    return cost, T, x


def adapt_mesh_equidistribute(x_old, T_old, N_new, alpha=20.0):
    """
    Build a new mesh with N_new nodes by equidistribution of a monitor function
    m = 1 + alpha * |dT/dx| (simple). This clusters mesh where gradient is large.
    """
    N_old = len(x_old)
    dTdx = np.zeros(N_old)
    for j in range(1, N_old-1):
        dTdx[j] = (T_old[j+1]-T_old[j-1])/(x_old[j+1]-x_old[j-1])
    m = 1.0 + alpha * np.abs(dTdx)
    c = np.zeros_like(m)
    c[0] = 0.0
    for j in range(1, N_old):
        c[j] = c[j-1] + 0.5*(m[j]+m[j-1])*(x_old[j]-x_old[j-1])
    Ctot = c[-1]
    new_c = np.linspace(0.0, Ctot, N_new)
    x_new = np.interp(new_c, c, x_old)
    return x_new


def compute_A_B_with_interpolation(basis_solutions, basis_meshes, T0_sol, T0_mesh, Target_sol, Target_mesh, x_ref=None):
    """
    Given basis solutions Tic on possibly different meshes, interpolate everything
    on a common reference mesh x_ref and compute A and B with composite trapezoid.
    """
    n = len(basis_solutions)
    if x_ref is None:
        all_meshes = basis_meshes + [T0_mesh, Target_mesh]
        maxN = max(len(m) for m in all_meshes)
        x_min = min(m[0] for m in all_meshes)
        x_max = max(m[-1] for m in all_meshes)
        x_ref = np.linspace(x_min, x_max, max(200, maxN*2))
    Target_ref = np.interp(x_ref, Target_mesh, Target_sol)
    T0_ref = np.interp(x_ref, T0_mesh, T0_sol)
    Tic_ref = [np.interp(x_ref, basis_meshes[i], basis_solutions[i]) for i in range(n)]
    A = np.zeros((n,n))
    B = np.zeros(n)
    for i in range(n):
        for j in range(i, n):
            integrand = Tic_ref[i] * Tic_ref[j]
            A[i,j] = np.trapz(integrand, x_ref)
            A[j,i] = A[i,j]
        integrandB = (Target_ref - T0_ref) * Tic_ref[i]
        B[i] = np.trapz(integrandB, x_ref)
    return A, B, x_ref


def run_adaptive_linearization(nbc=6, NX_init=30, NX_adapt=40, adapt_iters=2, alpha=30.0):
    """
    Full experiment: build Target via adaptive solves, compute basis responses with adaptation,
    interpolate to reference mesh and compute A,B, solve for xopt; compare with fixed fine reference.
    """
    xcible = np.arange(nbc) + 1.0
    x_coarse = np.linspace(0.0, 1.0, NX_init)
    x_curr = x_coarse.copy()
    Target = None
    for k in range(adapt_iters):
        cost, T, mesh = solve_ADRS_on_mesh(x_curr, xcible, np.zeros_like(x_curr))
        x_new = adapt_mesh_equidistribute(mesh, T, NX_adapt if k == adapt_iters-1 else len(mesh), alpha=alpha)
        x_curr = x_new
        Target = np.interp(x_curr, mesh, T)
    Target_mesh = x_curr.copy(); Target_sol = Target.copy()
    x0 = np.linspace(0.0, 1.0, NX_init)
    x_curr0 = x0.copy()
    for k in range(adapt_iters):
        cost0, T0, mesh0 = solve_ADRS_on_mesh(x_curr0, np.zeros(nbc), np.zeros_like(x_curr0))
        x_new0 = adapt_mesh_equidistribute(mesh0, T0, NX_adapt if k == adapt_iters-1 else len(mesh0), alpha=alpha)
        x_curr0 = x_new0
        T0 = np.interp(x_curr0, mesh0, T0)
    T0_mesh = x_curr0.copy(); T0_sol = T0.copy()
    basis_solutions = []
    basis_meshes = []
    for ic in range(nbc):
        xic = np.zeros(nbc); xic[ic] = 1.0
        x_curr_i = np.linspace(0.0, 1.0, NX_init)
        for k in range(adapt_iters):
            costi, Ti, meshi = solve_ADRS_on_mesh(x_curr_i, xic, np.interp(x_curr_i, Target_mesh, Target_sol))
            x_newi = adapt_mesh_equidistribute(meshi, Ti, NX_adapt if k == adapt_iters-1 else len(meshi), alpha=alpha)
            x_curr_i = x_newi
            Ti = np.interp(x_curr_i, meshi, Ti)
        basis_meshes.append(x_curr_i.copy())
        basis_solutions.append(Ti.copy())
    A_adapt, B_adapt, x_ref = compute_A_B_with_interpolation(basis_solutions, basis_meshes,
                                                            T0_sol, T0_mesh, Target_sol, Target_mesh, x_ref=None)
    xopt_adapt = np.linalg.solve(A_adapt + 1e-12*np.eye(nbc), B_adapt)
    cost_opt_adapt, T_opt_adapt, mesh_opt_adapt = solve_ADRS_on_mesh(np.linspace(0.0,1.0,NX_adapt), xopt_adapt, np.interp(np.linspace(0.0,1.0,NX_adapt), Target_mesh, Target_sol))
    NX_ref = 200
    x_ref_fixed = np.linspace(0.0,1.0,NX_ref)
    Target_ref_fixed = np.interp(x_ref_fixed, Target_mesh, Target_sol)
    basis_ref = []
    for ic in range(nbc):
        xic = np.zeros(nbc); xic[ic] = 1.0
        costi, Ti_ref, _ = solve_ADRS_on_mesh(x_ref_fixed, xic, Target_ref_fixed)
        basis_ref.append(Ti_ref.copy())
    cost0_ref, T0_ref, _ = solve_ADRS_on_mesh(x_ref_fixed, np.zeros(nbc), Target_ref_fixed)
    A_ref = np.zeros((nbc,nbc)); B_ref = np.zeros(nbc)
    for i in range(nbc):
        for j in range(i, nbc):
            A_ref[i,j] = np.trapz(basis_ref[i]*basis_ref[j], x_ref_fixed)
            A_ref[j,i] = A_ref[i,j]
        B_ref[i] = np.trapz((Target_ref_fixed - T0_ref) * basis_ref[i], x_ref_fixed)
    xopt_ref = np.linalg.solve(A_ref + 1e-12*np.eye(nbc), B_ref)
    cost_opt_ref, T_opt_ref, _ = solve_ADRS_on_mesh(x_ref_fixed, xopt_ref, Target_ref_fixed)
    results = {
        'Target_mesh': Target_mesh, 'Target': Target_sol,
        'T0_mesh': T0_mesh, 'T0': T0_sol,
        'basis_meshes': basis_meshes, 'basis_solutions': basis_solutions,
        'A_adapt': A_adapt, 'B_adapt': B_adapt, 'xopt_adapt': xopt_adapt, 'cost_opt_adapt': cost_opt_adapt,
        'x_ref': x_ref, 'A_ref': A_ref, 'B_ref': B_ref, 'xopt_ref': xopt_ref, 'cost_opt_ref': cost_opt_ref,
        'mesh_opt_adapt': mesh_opt_adapt, 'T_opt_adapt': T_opt_adapt, 'x_ref_fixed': x_ref_fixed, 'T_opt_ref': T_opt_ref
    }
    plt.figure(figsize=(10,6))
    plt.plot(Target_mesh, Target_sol, 'k.-', label='Target (adapt mesh)')
    plt.plot(T0_mesh, T0_sol, 'b.-', label='T0 (zero control, adapt mesh)')
    plt.plot(mesh_opt_adapt, T_opt_adapt, 'r.-', label='T_opt (from adapt linearization)')
    plt.plot(x_ref_fixed, T_opt_ref, 'g--', label='T_opt (reference fixed fine)')
    plt.legend()
    plt.title("Solutions comparison (Target, T0, T_opt adaptive, T_opt reference)")
    plt.xlabel("x"); plt.ylabel("T(x)");
    plt.grid(True)
    plt.show()
    print("xopt_adapt =", np.round(xopt_adapt,4))
    print("cost_opt_adapt =", cost_opt_adapt)
    print("xopt_ref   =", np.round(xopt_ref,4))
    print("cost_opt_ref   =", cost_opt_ref)
    plt.figure(figsize=(8,4))
    ic = np.arange(1, nbc+1)
    plt.plot(ic, xopt_adapt, 'ro-', label='xopt adapt')
    plt.plot(ic, xopt_ref, 'bs--', label='xopt ref')
    plt.xlabel("control index"); plt.ylabel("value"); plt.legend(); plt.grid(True)
    plt.title("Comparison of optimal controls")
    plt.show()
    return results

if __name__ == '__main__':
    res = run_adaptive_linearization(nbc=6, NX_init=30, NX_adapt=60, adapt_iters=2, alpha=30.0)
    # save results if desired
    # np.savez('adrs_adapt_results.npz', **res)
