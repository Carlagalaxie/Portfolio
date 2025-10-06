import numpy as np
import matplotlib.pyplot as plt

# -------------------- ADRS solver on non-uniform mesh --------------------
def solve_ADRS_on_mesh(x, xcontrol, Target, K=0.1, V=1.0, lamda=1.0,
                       NT=2000, eps=1e-6, dt_factor=0.4, max_iters=5000):
    NX = len(x)
    dxs = np.diff(x)
    L = x[-1] - x[0]

    # Source term
    F = np.zeros(NX)
    for ic, a in enumerate(xcontrol):
        center = L / (ic+1)
        F += a * np.exp(-100.0*(x - center)**2)

    # initial T
    T = np.zeros(NX)

    # compute stable dt
    dx_min = np.min(dxs)
    dt = dt_factor * dx_min**2 / (abs(V)*dx_min + 2*K + abs(np.max(F))*dx_min**2 + 1e-12)

    # second derivative on non-uniform mesh
    def second_deriv_nonuniform(T, x):
        Txx = np.zeros_like(T)
        for j in range(1, len(x)-1):
            h1 = x[j] - x[j-1]
            h2 = x[j+1] - x[j]
            Txx[j] = 2.0 * (((T[j+1]-T[j])/h2) - ((T[j]-T[j-1])/h1)) / (h1 + h2)
        return Txx

    # time loop
    res = 1.0
    res0 = None
    n = 0
    while n < max_iters and (res0 is None or res > eps*res0):
        n += 1
        Txx = second_deriv_nonuniform(T, x)
        # first derivative (non-uniform central)
        Tx = np.zeros_like(T)
        for j in range(1, NX-1):
            h1 = x[j] - x[j-1]
            h2 = x[j+1] - x[j]
            Tx[j] = ((T[j+1]-T[j])/h2*(h1/(h1+h2)) + (T[j]-T[j-1])/h1*(h2/(h1+h2)))
        # explicit update
        RHS = dt*(-V*Tx + K*Txx - lamda*T + F)
        res = np.sum(np.abs(RHS))
        if res0 is None:
            res0 = res + 1e-16
        T[1:-1] += RHS[1:-1]

    # compute cost integral
    integrand = (T - Target)**2
    cost = 0.5 * np.trapz(integrand, x)
    return cost, T, x

# -------------------- Compute A, B on reference mesh --------------------
def compute_A_B_with_interpolation(basis_solutions, basis_meshes, T0_sol, T0_mesh, udes, x_ref=None):
    n = len(basis_solutions)
    # reference mesh
    if x_ref is None:
        all_meshes = basis_meshes + [T0_mesh]
        maxN = max(len(m) for m in all_meshes)
        x_min = min(m[0] for m in all_meshes)
        x_max = max(m[-1] for m in all_meshes)
        x_ref = np.linspace(x_min, x_max, max(200, maxN*2))

    # interpolate all quantities on reference mesh
    udes_ref = np.interp(x_ref, T0_mesh, udes)
    T0_ref = np.interp(x_ref, T0_mesh, T0_sol)
    Tic_ref = [np.interp(x_ref, basis_meshes[i], basis_solutions[i]) for i in range(n)]

    # compute A and B
    A = np.zeros((n,n))
    B = np.zeros(n)
    for i in range(n):
        for j in range(i, n):
            A[i,j] = np.trapz(Tic_ref[i]*Tic_ref[j], x_ref)
            A[j,i] = A[i,j]
        B[i] = np.trapz((udes_ref - T0_ref)*Tic_ref[i], x_ref)

    return A, B, x_ref

# -------------------- Solve linear inverse problem --------------------
def solve_linear_inverse(basis_solutions, basis_meshes, T0_sol, T0_mesh, udes, x_ref=None):
    A, B, x_ref = compute_A_B_with_interpolation(
        basis_solutions, basis_meshes, T0_sol, T0_mesh, udes, x_ref
    )
    x_star = np.linalg.solve(A + 1e-12*np.eye(len(B)), B)
    # compute resulting solution
    T_star = np.interp(x_ref, T0_mesh, T0_sol)
    for j in range(len(B)):
        T_star += x_star[j]*np.interp(x_ref, basis_meshes[j], basis_solutions[j])
    # interpolate udes on same mesh
    udes_ref = np.interp(x_ref, T0_mesh, udes)
    # cost
    cost = 0.5 * np.trapz((T_star - udes_ref)**2, x_ref)
    return x_star, T_star, cost

# -------------------- Refinement loop --------------------
def refinement_study(Xopt_true=None, nbc=4, mesh_sizes=[20,40,80,160]):
    errors = []
    costs = []
    x_stars_all = []

    for NX in mesh_sizes:
        x_mesh = np.linspace(0,1,NX)
        # Target solution
        if Xopt_true is not None:
            _, udes, _ = solve_ADRS_on_mesh(x_mesh, Xopt_true, np.zeros_like(x_mesh))
        else:
            udes = np.ones_like(x_mesh)

        # Zero control solution
        _, T0_sol, T0_mesh = solve_ADRS_on_mesh(x_mesh, np.zeros(nbc), np.zeros_like(x_mesh))

        # Basis solutions
        basis_sols = []
        basis_meshes = []
        for j in range(nbc):
            ei = np.zeros(nbc); ei[j]=1.0
            _, uj, _ = solve_ADRS_on_mesh(x_mesh, ei, udes)
            basis_sols.append(uj)
            basis_meshes.append(x_mesh)

        # Solve linear inverse
        x_star, T_star, cost = solve_linear_inverse(basis_sols, basis_meshes, T0_sol, T0_mesh, udes)

        x_stars_all.append(x_star)
        costs.append(cost)
        if Xopt_true is not None:
            errors.append(np.linalg.norm(x_star - Xopt_true))
        else:
            errors.append(np.nan)  # unknown true Xopt

        print(f"NX={NX}, x*={np.round(x_star,4)}, cost={cost:.4e}")

    # Convert to arrays
    x_stars_all = np.array(x_stars_all)
    costs = np.array(costs)
    errors = np.array(errors)

    # Plot convergence
    plt.figure(figsize=(10,5))
    for j in range(nbc):
        plt.plot(mesh_sizes, x_stars_all[:,j], '-o', label=f"x*_{j+1}")
    plt.xlabel("Mesh size NX")
    plt.ylabel("Optimal control x*")
    plt.title("Convergence of x* components with mesh refinement")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(mesh_sizes, costs, 'r-o')
    plt.xlabel("Mesh size NX")
    plt.ylabel("Cost J(x*)")
    plt.title("Cost vs mesh refinement")
    plt.grid(True)
    plt.show()

    if Xopt_true is not None:
        plt.figure(figsize=(8,4))
        plt.plot(mesh_sizes, errors, 'b-o')
        plt.xlabel("Mesh size NX")
        plt.ylabel("Error ||x* - Xopt||")
        plt.title("Error vs mesh refinement")
        plt.grid(True)
        plt.show()

# -------------------- Run experiments --------------------
if __name__ == '__main__':
    print("=== Case 1: Known Xopt ===")
    refinement_study(Xopt_true=np.array([1.0,2.0,3.0,4.0]), nbc=4, mesh_sizes=[20,40,80,160])

    print("=== Case 2: Unknown Xopt, udes=1 ===")
    refinement_study(Xopt_true=None, nbc=4, mesh_sizes=[20,40,80,160])
