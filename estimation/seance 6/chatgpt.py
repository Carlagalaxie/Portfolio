import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

#-----------------------------------
# 1. ADRS solver 1D
#-----------------------------------
def ADRS(NX, xcontrol, Target=None, L=1.0, V=1.0, K=0.1, lamda=1.0, Time=20.0):
    """
    Résolution ADRS 1D avec nb contrôles sources.
    """
    dx = L / (NX - 1)
    x = np.linspace(0, L, NX)
    T = np.zeros(NX)
    F = np.zeros(NX)
    nbc = len(xcontrol)

    # Construction du second membre F
    for j in range(1, NX-1):
        for ic in range(nbc):
            F[j] += xcontrol[ic] * np.exp(-100*(x[j]-L/(ic+1))**2)

    dt = 0.5*dx**2/(V*dx + 2*K + np.max(np.abs(F))*dx**2)
    NT = int(Time/dt)
    
    # Boucle temporelle
    for n in range(NT):
        Tx = (T[2:] - T[:-2])/(2*dx)
        Txx = (T[:-2] - 2*T[1:-1] + T[2:])/(dx**2)
        xnu = K + 0.5*dx*abs(V)
        RHS = dt*(-V*Tx + xnu*Txx - lamda*T[1:-1] + F[1:-1])
        T[1:-1] += RHS

    cost = None
    if Target is not None:
        cost = 0.5 * np.sum((T - Target)**2) * dx
    return cost, T, x

#-----------------------------------
# 2. Calcul des solutions élémentaires
#-----------------------------------
def compute_elementary_solutions(NX, nbc, Xopt=None, refine=False):
    """
    Calcule u0 et uj pour j=1..nbc et fait interpolation si maillage raffiné.
    """
    NX_fine = NX*2 if refine else NX
    x_fine = np.linspace(0, 1, NX_fine)

    # u_des = solution pour Xopt si fourni
    if Xopt is not None:
        _, udes_fine, _ = ADRS(NX_fine, Xopt)
    else:
        udes_fine = np.ones(NX_fine)

    # u0 = solution pour xcontrol = 0
    _, u0_fine, _ = ADRS(NX_fine, np.zeros(nbc))

    # Solutions élémentaires
    uj_fine = []
    for j in range(nbc):
        xj = np.zeros(nbc)
        xj[j] = 1.0
        _, uj, _ = ADRS(NX_fine, xj)
        uj_fine.append(uj)

    # Interpolation sur maillage commun
    x_common = np.linspace(0, 1, NX)
    udes = np.interp(x_common, x_fine, udes_fine)
    u0 = np.interp(x_common, x_fine, u0_fine)
    uj = [np.interp(x_common, x_fine, u) for u in uj_fine]
    dx_common = 1.0/(NX-1)
    return udes, u0, uj, dx_common, x_common

#-----------------------------------
# 3. Résolution du problème inverse linéaire
#-----------------------------------
def solve_inverse_problem(udes, u0, uj, dx):
    nbc = len(uj)
    A = np.zeros((nbc, nbc))
    b = np.zeros(nbc)
    for i in range(nbc):
        for j in range(nbc):
            A[i,j] = np.sum(uj[i]*uj[j])*dx
        b[i] = np.sum(uj[i]*(udes - u0))*dx
    x_opt = np.linalg.solve(A, b)
    return x_opt

#-----------------------------------
# 4. Boucle de raffinement et comparaison avec maillage fixe
#-----------------------------------
def refinement_loop(nbc, Xopt=None, NX_list=[20,30,40,50,60,70], refine=True):
    x_opt_list = []
    J_list = []

    for NX in NX_list:
        udes, u0, uj, dx, xmesh = compute_elementary_solutions(NX, nbc, Xopt=Xopt, refine=refine)
        x_opt = solve_inverse_problem(udes, u0, uj, dx)
        x_opt_list.append(x_opt)

        # Calcul du coût
        cost, u_sol, _ = ADRS(NX, x_opt, Target=udes)
        J_list.append(cost)
        print(f"NX={NX}, x_opt={x_opt}, J={cost}")

    x_opt_array = np.array(x_opt_list)
    plt.figure()
    for i in range(nbc):
        plt.plot(NX_list, x_opt_array[:,i], '-o', label=f"x{i+1}")
    plt.xlabel("NX")
    plt.ylabel("x_opt")
    plt.title("Convergence des composantes de x_opt")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(NX_list, J_list, '-o')
    plt.xlabel("NX")
    plt.ylabel("J(x_opt)")
    plt.title("Coût en fonction du raffinement")
    plt.show()

#-----------------------------------
# 5. Surface 3D du coût pour 2 contrôles
#-----------------------------------
def plot_cost_surface(NX=28, nbc=4, fixed=None, x1_range=(-3,3), x2_range=(-3,3), ngrid=30):
    xcible = np.arange(nbc) + 1.0
    Target, _, _, _, _ = compute_elementary_solutions(NX, nbc, Xopt=xcible, refine=True)

    if fixed is None:
        fixed = np.zeros(nbc-2)

    x1_vals = np.linspace(x1_range[0], x1_range[1], ngrid)
    x2_vals = np.linspace(x2_range[0], x2_range[1], ngrid)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    J = np.zeros_like(X1)

    for i in range(ngrid):
        for j in range(ngrid):
            xcontrol = np.zeros(nbc)
            xcontrol[0] = X1[i,j]
            xcontrol[1] = X2[i,j]
            xcontrol[2:] = fixed
            J[i,j], _, _ = ADRS(NX, xcontrol, Target)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, J.T, cmap='viridis', alpha=0.8)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("J(x1,x2)")
    ax.view_init(elev=30, azim=135)
    plt.show()

#-----------------------------------
# 6. Exemple d'utilisation
#-----------------------------------
nbc = 4
Xopt = np.array([1.0, 2.0, 3.0, 4.0])

# Raffinement et convergence vers Xopt
refinement_loop(nbc, Xopt=Xopt, NX_list=[20,30,40,50,60,70], refine=True)

# Cas udes inconnu = 1
print("\nCas udes inconnu = 1")
refinement_loop(nbc, Xopt=None, NX_list=[20,30,40,50,60,70], refine=True)

# Surface de coût pour les deux premiers contrôles
plot_cost_surface(NX=28, nbc=4, fixed=np.zeros(2), x1_range=(-3,3), x2_range=(-3,3), ngrid=40)
