import numpy as np
import matplotlib.pyplot as plt

# ---------------------
# Paramètres physiques
# ---------------------
K = 0.01
V = 1.0
lamda = 1.0
xmin, xmax = 0.0, 1.0
Time = 10.0

# ---------------------
# Paramètres de maillage
# ---------------------
hmin = 0.02
hmax = 0.15
NXmax = 200
niter_refinement = 50
NXtarget = 20   # Nombre de points minimum exigé

# ---------------------
# Paramètres numériques
# ---------------------
NT = 10000
eps = 1e-3
tol_residual = 1e-6

# ---------------------
# Condition initiale
# ---------------------
def initial_tex(x):
    return 2*np.exp(-100*(x - 0.25)**2) + np.exp(-200*(x - 0.65)**2)

# ---------------------
# Solveur ADR explicite
# ---------------------
def adrs(x, Tex, K, V, lamda, dt, NT, eps):
    N = len(x)
    T = Tex.copy()
    RHS = np.zeros(N)

    res = 1.0
    res0 = 1.0
    t = 0.0
    n = 0

    while n < NT and (res0 == 0 or res/res0 > eps) and t < Time:
        n += 1
        t += dt
        res = 0.0

        for j in range(1, N-1):
            dx_forward = x[j+1] - x[j]
            dx_backward = x[j] - x[j-1]
            dx_center = 0.5*(dx_forward + dx_backward)

            # dérivées
            Tx = (T[j+1] - T[j-1])/ (dx_forward + dx_backward)
            Txip1 = (T[j+1] - T[j]) / dx_forward
            Txim1 = (T[j] - T[j-1]) / dx_backward
            Txx = (Txip1 - Txim1) / dx_center

            # viscosité numérique
            visnum = 0.25*(dx_forward + dx_backward)*abs(V)
            xnu = K + visnum

            # terme de flux
            F = V*((Tex[j+1]-Tex[j-1])/(dx_forward + dx_backward)) - K*Txx + lamda*Tex[j]

            RHS[j] = dt * (-V*Tx + xnu*Txx - lamda*T[j] + F)
            res += abs(RHS[j])

        T[1:-1] += RHS[1:-1]
        T[-1] = T[-2]  # Neumann à droite

        if n == 1:
            res0 = res if res != 0 else 1e-16

    return T, res

# ---------------------
# Calcul du metric pour adaptation
# ---------------------
def metric(x, T, Tex, errtol, hmin, hmax):
    N = len(x)
    metric = np.ones(N)

    for j in range(1, N-1):
        dx_forward = x[j+1] - x[j]
        dx_backward = x[j] - x[j-1]
        dx_center = 0.5*(dx_forward + dx_backward)

        Txip1 = (T[j+1] - T[j]) / dx_forward
        Txim1 = (T[j] - T[j-1]) / dx_backward
        Txx = (Txip1 - Txim1) / dx_center

        metric[j] = min(1./hmin**2, max(1./hmax**2, abs(Txx)/errtol))

    metric[0] = metric[1]
    metric[-1] = metric[-2]

    # lissage
    metric[:-1] = 0.5*(metric[:-1] + metric[1:])
    metric[-1] = metric[-2]

    hloc = np.sqrt(1.0 / metric)
    return metric, hloc

# ---------------------
# Mesh adaptation
# ---------------------
def mesh(x, metric, hmin, hmax):
    xnew = [x[0]]
    N = len(x)
    for i in range(N-1):
        dx = x[i+1] - x[i]
        length = 0.5 * dx**2 * (metric[i] + metric[i+1])
        nseg = max(1, int(np.ceil(np.sqrt(length))))
        for k in range(1, nseg+1):
            newpt = x[i] + k*dx/nseg
            if newpt < x[-1] - 1e-12:
                xnew.append(newpt)
    xnew.append(x[-1])
    return np.array(xnew)

# ---------------------
# Erreurs L2 et H1
# ---------------------
def compute_errors(x, T, Tex):
    errL2 = 0.0
    errH1 = 0.0
    for j in range(1, len(x)-1):
        dx_center = 0.5*(x[j+1] - x[j-1])
        Tx = (T[j+1] - T[j-1]) / (x[j+1] - x[j-1])
        Texx = (Tex[j+1] - Tex[j-1]) / (x[j+1] - x[j-1])
        errL2 += dx_center * (T[j] - Tex[j])**2
        errH1 += dx_center * (Tx - Texx)**2
    return errL2, errH1

# ---------------------
# MAIN avec visualisation
# ---------------------
err_list = [0.04, 0.02, 0.01, 0.005, 0.0025]
results = []

for errtol in err_list:
    print(f"\n===== Tolérance err = {errtol} =====")
    x = np.linspace(xmin, xmax, 3)
    Tex = initial_tex(x)
    dt = 0.01

    converged = False
    it = 0
    prev_err = np.inf
    while not converged and it < niter_refinement:
        it += 1
        T, res = adrs(x, Tex, K, V, lamda, dt, NT, eps)
        errL2, errH1 = compute_errors(x, T, Tex)
        print(f" Iter {it}: NX={len(x)}, L2={errL2:.4e}")

        # Affichage T(x) et maillage
        plt.figure(figsize=(6,3))
        plt.plot(x, T, 'b-', label='T(x)')
        plt.plot(x, T, 'ro', label='maillage')
        plt.xlabel('x')
        plt.ylabel('T')
        plt.title(f'Err tol={errtol}, Iter {it}, NX={len(x)}')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Arrêt si erreur stagne
        if abs(prev_err - errL2) < tol_residual:
            print("  -> Erreur stagne, arrêt adaptation")
            converged = True
            results.append((errtol, len(x), errL2))
            break
        prev_err = errL2

        # Critère mixte
        if (errL2 < errtol) and (len(x) >= NXtarget):
            converged = True
            results.append((errtol, len(x), errL2))
            break
        else:
            M, hloc = metric(x, T, Tex, errtol, hmin, hmax)
            x = mesh(x, M, hmin, hmax)
            Tex = initial_tex(x)

# ---------------------
# Plot NX(err)
# ---------------------
errs, Npts, errs_eff = zip(*results)
plt.figure(figsize=(6,4))
plt.loglog(errs, Npts, 'o-', label='NX vs err')
plt.xlabel('Tolérance err')
plt.ylabel('Nombre de points NX')
plt.title('Relation NX(err)')
plt.grid(True, which="both")
plt.legend()
plt.show()
