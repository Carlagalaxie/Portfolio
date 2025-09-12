import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Equation : u_t + v u_s - nu u_ss + lambda u = f(s)
# Solution exacte : u_ex(s) = exp(-10*(s - 0.5)^2)
# ============================================================

# --- PARAMETRES PHYSIQUES ---
nu = 0.01     # coefficient de diffusion
L = 1.0       # taille du domaine spatial
Time = 20.0   # temps d'intégration
v = 1.0       # vitesse d’advection
lamda = 1.0   # coefficient de réaction

# --- PARAMETRES NUMERIQUES ---
NX = 2                  # nb de points (augmentera à chaque raffinement)
NT = 10000              # nb max de pas de temps
ifre = 1000000          # fréquence d'affichage
eps = 1e-3              # tolérance convergence
niter_refinement = 10   # nb de raffinement

error = np.zeros((niter_refinement))

for iter in range(niter_refinement):
    NX = NX + 3
    dx = L/(NX-1)  # pas d’espace

    # condition CFL
    dt = dx**2 / (v*dx + nu + dx**2)
    print(f"dx={dx:.5f}, dt={dt:.5e}")

    # --- INITIALISATION ---
    x = np.linspace(0.0, 1.0, NX)
    T = np.zeros((NX))    # solution numérique
    F = np.zeros((NX))    # terme source
    RHS = np.zeros((NX))  # résidu
    rest = []

    # --- SOLUTION EXACTE ---
    Tex = np.zeros((NX))
    Texx = np.zeros((NX))

    for j in range(NX):
        s = x[j]
        Tex[j] = np.exp(-10*(s - 0.5)**2)

        # dérivées exactes
        u1 = -20*(s - 0.5)*Tex[j]                       # u'
        u2 = (-20 + 400*(s - 0.5)**2)*Tex[j]            # u''

        # terme source f(s) = v u' - nu u'' + lambda u
        F[j] = v*u1 - nu*u2 + lamda*Tex[j]

        # pour l'erreur H1
        Texx[j] = u1

    # recalcul dt avec F (stabilité)
    dt = dx**2 / (v*dx + 2*nu + abs(np.max(F))*dx**2)

    # --- BOUCLE EN TEMPS ---
    n = 0
    res = 1
    res0 = 1

    while (n < NT and res/res0 > eps):
        n += 1
        res = 0

        # calcul RHS
        for j in range(1, NX-1):
            xnu = nu + 0.5*dx*abs(v)
            Tx = (T[j+1] - T[j-1])/(2*dx)
            Txx = (T[j-1] - 2*T[j] + T[j+1])/(dx**2)

            RHS[j] = dt * (-v*Tx + xnu*Txx - lamda*T[j] + F[j])
            res += abs(RHS[j])

        # mise à jour solution
        for j in range(1, NX-1):
            T[j] += RHS[j]
            RHS[j] = 0

        if n == 1:
            res0 = res
        rest.append(res)

        # affichage intermédiaire
        if (n % ifre == 0 or (res/res0) < eps):
            print(f"it={n}, res={res:.3e}")

    print(f"Convergence en {n} itérations, res={res:.3e}")

    # --- PLOTS ---
    plt.figure(1)
    plt.plot(x, T, label=f"NX={NX}")
    plt.plot(x, Tex, '--', label="Texacte")

    plt.figure(2)
    plt.plot(np.log10(rest/ rest[0]), label=f"NX={NX}")

    # --- ERREUR ---
    err = np.dot(T-Tex, T-Tex)
    errh1 = 0
    for j in range(1, NX-1):
        errh1 += (Texx[j] - (T[j+1]-T[j-1])/(2*dx))**2

    error[iter] = np.sqrt(err)
    print('Erreur L2 =', error[iter])

# --- AFFICHAGE FINAL ---
plt.figure(1)
plt.xlabel('s')
plt.ylabel('u')
plt.title('ADRS 1D : solution numérique vs exacte')
plt.legend()

plt.figure(2)
plt.xlabel('itérations')
plt.ylabel('log10(residu relatif)')
plt.title('Convergence du résidu')
plt.legend()
plt.show()
