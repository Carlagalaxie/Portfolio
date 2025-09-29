import numpy as np
import matplotlib.pyplot as plt

# ================================
# PARAMÈTRES PHYSIQUES
# ================================
K = 0.1       # diffusion
L = 1.0       # taille du domaine
V = 1.0       # advection
lamda = 1.0   # réaction
Time = 1.0    # temps final

# ================================
# PARAMÈTRES NUMÉRIQUES
# ================================
NX = 50              # nombre de points initiaux
dx = L/(NX-1)
dt = 0.5*dx**2/(K + V*dx/2)  # CFL simplifié
NT = int(Time/dt) + 1

# Grille initiale
x = np.linspace(0, L, NX)

# Fonction spatiale exacte
v = np.exp(-20*(x-0.5)**2)

# Initialisation
T = np.zeros(NX)

# Stockage pour snapshots
snapshots = []
M_list = []  # liste des métriques temporelles

# ================================
# BOUCLE TEMPORELLE
# ================================
for n in range(NT):
    t = n*dt

    # Solution exacte au temps t
    Tex = np.sin(4*np.pi*t) * v

    # Dérivées spatiales de Tex
    Texx = np.zeros_like(Tex)
    Texx[1:-1] = (Tex[2:] - 2*Tex[1:-1] + Tex[:-2])/(dx**2)

    # ----- Terme source f(x,t) -----
    dUdt = 4*np.pi*np.cos(4*np.pi*t) * v
    v_x = np.zeros_like(v)
    v_x[1:-1] = (v[2:] - v[:-2])/(2*dx)
    v_xx = np.zeros_like(v)
    v_xx[1:-1] = (v[2:] - 2*v[1:-1] + v[:-2])/(dx**2)
    F = dUdt + V*np.sin(4*np.pi*t)*v_x - K*np.sin(4*np.pi*t)*v_xx + lamda*np.sin(4*np.pi*t)*v

    # ----- Schéma explicite -----
    Tx = np.zeros_like(T)
    Txx = np.zeros_like(T)
    Tx[1:-1] = (T[2:] - T[:-2])/(2*dx)
    Txx[1:-1] = (T[2:] - 2*T[1:-1] + T[:-2])/(dx**2)

    xnu = K + 0.5*dx*abs(V)
    RHS = np.zeros_like(T)
    RHS[1:-1] = dt*(-V*Tx[1:-1] + xnu*Txx[1:-1] - lamda*T[1:-1] + F[1:-1])
    T[1:-1] += RHS[1:-1]

    # ----- Metrqiue locale (curvature) -----
    M_local = np.abs(Texx)
    M_list.append(M_local)

    # Stocker quelques snapshots
    if n % max(1, NT//5) == 0:
        snapshots.append((t, T.copy(), Tex.copy()))

# ================================
# MÉTRIQUE GLOBALE
# ================================
M_list = np.array(M_list)
M_mean = np.mean(M_list, axis=0)   # moyenne temporelle
M_max  = np.max(M_list, axis=0)    # intersection temporelle

# ================================
# MAILLAGE ADAPTATIF
# ================================
def build_adapted_mesh(x, M, Ntarget=50):
    """Construit un maillage 1D adaptatif basé sur une métrique M(x)."""
    M = M/np.max(M) + 1e-10  # normalisation
    density = M/np.trapz(M, x)   # densité relative
    # fonction de répartition cumulative
    cdf = np.cumsum(density) / np.sum(density)
    # interpolation pour obtenir Ntarget points
    x_adapt = np.interp(np.linspace(0,1,Ntarget), cdf, x)
    return x_adapt

# Construire un maillage basé sur la métrique moyenne
x_adapt_mean = build_adapted_mesh(x, M_mean, Ntarget=50)
x_adapt_max  = build_adapted_mesh(x, M_max,  Ntarget=50)

# ================================
# PLOTS
# ================================

# Snapshots de la solution numérique
plt.figure(figsize=(8,5))
for t, Tnum, Tex in snapshots:
    plt.plot(x, Tnum, label=f'num t={t:.2f}')
    plt.plot(x, Tex, 'k--', alpha=0.5)
plt.title("Solution numérique vs exacte (snapshots)")
plt.xlabel("x"); plt.ylabel("T")
plt.legend()
plt.show()

# Metrqiues
plt.figure(figsize=(8,5))
plt.plot(x, M_mean, label="Métrique moyenne (temps)")
plt.plot(x, M_max,  label="Métrique intersection (max en temps)")
plt.title("Métriques temporelles")
plt.xlabel("x"); plt.ylabel("M(x)")
plt.legend()
plt.show()

# Maillages adaptés
plt.figure(figsize=(8,5))
plt.plot(x, np.zeros_like(x), 'k-', lw=1, label="Maillage uniforme")
plt.plot(x_adapt_mean, np.zeros_like(x_adapt_mean)+0.05, 'ro', label="Adapté (moyenne)")
plt.plot(x_adapt_max, np.zeros_like(x_adapt_max)+0.1, 'bs', label="Adapté (max)")
plt.title("Maillage adaptatif basé sur la métrique")
plt.yticks([])
plt.legend()
plt.show()
