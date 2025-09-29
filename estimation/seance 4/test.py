import math
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Fonction: solution exacte manufacturée et source
# --------------------------------------------------------
def fex(NX, dx, time):
    F = np.zeros((NX))
    Tex = np.zeros((NX))
    Text = np.zeros((NX))
    Texx = np.zeros((NX))
    for j in range(1, NX-1):
        v = (np.exp(-1000*((j-NX/3)/NX)**2)
             + np.exp(-10*np.exp(-1000*((j-NX/3)/NX)**2))) \
            * np.sin(5*j*math.pi/NX)
        Tex[j] = np.sin(4*math.pi*time) * v
        Text[j] = 4*math.pi*np.cos(4*math.pi*time) * v

    for j in range(1, NX-1):
        Texx[j] = (Tex[j+1]-Tex[j-1])/(2*dx)
        Txx = (Tex[j+1]-2*Tex[j]+Tex[j-1])/(dx**2)
        F[j] = V*Texx[j] - K*Txx + lamda*Tex[j] + Text[j]
    return F, Tex, Texx

# --------------------------------------------------------
# Paramètres physiques
# --------------------------------------------------------
K = 0.1     # Diffusion
L = 1.0     # Taille du domaine
V = 1.0     # Vitesse
lamda = 1.0 # Réaction
time_total = 1.0  # Temps final

# --------------------------------------------------------
# Paramètres numériques
# --------------------------------------------------------
NT = 10000
eps = 1e-3
niter_refinement = 6    # nombre de raffinements

# --------------------------------------------------------
# Étude 1 : convergence spatiale (erreur L2 à T/2 et T)
# --------------------------------------------------------
NX_tab = []
Err_tab_half = []
Err_tab_final = []

for iter in range(niter_refinement):
    NX = 10 + 5*iter
    dx = L/(NX-1)
    dt = dx**2/(V*dx+K+dx**2)

    x = np.linspace(0.0, 1.0, NX)
    T = np.zeros(NX)

    time = 0
    err_half = None
    while time < time_total:
        F, Tex, Texx = fex(NX, dx, time)
        dt = dx**2/(V*dx+2*K+abs(np.max(F))*dx**2)
        if time+dt > time_total:
            dt = time_total-time
        time += dt

        T0 = T.copy()
        # RK4
        alpha = [1/4, 1/3, 1/2, 1.0]
        for irk in range(4):
            for j in range(1, NX-1):
                xnu = K+0.5*dx*abs(V)
                Tx = (T[j+1]-T[j-1])/(2*dx)
                Txx = (T[j-1]-2*T[j]+T[j+1])/(dx**2)
                RHS = dt*(-V*Tx + xnu*Txx - lamda*T[j] + F[j])
                T[j] = T0[j] + RHS*alpha[irk]

        # Calcul des erreurs
        err = np.sqrt(np.dot(T-Tex, T-Tex)*dx)/NX
        if abs(time - time_total/2) < dt/2:
            err_half = err
        if abs(time - time_total) < dt/2:
            err_final = err

    NX_tab.append(NX)
    Err_tab_half.append(err_half)
    Err_tab_final.append(err_final)

plt.figure(figsize=(6,4))
plt.plot(NX_tab, Err_tab_half, '-o', label="Erreur L2 à T/2")
plt.plot(NX_tab, Err_tab_final, '-s', label="Erreur L2 à T")
plt.xlabel("Nombre de points NX")
plt.ylabel("Erreur L2")
plt.title("Erreur L2 vs raffinement spatial")
plt.legend()
plt.grid(True)

# --------------------------------------------------------
# Étude 2 : erreur au point milieu pour différents RK
# --------------------------------------------------------
RK_orders = [1,2,3,4]
errors_mid = {}

NX = 50   # maillage fixe pour comparaison
dx = L/(NX-1)
x = np.linspace(0.0, 1.0, NX)
mid = NX//2

for rk_order in RK_orders:
    T = np.zeros(NX)
    time = 0
    dt = dx**2/(V*dx+K+dx**2)
    err_mid = []
    time_tab = []

    while time < time_total:
        F, Tex, Texx = fex(NX, dx, time)
        dt = dx**2/(V*dx+2*K+abs(np.max(F))*dx**2)
        if time+dt > time_total:
            dt = time_total-time
        time += dt
        T0 = T.copy()

        alpha = [1/(rk_order-k) for k in range(rk_order)]
        for irk in range(rk_order):
            for j in range(1, NX-1):
                xnu = K+0.5*dx*abs(V)
                Tx = (T[j+1]-T[j-1])/(2*dx)
                Txx = (T[j-1]-2*T[j]+T[j+1])/(dx**2)
                RHS = dt*(-V*Tx + xnu*Txx - lamda*T[j] + F[j])
                T[j] = T0[j] + RHS*alpha[irk]

        err_mid.append(abs(T[mid]-Tex[mid]))
        time_tab.append(time)

    errors_mid[rk_order] = (time_tab, err_mid)

plt.figure(figsize=(6,4))
for rk_order in RK_orders:
    time_tab, err_mid = errors_mid[rk_order]
    plt.plot(time_tab, err_mid, label=f"RK{rk_order}")
plt.xlabel("Temps")
plt.ylabel("Erreur au milieu du domaine")
plt.title("Erreur au point milieu vs ordre de Runge-Kutta")
plt.legend()
plt.grid(True)

plt.show()
