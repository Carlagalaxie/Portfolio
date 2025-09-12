#!/usr/bin/env python3
# identification_parameters.py
import math
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Paramètres physiques et numeriques 
# ----------------------------
K = 0.1
L = 1.0
V = 1.0
lamda = 1.0

NT = 200000
eps_time = 1e-8

# solution exacte  et ses dérivées analytiques
def exact_solution(x):
    u = np.exp(-20.0 * (x - 0.5)**2)            # u
    u1 = -40.0 * (x - 0.5) * u                  # u'
    u2 = (-40.0 + 1600.0*(x-0.5)**2) * u        # u''
    return u, u1, u2

# Solveur vers la stationnaire 
def solve_to_stationary(NX, dt_factor=1.0, verbose=False):
    x = np.linspace(0.0, L, NX)
    dx = L/(NX-1)

    Tex, Tex1, Tex2 = exact_solution(x)
    F = V*Tex1 - K*Tex2 + lamda*Tex

    # CFL
    dt = dx**2 / (V*dx + 4*K + dx**2 + 1e-16)
    dt *= dt_factor

    T = np.zeros(NX)

    # Dirichlet
    T[0] = Tex[0]
    T[-1] = Tex[-1]

    Told = T.copy()
    diff0 = None
    n = 0
    while n < NT:
        n += 1
        RHS = np.zeros(NX)
        for j in range(1, NX-1):
            xnu = K + 0.5*dx*abs(V)
            Tx = (T[j+1] - T[j-1])/(2.0*dx)
            Txx = (T[j-1] - 2.0*T[j] + T[j+1])/(dx*dx)
            RHS[j] = dt * (-V*Tx + xnu*Txx - lamda*T[j] + F[j])


        # mise à jour 
        T[1:-1] += RHS[1:-1]

        diff = math.sqrt(dx * np.dot(T - Told, T - Told))
        if diff0 is None and diff > 0:
            diff0 = diff
        if diff0 is None:
            diff0 = 1e-16
        if (diff / diff0) < eps_time:
            break
        Told[:] = T
    if verbose:
        print(f"NX={NX} converged in {n} steps, diff_rel={diff/diff0:.3e}, dx={dx:.3e}")
    return x, dx, T, Tex, Tex1, Tex2

# ----------------------------
# Choix des maillages (5 maillages) - partir de N=3
# ----------------------------
N_list = [3, 23, 43, 63, 83]   
hs = []
E0s = []     
E1s = []     
S2s = []     

for NX in N_list:
    x, dx, Tnum, Tex, Tex1, Tex2 = solve_to_stationary(NX, verbose=True)
    hs.append(dx)

    # E0 = L2 
    diff_vec = Tnum - Tex
    E0_sq = dx * np.sum(diff_vec[1:-1]**2) + dx*( (diff_vec[0]**2 + diff_vec[-1]**2)/2.0 )
    E0 = math.sqrt(E0_sq)

    # E1 
    
    u1_num = np.zeros_like(Tnum)
    u1_num[1:-1] = (Tnum[2:] - Tnum[:-2])/(2.0*dx)
    
    E1_sq = dx * np.sum((u1_num[1:-1] - Tex1[1:-1])**2)
    E1 = math.sqrt(E1_sq)

    # S2 
    S2_sq = dx * np.sum(Tex2[1:-1]**2)
    S2 = math.sqrt(S2_sq)

    E0s.append(E0)
    E1s.append(E1)
    S2s.append(S2)

    print(f"NX={NX:3d} dx={dx:.3e}  E0={E0:.4e}  E1={E1:.4e}  S2={S2:.4e}")

hs = np.array(hs)
E0s = np.array(E0s)
E1s = np.array(E1s)
S2s = np.array(S2s)

# normaliser
E0_norm = E0s / S2s

# ----------------------------
# Identification (C, k) par une régression :
# ----------------------------
mask = (E0_norm > 0) & (hs > 0)
logh = np.log(hs[mask])
logE = np.log(E0_norm[mask])


p = np.polyfit(logh, logE, 1)
k_est = p[0]
logC_est = p[1]
C_est = math.exp(logC_est)


M_emp = np.max(E0_norm / (hs**k_est))

print("\nIdentification results:")
print(f"Estimated k = {k_est:.4f}")
print(f"Estimated C = {C_est:.4e}")
print(f"Empirical M = {M_emp:.4e}")

# ----------------------------
# Superposition des courbes
# ----------------------------
h_plot = np.linspace(np.min(hs)*0.8, np.max(hs)*1.2, 50)
curve_k = C_est * h_plot**(k_est)
curve_kp1 = C_est * h_plot**(k_est + 1.0)

plt.figure(figsize=(7,5))
plt.loglog(hs, E0_norm, 'o', label=r'$\|u-u_h\|_{0,2} / \|u\|_{2,2}$ (data)')
plt.loglog(h_plot, curve_k, '-', label=f'C*h^k (k={k_est:.3f})')
plt.loglog(h_plot, curve_kp1, '--', label=f'C*h^{{k+1}}')
plt.xlabel('h')
plt.ylabel('Normalized error')
plt.title('Identification (C,k) and comparison')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.gca().invert_xaxis()
plt.tight_layout()

# ----------------------------
# Afficher l'erreur L2 et H1 
# ----------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.loglog(hs, E0s, 'o-', label='E0 (L2)')
plt.xlabel('h'); plt.ylabel('E0'); plt.title('E0 vs h'); plt.grid(True)
plt.subplot(1,2,2)
plt.loglog(hs, E1s, 's-', label='E1 (H1 semi)')
plt.xlabel('h'); plt.ylabel('E1'); plt.title('E1 vs h'); plt.grid(True)
plt.tight_layout()
plt.show()
