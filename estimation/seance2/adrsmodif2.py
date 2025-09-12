#!/usr/bin/env python3
# adrs_convergence.py
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
v = 1.0       # vitesse d’advection
lamda = 1.0   # coefficient de réaction

# --- PARAMETRES TEMPS ---
NT = 200000         # nb max de pas de temps (très grand pour laisser converger)
eps_time = 1e-6     # tolérance pour ||u^{n+1}-u^n||_L2 relative
# (on arrêtera aussi si on dépasse NT)

# --- FONCTIONS UTILITAIRES ---
def exact_solution(x):
    """Solution exacte et ses dérivées (u, u', u'')."""
    u = np.exp(-10.0 * (x - 0.5)**2)
    u1 = -20.0 * (x - 0.5) * u             # u'
    u2 = (-20.0 + 400.0 * (x - 0.5)**2) * u  # u''
    return u, u1, u2

def build_rhs_and_tex(NX):
    """Construit la grille, Tex (exact), et le terme source F discretisé."""
    x = np.linspace(0.0, L, NX)
    Tex, Tex1, Tex2 = exact_solution(x)
    # f(s) = v u' - nu u'' + lambda u (stationnaire)
    F = v * Tex1 - nu * Tex2 + lamda * Tex
    return x, Tex, Tex1, Tex2, F

def solve_to_stationary(NX, verbose=False):
    """
    Résout le problème en temps explicite jusqu'à la stationnaire.
    Retourne T_final, Tex, x, diffs_rel (liste des ||u^{n+1}-u^n||_L2 / init).
    """
    x, Tex, Tex1, Tex2, F = build_rhs_and_tex(NX)
    dx = L/(NX-1)

    # choix heuristique de dt (CFL-like)
    dt = dx**2 / (v*dx + 2*nu + abs(np.max(F))*dx**2 + 1e-16)

    # initialisation
    T = np.zeros(NX)
    # imposer Dirichlet = valeurs exactes aux bords pour retrouver Tex
    T[0] = Tex[0]
    T[-1] = Tex[-1]

    Told = T.copy()
    diffs_rel = []

    # norme initiale pour normalisation (sera fixée au premier pas)
    diff0 = None

    n = 0
    while n < NT:
        n += 1
        # calcul du RHS et mise à jour
        RHS = np.zeros(NX)
        for j in range(1, NX-1):
            xnu = nu + 0.5*dx*abs(v)
            Tx = (T[j+1] - T[j-1])/(2.0*dx)
            Txx = (T[j-1] - 2.0*T[j] + T[j+1])/(dx*dx)

            RHS[j] = dt * (-v * Tx + xnu * Txx - lamda * T[j] + F[j])

        # appliquer mise à jour 
        T[1:-1] += RHS[1:-1]

        # calcul norme L2 de la différence entre itérations
        diff = np.sqrt(dx * np.dot(T - Told, T - Told))
        if diff0 is None and diff > 0:
            diff0 = diff
        if diff0 is None:
            diff0 = 1e-16  # sécurité si tout reste nul
        diffs_rel.append(diff / diff0)

        # condition d'arrêt
        if (diff / diff0) < eps_time:
            if verbose:
                print(f"[NX={NX}] Converged in {n} steps, diff_rel={diff/diff0:.3e}")
            break

        Told[:] = T
        
    else:
        if verbose:
            print(f"[NX={NX}] Warning: reached NT={NT} without reaching eps_time")

    return T, Tex, x, dx, np.array(diffs_rel)

# -----------------------------
# 1) Vérifier convergence pour NX = 100
# -----------------------------
NX_check = 100
T100, Tex100, x100, dx100, diffs100 = solve_to_stationary(NX_check, verbose=True)

# Tracé de la norme normalisée 
plt.figure(figsize=(8,4))
plt.semilogy(diffs100, '-', lw=1.2)
plt.xlabel('itération temporelle')
plt.ylabel(r'$\|u^{n+1}-u^n\|_{L^2} / \|u^{1}-u^{0}\|_{L^2}$')
plt.title(f'Convergence temporelle vers la stationnaire (NX={NX_check})')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()

# Afficher solution finale vs exacte
plt.figure(figsize=(8,4))
plt.plot(x100, T100, 'b-', label='T_num final')
plt.plot(x100, Tex100, 'k--', label='Tex (exact)')
plt.xlabel('s')
plt.ylabel('u')
plt.title(f'Solution numérique vs exacte (NX={NX_check})')
plt.legend()
plt.grid(True, ls='--', alpha=0.4)
plt.tight_layout()

# -----------------------------
# 2) Calculer erreurs L2 et H1 pour 5 maillages (à partir de N=3)
# -----------------------------
# Choix des maillages : partant de 3 points, on prend 5 maillages de pas réguliers
# tu peux ajuster la progression si tu préfères une autre série

N_list = [3, 23, 43, 63, 83]  # 5 maillages (commence à 3)
hs = []
errors_L2 = []
errors_H1 = []

for NX in N_list:
    T, Tex, x, dx, diffs = solve_to_stationary(NX, verbose=True)

    # Erreur L2 
    err_L2 = np.sqrt(dx * np.dot(T - Tex, T - Tex))

    # Erreur H1 
    err_h1 = 0.0
    for j in range(1, NX-1):
        u1_num = (T[j+1] - T[j-1])/(2.0*dx)
        u1_ex  = -20.0*(x[j] - 0.5)*Tex[j]
        err_h1 += (u1_ex - u1_num)**2
    err_H1 = np.sqrt(dx * err_h1)

    hs.append(dx)
    errors_L2.append(err_L2)
    errors_H1.append(err_H1)

    print(f"NX={NX:3d}  h={dx:.4e}  Err_L2={err_L2:.4e}  Err_H1={err_H1:.4e}")

hs = np.array(hs)
errors_L2 = np.array(errors_L2)
errors_H1 = np.array(errors_H1)

# -----------------------------
# 3) Tracer Erreur L2 et H1 en fonction de h (log-log), images côte à côte
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12,5))

axes[0].loglog(hs, errors_L2, 'o-', label='Erreur L2')
axes[0].set_xlabel('h')
axes[0].set_ylabel('Erreur L2')
axes[0].set_title('Convergence en norme L2')
axes[0].grid(True, which='both', ls='--', alpha=0.4)
axes[0].invert_xaxis()  
for i,h in enumerate(hs):
    axes[0].text(h, errors_L2[i], f" N={N_list[i]}", fontsize=8)

axes[1].loglog(hs, errors_H1, 's-', label='Erreur H1')
axes[1].set_xlabel('h')
axes[1].set_ylabel('Erreur H1')
axes[1].set_title('Convergence en norme H1')
axes[1].grid(True, which='both', ls='--', alpha=0.4)
for i,h in enumerate(hs):
    axes[1].text(h, errors_H1[i], f" N={N_list[i]}", fontsize=8)

plt.suptitle('Erreurs numériques en fonction du pas h')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# -----------------------------
# 4) Afficher 
# -----------------------------
plt.show()

