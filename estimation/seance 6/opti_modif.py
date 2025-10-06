# -*- coding: utf-8 -*-
"""
ADRS - version avec 'adaptation' (résolutions variables),
interpolation sur maillage de référence et calcul matriciel A,B.

Instructions :
 - Exécuter ce script ; il trace également la surface J(x1,x2) pour les deux
   premiers contrôles et compare le contrôle obtenu par résolution linéaire
   (avec maillages adaptifs) et la référence sur maillage fixe fin.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# ---------------------------
# Solveur ADRS (uniform mesh)
# ---------------------------
def ADRS_uniform(NX, xcontrol, Target, L=1.0, K=0.1, V=1.0, lamda=1.0,
                 NT=1000, eps=1e-4):
    """
    Résout l'équation ADRS sur un maillage uniforme de NX points.
    Renvoie : x (grille), T (solution au temps final), cost
    """
    dx = L / (NX - 1)
    # CFL-ish timestep stable for explicit scheme
    dt = 0.5 * dx**2 / (V * dx + 2 * K + max(1e-12, np.max(np.abs(xcontrol))) * dx**2)

    x = np.linspace(0.0, L, NX)
    T = np.zeros(NX)
    F = np.zeros(NX)

    # Construire le second membre F selon xcontrol (sources gaussiennes)
    for j in range(1, NX-1):
        for ic in range(len(xcontrol)):
            center = L / (ic + 1)
            F[j] += xcontrol[ic] * np.exp(-100 * (x[j] - center)**2)

    n = 0
    res = 1.0
    res0 = 1.0
    rest = []

    while (n < NT and res > eps * res0):
        n += 1
        res = 0.0
        RHS = np.zeros(NX)
        for j in range(1, NX-1):
            xnu = K + 0.5 * dx * abs(V)
            Tx = (T[j+1] - T[j-1]) / (2.0 * dx)
            Txx = (T[j-1] - 2.0 * T[j] + T[j+1]) / (dx**2)
            RHS[j] = dt * (-V * Tx + xnu * Txx - lamda * T[j] + F[j])
            res += abs(RHS[j])

        T[1:NX-1] += RHS[1:NX-1]

        if n == 1:
            res0 = res
        rest.append(res)

    cost = np.dot(T - Target, T - Target) * dx
    return x, T, cost

# ----------------------------------------------
# Gestion d'une 'adaptation' simple de la résolution
# ----------------------------------------------
def choose_NX_adaptive(xcontrol, NX_min=28, NX_max=200, scale_max=5.0):
    """
    Choisit une NX en fonction de la complexité/amplitude de xcontrol.
    scale_max est l'amplitude de référence au-delà de laquelle on prendra NX_max.
    C'est un critère simple : plus la norme infinie de xcontrol est grande, plus NX est élevé.
    """
    amp = np.max(np.abs(xcontrol))
    frac = min(1.0, amp / scale_max)
    NX = int(NX_min + frac * (NX_max - NX_min))
    # s'assurer NX impair raisonnable:
    NX = max(NX_min, min(NX_max, NX))
    return NX

# ----------------------------------------------
# Calcul A, B via interpolation sur maillage de référence
# ----------------------------------------------
def build_A_B_adaptive(nbc, xcontrol_zero, Target_func_for_ref, NX_ref=400, L=1.0):
    """
    Construit A (nbc x nbc) et B (nbc) en calculant les réponses pour
    vecteurs de base de contrôle (1 sur un contrôle, 0 ailleurs),
    en calculant la solution T sur un maillage choisi automatiquement (adaptatif),
    puis en interpolant toutes les solutions sur x_ref commun et en intégrant.
    Returns: A, B, T0_ref (solution pour xcontrol=0 interpolée sur x_ref),
             x_ref (maillage de référence)
    """
    x_ref = np.linspace(0.0, L, NX_ref)
    # Construire Target sur x_ref (utilise Target_func_for_ref qui renvoie Target sur un NX donné)
    # Ici Target_func_for_ref retourne Target sur la grille demandée (uniform)
    Target_ref = Target_func_for_ref(NX_ref)

    # 1) Calculer T0 (xcontrol = 0) sur maillage adaptatif et interpoler
    x0 = xcontrol_zero.copy()
    NX0 = choose_NX_adaptive(x0)
    x_T0, T0_native, cost_dummy = ADRS_uniform(NX0, x0, Target_func_for_ref(NX0))
    T0_ref = np.interp(x_ref, x_T0, T0_native)

    # 2) Pour chaque ic, calculer la réponse Tic à xic (base)
    T_basis_ref = np.zeros((nbc, NX_ref))
    for ic in range(nbc):
        xic = np.zeros(nbc)
        xic[ic] = 1.0
        NX_i = choose_NX_adaptive(xic)
        x_native, T_native, _ = ADRS_uniform(NX_i, xic, Target_func_for_ref(NX_i))
        # interpolation sur maillage de référence
        T_basis_ref[ic, :] = np.interp(x_ref, x_native, T_native)

    # 3) Calcul numérique de A_ij et B_i par trapèzes sur x_ref
    A = np.zeros((nbc, nbc))
    B = np.zeros(nbc)

    for i in range(nbc):
        for j in range(i, nbc):
            A[i, j] = np.trapz(T_basis_ref[i, :] * T_basis_ref[j, :], x_ref)
            A[j, i] = A[i, j]
    for i in range(nbc):
        B[i] = np.trapz((Target_ref - T0_ref) * T_basis_ref[i, :], x_ref)

    return A, B, x_ref, Target_ref, T0_ref, T_basis_ref

# ----------------------------------------------
# Fonction utilitaire pour construire Target (on garde la même définition que toi)
# ----------------------------------------------
def make_target_by_sources(NX, nbc, L=1.0):
    """
    Renvoie un vecteur Target de taille NX construit comme dans ton code,
    c'est-à-dire la solution obtenue en utilisant des 'xcible' = [1..nbc].
    """
    # on simule la génération du Target comme dans ton code (solution for xcible = [1,2,...])
    xcible = np.arange(nbc) + 1
    x, T, cost = ADRS_uniform(NX, xcible, np.zeros(NX))
    return T

# ----------------------------------------------
# Routine principale : construction, résolution, tracés
# ----------------------------------------------
def main():
    nbc = 6
    # paramètres de maillage de référence (pour intégration précise)
    NX_ref = 800  # maillage de référence pour intégration (augmente si nécessaire)
    L = 1.0

    # Construire Target sur NX_ref (référence pour intégration)
    Target_ref = make_target_by_sources(NX_ref, nbc, L=L)

    # On veut résoudre linéairement A x = B
    # Pour la construction des A,B, on fournit une fonction qui construit Target sur n'importe quel NX
    Target_func = lambda NX: make_target_by_sources(NX, nbc, L=L)

    # Calcul A,B avec maillages adaptatifs (choisis automatiquement par choose_NX_adaptive)
    print("Construction de A,B (maillages adaptatifs) ...")
    t0 = time.time()
    A, B, x_ref, Target_ref2, T0_ref, T_basis_ref = build_A_B_adaptive(nbc, np.zeros(nbc), Target_func, NX_ref=NX_ref, L=L)
    print("done in {:.2f} s".format(time.time() - t0))

    # Résolution linéaire
    xopt_adapt = np.linalg.solve(A, B)
    print("xopt (adaptatif, linéaire) =", xopt_adapt)

    # Calculer coût pour xopt en interpolant la solution correspondante sur x_ref
    NX_xopt = choose_NX_adaptive(xopt_adapt)
    x_native, T_native, cost_native = ADRS_uniform(NX_xopt, xopt_adapt, Target_func(NX_xopt))
    T_xopt_ref = np.interp(x_ref, x_native, T_native)
    cost_xopt_ref = np.trapz((T_xopt_ref - Target_ref)**2, x_ref)
    print("cost_xopt (sur maillage ref) =", cost_xopt_ref)

    # Comparaison avec 'référence' : résolution sur maillage fixe très fin
    NX_ref_solution = 1200  # maillage fixe très fin pour référence
    print("Calcul de la solution de référence (maillage fixe très fin, NX = {}) ...".format(NX_ref_solution))
    xr_ref, T_ref_solution, cost_ref_dummy = ADRS_uniform(NX_ref_solution, np.arange(nbc)+1, make_target_by_sources(NX_ref_solution, nbc))
    # ici la 'référence' est la solution issue des xcible = [1..nbc] ; on peut aussi calculer la solution
    # correspondant à xopt mais sur NX_ref_solution (plus juste pour comparaison)
    _, T_xopt_on_ref, _ = ADRS_uniform(NX_ref_solution, xopt_adapt, make_target_by_sources(NX_ref_solution, nbc))
    cost_xopt_on_ref = np.trapz((T_xopt_on_ref - make_target_by_sources(NX_ref_solution, nbc))**2, xr_ref)
    print("cost_xopt (évalué sur maillage de référence fixe) =", cost_xopt_on_ref)

    # ---------------------------
    # Tracé J(x1,x2) en échantillonnant x1,x2 et gardant les autres contrôles = 0
    # ---------------------------
    print("Échantillonnage de J(x1,x2) ...")
    ngrid = 40
    x1_vals = np.linspace(0.0, 2.0, ngrid)
    x2_vals = np.linspace(0.0, 2.0, ngrid)
    Jsurf = np.zeros((ngrid, ngrid))

    # Les autres contrôles fixés à zéro
    x_fixed = np.zeros(nbc)
    for i1, v1 in enumerate(x1_vals):
        for i2, v2 in enumerate(x2_vals):
            x_try = x_fixed.copy()
            x_try[0] = v1
            x_try[1] = v2
            NX_try = choose_NX_adaptive(x_try)
            x_native, T_native, _ = ADRS_uniform(NX_try, x_try, make_target_by_sources(NX_try, nbc))
            T_try_ref = np.interp(x_ref, x_native, T_native)
            Jsurf[i2, i1] = np.trapz((T_try_ref - Target_ref)**2, x_ref)  # axis ordering for plotting

    # Plot 2D contour et surface 3D
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    plt.figure(figsize=(8,6))
    cp = plt.contourf(X1, X2, Jsurf, levels=30)
    plt.colorbar(cp)
    plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Surface J(x1,x2) (autres controles = 0)')
    plt.scatter([xopt_adapt[0]], [xopt_adapt[1]], color='white', marker='x', label='xopt adapt')
    plt.legend()
    plt.show()

    # ---------------------------
    # Affichage comparaison solution target / solution xopt (sur le maillage de référence x_ref)
    # ---------------------------
    plt.figure(figsize=(8,5))
    plt.plot(x_ref, Target_ref, label='Target (maillage ref)')
    plt.plot(x_ref, T_xopt_ref, label='T(xopt) (interpolée sur ref)')
    plt.plot(xr_ref, T_xopt_on_ref, '--', label='T(xopt) sur maillage fixe très fin')
    plt.legend()
    plt.title('Comparaison Target / solution contrôlée')
    plt.xlabel('x'); plt.ylabel('T(x)')
    plt.show()

    # Impression finale
    print("Résumé:")
    print(" - xopt adapté (linéaire) :", xopt_adapt)
    print(" - coût xopt (évalué sur maillage ref) :", cost_xopt_ref)
    print(" - coût xopt (évalué sur maillage fixe très fin) :", cost_xopt_on_ref)

if __name__ == "__main__":
    main()
