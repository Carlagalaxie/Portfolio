import numpy as np
import matplotlib.pyplot as plt

#============================
# PHYSICAL PARAMETERS
#============================
K = 0.1     # Diffusion coefficient
L = 1.0     # Domain size
V = 1.0
lamda = 1.0

#============================
# NUMERICAL PARAMETERS
#============================
NX_values = [11, 21, 41, 81, 161, 321]  # Maillages réguliers
eps = 1e-6  # Critère stationnaire

# Pour stocker les résultats
errL2_values = []
errH1_values = []
semiH2_values = []
h_values = []

#============================
# SOLUTION FINE POUR INTERPOLATION
#============================
NX_fine = 2001
x_fine = np.linspace(0, L, NX_fine)
Tex_fine = np.exp(-20*(x_fine - 0.5)**2)

#============================
# BOUCLE SUR LES MAILLAGES
#============================
for NX in NX_values:
    dx = L/(NX-1)
    dt = 0.5 * dx**2 / (K + 0.5*V*dx)  # CFL
    x = np.linspace(0, L, NX)

    # Initialisation
    T = np.zeros(NX)
    Tex = np.zeros(NX)
    F = np.zeros(NX)

    # Solution exacte et source
    for j in range(1, NX-1):
        Tex[j] = np.exp(-20*(x[j]-0.5)**2)
    for j in range(1, NX-1):
        Tx = (Tex[j+1]-Tex[j-1])/(2*dx)
        Txx = (Tex[j+1]-2*Tex[j]+Tex[j-1])/(dx**2)
        F[j] = V*Tx - K*Txx + lamda*Tex[j]

    #============================
    # ITERATION EN TEMPS JUSQU'A STATIONNAIRE
    #============================
    n = 0
    res0 = 1
    res = 1
    while res/res0 > eps and n < 100000:
        n += 1
        res = 0
        RHS = np.zeros(NX)
        for j in range(1, NX-1):
            xnu = K + 0.5*dx*abs(V)
            Tx = (T[j+1]-T[j-1])/(2*dx)
            Txx = (T[j-1]-2*T[j]+T[j+1])/(dx**2)
            RHS[j] = dt*(-V*Tx + xnu*Txx - lamda*T[j] + F[j])
            res += abs(RHS[j])
        T[1:-1] += RHS[1:-1]
        if n == 1:
            res0 = res

    #============================
    # CALCUL DES ERREURS
    #============================
    errL2 = np.sqrt(np.sum((T - Tex)**2) * dx)
    # H1
    Tx_num = (T[2:] - T[:-2]) / (2*dx)
    Tx_ex = (Tex[2:] - Tex[:-2]) / (2*dx)
    errH1 = np.sqrt(errL2**2 + np.sum((Tx_num - Tx_ex)**2) * dx)
    # semi-H2
    Txx_ex = (Tex[2:] - 2*Tex[1:-1] + Tex[:-2]) / (dx**2)
    semiH2 = np.sqrt(np.sum(Txx_ex**2) * dx)

    errL2_values.append(errL2)
    errH1_values.append(errH1)
    semiH2_values.append(semiH2)
    h_values.append(dx)

    print(f"NX={NX}, h={dx:.5f}, niter={n}, L2={errL2:.3e}, H1={errH1:.3e}, semiH2={semiH2:.3e}")

#============================
# IDENTIFICATION C et k
#============================
h_values = np.array(h_values)
log_h = np.log(h_values)

errL2_values = np.array(errL2_values)
errH1_values = np.array(errH1_values)

# L2
coeff_L2 = np.polyfit(log_h, np.log(errL2_values), 1)
k_plus_1 = coeff_L2[0]
C = np.exp(coeff_L2[1])
print(f"L2: ordre observé k+1 = {k_plus_1:.3f}, C = {C:.3e}")

# H1
coeff_H1 = np.polyfit(log_h, np.log(errH1_values), 1)
k_H1 = coeff_H1[0]
C_H1 = np.exp(coeff_H1[1])
print(f"H1: ordre observé k = {k_H1:.3f}, C = {C_H1:.3e}")

#============================
# CONSTANTE M POUR INTERPOLATION P1
#============================
M_values = []
for NX in NX_values:
    dx = L/(NX-1)
    x = np.linspace(0, L, NX)
    # Interpolation linéaire de la solution exacte
    Ph = np.interp(x, x_fine, Tex_fine)
    # Simulation correspondante (déjà calculée comme T)
    # Pour retrouver T, refaire simulation rapide
    T = np.zeros(NX)
    Tex = np.zeros(NX)
    F = np.zeros(NX)
    for j in range(1, NX-1):
        Tex[j] = np.exp(-20*(x[j]-0.5)**2)
    for j in range(1, NX-1):
        Tx = (Tex[j+1]-Tex[j-1])/(2*dx)
        Txx = (Tex[j+1]-2*Tex[j]+Tex[j-1])/(dx**2)
        F[j] = V*Tx - K*Txx + lamda*Tex[j]
    n = 0
    res0 = 1
    res = 1
    while res/res0 > eps and n < 100000:
        n += 1
        res = 0
        RHS = np.zeros(NX)
        for j in range(1, NX-1):
            xnu = K + 0.5*dx*abs(V)
            Tx = (T[j+1]-T[j-1])/(2*dx)
            Txx = (T[j-1]-2*T[j]+T[j+1])/(dx**2)
            RHS[j] = dt*(-V*Tx + xnu*Txx - lamda*T[j] + F[j])
            res += abs(RHS[j])
        T[1:-1] += RHS[1:-1]
        if n == 1:
            res0 = res
    # Constante M
    err_num = np.sqrt(np.sum((T - Tex)**2) * dx)
    err_interp = np.sqrt(np.sum((Tex - Ph)**2) * dx)
    M = err_num / err_interp
    M_values.append(M)

#============================
# PLOTS
#============================
plt.figure(figsize=(6,5))
plt.loglog(h_values, errL2_values, 'o-', label='L2')
plt.loglog(h_values, errH1_values, 's-', label='H1')
plt.xlabel('h')
plt.ylabel('Erreur')
plt.title('Erreur stationnaire vs h')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.show()

plt.figure(figsize=(6,5))
plt.plot(NX_values, M_values, 'o-')
plt.xlabel('NX')
plt.ylabel('Constante M')
plt.title('Constante M pour interpolation P1')
plt.grid(True)
plt.show()
