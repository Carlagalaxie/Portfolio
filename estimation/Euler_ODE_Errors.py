# Euler_ODE_Errors.py

import numpy as np
import matplotlib.pyplot as plt

# Paramètres
u0 = 1.0
lambda_ = 1.0
T = 60.0  
Dt_values = np.logspace(0, -3, 20)  # 20 pas de temps de 1 à 0.001 s


def f(u):
    return -lambda_ * u

# Sol exacte
def u_exact(t):
    return u0 * np.exp(-lambda_ * t)

# Euler explicite
def euler_explicit(Dt):
    N = int(T / Dt) + 1
    t = np.linspace(0, T, N)
    u = np.zeros(N)
    u[0] = u0
    for n in range(N-1):
        u[n+1] = u[n] + Dt * f(u[n])
    return t, u

#Solution exacte, numérique et erreur temporelle pour Dt=1s
Dt = 1.0
t, u_num = euler_explicit(Dt)
u_ex = u_exact(t)
err_t = np.abs(u_num - u_ex)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, u_ex, 'k-', label='Exacte')
plt.plot(t, u_num, 'r--', label=f'Euler Dt={Dt}s')
plt.xlabel('Temps [s]')
plt.ylabel('u(t)')
plt.title('Solution exacte vs numérique')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, err_t, 'b-')
plt.xlabel('Temps [s]')
plt.ylabel('Erreur |u_num - u_exact|')
plt.title('Erreur temporelle')
plt.grid(True)

plt.tight_layout()
plt.show()

#Erreur L2 en fonction du pas de temps
errors_u = []
errors_du = []

for Dt in Dt_values:
    t, u_num = euler_explicit(Dt)
    u_ex = u_exact(t)
    # Erreur L2 de la fonction
    err_u = np.sqrt(np.sum((u_num - u_ex)**2) * Dt)
    # Erreur L2 de la dérivée
    du_num = np.diff(u_num) / Dt
    du_ex = np.diff(u_ex) / Dt
    err_du = np.sqrt(np.sum((du_num - du_ex)**2) * Dt)
    errors_u.append(err_u)
    errors_du.append(err_du)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.loglog(Dt_values, errors_u, 'o-', label='Erreur L2 de u')
plt.xlabel('Dt [s]')
plt.ylabel('Erreur L2')
plt.title('Erreur L2 de la fonction')
plt.grid(True, which='both')
plt.legend()

plt.subplot(1, 2, 2)
plt.loglog(Dt_values, errors_du, 's-', label='Erreur L2 de du/dt')
plt.xlabel('Dt [s]')
plt.ylabel('Erreur L2')
plt.title('Erreur L2 de la dérivée')
plt.grid(True, which='both')
plt.legend()

plt.tight_layout()
plt.show()

