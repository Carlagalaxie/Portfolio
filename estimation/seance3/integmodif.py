import numpy as np
import matplotlib.pyplot as plt

# Intervalle pour theta
theta_min, theta_max = -np.pi/2, np.pi/2

exact = np.pi

# Nombres de points
N_points = np.logspace(1, 5, 20, dtype=int)

# Tableaux pour stocker les erreurs
errors_riemann = []
errors_montecarlo = []
errors_lebesgue = []

# Valeurs de l'intégrale pour la dernière itération
integral_riemann_val = 0
integral_montecarlo_val = 0
integral_lebesgue_val = 0

for N in N_points:
    # Riemann (méthode du milieu)
    theta_riemann = np.linspace(theta_min, theta_max, N)
    dtheta = theta_riemann[1] - theta_riemann[0]
    integral_riemann = np.sum(np.ones_like(theta_riemann)) * dtheta  # f(theta) = 1
    errors_riemann.append(abs(integral_riemann - exact))
    integral_riemann_val = integral_riemann
    
    # Monte Carlo
    theta_mc = np.random.uniform(theta_min, theta_max, N)
    integral_mc = (theta_max - theta_min) * np.mean(np.ones_like(theta_mc))  # f(theta)=1
    errors_montecarlo.append(abs(integral_mc - exact))
    integral_montecarlo_val = integral_mc
    
    # Lebesgue (approx Riemann)
    theta_lebesgue = np.linspace(theta_min, theta_max, N)
    dtheta = theta_lebesgue[1] - theta_lebesgue[0]
    integral_lebesgue = np.sum(np.ones_like(theta_lebesgue)) * dtheta
    errors_lebesgue.append(abs(integral_lebesgue - exact))
    integral_lebesgue_val = integral_lebesgue

# Affichage des valeurs approchées
print(f"Intégrale approchée par Riemann : {integral_riemann_val}")
print(f"Intégrale approchée par Monte Carlo : {integral_montecarlo_val}")
print(f"Intégrale approchée par Lebesgue (approx) : {integral_lebesgue_val}")
print(f"Valeur exacte : {exact}")

# Tracé des erreurs
plt.figure(figsize=(8,6))
plt.loglog(N_points, errors_riemann, 'o-', label='Riemann')
plt.loglog(N_points, errors_montecarlo, 's-', label='Monte Carlo')
plt.loglog(N_points, errors_lebesgue, '^-', label='Lebesgue (approx)')
plt.xlabel("Nombre d'évaluations N")
plt.ylabel("Erreur absolue")
plt.title("Erreur de l'intégration après substitution trigonométrique")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
