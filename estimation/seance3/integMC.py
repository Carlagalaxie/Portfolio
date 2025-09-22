import numpy as np
import matplotlib.pyplot as plt

# Fonction
f = lambda x: 1/np.sqrt(1 - x**2)
exact = np.pi

# Nombres de points
N_points = np.logspace(1, 5, 20, dtype=int)

# Stockage des erreurs
errors_riemann = []
errors_montecarlo = []
errors_lebesgue = []

eps = 1e-10  # éviter les extrémités

# Valeur de l'intégrale 
integral_riemann_val = 0
integral_montecarlo_val = 0
integral_lebesgue_val = 0

for N in N_points:
    # Riemann 
    x_riemann = np.linspace(-1+eps, 1-eps, N)
    dx = x_riemann[1] - x_riemann[0]
    integral_riemann = np.sum(f(x_riemann)) * dx
    errors_riemann.append(abs(integral_riemann - exact))
    integral_riemann_val = integral_riemann  # dernière valeur
    
    # Monte Carlo
    x_mc = np.random.uniform(-1+eps, 1-eps, N)
    integral_mc = 2 * np.mean(f(x_mc))
    errors_montecarlo.append(abs(integral_mc - exact))
    integral_montecarlo_val = integral_mc  # dernière valeur
    
    # Lebesgue 
    x_lebesgue = np.linspace(-1+eps, 1-eps, N)
    dx = x_lebesgue[1] - x_lebesgue[0]
    fx = f(x_lebesgue)
    integral_lebesgue = np.sum(fx) * dx
    errors_lebesgue.append(abs(integral_lebesgue - exact))
    integral_lebesgue_val = integral_lebesgue  # dernière valeur

# Affichage des valeurs
print(f"Intégrale approchée par Riemann : {integral_riemann_val}")
print(f"Intégrale approchée par Monte Carlo : {integral_montecarlo_val}")
print(f"Intégrale approchée par Lebesgue (approx) : {integral_lebesgue_val}")
print(f"Valeur exacte : {exact}")

# Erreurs
plt.figure(figsize=(8,6))
plt.loglog(N_points, errors_riemann, 'o-', label='Riemann')
plt.loglog(N_points, errors_montecarlo, 's-', label='Monte Carlo')
plt.loglog(N_points, errors_lebesgue, '^-', label='Lebesgue (approx)')
plt.xlabel("Nombre d'évaluations N")
plt.ylabel("Erreur absolue")
plt.title("Erreur de l'intégration de 1/sqrt(1-x^2)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
