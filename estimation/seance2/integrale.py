import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, pi

# fonction à intégrer
def f(x):
    return np.exp(-50*x**2)

# borne d'intégration
a, b = -1, 1
# valeur exacte (via erf)
I_exact = sqrt(pi/50) * erf(sqrt(50))
print("Valeur exacte :", I_exact)

# ==========================
# 1) Intégration de Riemann
# ==========================
N = 10000
x = np.linspace(a, b, N)
dx = (b - a) / (N-1)
I_riemann = np.sum(f(x)) * dx
print("Intégrale Riemann =", I_riemann)

# ==========================
# 2) Intégration "Lebesgue"
# ==========================
M = 1000
y_vals = np.linspace(0, 1, M)
dy = y_vals[1] - y_vals[0]
lengths = []
for y in y_vals:
    mask = f(x) > y
    if np.any(mask):
        indices = np.where(mask)[0]
        x_len = (indices[-1] - indices[0]) * dx
    else:
        x_len = 0
    lengths.append(x_len)

I_lebesgue = np.sum(lengths) * dy
print("Intégrale Lebesgue  =", I_lebesgue)

# ==========================
# 3) Intégration Monte Carlo
# ==========================
N_mc = 100000
samples = np.random.uniform(a, b, N_mc)
I_mc = (b - a) * np.mean(f(samples))
print("Intégrale Monte Carlo =", I_mc)

