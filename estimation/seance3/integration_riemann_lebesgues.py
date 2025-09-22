import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, pi


def f(x, a=0.5, b=10, c=3):
    return a*x**2 + b*x + c*np.sin(4*np.pi*x) + 10*np.exp(-100*(x-0.5)**2)

a, b = 0, 1


# 1) Intégration de Riemann

N = 10000
x = np.linspace(a, b, N)
dx = (b - a) / (N-1)
I_riemann = np.sum(f(x)) * dx
print("Intégrale Riemann =", I_riemann)


# 2) Intégration "Lebesgue" 

M = 1000
y_vals = np.linspace(f(x).min(), f(x).max(), M)
dy = y_vals[1] - y_vals[0]
lengths = []

fx = f(x)

for y in y_vals:
    mask = fx > y
    dx_array = np.diff(x)
    # Somme des dx correspondant aux intervalles où f(x) > y
    x_len = dx_array[mask[:-1]].sum() if np.any(mask) else 0
    lengths.append(x_len)

I_lebesgue = np.sum(lengths) * dy
print("Intégrale Lebesgue  =", I_lebesgue)



# 3) Intégration façon Lebesgue (pas uniforme en y = f(x))

N_y = 1000
y_vals = np.linspace(fx.min(), fx.max(), N_y)
dy = y_vals[1] - y_vals[0]

dx_array = np.diff(x)
I_lebesgue = 0
for k in range(N_y-1):
    y_low, y_high = y_vals[k], y_vals[k+1]
    # masque des intervalles en x correspondant à ce niveau de y
    mask = (fx[:-1] >= y_low) & (fx[:-1] < y_high)
    # somme des dx correspondants
    measure_x = dx_array[mask].sum()
    # contribution à l'intégrale (trapèze sur y)
    I_lebesgue += (y_low + y_high)/2 * measure_x

print("Intégrale Lebesgue (pas uniforme en y) =", I_lebesgue)


# 3) Visualisation

plt.plot(x, fx, label="f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Fonction à intégrer")
plt.legend()
plt.show()