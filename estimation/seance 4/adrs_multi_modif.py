import numpy as np
import matplotlib.pyplot as plt

# PHYSICAL PARAMETERS
K = 0.1
L = 1.0
V = 1.0
lamda = 1.0
Time = 1.0

# NUMERICAL PARAMETERS
NX = 50
dx = L/(NX-1)
dt = 0.5*dx**2/(K + V*dx/2)  # CFL
NT = int(Time/dt) + 1

# Spatial grid
x = np.linspace(0, L, NX)

# Exact solution in space
v = np.exp(-20*(x-0.5)**2)

# Initialize solution
T = np.zeros(NX)
Tex = np.zeros(NX)

# Storage for plotting
snapshots = []

# Time loop
for n in range(NT):
    t = n*dt
    # Exact solution at this time
    Tex = np.sin(4*np.pi*t) * v

    # Compute spatial derivatives
    Tx = np.zeros(NX)
    Txx = np.zeros(NX)
    Tx[1:-1] = (T[2:] - T[:-2])/(2*dx)
    Txx[1:-1] = (T[2:] - 2*T[1:-1] + T[:-2])/(dx**2)

    # Source term f(x,t)
    dUdt = 4*np.pi*np.cos(4*np.pi*t) * v
    v_x = np.zeros(NX)
    v_x[1:-1] = (v[2:] - v[:-2])/(2*dx)
    v_xx = np.zeros(NX)
    v_xx[1:-1] = (v[2:] - 2*v[1:-1] + v[:-2])/(dx**2)
    F = dUdt + V*np.sin(4*np.pi*t)*v_x - K*np.sin(4*np.pi*t)*v_xx + lamda*np.sin(4*np.pi*t)*v

    # Update solution (Euler explicite)
    xnu = K + 0.5*dx*abs(V)
    RHS = np.zeros(NX)
    RHS[1:-1] = dt*(-V*Tx[1:-1] + xnu*Txx[1:-1] - lamda*T[1:-1] + F[1:-1])
    T[1:-1] += RHS[1:-1]

    # Store snapshots at selected times
    if n % max(1, NT//5) == 0:
        snapshots.append((t, T.copy()))

# Plot snapshots
plt.figure(figsize=(8,5))
for t, Tsnap in snapshots:
    plt.plot(x, Tsnap, label=f't={t:.3f}s')
plt.plot(x, np.sin(4*np.pi*Time)*v, 'k--', label='Exact t=1s')
plt.xlabel('x')
plt.ylabel('T')
plt.title('Solution instationnaire ADRS 1D')
plt.legend()
plt.show()
