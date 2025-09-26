import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------
# Paramètres
# ------------------------------
m = 40
p = 40
x_vals = np.linspace(0, 1, m)
y_vals = np.linspace(0, 1, p)

Lx = [0, 1]
Ly = [0, 1]
dx = 1 / 40
dy = 1 / 40

CFL = 0.45
dt = (dx**2) * CFL / 2
T = 5
Trad = 40
T_ext = 5

Nx = int((Lx[1] - Lx[0]) / dx)
Ny = int((Ly[1] - Ly[0]) / dy)
Nt = int(T // dt)
Lambda = dt / dx**2

Fy = [int(0.4 * Ny), int(0.6 * Ny)]  # fenêtre en y
X = np.linspace(Lx[0], Lx[1], Nx + 1)
Y = np.linspace(Ly[0], Ly[1], Ny + 1)

c = 2  # position du radiateur

# Position du radiateur
if c == 1:
    Wx = [int(0.9 * Nx), int(1 * Nx)]
    Wy = [int(0.4 * Ny), int(0.6 * Ny)]
elif c == 2:
    Wx = [int(0.45 * Nx), int(0.55 * Nx)]
    Wy = [int(0.4 * Ny), int(0.6 * Ny)]
elif c == 3:
    Wx = [int(0 * Nx), int(0.1 * Nx)]
    Wy = [int(0.4 * Ny), int(0.6 * Ny)]

# ------------------------------
# Initialisation
# ------------------------------
def f0(x, y):
    if x == 0 and abs(y - 0.5) <= 0.1:
        return T_ext
    else:
        return 20

U = np.zeros((Nx + 1, Ny + 1, Nt + 1), float)
A = np.zeros((Nx + 1, Ny + 1), float)
P = np.zeros((Nx + 1, Ny + 1), float)
FuturA = np.zeros((Nx + 1, Ny + 1), float)

for i in range(Nx + 1):
    for j in range(Ny + 1):
        A[i, j] = f0(X[i], Y[j])
        if c != 0 and Wx[0] <= i < Wx[1] and Wy[0] <= j < Wy[1]:
            P[i, j] = (Trad - A[i, j])**3

U[:, :, 0] = A.copy()

# ------------------------------
# Boucle temporelle
# ------------------------------
N_points = (Nx+1)*(Ny+1)

for t in range(1, Nt + 1):
    A = U[:, :, t - 1].copy()
    P.fill(0)

    if c != 0:
        P[Wx[0]:Wx[1], Wy[0]:Wy[1]] = (Trad - A[Wx[0]:Wx[1], Wy[0]:Wy[1]])**3

    FuturA[1:Nx, 1:Ny] = (
        (1 - 4 * Lambda) * A[1:Nx, 1:Ny]
        + Lambda * (A[0:Nx-1, 1:Ny] + A[2:Nx+1, 1:Ny] + A[1:Nx, 0:Ny-1] + A[1:Nx, 2:Ny+1])
        + dt * P[1:Nx, 1:Ny]
    )

    # Conditions aux limites
    FuturA[Nx, :] = (4 / 3) * FuturA[Nx - 1, :] - (1 / 3) * FuturA[Nx - 2, :]
    FuturA[:, 0] = (4 / 3) * FuturA[:, 1] - (1 / 3) * FuturA[:, 2]
    FuturA[:, Ny] = (4 / 3) * FuturA[:, Ny - 1] - (1 / 3) * FuturA[:, Ny - 2]

    FuturA[0, :Fy[0]] = (4 / 3) * FuturA[1, :Fy[0]] - (1 / 3) * FuturA[2, :Fy[0]]
    FuturA[0, Fy[1]:] = (4 / 3) * FuturA[1, Fy[1]:] - (1 / 3) * FuturA[2, Fy[1]:]
    FuturA[0, Fy[0]:Fy[1]] = T_ext

    U[:, :, t] = FuturA.copy()

# ------------------------------
# Animation 2D
# ------------------------------
fig, ax = plt.subplots(figsize=(7,6))
heatmap = ax.pcolormesh(X, Y, U[:,:,0].T, shading='nearest', cmap='inferno', vmin=5, vmax=Trad)
cbar = plt.colorbar(heatmap, ax=ax)
cbar.set_label("Température (°C)", fontsize=12)

ax.set_title("Diffusion de la chaleur dans la pièce", fontsize=14, weight='bold')
ax.set_xlabel("x (position)")
ax.set_ylabel("y (position)")
ax.set_aspect('equal')

# Ajouter fenêtre et radiateur
fenetre = Rectangle((0, Fy[0]/Ny), dx, (Fy[1]-Fy[0])/Ny,
                    linewidth=2, edgecolor='cyan', facecolor='none', label="Fenêtre")
ax.add_patch(fenetre)
radiator = Rectangle((Wx[0]/Nx, Wy[0]/Ny),
                     (Wx[1]-Wx[0])/Nx, (Wy[1]-Wy[0])/Ny,
                     linewidth=2, edgecolor='red', facecolor='none', label="Radiateur")
ax.add_patch(radiator)
ax.legend()

# Fonction de mise à jour
def update(frame):
    heatmap.set_array(U[:,:,frame].T.ravel())
    ax.set_title(f"Diffusion de la chaleur (t = {frame*dt:.2f} s)")
    return heatmap,

ani = FuncAnimation(fig, update, frames=Nt, interval=30, blit=False)

plt.show()
