import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------
# Paramètres
# ------------------------------
Lx, Ly = [0, 1], [0, 1]
Nx, Ny = 40, 40
dx, dy = (Lx[1]-Lx[0])/Nx, (Ly[1]-Ly[0])/Ny

CFL = 0.45
dt = (dx**2)*CFL/2
T = 5
Trad, T_ext = 40, 5

Nt = int(T//dt)
Lambda = dt/dx**2

# Fenêtre
Fy = [int(0.4*Ny), int(0.6*Ny)]

# Radiateur
c = 2
if c==1:
    Wx = [int(0.9*Nx), Nx]
    Wy = [int(0.4*Ny), int(0.6*Ny)]
elif c==2:
    Wx = [int(0.45*Nx), int(0.55*Nx)]
    Wy = [int(0.4*Ny), int(0.6*Ny)]
elif c==3:
    Wx = [0, int(0.1*Nx)]
    Wy = [int(0.4*Ny), int(0.6*Ny)]

X = np.linspace(Lx[0], Lx[1], Nx+1)
Y = np.linspace(Ly[0], Ly[1], Ny+1)

# ------------------------------
# Initialisation
# ------------------------------
def f0(x,y):
    if x==0 and abs(y-0.5)<=0.1:
        return T_ext
    else:
        return 20

U = np.zeros((Nx+1, Ny+1, Nt+1))
A = np.zeros((Nx+1, Ny+1))
P = np.zeros((Nx+1, Ny+1))
FuturA = np.zeros((Nx+1, Ny+1))

for i in range(Nx+1):
    for j in range(Ny+1):
        A[i,j] = f0(X[i], Y[j])
U[:,:,0] = A.copy()

# Facteur d'accélération du radiateur
facteur_source = 2

# ------------------------------
# Boucle temporelle
# ------------------------------
for t in range(1, Nt+1):
    A = U[:,:,t-1].copy()
    P.fill(0)
    P[Wx[0]:Wx[1], Wy[0]:Wy[1]] = facteur_source*(Trad - A[Wx[0]:Wx[1], Wy[0]:Wy[1]])**3

    FuturA[1:Nx,1:Ny] = (
        (1-4*Lambda)*A[1:Nx,1:Ny] +
        Lambda*(A[0:Nx-1,1:Ny] + A[2:Nx+1,1:Ny] + A[1:Nx,0:Ny-1] + A[1:Nx,2:Ny+1]) +
        dt*P[1:Nx,1:Ny]
    )

    # Conditions aux limites ordre 2
    FuturA[Nx,:] = (4/3)*FuturA[Nx-1,:] - (1/3)*FuturA[Nx-2,:]
    FuturA[:,0] = (4/3)*FuturA[:,1] - (1/3)*FuturA[:,2]
    FuturA[:,Ny] = (4/3)*FuturA[:,Ny-1] - (1/3)*FuturA[:,Ny-2]
    FuturA[0,:Fy[0]] = (4/3)*FuturA[1,:Fy[0]] - (1/3)*FuturA[2,:Fy[0]]
    FuturA[0,Fy[1]:] = (4/3)*FuturA[1,Fy[1]:] - (1/3)*FuturA[2,Fy[1]:]
    FuturA[0,Fy[0]:Fy[1]] = T_ext

    U[:,:,t] = FuturA.copy()

# ------------------------------
# Animation 3D accélérée
# ------------------------------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
X_mesh, Y_mesh = np.meshgrid(X,Y)

ax.set_zlim(T_ext, Trad+5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Température (°C)")

# Affichage initial
surf = [ax.plot_surface(X_mesh, Y_mesh, U[:,:,0].T, cmap='inferno', edgecolor='none')]

# Paramètres pour accélérer l'animation
step = 40  # saute 5 pas de temps à chaque frame
interval_ms = 5  # intervalle entre frames (ms)

def update(frame):
    ax.clear()
    ax.plot_surface(X_mesh, Y_mesh, U[:,:,frame*step].T, cmap='inferno', edgecolor='none')
    ax.set_zlim(T_ext, Trad+5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Température (°C)")
    ax.set_title(f"Diffusion de la chaleur (t={frame*step*dt:.2f}s)")
    return

frames_to_show = Nt//step

ani = FuncAnimation(fig, update, frames=frames_to_show, interval=interval_ms, blit=False)

# ------------------------------
# Sauvegarde de l'animation
# ------------------------------
ani.save("diffusion_chaleur2.gif", writer="pillow", fps=30)  # <- ici, tu sauvegardes en GIF
# ou en MP4 :
# ani.save("diffusion_chaleur.mp4", writer="ffmpeg", fps=30)


plt.show()

