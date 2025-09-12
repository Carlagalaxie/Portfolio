import numpy as np
import matplotlib.pyplot as plt


# Paramètres du problème
Nx, Ny = 50, 50          
Lx, Ly = 1.0, 1.0        
dx, dy = Lx/(Nx-1), Ly/(Ny-1)
dt = 0.001              
Tfinal = 0.1             
nu = 0.01                
lambda_ = 1.0           
Tc = 1.0
k = 50.0
sc = np.array([0.5, 0.5]) # centre de la source

# Vitesse
V = np.array([1.0, 0.5])  # (v1, v2)


# Grille

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Source f(t,s)
def f(X, Y):
    d2 = (X - sc[0])**2 + (Y - sc[1])**2
    return Tc * np.exp(-k * d2)


# Solution exacte pour test (ici on prend 0 pour illustration)
def u_exact(X,Y,t):
    return np.zeros_like(X)


# Initialisation
u = np.zeros((Nx, Ny))      # solution init
u_new = np.zeros_like(u)


# Schéma explicite simple
Nt = int(Tfinal/dt)

for n in range(Nt):
    # Diffusion + réaction
    uxx = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0))/dx**2
    uyy = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1))/dy**2
    laplace_u = uxx + uyy
    
    # Convection
    ux = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0))/(2*dx)
    uy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1))/(2*dy)
    
    u_new = u + dt * (-lambda_ * u + f(X,Y) - (V[0]*ux + V[1]*uy) + nu*laplace_u)
    
    # Cond limites : Dirichlet sur les bords entrants
    if V[0] > 0:
        u_new[0,:] = 0  # bord gauche
    else:
        u_new[-1,:] = 0 # bord droit
    
    if V[1] > 0:
        u_new[:,0] = 0  # bord bas
    else:
        u_new[:,-1] = 0 # bord haut
    
    u[:] = u_new


#Erreur L2

uex = u_exact(X,Y,Tfinal)
err = u - uex
L2_err = np.sqrt(np.sum(err**2)*dx*dy)

# Gradient
ux, uy = np.gradient(u, dx, dy)
grad_norm = np.sqrt(ux**2 + uy**2)

#Graphes
fig, axs = plt.subplots(1, 3, figsize=(18,5))

# Solution
im0 = axs[0].imshow(u, origin='lower', extent=[0,Lx,0,Ly], cmap='viridis')
axs[0].set_title("Solution numérique")
fig.colorbar(im0, ax=axs[0])

# Erreur L2
im1 = axs[1].imshow(err, origin='lower', extent=[0,Lx,0,Ly], cmap='coolwarm')
axs[1].set_title(f"Erreur L2 = {L2_err:.3e}")
fig.colorbar(im1, ax=axs[1])

# Norme du gradient
im2 = axs[2].imshow(grad_norm, origin='lower', extent=[0,Lx,0,Ly], cmap='plasma')
axs[2].set_title("Norme du gradient")
fig.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()
