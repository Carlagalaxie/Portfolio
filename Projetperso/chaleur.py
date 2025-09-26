#représentation de la position du radiateur

import matplotlib.pyplot as plt
import numpy as np


m = 40
p = 40
x_vals = np.linspace(0, 1, m)
y_vals = np.linspace(0, 1, p)


omega = np.zeros((p, m), dtype=bool)
#x_start = int(0.9 * m)
x_start = int(0 * m)
#x_end = m
x_end = int(0.1 * m)
y_start = int(0.4 * p)
y_end = int(0.6 * p)
omega[y_start:y_end, x_start:x_end] = True

plt.imshow(omega, extent=[0, 1, 0, 1], origin="lower", cmap="Greys")
plt.title("Position du radiateur")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Radiateur ")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




#Simulation de la diffusion de la chaleur dans une pièce carrée avec un radiateur et une fenêtre

# Maillage
Lx = [0, 1]  # Domaines spatial [0,1] en x
Ly = [0,1]
dx = 1 /40  #pas d'espace en x
dy = 1/40

CFL = 0.45
dt = (dx**2) * CFL / 2     #CFL : λx​+λy​≤CFL cad  2*(dx**2)≤ CFL  Δt≤  CFL/2 * dx**2


T = 5   #temps final
Trad = 40  #température radiateur
T_ext = 5  #température extérieur


Nx = int((Lx[1] - Lx[0]) / dx)  #Nombre de points  calculés en fonction de la taille de la cellule dx.
Ny = int((Lx[1] - Lx[0]) / dx)   #même nombre  car dx=dy
Nt = int(T // dt)   # // donne un résultat entier

Lambda = dt / dx**2

Fy = [int(0.4 * Ny), int(0.6 * Ny)]  #Indices correspondant à la fenêtre en y

X = np.linspace(Lx[0], Lx[1], Nx + 1)
Y = np.linspace(Ly[0], Ly[1], Ny + 1)

c = 1 #position de radiateur à choisir

# Position du radiateur selon c
if c == 1:
    Wx = [int(0.9 * Nx), int(1 * Nx)]
    Wy = [int(0.4 * Ny), int(0.6 * Ny)]
elif c == 2:
    Wx = [int(0.45 * Nx), int(0.55 * Nx)]
    Wy = [int(0.4 * Ny), int(0.6 * Ny)]
elif c == 3:
    Wx = [int(0 * Nx), int(0.1 * Nx)]
    Wy = [int(0.4 * Ny), int(0.6 * Ny)]


# Fonction température initiale
def f0(x, y):
    if x == 0 and abs(y - 0.5) <= 0.1:     #température initiale =5 degré sous la fenetre =20 sinon
        return T_ext
    else:
        return 20

# Matrices de calcul
U = np.zeros((Nx + 1, Ny + 1, Nt + 1), float)   # U tensor 3D stockant la température pour toutes les positions (x,y) à chaque instant t.
A = np.zeros((Nx + 1, Ny + 1), float)           # A matrice Température à l'instant initial, définie par f0(x,y)
P = np.zeros((Nx + 1, Ny + 1), float)           # P matrice Terme source dû au radiateur

FuturA = np.zeros((Nx + 1, Ny + 1), float)  #FuturA matrice qui contient les températures pour le prochain instant


# Initialisation
for i in range(Nx + 1):
    for j in range(Ny + 1):
        A[i, j] = f0(X[i], Y[j])      #mets la matrice au condition initiale

        if c != 0 and Wx[0] <= i < Wx[1] and Wy[0] <= j < Wy[1]:    #Si un radiateur est présent (c≠0)
            P[i, j] = (Trad - A[i, j])**3                           #Wx et Wy définissent la zone ω où le radiateur agit
U[:, :, 0] = A.copy()                                               #Si le point (i,j) appartient à cette zone, le terme source P[i,j] est mis à jour par la formule


# Boucle temporelle
for t in range(1, Nt + 1):
    A = U[:, :, t - 1].copy()  #On copie les valeurs de température de l'instant précédent t−1 dans une matrice temporaire A
    P.fill(0)                  #réinitialisation à à zéro

    if c != 0:

        for i in range(Wx[0], Wx[1]):         #on parcourt les indices correspondant à la région ω (zone occupée par le radiateur).
            for j in range(Wy[0], Wy[1]):
                P[i, j] = (Trad - A[i, j])**3



    FuturA[1:Nx, 1:Ny] = (
        (1 - 4 * Lambda) * A[1:Nx, 1:Ny]
        + Lambda * (A[0:Nx-1, 1:Ny] + A[2:Nx+1, 1:Ny] + A[1:Nx, 0:Ny-1] + A[1:Nx, 2:Ny+1])      # discrétisation explicite : voir schéma TP2 dans le cas lambda_x = lambda_y
        + dt * P[1:Nx, 1:Ny]                                                                    # FuturA[i,j]= (1−4λ)A[i,j] + λ(A[i−1,j] + A[i+1,j] + A[i,j−1] + A[i,j+1]) + Δt⋅P[i,j]
    )


    # Conditions aux limites

    #à l'ordre 1:  (le point est égale à son voisin)

    #FuturA[Nx, :] = FuturA[Nx - 1, :] # Bord droit (x = Nx)
    #FuturA[:, 0] = FuturA[:, 1]       # Bord gauche (y = 0)
    #FuturA[:, Ny] = FuturA[:, Ny - 1] # Bord supérieur (y = Ny)

    # Fenêtre (x = 0, entre Fy[0] et Fy[1])
    #FuturA[0, :Fy[0]] = FuturA[1, :Fy[0]]
    #FuturA[0, Fy[1]:] = FuturA[1, Fy[1]:]
    #FuturA[0, Fy[0]:Fy[1]] = T_ext


    # à l'ordre 2 :

    FuturA[Nx, :] = (4 / 3) * FuturA[Nx - 1, :] - (1 / 3) * FuturA[Nx - 2, :]       #bord droit  pour x = Nx, qui représente la colonne la plus à droite de la grille
    FuturA[:, 0] = (4 / 3) * FuturA[:, 1] - (1 / 3) * FuturA[:, 2]                  #bord gauche
    FuturA[:, Ny] = (4 / 3) * FuturA[:, Ny - 1] - (1 / 3) * FuturA[:, Ny - 2]       #bord supérieur pour y=Ny

    # Fenêtre (x = 0, entre Fy[0] et Fy[1])
    FuturA[0, :Fy[0]] = (4 / 3) * FuturA[1, :Fy[0]] - (1 / 3) * FuturA[2, :Fy[0]]
    FuturA[0, Fy[1]:] = (4 / 3) * FuturA[1, Fy[1]:] - (1 / 3) * FuturA[2, Fy[1]:]
    FuturA[0, Fy[0]:Fy[1]] = T_ext

    U[:, :, t] = FuturA.copy()

    



#visualisation 

def visualize_temperature(U, X, Y):
    fig = plt.figure(figsize=(18, 8))

    # Carte de chaleur (Heatmap)
    ax1 = fig.add_subplot(121)
    heatmap = ax1.pcolormesh(X, Y, U[:, :, -1].T, shading='nearest', cmap='inferno')
    ax1.set_title(f"Carte de température à t = {T:.2f}", fontsize=14, weight='bold')
    ax1.set_xlabel("x (position)", fontsize=12)
    ax1.set_ylabel("y (position)", fontsize=12)
    cbar = fig.colorbar(heatmap, ax=ax1)
    cbar.set_label("Température (°C)", fontsize=12)
    ax1.set_aspect('equal')

    # Ajout des contours pour une meilleure distinction
    contour = ax1.contour(X, Y, U[:, :, -1].T, levels=10, colors='white', linewidths=0.5)
    ax1.clabel(contour, inline=True, fontsize=8, fmt="%.1f")

    # Vue 3D (Surface plot)
    ax2 = fig.add_subplot(122, projection='3d')
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    surface = ax2.plot_surface(
        X_mesh, Y_mesh, U[:, :, -1].T,
        cmap='inferno', edgecolor='none', alpha=0.9
    )
    ax2.set_title(f"Vue 3D de la température à t = {T:.2f} (CFL = 0.45)", fontsize=14, weight='bold')
    ax2.set_xlabel("x (position)", fontsize=12)
    ax2.set_ylabel("y (position)", fontsize=12)
    ax2.set_zlabel("Température (°C)", fontsize=12)
    fig.colorbar(surface, ax=ax2, shrink=0.6, aspect=10, label="Température (°C)")

    # Ajustements pour une meilleure lisibilité
    ax2.view_init(elev=30, azim=135)  # Angle de vue
    ax2.grid(True)
    ax2.set_box_aspect([1, 1, 0.6])  # Aspect proportionnel

    plt.tight_layout()
    plt.show()

# Visualisation améliorée
visualize_temperature(U, X, Y)

#calcul de la température moyenne:
temperature_moy= sum([U[i,j,-1]/(Nx*Ny) for i in range(0,Nx+1) for j in range(0,Ny+1)])
print(f"La valeur moyenne de la température est : { temperature_moy} degré")

ecart_type= np.sqrt((1/(Nx*Ny))*sum([(U[i,j,-1]-temperature_moy)**2 for i in range(0,Nx+1) for j in range(0,Ny+1)]))  #La formule de l'écart type est sigma =sqrt( (1/n)* ∑(xi​−moyenne)**2 )

print("l'ecart type est:", ecart_type)
