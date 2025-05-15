import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x_data = np.array([1, 2, 3, 4])
y_data = np.array([3, 5, 7, 9])

def cost_function(a, b, x, y):
    m = len(x)
    predictions = a * x + b
    return (1/(2*m)) * np.sum((predictions - y)**2)

# grille 
a_vals = np.linspace(0, 4, 100)
b_vals = np.linspace(-2, 4, 100)

# Calcul du coût combinaison (a,b)
A, B = np.meshgrid(a_vals, b_vals)
J = np.zeros(A.shape)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        J[i,j] = cost_function(A[i,j], B[i,j], x_data, y_data)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(A, B, J, cmap='viridis', alpha=0.8)
fig.colorbar(surf, shrink=0.5, aspect=5)


optimal_idx = np.unravel_index(np.argmin(J), J.shape)
ax.scatter(A[optimal_idx], B[optimal_idx], J[optimal_idx], color='red', s=100, label='Minimum')

ax.set_xlabel('Paramètre a (pente)')
ax.set_ylabel('Paramètre b (biais)')
ax.set_zlabel('Fonction de coût J(a,b)')
ax.set_title('Fonction de Coût pour la Régression Linéaire')
ax.legend()
plt.show()