import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# --- Paramètres ---
L = 2.0             # longueur totale du domaine
nx = 1000           # nombre de points
dx = L / nx         # pas d'espace
x = np.linspace(0, L, nx)

V0 = -4000          # profondeur du puits (négatif)
a = 0.1             # largeur du puits
x0 = 0.8            # position du début du puits

# --- Potentiel ---
V = np.zeros(nx)
V[(x >= x0) & (x <= x0 + a)] = V0

# --- Construction du Hamiltonien (différences finies) ---
hbar2_2m = 1.0      # on prend hbar²/2m = 1 en unités réduites
main_diag = hbar2_2m * 2 / dx**2 + V
off_diag = -hbar2_2m / dx**2 * np.ones(nx - 1)

# --- Diagonalisation : résolution Hψ = Eψ ---
# on cherche les n premiers états propres
n_states = 5
energies, wavefuncs = eigh_tridiagonal(main_diag, off_diag, select='i', select_range=(0, n_states - 1))

# --- Normalisation et affichage ---
plt.figure(figsize=(10,6))
for i in range(n_states):
    psi = wavefuncs[:, i]
    psi /= np.sqrt(np.trapz(psi**2, x))  # normalisation
    plt.plot(x, psi + energies[i], label=f'n={i}, E={energies[i]:.2f}')
plt.plot(x, V, 'k--', label='Potentiel V(x)')
plt.xlabel("x")
plt.ylabel("Énergie / fonction d'onde")
plt.title("États stationnaires d'un puits de potentiel fini")
plt.legend()
plt.grid()
plt.show()
