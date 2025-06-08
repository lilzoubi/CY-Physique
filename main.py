import numpy as np
import matplotlib.pyplot as plt

# --- Constantes physiques ---
h_bar = 1.0545718e-34  # Constante de Planck réduite (J·s)
electron_mass = 9.10938356e-31  # Masse de l'électron (kg)
electron_volt = 1.60218e-19  # Conversion 1 eV en joules

# --- Paramètres du potentiel ---
puits_profondeur = 10 * electron_volt  # Profondeur du puits (J)
demi_largeur = 1e-10  # Demi-largeur du puits (m)

# --- Énergie de la particule incidente ---
energie_particule = 50 * electron_volt

# --- Constantes d'onde dans chaque région ---
k_gauche = np.sqrt(2 * electron_mass * energie_particule) / h_bar
k_puits = np.sqrt(2 * electron_mass * (energie_particule + puits_profondeur)) / h_bar
k_droite = k_gauche

# --- Amplitudes ---
A_gauche = 1

# Calcul de A2, B2, A3 selon continuité aux frontières
numerateur = 2j * k_gauche * np.exp(-1j * k_gauche * demi_largeur)
denominateur = (1j * k_puits * (np.exp(-1j * k_puits * demi_largeur)
    - np.exp(3j * k_puits * demi_largeur) * (2j * k_puits / (1j * k_gauche + 1j * k_puits) - 1))
    + 1j * k_gauche * (np.exp(-1j * k_puits * demi_largeur)
    + np.exp(3j * k_puits * demi_largeur) * (2j * k_puits / (1j * k_gauche + 1j * k_puits) - 1)))
A_puits = numerateur / denominateur

B_puits = A_puits * np.exp(2j * k_puits * demi_largeur) * (2j * k_puits / (1j * k_gauche + 1j * k_puits) - 1)
A_droite = (2j * k_puits / (1j * k_gauche + 1j * k_puits)) * A_puits * np.exp(1j * k_puits * demi_largeur - 1j * k_gauche * demi_largeur)
B_gauche = A_gauche - A_puits * np.exp(-2j * k_gauche * demi_largeur) - B_puits * np.exp(-2j * k_gauche * demi_largeur)

# --- Fonctions d'onde par région (réelle uniquement) ---
def onde_gauche(x):  # x < -a
    return np.real(A_gauche * np.exp(1j * k_gauche * x) + B_gauche * np.exp(-1j * k_gauche * x))

def onde_puits(x):   # |x| <= a
    return np.real(A_puits * np.exp(1j * k_puits * x) + B_puits * np.exp(-1j * k_puits * x))

def onde_droite(x):  # x > a
    return np.real(A_droite * np.exp(1j * k_droite * x))

# --- Domaine spatial et potentiel ---
x_vals = np.linspace(-8 * demi_largeur, 8 * demi_largeur, 1000)
potentiel = np.zeros_like(x_vals)
potentiel[np.abs(x_vals) <= demi_largeur] = -puits_profondeur

# --- Tracé ---
plt.figure(figsize=(10, 6))
plt.plot(x_vals, potentiel / puits_profondeur, color='black', label='Potentiel (V)', linewidth=2)
plt.plot(x_vals[x_vals < -demi_largeur], onde_gauche(x_vals[x_vals < -demi_largeur]), label=r'$\psi_1(x)$ (gauche)', color='blue')
plt.plot(x_vals[np.abs(x_vals) <= demi_largeur], onde_puits(x_vals[np.abs(x_vals) <= demi_largeur]), label=r'$\psi_2(x)$ (puits)', color='green')
plt.plot(x_vals[x_vals > demi_largeur], onde_droite(x_vals[x_vals > demi_largeur]), label=r'$\psi_3(x)$ (droite)', color='red')

plt.axvline(-demi_largeur, color='gray', linestyle='--')
plt.axvline(demi_largeur, color='gray', linestyle='--')

plt.title("Fonctions d'onde pour un puits de potentiel carré fini")
plt.xlabel("Position $x$ (m)")
plt.ylabel("Parties réelles de $\\psi(x)$ et $V(x)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
