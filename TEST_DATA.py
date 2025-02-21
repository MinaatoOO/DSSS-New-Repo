import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Charger les données
X = np.load("dataset_dsss/X.npy")
y = np.load("dataset_dsss/y.npy")

print(f"X shape : {X.shape}")  # (nombre_samples, 2048, 1, 2)
print(f"y shape : {y.shape}")  # (nombre_samples, )

# 2️⃣ Sélectionner un échantillon DSSS et un bruit
idx_dsss = np.where(y == 1)[0][0]  # Premier échantillon DSSS
idx_noise = np.where(y == 0)[0][1]  # Premier échantillon bruit



# Extraire les signaux
signal_dsss = X[idx_dsss]
signal_noise = X[idx_noise]
print(f"avant reshape signal_dsss et signal_noise sont de la forme {signal_dsss.shape} et {signal_noise.shape}")
signal_dsss = X[idx_dsss].reshape(-1, 2)  # (2048, 2)
signal_noise = X[idx_noise].reshape(-1, 2)
print(f"avant reshape signal_dsss et signal_noise sont de la forme {signal_dsss.shape} et {signal_noise.shape}")

I=signal_dsss[:,0]
I_bruit=signal_noise[:,0]

# Création des subplots

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 1️⃣ Affichage du Modulated Signal
axs[0].plot(I, color='purple')
axs[0].set_title('I')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

# 2️⃣ Affichage du Filtered Signal
axs[1].plot(I_bruit, color='orange')
axs[1].set_title('I_bruit')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)
# Ajustement et affichage
plt.tight_layout()
plt.show()

