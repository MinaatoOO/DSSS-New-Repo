import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
import os
import matplotlib.pyplot as plt

'''
Remarque :
-->Le code PN semble aléatoire, mais il est en réalité déterministe : il est généré par un algorithme ou un circuit 
logique (comme un registre à décalage à rétroaction linéaire, ou LFSR).
Il est périodique, c'est-à-dire qu'il se répète après un certain nombre de bits (la période du code).
Sur une période complète, le nombre de 1 et de 0 dans la séquence PN est à peu près égal (à un près).
Cela garantit que le signal a une composante continue nulle.
Les codes PN doivent être faciles à générer à l'aide de circuits simples,
comme des registres à décalage à rétroaction linéaire (LFSR).Ces circuits sont peu coûteux et efficaces
 pour générer des séquences longues et complexes.

'''

# ================================
# 1️⃣ Fonction pour générer le code PN avec LFSR (code PN fixe)
# ================================

def generate_pn_sequence_lfsr(length, seed=1, taps=[3, 2]):
    """
    Génère une séquence pseudo-aléatoire (PN) en utilisant un registre à décalage à rétroaction linéaire (LFSR).
    
    Args:
        length (int): Longueur de la séquence PN souhaitée.
        seed (int): Valeur initiale du registre (doit être non nul).
        taps (list): Positions des registres utilisés pour la rétroaction(opération XOR entre les deux bits).

    Returns:
        np.array: Séquence PN composée de +1 et -1.
    """
    lfsr = seed
    pn_sequence = []

    for _ in range(length):
        # Calcul du bit de rétroaction via XOR des taps
        feedback = 0
        for t in taps:
            feedback ^= (lfsr >> (t - 1)) & 1  # Décale et récupère le bit

        # Récupérer le bit de sortie (dernier bit)
        bit = lfsr & 1 # Seul le bit le plus à droite est conservé, c'est la sortie de registre a decalage 
        pn_sequence.append(1 if bit == 1 else -1)  # Convertit 0 -> -1 pour la modulation

        # Décale le registre et ajoute le bit de rétroaction
        lfsr = (lfsr >> 1) | (feedback << (max(taps) - 1))

    return np.array(pn_sequence)

# ================================
# 2️⃣ Génération des séquences de bits et étalement du spectre
# ================================

def generate_information_sequence(N):
    """
    Génère une séquence aléatoire de bits d'information (+1/-1).

    Args:
        N (int): Nombre de bits à générer.

    Returns:
        np.array: Séquence de bits aléatoires.
    """
    return np.random.choice([-1, 1], size=N)

def spread_spectrum(info_seq, pn_seq):
    """
    Applique l'étalement du spectre en multipliant la séquence d'information par la séquence PN.

    Args:
        info_seq (np.array): Séquence d'information.
        pn_seq (np.array): Séquence PN étendue.

    Returns:
        np.array: Signal étalé.
    """
    return info_seq * pn_seq

# ================================
# 3️⃣ Modulation et traitement du signal
# ================================

def modulate_bpsk_with_phase(spread_signal, fc=2e3, fs=1e4):
    """
    Modulation BPSK avec un décalage de phase aléatoire.

    Args:
        spread_signal (np.array): Signal étalé.
        fc (float): Fréquence porteuse (2 kHz par défaut).
        fs (float): Fréquence d'échantillonnage (10 kHz par défaut).

    Returns:
        tuple: Signal modulé et axe temporel.
    """
    t = np.arange(len(spread_signal)) / fs  # Axe temporel a un pas de 1/Fs
    phase_shift = np.random.uniform(0, 2 * np.pi)  # Décalage de phase aléatoire
    carrier = np.cos(2 * np.pi * fc * t + phase_shift)  # Porteuse avec décalage de phase
    modulated_signal = spread_signal * carrier  # Modulation BPSK
    return modulated_signal, t

def raised_cosine_filter(beta, T, fs, num_taps=101):
    """
    Génère un filtre en cosinus surélevé.

    Args:
        beta (float): Facteur de roll-off.
        T (float): Durée d'un symbole.
        fs (float): Fréquence d'échantillonnage.
        num_taps (int): Nombre de coefficients du filtre.

    Returns:
        np.array: Coefficients du filtre en cosinus surélevé.
    """
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / fs
    h = np.sinc(t / T) * np.cos(np.pi * beta * t / T) / (1 - (2 * beta * t / T) ** 2)
    h[np.isnan(h)] = 0  # Correction de division par zéro
    return h

def apply_filter(signal, filter_taps):
    """
    Applique un filtre à un signal.

    Args:
        signal (np.array): Signal à filtrer.
        filter_taps (np.array): Coefficients du filtre.

    Returns:
        np.array: Signal filtré.
    """
    return lfilter(filter_taps, 1.0, signal)

# ================================
# 4️⃣ Ajout de bruit et normalisation
# ================================

def add_awgn_noise(signal, snr_db):
    """
    Ajoute du bruit AWGN à un signal donné en fonction du SNR.
    Si le signal est nul, génère du bruit pur basé sur la puissance spécifiée.
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)

    if signal_power == 0:
        # Si le signal est nul, générer du bruit pur
        noise_power = 1  # ou ajuste selon la plage souhaitée
    else:
        noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise


def normalize_signal(signal):
    """
    Normalise le signal entre -1 et 1.

    Args:
        signal (np.array): Signal à normaliser.

    Returns:
        np.array: Signal normalisé.
    """
    return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1



signal_length=2048
spreading_factor=7
snr=-15 # c'est en dB

info_sequence=generate_information_sequence(signal_length//spreading_factor)
print(f"au départ on genere un signale dinformation binaire de taille {len(info_sequence)}")

spread_message=np.repeat(info_sequence, spreading_factor+1)[:signal_length]
print(f"aprés avoir appliquer np.repeat, le signale dinformation binaire est de taille {len(spread_message)}")

pn_sequence = generate_pn_sequence_lfsr(spreading_factor,seed=0b0101, taps=[3, 1])
extended_pn_sequence = np.tile(pn_sequence, (signal_length // spreading_factor) + 1)[:signal_length]
print(f"aprés avoir appliquer np.tile sur le PN code, le signale  est de taille {len(extended_pn_sequence)}")

spread_signal = spread_spectrum(spread_message, extended_pn_sequence)
print(f"le signale resultant de la multiplication du pn code et de signale d'information est de taille {len(spread_signal)}")

modulated_signal, _ = modulate_bpsk_with_phase(spread_signal)
print(f"signale modulé en BPSK  est de taille {len(modulated_signal)}")

filter_taps = raised_cosine_filter(beta=0.3, T=1e-3, fs=1e4)
print(f"la variable filter_taps contient : {filter_taps}")

filtered_signal = apply_filter(modulated_signal, filter_taps)
max__value = np.max(filtered_signal)
min__value = np.min(filtered_signal)
print(f"le signale filtré (filtred_signal ) est de taille  : {len(filtered_signal)} et de valeur max et min respectivement : {max__value} et {min__value}")


noisy_signal = add_awgn_noise(filtered_signal, snr)
max_value = np.max(noisy_signal)
min_value = np.min(noisy_signal)
print(f"le signale bruité est de taille  : {len(filtered_signal)} et de valeur max et min respectivement : {max_value} et {min_value}")


normalized_signal = normalize_signal(noisy_signal)


signal_sample = np.stack([np.real(normalized_signal), np.imag(normalized_signal)], axis=-1).reshape(signal_length, 1, 2)

print(f"Shape du signal normalisé : {normalized_signal.shape}")
print(f"Shape du signal I/Q (signal_sample) : {signal_sample.shape}")


#Générer un signal de bruit pur
noise = add_awgn_noise(np.zeros(signal_length), snr)
noise_sample = np.stack([np.real(noise), np.imag(noise)], axis=-1).reshape(signal_length, 1, 2)


#Afichage ::::::

# Tracé BRUIT seul
plt.figure(figsize=(10, 5))
plt.plot(noise,  color='green', label='bruit')
plt.title('Test de generation bruit ')
plt.xlabel('Échantillons')
plt.ylabel('amplitude')
#plt.ylim(-1.5, 1.5)  
plt.legend()
plt.show()

# Calcul de l'autocorrélation du PN sequence
autocorr = np.correlate(pn_sequence, pn_sequence, mode='full')
lags = np.arange(-len(pn_sequence) + 1, len(pn_sequence))

# Création des subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# 1️⃣ Tracé du PN Sequence
axs[0].plot(pn_sequence, drawstyle='steps-pre', marker='o', color='green', label='PN Sequence')
axs[0].set_title('PN Sequence')
axs[0].set_xlabel('Échantillons')
axs[0].set_ylabel('+1 / -1')
axs[0].set_ylim(-1.5, 1.5)
axs[0].grid(True)
axs[0].legend()

# 2️⃣ Tracé de l'Autocorrélation
axs[1].stem(autocorr, basefmt=" ", linefmt='blue', markerfmt='bo', label='Autocorrélation')
axs[1].set_title('Autocorrélation du PN Sequence')
axs[1].set_xlabel('Décalage (lags)')
axs[1].set_ylabel('Valeur de l\'autocorrélation')
axs[1].grid(True)
axs[1].legend()

# Ajustement de l'espacement
plt.tight_layout()
plt.show()


# Création des sous-graphiques
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Affichage du Spread Message
axs[0].plot(spread_message, color='blue', drawstyle='steps-pre')
axs[0].set_title('Spread Message (Séquence d\'information étendue)')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

# Affichage du Extended PN Sequence
axs[1].plot(extended_pn_sequence, color='green', drawstyle='steps-pre')
axs[1].set_title('Extended PN Sequence (Code PN Étendu)')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)

# Affichage du Spread Signal (Produit des deux)
axs[2].plot(spread_signal, color='red', drawstyle='steps-pre')
axs[2].set_title('Spread Signal (Message Étendu avec Code PN)')
axs[2].set_xlabel('Échantillons')
axs[2].set_ylabel('Amplitude')
axs[2].grid(True)

# Ajustement et affichage
plt.tight_layout()
plt.show()


# Création des sous-graphiques
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# 1️⃣ Affichage du Modulated Signal
axs[0].plot(modulated_signal, color='purple')
axs[0].set_title('Modulated Signal (BPSK avec décalage de phase)')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

# 2️⃣ Affichage du Filtered Signal
axs[1].plot(filtered_signal, color='orange')
axs[1].set_title('Filtered Signal (Filtré avec Cosinus Surélevé)')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)

# 3️⃣ Affichage du Noisy Signal
axs[2].plot(noisy_signal, color='red')
axs[2].set_title(f'Noisy Signal (AWGN ajouté, SNR={snr} dB)')
axs[2].set_xlabel('Échantillons')
axs[2].set_ylabel('Amplitude')
axs[2].grid(True)

# Ajustement et affichage
plt.tight_layout()
plt.show()

# Séparation des composantes I et Q
I = signal_sample[:, 0, 0]  # Composante In-Phase
Q = signal_sample[:, 0, 1]  # Composante Quadrature

# Création des subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# 1️⃣ Diagramme en constellation (I/Q)
axs[0].scatter(I, Q, color='purple', s=10, alpha=0.7)
axs[0].set_title('Diagramme en Constellation (I/Q)')
axs[0].set_xlabel('In-Phase (I)')
axs[0].set_ylabel('Quadrature (Q)')
axs[0].grid(True)
axs[0].axis('equal')  # Échelle égale pour I et Q

# 2️⃣ Tracé des composantes I et Q séparées
axs[1].plot(I, label='Composante I (In-Phase)', color='blue')
axs[1].plot(Q, label='Composante Q (Quadrature)', color='red', alpha=0.7)
axs[1].set_title('Composantes I et Q en fonction des échantillons')
axs[1].set_xlabel('Échantillons')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
axs[1].grid(True)

# Ajustement et affichage
plt.tight_layout()
plt.show()