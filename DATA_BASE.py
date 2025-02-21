import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
import os


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
    t = np.arange(len(spread_signal)) / fs  # Axe temporel a un pas de 1/Fs, qui est Ts
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
    Normalise la puissance du signal selon la formule (14) du document.

    Args:
        signal (np.array): Signal à normaliser.

    Returns:
        np.array: Signal normalisé en puissance.
    avec cette formule 
    """
    power = np.mean(np.abs(signal) ** 2)  # Calcul de la puissance moyenne
    if power == 0:
        return signal  # Éviter la division par zéro
    return signal / power

# ================================
# 5️⃣ Génération du Dataset Complet
# ================================

def generate_dataset(num_samples_per_snr, snr_range, signal_length=2048, spreading_factor=7, save_path="dataset_dsss"):
    """
    Génère une base de données de signaux DSSS et de bruit pur pour l'entraînement d'un CNN.

    Args:
        num_samples_per_snr (int): Nombre d'échantillons par SNR.
        snr_range (tuple): Intervalle du SNR en dB (ex: (-30, 0)).
        signal_length (int): Longueur du signal (2048 par défaut).
        spreading_factor (int): Facteur d'étalement (7 par défaut).
        save_path (str): Dossier de sauvegarde des données.
    """
    dataset = []
    labels = []

    for snr in range(snr_range[0], snr_range[1] + 1, 2):
        for _ in range(num_samples_per_snr):
            # 1️⃣ Générer la séquence d'information
            info_sequence = generate_information_sequence(signal_length // spreading_factor)

            # 2️⃣ Générer un code PN via LFSR et l'étendre
            pn_sequence = generate_pn_sequence_lfsr(spreading_factor,seed=0b0101, taps=[3, 1])
            extended_pn_sequence = np.tile(pn_sequence, (signal_length // spreading_factor) + 1)[:signal_length]

            # 3️⃣ Étaler le signal
            spread_message = np.repeat(info_sequence, spreading_factor+1)[:signal_length] #augumenter la periode du signale 
            spread_signal = spread_spectrum(spread_message, extended_pn_sequence)
            '''
            np.tile : repete le tableau en entier , ca sert a la périodisation 
            np.repeat: repete element par element dans un tableau un nombre de fois donné , ca sert par exemple d'augumenter la duré ou la periode d'un signale 
            '''

            # 4️⃣ Modulation BPSK avec décalage de phase
            modulated_signal, _ = modulate_bpsk_with_phase(spread_signal)

            # 5️⃣ Filtrage en cosinus surélevé
            filter_taps = raised_cosine_filter(beta=0.3, T=1e-3, fs=1e4)
            filtered_signal = apply_filter(modulated_signal, filter_taps)

            # 6️⃣ Ajout de bruit AWGN
            noisy_signal = add_awgn_noise(filtered_signal, snr)

            # 7️⃣ Normalisation
            normalized_signal = normalize_signal(noisy_signal)

            # 8️⃣ Représentation I/Q
            signal_sample = np.stack([np.real(normalized_signal), np.imag(normalized_signal)], axis=-1).reshape(signal_length, 1, 2)
            '''
            

            '''
            # 9️⃣ Ajout au dataset (signal DSSS)
            dataset.append(signal_sample)
            labels.append(1)

            # 🔟 Générer un signal de bruit pur
            noise = add_awgn_noise(np.zeros(signal_length), snr)
            noise_sample = np.stack([np.real(noise), np.imag(noise)], axis=-1).reshape(signal_length, 1, 2)
            dataset.append(noise_sample)
            labels.append(0)

    # ✅ Conversion en tableau numpy et sauvegarde
    dataset = np.array(dataset, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "X.npy"), dataset)
    np.save(os.path.join(save_path, "y.npy"), labels)

    print(f"✅ Dataset généré et sauvegardé dans {save_path}/")
    print(f"  - Nombre total d'échantillons : {dataset.shape[0]}")
    print(f"  - Format des données : {dataset.shape} (2048, 1, 2)")
    print(f"  - Nombre de classes : {np.unique(labels, return_counts=True)}")

# ================================
# Test de génération du Dataset
# ================================

if __name__ == "__main__":
    # Paramètres
    num_samples_per_snr = 1000
    snr_range = (-10, 10)
    signal_length = 2048
    spreading_factor = 7

    # Générer le dataset
    generate_dataset(num_samples_per_snr, snr_range, signal_length, spreading_factor)
