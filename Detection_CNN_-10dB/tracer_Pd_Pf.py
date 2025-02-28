import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Charger le modèle entraîné
model = tf.keras.models.load_model("dsss_cnn_model.h5")

# Charger la base de données de test
X_test = np.load("dataset_dsss/X.npy")  # Données d'entrée
y_test = np.load("dataset_dsss/y.npy")  # Labels réels (0 : bruit, 1 : DSSS)

# Liste des SNR utilisés lors de la génération des données
snr_values = np.arange(-30, 12, 2)  # SNR de -30 à 10 dB avec un pas de 2
num_samples_per_snr = 1000  # Nombre d'exemples par SNR

# Prédiction des classes (0 ou 1) sur X_test
y_pred_probs = model.predict(X_test)  # Probabilités (sortie de MLP)
print(y_pred_probs)
y_pred_classes = np.argmax(y_pred_probs, axis=1)  # Classes prédictes(si Pj<Pi on choisi la classe i)
print(y_pred_classes)

# Initialisation des listes pour stocker les métriques
Pd_values = []
Pfa_values = []
F1_values = []

# Calcul des métriques pour chaque SNR
for i, snr in enumerate(snr_values):
    start_idx = i * num_samples_per_snr  # Index de début
    end_idx = start_idx + num_samples_per_snr  # Index de fin

    # Récupérer les labels et prédictions pour ce SNR
    y_true_snr = y_test[start_idx:end_idx]
    y_pred_snr = y_pred_classes[start_idx:end_idx]

    # Calcul des métriques
    TP = np.sum((y_pred_snr == 1) & (y_true_snr == 1))  # Vrai Positifs
    FN = np.sum((y_pred_snr == 0) & (y_true_snr == 1))  # Faux Négatifs
    FP = np.sum((y_pred_snr == 1) & (y_true_snr == 0))  # Faux Positifs
    TN = np.sum((y_pred_snr == 0) & (y_true_snr == 0))  # Vrai Négatifs

    # Calcul Pd et Pfa
    Pd = TP / (TP + FN) if (TP + FN) > 0 else 0  # Probabilité de Détection
    Pfa = FP / (FP + TN) if (FP + TN) > 0 else 0  # Probabilité de Fausse Alarme

    # Calcul du F1-Score
    F1 = f1_score(y_true_snr, y_pred_snr, zero_division=0)

    # Stockage des valeurs
    Pd_values.append(Pd)
    Pfa_values.append(Pfa)
    F1_values.append(F1)

# Tracé des courbes
plt.figure(figsize=(10, 6))
plt.plot(snr_values, Pd_values, marker='o', linestyle='-', label="Probabilité de Détection (Pd)", color='blue')
plt.plot(snr_values, Pfa_values, marker='s', linestyle='-', label="Probabilité de Fausse Alarme (Pfa)", color='red')
plt.plot(snr_values, F1_values, marker='d', linestyle='-', label="F1-Score", color='green')

# Configuration du graphique
plt.xlabel("SNR (dB)")
plt.ylabel("Probabilité / Score")
plt.title("Performance du modèle en fonction du SNR")
plt.legend()
plt.grid()
plt.show()
